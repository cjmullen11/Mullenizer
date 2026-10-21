"""Run Mullenizer's three-stage PDF-to-review pipeline.

Render, deskew, and OCR each PDF, then retain up to five distinct pages per
metric ranked by OCR evidence. Only OCR hits in the shortest ranked prefix
needed to reach those pages remain highlightable in the GUI. Reviewers identify and interpret graphs manually.
Each PDF is processed in its own workspace; completed pages remain browsable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# PIPELINE CONSTANTS
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT = "output/mullenizer"
DEFAULT_RENDER_DPI = 200

# The current review design retains at most five areas for each of the three
# targets: AFib burden, average heart rate, and average daily activity.
TOP_CANDIDATES_PER_TARGET = 5

# Keep per-PDF directory components short so long source filenames cannot create
# risky Windows paths. Long names retain a readable prefix plus a deterministic
# hash; duplicate source stems are also bounded after their path-hash suffix.
MAX_JOB_NAME_CHARS = 56
JOB_NAME_HASH_CHARS = 16
DUPLICATE_PATH_HASH_CHARS = 8

# Auto-parallelization leaves approximately one quarter of logical CPU cores
# unused, with at least one core reserved on systems having more than one core.
AUTO_RESERVED_CPU_FRACTION = 0.25
AUTO_MIN_RESERVED_CORES_ON_LARGE_SYSTEM = 2
AUTO_LARGE_SYSTEM_CORE_THRESHOLD = 8
PIPELINE_EVENT_PREFIX = "MULLENIZER_EVENT "

# External libraries used by OCR and image processing may otherwise create
# extra thread pools inside every parallel PDF worker. These limits prevent
# multiplying the chosen worker count by several hidden compute threads.
SUBPROCESS_THREAD_LIMITS = {
    "OMP_THREAD_LIMIT": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

SCRIPT_FILENAMES = {
    "render": "render_pdf.py",
    "deskew": "deskew_pages.py",
    "ocr": "ocr_report_targets.py",
}

BATCH_RESULTS_FILENAME = "batch_results.json"
PIPELINE_LOG_FILENAME = "pipeline.log"
FINAL_RESULTS_FILENAME = "results.json"
DESKEWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def running_frozen() -> bool:
    """Return True when running from a PyInstaller executable."""
    return bool(getattr(sys, "frozen", False))


def resolve_tesseract_command(explicit_command: str | None) -> str | None:
    """
    Resolve Tesseract once before any per-PDF worker changes directories.

    Packaged uses the portable Tesseract copy located beside
    .exe. An explicitly supplied relative path is resolved against the
    launch directory before workers switch into isolated .work directories.

    Source/development execution may still return None so the OCR stage can use
    a normal system-PATH Tesseract installation.
    """
    if explicit_command:
        candidate = Path(explicit_command).expanduser()

        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate

        candidate = candidate.resolve()

        if not candidate.is_file():
            raise FileNotFoundError(
                f"Tesseract executable does not exist: {candidate}"
            )

        return str(candidate)

    if running_frozen():
        candidate = (
            Path(sys.executable).resolve().parent
            / "tesseract"
            / "tesseract.exe"
        ).resolve()

        if not candidate.is_file():
            raise FileNotFoundError(
                "Packaged Mullenizer could find its bundled Tesseract "
                f"executable at: {candidate}"
            )

        return str(candidate)

    return None


def hidden_subprocess_creation_flags() -> int:
    """Hide Windows child consoles and give compute work background priority."""
    if not sys.platform.startswith("win"):
        return 0

    return (
        int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        | int(getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    )


def lower_current_process_priority() -> None:
    """Let interactive desktop work preempt background PDF processing."""
    if sys.platform.startswith("win"):
        try:
            import ctypes

            below_normal_priority_class = 0x00004000
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(
                kernel32.GetCurrentProcess(),
                below_normal_priority_class,
            )
        except Exception:
            pass
        return

    if hasattr(os, "nice"):
        try:
            os.nice(5)
        except OSError:
            pass


def build_stage_command(
    stage_name: str,
    script_path: Path,
    arguments: list[str],
) -> list[str]:
    """Build one pipeline-stage command for source or frozen execution."""
    if running_frozen():
        return [
            sys.executable,
            "--mullenizer-stage",
            stage_name,
            *arguments,
        ]

    return [
        sys.executable,
        str(script_path),
        *arguments,
    ]


# -----------------------------------------------------------------------------
# GENERIC FILE AND JSON HELPERS
# -----------------------------------------------------------------------------


def utc_now_text() -> str:
    """Return the current UTC timestamp in a stable ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()


def emit_pipeline_event(event: dict[str, Any]) -> None:
    """Emit one machine-readable progress event for the GUI."""
    print(
        PIPELINE_EVENT_PREFIX
        + json.dumps(event, separators=(",", ":"), default=str),
        flush=True,
    )


def load_json(json_path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save one readable UTF-8 JSON object."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def remove_path(path: Path) -> None:
    """Remove one file, symlink, or directory when it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink()
        return

    if path.is_dir():
        shutil.rmtree(path)


def safe_slug(text: str) -> str:
    """Convert one source stem into a bounded readable prefix plus hash."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    normalized = normalized.strip("._-") or "report"

    if len(normalized) <= MAX_JOB_NAME_CHARS:
        return normalized

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:JOB_NAME_HASH_CHARS]
    prefix_length = MAX_JOB_NAME_CHARS - JOB_NAME_HASH_CHARS - 1
    return f"{normalized[:prefix_length]}_{digest}"


def short_path_hash(path: Path) -> str:
    """Return a short deterministic hash for distinguishing duplicate source names."""
    return hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:DUPLICATE_PATH_HASH_CHARS]


def make_unique_job_names(pdf_paths: list[Path]) -> list[str]:
    """Create deterministic unique job names without exceeding the component cap."""
    base_names = [safe_slug(pdf_path.stem) for pdf_path in pdf_paths]
    counts: dict[str, int] = {}

    for base_name in base_names:
        counts[base_name] = counts.get(base_name, 0) + 1

    job_names: list[str] = []

    for pdf_path, base_name in zip(pdf_paths, base_names, strict=True):
        if counts[base_name] == 1:
            job_names.append(base_name)
            continue

        suffix = f"_{short_path_hash(pdf_path)}"
        prefix_length = MAX_JOB_NAME_CHARS - len(suffix)
        job_names.append(f"{base_name[:prefix_length]}{suffix}")

    return job_names


def validate_pdf_paths(raw_paths: list[str]) -> list[Path]:
    """Validate and resolve all PDF inputs in the user-supplied order."""
    validated: list[Path] = []

    for raw_path in raw_paths:
        pdf_path = Path(raw_path).expanduser().resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

        if not pdf_path.is_file():
            raise ValueError(f"PDF path is not a file: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input is not a PDF file: {pdf_path}")

        validated.append(pdf_path)

    return validated


def resolve_stage_scripts(script_dir: Path) -> dict[str, Path]:
    """Resolve stage scripts in source mode or module placeholders when frozen."""
    if running_frozen():
        return {
            stage_name: Path(filename)
            for stage_name, filename in SCRIPT_FILENAMES.items()
        }

    resolved: dict[str, Path] = {}

    for stage_name, filename in SCRIPT_FILENAMES.items():
        script_path = (script_dir / filename).resolve()

        if not script_path.exists() or not script_path.is_file():
            raise FileNotFoundError(
                f"Required {stage_name} script is missing: {script_path}"
            )

        resolved[stage_name] = script_path

    return resolved


# -----------------------------------------------------------------------------
# PARALLEL WORKER SELECTION
# -----------------------------------------------------------------------------


def choose_worker_count(
    requested_workers: int,
    pdf_count: int,
) -> tuple[int, dict[str, int]]:
    """Choose a PDF-level worker count while leaving CPU capacity unused."""
    if pdf_count <= 0:
        raise ValueError("At least one PDF is required.")

    cpu_count = os.cpu_count() or 1

    if requested_workers < 0:
        raise ValueError("--workers cannot be negative.")

    if requested_workers > 0:
        worker_count = min(requested_workers, pdf_count)
        reserved_cores = max(0, cpu_count - worker_count)

        return worker_count, {
            "logical_cpu_count": cpu_count,
            "reserved_logical_cores": reserved_cores,
            "worker_count": worker_count,
        }

    if cpu_count <= 1:
        reserved_cores = 0
    else:
        reserved_cores = max(
            1,
            int(math.ceil(cpu_count * AUTO_RESERVED_CPU_FRACTION)),
        )

        if cpu_count >= AUTO_LARGE_SYSTEM_CORE_THRESHOLD:
            reserved_cores = max(
                reserved_cores,
                AUTO_MIN_RESERVED_CORES_ON_LARGE_SYSTEM,
            )

    usable_cores = max(1, cpu_count - reserved_cores)
    worker_count = min(pdf_count, usable_cores)

    return worker_count, {
        "logical_cpu_count": cpu_count,
        "reserved_logical_cores": reserved_cores,
        "worker_count": worker_count,
    }


def make_subprocess_environment() -> dict[str, str]:
    """Build an environment that avoids nested thread oversubscription."""
    environment = os.environ.copy()
    environment.update(SUBPROCESS_THREAD_LIMITS)
    environment["PYTHONUNBUFFERED"] = "1"

    return environment


# -----------------------------------------------------------------------------
# STAGE EXECUTION AND LOGGING
# -----------------------------------------------------------------------------


def append_log(log_path: Path, text: str) -> None:
    """Append plain text to one per-PDF pipeline log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(text)

        if not text.endswith("\n"):
            log_file.write("\n")


def run_stage(
    stage_name: str,
    command: list[str],
    working_dir: Path,
    log_path: Path,
    environment: dict[str, str],
) -> float:
    """Run one stage, append all output to the job log, and return its time."""
    stage_start = time.perf_counter()

    append_log(
        log_path,
        (
            "\n"
            f"{'=' * 80}\n"
            f"Stage: {stage_name}\n"
            f"Started: {utc_now_text()}\n"
            f"Command: {shlex.join(command)}\n"
            f"{'=' * 80}\n"
        ),
    )

    with log_path.open("a", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=working_dir,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            creationflags=hidden_subprocess_creation_flags(),
        )

    elapsed_seconds = time.perf_counter() - stage_start

    append_log(
        log_path,
        (
            f"Stage exit code: {completed.returncode}\n"
            f"Stage seconds: {elapsed_seconds:.3f}\n"
        ),
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage_name} failed with exit code {completed.returncode}. "
            f"See: {log_path}"
        )

    return elapsed_seconds


def read_log_tail(log_path: Path, maximum_lines: int = 40) -> str:
    """Read the final lines of a log for inclusion in a batch failure result."""
    if not log_path.exists():
        return ""

    lines = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    return "\n".join(lines[-maximum_lines:])


# -----------------------------------------------------------------------------
# RETAINED CANDIDATE AND PAGE COLLECTION
# -----------------------------------------------------------------------------


def resolve_job_file(stored_path: str, job_root: Path) -> Path:
    """Resolve a path written by a stage running inside one job directory."""
    path = Path(stored_path).expanduser()

    if not path.is_absolute():
        path = job_root / path

    resolved = path.resolve()

    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(
            f"Referenced pipeline file does not exist: {stored_path} "
            f"(resolved as {resolved})"
        )

    return resolved


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float]:
    """Sort reviewable candidates by explicit rank, then descending score."""
    raw_rank = candidate.get("review_rank")
    rank = int(raw_rank) if raw_rank is not None else 1_000_000
    score = float(candidate.get("score", 0.0))

    return rank, -score


def retained_ocr_candidates(
    ocr_results: dict[str, Any],
    target_key: str,
    source_field: str = "candidates",
) -> list[dict[str, Any]]:
    """Return up to five ranked review pages from a target's OCR results."""
    raw_candidates = ocr_results.get(target_key, {}).get(source_field, [])
    candidates = [
        candidate for candidate in raw_candidates
        if candidate.get("review_crop_bbox") and candidate.get("highlight_regions")
    ]
    return sorted(candidates, key=candidate_sort_key)[:TOP_CANDIDATES_PER_TARGET]


def candidate_identity(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Build a stable identity for matching one OCR selected result."""
    return (
        candidate.get("page_number"),
        candidate.get("candidate_type"),
        candidate.get("value_kind"),
        candidate.get("display_value"),
        candidate.get("value_percent"),
        candidate.get("birth_year"),
        candidate.get("initials"),
        candidate.get("first_initial"),
        candidate.get("last_initial"),
        tuple(candidate.get("label_bbox") or []),
        tuple(candidate.get("value_bbox") or []),
    )


def copy_all_deskewed_pages(
    job_root: Path,
    final_stage: Path,
    retained_pages: dict[Path, str],
) -> None:
    """Copy every deskewed page into the final result for GUI manual browsing."""
    source_dir = job_root / "intermediates" / "deskewed_pages"

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Deskewed page directory does not exist: {source_dir}"
        )

    page_sources = sorted(
        (
            path.resolve()
            for path in source_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in DESKEWED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )

    if not page_sources:
        raise RuntimeError(
            f"Deskew stage produced no page images in: {source_dir}"
        )

    destination_dir = final_stage / "deskewed_pages"
    destination_dir.mkdir(parents=True, exist_ok=True)

    for page_source in page_sources:
        relative_path = Path("deskewed_pages") / page_source.name
        shutil.copy2(page_source, final_stage / relative_path)
        retained_pages[page_source] = relative_path.as_posix()


def copy_candidate_page(
    candidate: dict[str, Any],
    job_root: Path,
    final_stage: Path,
    retained_pages: dict[Path, str],
) -> dict[str, Any]:
    """Copy one source deskewed page and rewrite the candidate's final path."""
    candidate_copy = copy.deepcopy(candidate)
    page_source = resolve_job_file(
        stored_path=str(candidate["image_path"]),
        job_root=job_root,
    )

    if page_source not in retained_pages:
        page_relative_path = Path("deskewed_pages") / page_source.name
        page_destination = final_stage / page_relative_path
        page_destination.parent.mkdir(parents=True, exist_ok=True)

        # The full deskewed page is the only retained candidate image artifact.
        shutil.copy2(
            page_source,
            page_destination,
        )

        retained_pages[page_source] = page_relative_path.as_posix()

    candidate_copy["image_path"] = retained_pages[page_source]

    # Remove legacy fields if older stage files are accidentally mixed in.
    for legacy_field in (
        "review_crop_path",
        "label_bbox_in_crop",
        "value_bbox_in_crop",
    ):
        candidate_copy.pop(legacy_field, None)

    return candidate_copy


def rewrite_selected_ocr(
    selected_candidate: dict[str, Any] | None,
    retained_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return one selected OCR result using retained final artifact paths."""
    if selected_candidate is None:
        return None

    selected_identity = candidate_identity(selected_candidate)

    for retained_candidate in retained_candidates:
        if candidate_identity(retained_candidate) == selected_identity:
            return copy.deepcopy(retained_candidate)

    # A different OCR hit on the same selected page may provide the extracted
    # demographic value. Preserve its fields but use the retained page path.
    for retained_candidate in retained_candidates:
        if retained_candidate["page_number"] == selected_candidate["page_number"]:
            selected_copy = copy.deepcopy(selected_candidate)
            selected_copy["image_path"] = retained_candidate["image_path"]
            return selected_copy

    # Preserve the extracted value even if its page fell outside review slots.
    selected_copy = copy.deepcopy(selected_candidate)
    selected_copy["image_path"] = None

    return selected_copy


def collect_final_outputs(
    source_pdf: Path,
    job_root: Path,
    final_stage: Path,
    ocr_json_path: Path,
    stage_seconds: dict[str, float],
    total_seconds: float,
) -> dict[str, Any]:
    """Collect saved pages and OCR results into one final review manifest.

    Retain up to five distinct pages per metric, with selected OCR highlights.
    """
    ocr_results = load_json(ocr_json_path)

    retained_pages: dict[Path, str] = {}
    copy_all_deskewed_pages(
        job_root=job_root,
        final_stage=final_stage,
        retained_pages=retained_pages,
    )

    patient_name_candidates = [
        copy_candidate_page(
            candidate=candidate,
            job_root=job_root,
            final_stage=final_stage,
            retained_pages=retained_pages,
        )
        for candidate in retained_ocr_candidates(
            ocr_results,
            "patient_name",
        )
    ]

    date_of_birth_candidates = [
        copy_candidate_page(
            candidate=candidate,
            job_root=job_root,
            final_stage=final_stage,
            retained_pages=retained_pages,
        )
        for candidate in retained_ocr_candidates(
            ocr_results,
            "date_of_birth",
        )
    ]

    afib_candidates = [
        copy_candidate_page(
            candidate=candidate,
            job_root=job_root,
            final_stage=final_stage,
            retained_pages=retained_pages,
        )
        for candidate in retained_ocr_candidates(ocr_results, "afib_burden")
    ]

    heart_rate_candidates = [
        copy_candidate_page(
            candidate=candidate,
            job_root=job_root,
            final_stage=final_stage,
            retained_pages=retained_pages,
        )
        for candidate in retained_ocr_candidates(
            ocr_results,
            "average_heart_rate",
            source_field="guides",
        )
    ]

    activity_candidates = [
        copy_candidate_page(
            candidate=candidate,
            job_root=job_root,
            final_stage=final_stage,
            retained_pages=retained_pages,
        )
        for candidate in retained_ocr_candidates(
            ocr_results,
            "average_daily_activity",
            source_field="guides",
        )
    ]

    selected_patient_name = rewrite_selected_ocr(
        selected_candidate=ocr_results.get("patient_name", {}).get("selected"),
        retained_candidates=patient_name_candidates,
    )
    selected_date_of_birth = rewrite_selected_ocr(
        selected_candidate=ocr_results.get("date_of_birth", {}).get("selected"),
        retained_candidates=date_of_birth_candidates,
    )
    selected_afib = rewrite_selected_ocr(
        selected_candidate=ocr_results.get("afib_burden", {}).get("selected"),
        retained_candidates=afib_candidates,
    )

    retained_page_paths = sorted(retained_pages.values())

    results = {
        "pipeline_version": "Mullenizer 1.0",
        "completed_at_utc": utc_now_text(),
        "source_pdf": str(source_pdf),
        "source_pdf_name": source_pdf.name,
        "top_candidates_per_target": TOP_CANDIDATES_PER_TARGET,
        "pipeline_seconds": round(total_seconds, 6),
        "stage_seconds": {
            stage_name: round(seconds, 6)
            for stage_name, seconds in stage_seconds.items()
        },
        "patient_name": {
            "status": (
                "found"
                if selected_patient_name is not None
                else "not_found"
            ),
            "selected": selected_patient_name,
            "candidate_count": len(patient_name_candidates),
            "candidates": patient_name_candidates,
        },
        "date_of_birth": {
            "status": (
                "found"
                if selected_date_of_birth is not None
                else "not_found"
            ),
            "selected": selected_date_of_birth,
            "candidate_count": len(date_of_birth_candidates),
            "candidates": date_of_birth_candidates,
        },
        "afib_burden": {
            "status": (
                "found"
                if selected_afib is not None
                else "not_found"
            ),
            "selected": selected_afib,
            "candidate_count": len(afib_candidates),
            "candidates": afib_candidates,
        },
        "average_heart_rate": {
            "status": (
                "candidates_found"
                if heart_rate_candidates
                else "not_found"
            ),
            "candidate_count": len(heart_rate_candidates),
            "candidates": heart_rate_candidates,
        },
        "average_daily_activity": {
            "status": (
                "candidates_found"
                if activity_candidates
                else "not_found"
            ),
            "candidate_count": len(activity_candidates),
            "candidates": activity_candidates,
        },
        "deskewed_page_count": len(retained_page_paths),
        "deskewed_pages": retained_page_paths,
        "retained_deskewed_page_count": len(retained_page_paths),
        "retained_deskewed_pages": retained_page_paths,
    }

    save_json(
        data=results,
        output_path=final_stage / FINAL_RESULTS_FILENAME,
    )

    return results


# -----------------------------------------------------------------------------
# FINAL DIRECTORY PUBLICATION AND DEBUG RETENTION
# -----------------------------------------------------------------------------


def publish_final_directory(
    final_stage: Path,
    final_destination: Path,
) -> None:
    """Replace one prior completed result only after the new result is ready."""
    final_destination.parent.mkdir(parents=True, exist_ok=True)

    if final_destination.exists() or final_destination.is_symlink():
        remove_path(final_destination)

    # move() is normally a rename because staging and output share one root.
    shutil.move(
        str(final_stage),
        str(final_destination),
    )


def retain_debug_workspace(
    intermediate_root: Path,
    final_stage: Path,
) -> None:
    """Move the complete intermediate workspace into the final result folder."""
    debug_destination = final_stage / "intermediates"

    if debug_destination.exists():
        remove_path(debug_destination)

    shutil.move(
        str(intermediate_root),
        str(debug_destination),
    )


def preserve_failed_debug_workspace(
    job_root: Path,
    output_root: Path,
    job_name: str,
) -> str:
    """Publish a failed workspace only when debug retention was requested."""
    failed_destination = output_root / f"{job_name}_failed_debug"

    if failed_destination.exists():
        remove_path(failed_destination)

    shutil.move(
        str(job_root),
        str(failed_destination),
    )

    return str(failed_destination)


# -----------------------------------------------------------------------------
# ONE-PDF WORKER
# -----------------------------------------------------------------------------


def process_one_pdf(job: dict[str, Any]) -> dict[str, Any]:
    """Run rendering, deskewing and OCR for one PDF in an isolated workspace.

    Publish completed results and keep intermediates only when requested.
    """
    lower_current_process_priority()
    source_pdf = Path(job["source_pdf"])
    job_name = str(job["job_name"])
    output_root = Path(job["output_root"])
    script_paths = {
        stage_name: Path(path_text)
        for stage_name, path_text in job["script_paths"].items()
    }
    render_dpi = int(job["render_dpi"])
    keep_debug = bool(job["keep_debug"])
    tesseract_cmd = job.get("tesseract_cmd")

    job_start = time.perf_counter()
    work_parent = output_root / ".work"
    job_root = work_parent / job_name
    intermediate_root = job_root / "intermediates"
    final_stage = job_root / "final"
    final_destination = output_root / job_name
    log_path = job_root / PIPELINE_LOG_FILENAME

    # Remove only stale temporary work from a prior interrupted run. A previous
    # completed final directory remains untouched until the replacement succeeds.
    if job_root.exists():
        remove_path(job_root)

    intermediate_root.mkdir(parents=True, exist_ok=True)
    final_stage.mkdir(parents=True, exist_ok=True)

    environment = make_subprocess_environment()
    stage_seconds: dict[str, float] = {}

    relative_pages_dir = Path("intermediates/pages")
    relative_deskewed_dir = Path("intermediates/deskewed_pages")
    relative_deskew_debug_dir = Path("intermediates/deskew_debug")
    relative_ocr_json = Path("intermediates/ocr_targets.json")

    try:
        append_log(
            log_path,
            (
                f"Mullenizer 1.0 pipeline\n"
                f"PDF: {source_pdf}\n"
                f"Job: {job_name}\n"
                f"Started: {utc_now_text()}\n"
                f"Keep debug: {keep_debug}\n"
            ),
        )

        render_command = build_stage_command(
            stage_name="render",
            script_path=script_paths["render"],
            arguments=[
                str(source_pdf),
                "--out-dir",
                relative_pages_dir.as_posix(),
                "--dpi",
                str(render_dpi),
            ],
        )

        stage_seconds["render"] = run_stage(
            stage_name="render",
            command=render_command,
            working_dir=job_root,
            log_path=log_path,
            environment=environment,
        )

        deskew_command = build_stage_command(
            stage_name="deskew",
            script_path=script_paths["deskew"],
            arguments=[
                relative_pages_dir.as_posix(),
                "--out-dir",
                relative_deskewed_dir.as_posix(),
                "--debug-dir",
                relative_deskew_debug_dir.as_posix(),
            ],
        )

        if keep_debug:
            deskew_command.append("--debug")

        stage_seconds["deskew"] = run_stage(
            stage_name="deskew",
            command=deskew_command,
            working_dir=job_root,
            log_path=log_path,
            environment=environment,
        )

        if not keep_debug:
            # Rendered pages are no longer needed after deskewing. Removing this
            # directory now bounds temporary disk use during a large PDF batch.
            remove_path(job_root / relative_pages_dir)

        ocr_command = build_stage_command(
            stage_name="ocr",
            script_path=script_paths["ocr"],
            arguments=[
                relative_deskewed_dir.as_posix(),
                "--out",
                relative_ocr_json.as_posix(),
                "--top-guides",
                str(TOP_CANDIDATES_PER_TARGET),
                "--top-afib-candidates",
                str(TOP_CANDIDATES_PER_TARGET),
                "--top-demographic-candidates",
                str(TOP_CANDIDATES_PER_TARGET),
            ],
        )

        if tesseract_cmd:
            ocr_command.extend(
                [
                    "--tesseract-cmd",
                    str(tesseract_cmd),
                ]
            )

        stage_seconds["ocr"] = run_stage(
            stage_name="ocr",
            command=ocr_command,
            working_dir=job_root,
            log_path=log_path,
            environment=environment,
        )

        total_seconds = time.perf_counter() - job_start

        final_results = collect_final_outputs(
            source_pdf=source_pdf,
            job_root=job_root,
            final_stage=final_stage,
            ocr_json_path=job_root / relative_ocr_json,
            stage_seconds=stage_seconds,
            total_seconds=total_seconds,
        )

        # Copy the complete stage log into the final user-facing result folder.
        shutil.copy2(
            log_path,
            final_stage / PIPELINE_LOG_FILENAME,
        )

        if keep_debug:
            # Relative paths recorded by the stage JSON files remain valid
            # because the workspace is retained under final/intermediates/.
            retain_debug_workspace(
                intermediate_root=intermediate_root,
                final_stage=final_stage,
            )

        publish_final_directory(
            final_stage=final_stage,
            final_destination=final_destination,
        )

        # All normal intermediates have either been deleted or moved into the
        # final debug folder. Remove the now-empty isolated work directory.
        if job_root.exists():
            remove_path(job_root)

        return {
            "status": "success",
            "job_name": job_name,
            "source_pdf": str(source_pdf),
            "output_dir": str(final_destination),
            "pipeline_seconds": round(total_seconds, 6),
            "candidate_counts": {
                "patient_name": final_results["patient_name"][
                    "candidate_count"
                ],
                "date_of_birth": final_results["date_of_birth"][
                    "candidate_count"
                ],
                "afib_burden": final_results["afib_burden"][
                    "candidate_count"
                ],
                "average_heart_rate": final_results[
                    "average_heart_rate"
                ]["candidate_count"],
                "average_daily_activity": final_results[
                    "average_daily_activity"
                ]["candidate_count"],
            },
            "retained_deskewed_page_count": final_results[
                "retained_deskewed_page_count"
            ],
        }

    except Exception as error:
        append_log(
            log_path,
            (
                "\n"
                f"{'!' * 80}\n"
                f"Pipeline failed: {utc_now_text()}\n"
                f"{type(error).__name__}: {error}\n"
                f"{traceback.format_exc()}\n"
                f"{'!' * 80}\n"
            ),
        )

        log_tail = read_log_tail(log_path)
        failed_debug_dir: str | None = None

        if keep_debug and job_root.exists():
            failed_debug_dir = preserve_failed_debug_workspace(
                job_root=job_root,
                output_root=output_root,
                job_name=job_name,
            )
        elif job_root.exists():
            # Normal operation deletes partial rendered, deskewed, OCR, and
            # candidate files even after a failure.
            remove_path(job_root)

        return {
            "status": "failed",
            "job_name": job_name,
            "source_pdf": str(source_pdf),
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_debug_dir": failed_debug_dir,
            "log_tail": log_tail,
        }


# -----------------------------------------------------------------------------
# BATCH ORCHESTRATION
# -----------------------------------------------------------------------------


def run_pdf_batch(
    pdf_paths: list[Path],
    job_names: list[str],
    output_root: Path,
    script_paths: dict[str, Path],
    worker_count: int,
    render_dpi: int,
    keep_debug: bool,
    tesseract_cmd: str | None,
) -> list[dict[str, Any]]:
    """Process PDFs in parallel while publishing each result when ready."""
    jobs = [
        {
            "source_pdf": str(pdf_path),
            "job_name": job_name,
            "output_root": str(output_root),
            "script_paths": {
                stage_name: str(script_path)
                for stage_name, script_path in script_paths.items()
            },
            "render_dpi": render_dpi,
            "keep_debug": keep_debug,
            "tesseract_cmd": tesseract_cmd,
        }
        for pdf_path, job_name in zip(pdf_paths, job_names, strict=True)
    ]
    batch_position_by_job = {
        job_name: index
        for index, job_name in enumerate(job_names, start=1)
    }
    results: list[dict[str, Any]] = []

    # Each PDF owns its own work directory. A completed success has already moved
    # its immutable final directory into place before this parent receives it.
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {
            executor.submit(process_one_pdf, job): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]

            try:
                result = future.result()
            except Exception as error:
                result = {
                    "status": "failed",
                    "job_name": job["job_name"],
                    "source_pdf": job["source_pdf"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_debug_dir": None,
                    "log_tail": "",
                }

            results.append(result)
            job_name = str(result["job_name"])
            event = {
                "event": "job_finished",
                "status": result["status"],
                "job_name": job_name,
                "batch_position": batch_position_by_job[job_name],
                "source_pdf": result.get("source_pdf"),
            }

            if result["status"] == "success":
                counts = result["candidate_counts"]
                event.update(
                    {
                        "output_dir": result["output_dir"],
                        "pipeline_seconds": result["pipeline_seconds"],
                        "candidate_counts": counts,
                    }
                )
                emit_pipeline_event(event)
                print(
                    f"Completed {job_name}: "
                    f"AFib={counts['afib_burden']}, "
                    f"HR={counts['average_heart_rate']}, "
                    f"activity={counts['average_daily_activity']}, "
                    f"pages={result['retained_deskewed_page_count']}, "
                    f"{result['pipeline_seconds']:.1f}s",
                    flush=True,
                )
            else:
                event.update(
                    {
                        "error_type": result.get("error_type"),
                        "error": result.get("error"),
                    }
                )
                emit_pipeline_event(event)
                print(
                    f"FAILED {job_name}: "
                    f"{result['error_type']}: {result['error']}",
                    file=sys.stderr,
                    flush=True,
                )

    result_order = {
        job_name: index
        for index, job_name in enumerate(job_names)
    }
    results.sort(key=lambda result: result_order[result["job_name"]])
    return results


# -----------------------------------------------------------------------------
# COMMAND-LINE ENTRY POINT
# -----------------------------------------------------------------------------


def main() -> None:
    """Parse batch CLI arguments and run the three-stage PDF processing pipeline.

    Validate sources, manage worker counts and report the final batch status.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the parallel Mullenizer 1.0 OCR-first candidate pipeline."
        ),
    )

    parser.add_argument(
        "pdfs",
        nargs="+",
        help="One or more input PDF files.",
    )

    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory containing one compact result folder per PDF. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_RENDER_DPI,
        help=f"PDF rendering resolution. Default: {DEFAULT_RENDER_DPI}",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of PDFs processed concurrently. "
            "Use 0 for automatic CPU-aware selection. Default: 0"
        ),
    )

    parser.add_argument(
        "--keep-debug",
        action="store_true",
        help=(
            "Enable deskew debug overlays and retain all intermediate files "
            "inside each completed result directory."
        ),
    )

    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help=(
            "Optional explicit path to the Tesseract executable when it is "
            "not available on PATH."
        ),
    )

    args = parser.parse_args()

    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")

    # Validate all source PDFs before starting the parallel worker pool.
    pdf_paths = validate_pdf_paths(args.pdfs)

    # Resolve Tesseract before the worker pool changes into per-PDF work
    # directories. Every OCR child therefore receives one absolute path.
    tesseract_cmd = resolve_tesseract_command(args.tesseract_cmd)

    # Resolve the directory containing this main pipeline script.
    script_dir = Path(__file__).resolve().parent

    # Verify that all four required stage files are installed beside this file.
    script_paths = resolve_stage_scripts(script_dir)

    # Create the batch output root before workers create their isolated folders.
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Assign stable and collision-free output directory names to the input PDFs.
    job_names = make_unique_job_names(pdf_paths)

    # Choose PDF-level parallelism while leaving some cores for other tasks.
    worker_count, worker_details = choose_worker_count(
        requested_workers=args.workers,
        pdf_count=len(pdf_paths),
    )

    print(
        f"Processing {len(pdf_paths)} PDF(s) with {worker_count} worker(s) "
        f"on {worker_details['logical_cpu_count']} logical CPU core(s); "
        f"reserving approximately "
        f"{worker_details['reserved_logical_cores']} core(s)."
    )
    print(f"Output root: {output_root}")
    print(
        "Tesseract: "
        + (tesseract_cmd if tesseract_cmd is not None else "system PATH")
    )
    emit_pipeline_event(
        {
            "event": "batch_started",
            "total_pdf_count": len(pdf_paths),
            "worker_count": worker_count,
            "logical_cpu_count": worker_details["logical_cpu_count"],
            "reserved_logical_cores": worker_details["reserved_logical_cores"],
        }
    )
    print()

    batch_start = time.perf_counter()

    # Execute complete per-PDF pipelines concurrently.
    results = run_pdf_batch(
        pdf_paths=pdf_paths,
        job_names=job_names,
        output_root=output_root,
        script_paths=script_paths,
        worker_count=worker_count,
        render_dpi=args.dpi,
        keep_debug=args.keep_debug,
        tesseract_cmd=tesseract_cmd,
    )

    batch_seconds = time.perf_counter() - batch_start
    successful_count = sum(
        result["status"] == "success"
        for result in results
    )
    failed_count = len(results) - successful_count

    batch_summary = {
        "pipeline_version": "Mullenizer 1.0",
        "completed_at_utc": utc_now_text(),
        "output_root": str(output_root),
        "pdf_count": len(pdf_paths),
        "successful_count": successful_count,
        "failed_count": failed_count,
        "batch_seconds": round(batch_seconds, 6),
        "top_candidates_per_target": TOP_CANDIDATES_PER_TARGET,
        "keep_debug": bool(args.keep_debug),
        "render_dpi": int(args.dpi),
        "parallelization": worker_details,
        "results": results,
    }

    # Save one batch-level status file after all workers have completed.
    batch_results_path = output_root / BATCH_RESULTS_FILENAME
    save_json(
        data=batch_summary,
        output_path=batch_results_path,
    )

    emit_pipeline_event(
        {
            "event": "batch_finished",
            "total_pdf_count": len(pdf_paths),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "batch_seconds": round(batch_seconds, 6),
            "batch_results_path": str(batch_results_path),
        }
    )

    # Remove the shared work parent when no retained failed-debug job uses it.
    work_parent = output_root / ".work"

    if work_parent.exists() and not any(work_parent.iterdir()):
        work_parent.rmdir()

    print()
    print(
        f"Finished: {successful_count} succeeded, "
        f"{failed_count} failed, {batch_seconds:.1f}s total."
    )
    print(f"Batch results: {batch_results_path}")

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
