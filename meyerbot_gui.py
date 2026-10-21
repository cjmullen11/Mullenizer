"""Mullenizer v2 desktop GUI.

Keep the UI layer focused on orchestration: select PDFs, stream pipeline events,
review candidates, edit the tracker, render overlays, and invoke manual graph
interpretation. Processing algorithms remain in their dedicated stage modules."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from human_guided_graph_interpreter import DEFAULT_HISTOGRAM_BIN_WIDTH, adjust_histogram_column_height, digitize_histogram, digitize_line_graph, histogram_known_bin_count, save_json as save_digitization_json
WINDOW_TITLE = 'Mullenizer v2'
DEFAULT_WINDOW_SIZE = '1180x860'
MINIMUM_WINDOW_WIDTH = 900
MINIMUM_WINDOW_HEIGHT = 650
PIPELINE_FILENAME = 'run_detection_pipeline.py'
OUTPUT_DIRECTORY_NAME = 'mullenizer_results'
BATCH_RESULTS_FILENAME = 'batch_results.json'
PDF_RESULTS_FILENAME = 'results.json'
REVIEW_RESULTS_FILENAME = 'review_results.json'
TRACKER_CSV_FILENAME = 'mullenizer_tracker.csv'
TRACKER_CSV_COLUMNS = ('ID', 'AFib burden', 'HR', 'Activity')
TRACKER_NUMERIC_COLUMNS = ('AFib burden', 'HR', 'Activity')
TRACKER_NUMBER_PATTERN = re.compile('[-+]?(?:\\d+(?:[.,]\\d*)?|[.,]\\d+)')
MAX_JOB_NAME_CHARS = 56
JOB_NAME_HASH_CHARS = 16
DUPLICATE_PATH_HASH_CHARS = 8
ACTIVITY_UNIT_MINUTES_PER_DAY = 'min/d'
ACTIVITY_UNIT_HOURS_PER_DAY = 'h/d'
ACTIVITY_UNIT_PERCENT = '%'
ACTIVITY_CONVERSION_UNITS = (ACTIVITY_UNIT_MINUTES_PER_DAY, ACTIVITY_UNIT_HOURS_PER_DAY, ACTIVITY_UNIT_PERCENT)
EVENT_POLL_INTERVAL_MILLISECONDS = 100
DEFAULT_HIGHLIGHTS_VISIBLE = True
ZOOM_STEP = 1.25
MINIMUM_ZOOM_FACTOR = 0.4
MAXIMUM_ZOOM_FACTOR = 10.0
CANDIDATE_VIEW_MARGIN_FRACTION = 0.06
STAT_ORDER: tuple[tuple[str, str], ...] = (('patient_name', 'Patient name'), ('date_of_birth', 'Date of birth'), ('afib_burden', 'AFib burden'), ('average_heart_rate', 'Average heart rate'), ('average_daily_activity', 'Average daily activity'))
PIPELINE_EVENT_PREFIX = 'MULLENIZER_EVENT '
COMPLETION_PATTERN = re.compile('^(?P<status>Completed|FAILED)\\s+(?P<job_name>[^:]+):')
LINE_MEASUREMENT_OPTIONS: dict[str, dict[str, Any]] = {
    'Heart rate (bpm)': {
        'unit': 'bpm',
        'tracker_field': 'HR',
        'target': 'average_heart_rate',
        'top_value': 120.0,
        'bottom_value': 40.0,
    },
    'Activity (%)': {
        'unit': '%',
        'tracker_field': 'Activity',
        'target': 'average_daily_activity',
        'top_value': 5.0,
        'bottom_value': 0.0,
    },
    'Activity (h/d)': {
        'unit': 'h/d',
        'tracker_field': 'Activity',
        'target': 'average_daily_activity',
        'top_value': 8.0,
        'bottom_value': 0.0,
    },
}
HISTOGRAM_MEASUREMENT_OPTIONS: dict[str, dict[str, Any]] = {
    'Heart rate (bpm)': {
        'unit': 'bpm',
        'tracker_field': 'HR',
        'target': 'average_heart_rate',
        'minimum': 40.0,
        'maximum': 120.0,
    },
    'Activity': {
        'unit': 'min/d',
        'tracker_field': 'Activity',
        'target': 'average_daily_activity',
        'minimum': 0.0,
        'maximum': 480.0,
    },
}
DEFAULT_LINE_MEASUREMENT_BY_STAT = {
    'average_heart_rate': 'Heart rate (bpm)',
    'average_daily_activity': 'Activity (h/d)',
}
DEFAULT_HISTOGRAM_MEASUREMENT_BY_STAT = {
    'average_heart_rate': 'Heart rate (bpm)',
    'average_daily_activity': 'Activity',
}
MANUAL_SELECTION_OUTLINE = '#ff9f2f'
MANUAL_SELECTION_WIDTH = 3
MANUAL_SELECTION_DASH = (7, 4)
DIGITIZED_TRACE_COLOR = '#ff00aa'
DIGITIZED_TRACE_WIDTH = 3
DIGITIZED_BOUNDS_DASH = (8, 4)
DIGITIZATION_HANDLE_HIT_PIXELS = 12
DIGITIZATION_HANDLE_HALF_HEIGHT_PIXELS = 14
DIGITIZATION_HANDLE_HALF_WIDTH_PIXELS = 4
HISTOGRAM_DETECTED_BAR_COLOR = '#1740b8'
HISTOGRAM_DETECTED_BAR_FILL = (0, 216, 255, 55)
HISTOGRAM_DETECTED_BAR_WIDTH = 3
HISTOGRAM_AVERAGE_COLUMN_COLOR = '#ff00aa'
HISTOGRAM_AVERAGE_COLUMN_FILL = (255, 0, 170, 80)
HISTOGRAM_AVERAGE_COLUMN_WIDTH = 5
HISTOGRAM_MEDIAN_COLUMN_COLOR = '#ff9d00'
HISTOGRAM_MEDIAN_COLUMN_FILL = (255, 157, 0, 55)
HISTOGRAM_MEDIAN_COLUMN_WIDTH = 4
HISTOGRAM_BAR_LABEL_FONT = ('TkDefaultFont', 9, 'bold')
POPUP_RIGHT_OFFSET_PIXELS = 140
POPUP_SCREEN_MARGIN_PIXELS = 20
OVERLAY_CANVAS_TAGS = ('candidate_highlight', 'digitization_bounds', 'digitization_handles', 'digitized_trace', 'histogram_detected_bars', 'histogram_bar_handles', 'histogram_average_column', 'histogram_median_column', 'manual_selection')
MINIMUM_MANUAL_SELECTION_WIDTH_PIXELS = 60
MINIMUM_MANUAL_SELECTION_HEIGHT_PIXELS = 45
TEXT_HIGHLIGHT_PADDING_PIXELS = 10
TEXT_HIGHLIGHT_ROLES = frozenset({'patient_name_label', 'patient_name_value', 'date_of_birth_label', 'date_of_birth_value', 'afib_label', 'afib_value', 'ocr_guide'})
HIGHLIGHT_STYLES: dict[str, tuple[int, int, int, int]] = {'patient_name_label': (255, 220, 70, 85), 'patient_name_value': (105, 220, 120, 85), 'date_of_birth_label': (255, 220, 70, 85), 'date_of_birth_value': (105, 220, 120, 85), 'afib_label': (255, 220, 70, 85), 'afib_value': (105, 220, 120, 85), 'ocr_guide': (255, 220, 70, 85)}

def running_frozen() -> bool:
    """Return True when running from a PyInstaller executable."""
    return bool(getattr(sys, 'frozen', False))

def hidden_subprocess_creation_flags() -> int:
    """Hide the background pipeline console and favor interactive responsiveness."""
    if not sys.platform.startswith('win'):
        return 0
    return int(getattr(subprocess, 'CREATE_NO_WINDOW', 0)) | int(getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))

def build_pipeline_command(pipeline_path: Path, pdf_paths: list[Path], output_root: Path) -> list[str]:
    """Build the source-mode or frozen pipeline command."""
    if running_frozen():
        command = [sys.executable, '--meyerbot-pipeline']
    else:
        command = [sys.executable, '-u', str(pipeline_path)]
    command.extend((str(pdf_path) for pdf_path in pdf_paths))
    command.extend(['--output-root', str(output_root)])
    return command

def utc_now_text() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()

def safe_slug(text: str) -> str:
    """Convert one source stem into the same bounded job name as the pipeline."""
    normalized = re.sub('[^A-Za-z0-9._-]+', '_', text.strip())
    normalized = normalized.strip('._-') or 'report'
    if len(normalized) <= MAX_JOB_NAME_CHARS:
        return normalized
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()[:JOB_NAME_HASH_CHARS]
    prefix_length = MAX_JOB_NAME_CHARS - JOB_NAME_HASH_CHARS - 1
    return f'{normalized[:prefix_length]}_{digest}'

def short_path_hash(path: Path) -> str:
    """Match the pipeline's duplicate-source path hash."""
    return hashlib.sha256(str(path.resolve()).encode('utf-8')).hexdigest()[:DUPLICATE_PATH_HASH_CHARS]

def make_unique_job_names(pdf_paths: list[Path]) -> list[str]:
    """Reproduce the pipeline's bounded deterministic per-PDF job names."""
    base_names = [safe_slug(pdf_path.stem) for pdf_path in pdf_paths]
    counts: dict[str, int] = {}
    for base_name in base_names:
        counts[base_name] = counts.get(base_name, 0) + 1
    job_names: list[str] = []
    for pdf_path, base_name in zip(pdf_paths, base_names, strict=True):
        if counts[base_name] == 1:
            job_names.append(base_name)
            continue
        suffix = f'_{short_path_hash(pdf_path)}'
        prefix_length = MAX_JOB_NAME_CHARS - len(suffix)
        job_names.append(f'{base_name[:prefix_length]}{suffix}')
    return job_names

def load_json(json_path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    return json.loads(json_path.read_text(encoding='utf-8'))

def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Atomically save one readable UTF-8 JSON object."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'.{output_path.name}.tmp')
    try:
        temporary_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

def list_folder_pdfs(folder: Path) -> list[Path]:
    """Return PDFs directly inside one folder in case-insensitive name order."""
    if not folder.exists() or not folder.is_dir():
        return []
    pdf_paths = [path.resolve() for path in folder.iterdir() if path.is_file() and path.suffix.lower() == '.pdf']
    return sorted(pdf_paths, key=lambda path: path.name.casefold())

def normalize_activity_unit(unit: str) -> str | None:
    """Normalize percent, hours/day, and minutes/day time-share units."""
    normalized = str(unit).strip().lower().replace(' ', '')
    if normalized in {'min/d', 'mins/d', 'minute/day', 'minutes/day', 'minuteperday', 'minutesperday', 'min/day'}:
        return ACTIVITY_UNIT_MINUTES_PER_DAY
    if normalized in {'h/d', 'hr/d', 'hrs/d', 'hour/day', 'hours/day', 'hoursperday', 'hourperday'}:
        return ACTIVITY_UNIT_HOURS_PER_DAY
    if normalized in {'%', 'percent', 'percentage', 'pct'}:
        return ACTIVITY_UNIT_PERCENT
    return None

def convert_activity_value(value: float, source_unit: str, target_unit: str) -> float:
    """Convert a daily time share among percent, hours/day, and minutes/day."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError('Time-share value must be finite.')
    source = normalize_activity_unit(source_unit)
    target = normalize_activity_unit(target_unit)
    if source is None:
        raise ValueError(f'Unsupported source unit: {source_unit!r}')
    if target is None:
        raise ValueError(f'Unsupported target unit: {target_unit!r}')
    if source == target:
        return numeric_value
    if source == ACTIVITY_UNIT_PERCENT:
        minutes_per_day = numeric_value * 1440.0 / 100.0
    elif source == ACTIVITY_UNIT_HOURS_PER_DAY:
        minutes_per_day = numeric_value * 60.0
    else:
        minutes_per_day = numeric_value
    if target == ACTIVITY_UNIT_MINUTES_PER_DAY:
        return minutes_per_day
    if target == ACTIVITY_UNIT_HOURS_PER_DAY:
        return minutes_per_day / 60.0
    return minutes_per_day / 1440.0 * 100.0

def normalize_measurement_unit(unit: str) -> str:
    """Normalize units whose displayed/tracked precision is standardized."""
    normalized = str(unit).strip().lower().replace(' ', '')
    if normalized in {'bpm', 'beats/min', 'beatsperminute'}:
        return 'bpm'
    activity_unit = normalize_activity_unit(normalized)
    if activity_unit is not None:
        return activity_unit
    return normalized

def round_measurement_value(value: float, unit: str) -> float | int:
    """Apply final measurement precision.

    - percent: nearest whole percent
    - bpm: nearest whole bpm
    - min/d: nearest whole minute
    - h/d: two decimal places
    - other/unitless values: four decimals"""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError('Measurement value must be finite.')
    normalized_unit = normalize_measurement_unit(unit)
    if normalized_unit in {ACTIVITY_UNIT_PERCENT, ACTIVITY_UNIT_MINUTES_PER_DAY, 'bpm'}:
        return int(round(numeric))
    if normalized_unit == ACTIVITY_UNIT_HOURS_PER_DAY:
        return round(numeric, 2)
    return round(numeric, 4)

def format_measurement_value(value: float, unit: str) -> str:
    """Format one rounded measurement consistently for GUI/tracker display."""
    rounded = round_measurement_value(value, unit)
    normalized_unit = normalize_measurement_unit(unit)
    if normalized_unit in {ACTIVITY_UNIT_PERCENT, ACTIVITY_UNIT_MINUTES_PER_DAY, 'bpm'}:
        return f'{int(rounded)}'
    if normalized_unit == ACTIVITY_UNIT_HOURS_PER_DAY:
        return f'{float(rounded):.2f}'
    if isinstance(rounded, int):
        return str(rounded)
    return f'{float(rounded):.4f}'.rstrip('0').rstrip('.')

def histogram_within_bin_fraction(average_column_position: float) -> float:
    """Return the mean's fractional position within its reported histogram bin.

    A whole-number mean is treated as the center of that bin. Thus 3.5 is 50%
    through the reported mean-containing column, matching the review convention."""
    position = float(average_column_position)
    if not math.isfinite(position):
        raise ValueError('Histogram mean position must be finite.')
    fraction = position - math.floor(position)
    if math.isclose(fraction, 0.0, abs_tol=1e-09):
        fraction = 0.5
    return max(0.0, min(1.0, fraction))

def histogram_value_from_bin_bounds(average_column_position: float, lower_bound: float, upper_bound: float) -> float:
    """Map the fractional histogram mean position into one detected bin's bounds.

    Example: a mean position of 3.5 with column-4 bounds 120..130 maps to 125.
    A whole-number mean is treated as the center of its detected bin."""
    lower = float(lower_bound)
    upper = float(upper_bound)
    if not all((math.isfinite(value) for value in (lower, upper))):
        raise ValueError('Histogram bounds must be finite numbers.')
    if upper < lower:
        raise ValueError('Histogram upper bound must be at least the lower bound.')
    fraction = histogram_within_bin_fraction(average_column_position)
    return lower + fraction * (upper - lower)

def resolve_candidate_page(pdf_output_dir: Path, candidate: dict[str, Any]) -> Path | None:
    """Resolve one retained full deskewed page from a compact PDF result."""
    raw_path = candidate.get('image_path')
    if not raw_path:
        return None
    image_path = Path(str(raw_path)).expanduser()
    if not image_path.is_absolute():
        image_path = pdf_output_dir / image_path
    image_path = image_path.resolve()
    if not image_path.exists() or not image_path.is_file():
        return None
    return image_path

def deskewed_page_sort_key(path: Path) -> tuple[int, str]:
    """Sort deskewed page images by embedded page number."""
    matches = re.findall(r'(?:page|pg)[_-]?(\d+)', path.stem, flags=re.IGNORECASE)
    page_number = int(matches[-1]) if matches else 10**9
    return (page_number, path.name.casefold())


def resolve_deskewed_pages(
    pdf_output_dir: Path,
    pdf_results: dict[str, Any],
) -> list[Path]:
    """Resolve every deskewed page currently retained for one processed PDF."""
    discovered: list[Path] = []

    for field_name in (
        'deskewed_pages',
        'all_deskewed_pages',
        'retained_deskewed_pages',
    ):
        raw_items = pdf_results.get(field_name, [])
        if not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            raw_path = raw_item
            if isinstance(raw_item, dict):
                raw_path = (
                    raw_item.get('deskewed_image_path')
                    or raw_item.get('image_path')
                    or raw_item.get('path')
                )
            if not raw_path:
                continue
            page_path = Path(str(raw_path)).expanduser()
            if not page_path.is_absolute():
                page_path = pdf_output_dir / page_path
            if page_path.is_file():
                discovered.append(page_path.resolve())

    for folder in (
        pdf_output_dir / 'deskewed_pages',
        pdf_output_dir / 'intermediates' / 'deskewed_pages',
    ):
        if not folder.is_dir():
            continue
        for page_path in folder.iterdir():
            if (
                page_path.is_file()
                and page_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
            ):
                discovered.append(page_path.resolve())

    unique = {str(path): path for path in discovered}
    return sorted(unique.values(), key=deskewed_page_sort_key)


def open_with_default_application(path: Path) -> None:
    """Open one file through the operating system's registered application."""
    path = path.expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f'File does not exist: {path}')
    if sys.platform.startswith('win'):
        os.startfile(str(path))
        return
    command = ['open', str(path)] if sys.platform == 'darwin' else ['xdg-open', str(path)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

def valid_bbox(raw_bbox: Any) -> list[int] | None:
    """Return one normalized x1,y1,x2,y2 box or None when invalid."""
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    if not all((isinstance(value, (int, float)) for value in raw_bbox)):
        return None
    bbox = [int(round(value)) for value in raw_bbox]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox

def unique_page_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group candidates by page, merging their retained OCR highlight records."""
    pages: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        page_number = int(candidate["page_number"])
        if page_number not in pages:
            pages[page_number] = candidate
            continue

        # Also consolidate pre-revision results that contained multiple hits per page.
        retained = pages[page_number]
        existing = retained["_highlight_regions"]
        seen = {(region["role"], tuple(region["bbox"])) for region in existing}
        for region in candidate["_highlight_regions"]:
            key = (region["role"], tuple(region["bbox"]))
            if key not in seen:
                existing.append(region)
                seen.add(key)

    return list(pages.values())


def candidate_highlights(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract only text highlight records intended for reviewer display."""
    raw_regions = candidate.get('highlight_regions')
    regions: list[dict[str, Any]] = []
    if isinstance(raw_regions, list):
        for raw_region in raw_regions:
            if not isinstance(raw_region, dict):
                continue
            role = str(raw_region.get('role', ''))
            if role not in TEXT_HIGHLIGHT_ROLES:
                continue
            bbox = valid_bbox(raw_region.get('bbox'))
            if bbox is not None:
                regions.append({'role': role, 'bbox': bbox})
    if regions:
        return regions
    for field_name, role in (('ocr_guide_bbox', 'ocr_guide'), ('label_bbox', 'afib_label'), ('value_bbox', 'afib_value')):
        bbox = valid_bbox(candidate.get(field_name))
        if bbox is not None:
            regions.append({'role': role, 'bbox': bbox})
    return regions

def padded_text_bbox(bbox: list[int], image_width: int, image_height: int) -> list[int]:
    """Pad a text highlight slightly while keeping it inside the page."""
    padding = TEXT_HIGHLIGHT_PADDING_PIXELS
    return [max(0, bbox[0] - padding), max(0, bbox[1] - padding), min(image_width, bbox[2] + padding), min(image_height, bbox[3] + padding)]

def union_bboxes(bboxes: list[list[int]]) -> list[int] | None:
    """Return the smallest box containing all supplied valid boxes."""
    if not bboxes:
        return None
    return [min((bbox[0] for bbox in bboxes)), min((bbox[1] for bbox in bboxes)), max((bbox[2] for bbox in bboxes)), max((bbox[3] for bbox in bboxes))]

def name_initials_from_candidate(name_candidate: dict[str, Any] | None) -> str:
    """Return every detected name initial, including hyphenated name pieces."""
    if not isinstance(name_candidate, dict):
        return ''
    initials = str(name_candidate.get('initials') or '').strip().upper()
    if initials:
        return ''.join((character for character in initials if character.isalpha()))
    first_initial = str(name_candidate.get('first_initial') or '').strip().upper()
    last_initial = str(name_candidate.get('last_initial') or '').strip().upper()
    return ''.join((character for character in f'{first_initial}{last_initial}' if character.isalpha()))

def birth_year_from_candidate(dob_candidate: dict[str, Any] | None) -> str:
    """Return a validated four-digit birth year from one DOB candidate."""
    if not isinstance(dob_candidate, dict):
        return ''
    try:
        year = int(dob_candidate.get('birth_year'))
    except (TypeError, ValueError):
        return ''
    if not 1900 <= year <= 2100:
        return ''
    return f'{year:04d}'

def compose_tracker_id(initials: str, birth_year: str, prefix: str='', suffix: str='') -> str:
    """Compose the single CSV ID from the separately editable tracker components.

    Empty pieces are omitted. Examples:
    - JAS + 1984 -> JAS_1984
    - prefix ABC, JAS, 1984, suffix X -> ABC_JAS_1984_X"""
    normalized_initials = ''.join((character for character in str(initials).strip().upper() if character.isalpha()))
    normalized_year = str(birth_year).strip()
    normalized_prefix = str(prefix).strip()
    normalized_suffix = str(suffix).strip()
    parts = [part for part in (normalized_prefix, normalized_initials, normalized_year, normalized_suffix) if part]
    return '_'.join(parts)

def patient_identity_parts_from_candidates(name_candidate: dict[str, Any] | None, dob_candidate: dict[str, Any] | None) -> tuple[str, str]:
    """Return independently editable initials and birth-year values."""
    return (name_initials_from_candidate(name_candidate), birth_year_from_candidate(dob_candidate))

def patient_id_from_candidates(name_candidate: dict[str, Any] | None, dob_candidate: dict[str, Any] | None) -> str:
    """Backward-compatible helper returning the default composed tracker ID."""
    initials, birth_year = patient_identity_parts_from_candidates(name_candidate=name_candidate, dob_candidate=dob_candidate)
    return compose_tracker_id(initials=initials, birth_year=birth_year)

def bounded_popup_position(parent_x: int, parent_y: int, parent_width: int, parent_height: int, popup_width: int, popup_height: int, screen_width: int, screen_height: int, x_offset: int=POPUP_RIGHT_OFFSET_PIXELS) -> tuple[int, int]:
    """Place a popup near the parent center, shifted right and kept on-screen."""
    x = parent_x + (parent_width - popup_width) // 2 + x_offset
    y = parent_y + (parent_height - popup_height) // 2
    margin = POPUP_SCREEN_MARGIN_PIXELS
    x = max(margin, min(x, screen_width - popup_width - margin))
    y = max(margin, min(y, screen_height - popup_height - margin))
    return (x, y)

def place_popup_right_of_center(window: tk.Toplevel, parent: tk.Misc, x_offset: int=POPUP_RIGHT_OFFSET_PIXELS) -> None:
    """Position a popup slightly right of center so it covers less of the report."""
    parent.update_idletasks()
    window.update_idletasks()
    x, y = bounded_popup_position(parent_x=parent.winfo_rootx(), parent_y=parent.winfo_rooty(), parent_width=max(1, parent.winfo_width()), parent_height=max(1, parent.winfo_height()), popup_width=max(1, window.winfo_reqwidth()), popup_height=max(1, window.winfo_reqheight()), screen_width=window.winfo_screenwidth(), screen_height=window.winfo_screenheight(), x_offset=x_offset)
    window.geometry(f'+{x}+{y}')

class GraphInterpretationDialog:
    """Choose graph mode, measurement type, and numeric calibration."""

    MODE_LABELS = {'Line graph': 'line', 'Histogram': 'histogram'}

    def __init__(
        self,
        parent: tk.Tk,
        stat_display_name: str,
        stat_key: str,
    ) -> None:
        """Collect only the calibration choices needed by the selected graph mode."""
        self.result: dict[str, Any] | None = None
        self.stat_key = stat_key
        self.window = tk.Toplevel(parent)
        self.window.title(f'Interpret {stat_display_name}')
        self.window.transient(parent)
        self.window.resizable(True, True)
        self.window.protocol('WM_DELETE_WINDOW', self._cancel)

        default_measurement = DEFAULT_LINE_MEASUREMENT_BY_STAT.get(
            stat_key,
            'Heart rate (bpm)',
        )
        default = LINE_MEASUREMENT_OPTIONS[default_measurement]

        self.mode_var = tk.StringVar(value='Histogram')
        self.measurement_var = tk.StringVar(value=default_measurement)
        self.first_var = tk.StringVar(value=f"{float(default['top_value']):g}")
        self.second_var = tk.StringVar(value=f"{float(default['bottom_value']):g}")
        self.bin_width_var = tk.StringVar(value=f'{DEFAULT_HISTOGRAM_BIN_WIDTH:g}')
        self.mode_help_var = tk.StringVar(
            value='Choose the measurement unit and enter the line graph y-axis bounds.'
        )

        outer = ttk.Frame(self.window, padding=10)
        outer.grid(row=0, column=0, sticky='nsew')
        outer.columnconfigure(1, weight=1)

        ttk.Label(outer, text='Graph type:').grid(
            row=0, column=0, padx=(0, 8), pady=2, sticky='e'
        )
        self.mode_combobox = ttk.Combobox(
            outer,
            textvariable=self.mode_var,
            values=list(self.MODE_LABELS),
            state='readonly',
            width=24,
        )
        self.mode_combobox.grid(row=0, column=1, pady=2, sticky='ew')
        self.mode_combobox.bind('<<ComboboxSelected>>', self._on_mode_changed)

        ttk.Label(
            outer,
            textvariable=self.mode_help_var,
            wraplength=620,
            justify='left',
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            pady=(1, 3),
            sticky='w',
        )

        self.measurement_label = ttk.Label(outer, text='Measurement:')
        self.measurement_label.grid(
            row=2, column=0, padx=(0, 8), pady=2, sticky='e'
        )
        self.measurement_combobox = ttk.Combobox(
            outer,
            textvariable=self.measurement_var,
            state='readonly',
            width=24,
        )
        self.measurement_combobox.grid(row=2, column=1, pady=2, sticky='ew')
        self.measurement_combobox.bind(
            '<<ComboboxSelected>>',
            self._apply_selected_measurement,
        )

        self.first_label = ttk.Label(outer, text='Top edge value:')
        self.first_label.grid(
            row=3, column=0, padx=(0, 8), pady=2, sticky='e'
        )
        self.first_entry = ttk.Entry(
            outer,
            textvariable=self.first_var,
            width=18,
        )
        self.first_entry.grid(row=3, column=1, pady=2, sticky='ew')

        self.second_label = ttk.Label(outer, text='Bottom edge value:')
        self.second_label.grid(
            row=4, column=0, padx=(0, 8), pady=2, sticky='e'
        )
        self.second_entry = ttk.Entry(
            outer,
            textvariable=self.second_var,
            width=18,
        )
        self.second_entry.grid(row=4, column=1, pady=2, sticky='ew')

        self.bin_width_label = ttk.Label(outer, text='Bin width:')
        self.bin_width_label.grid(row=5, column=0, padx=(0, 8), pady=2, sticky='e')
        self.bin_width_entry = ttk.Entry(outer, textvariable=self.bin_width_var, width=18)
        self.bin_width_entry.grid(row=5, column=1, pady=2, sticky='ew')

        button_frame = ttk.Frame(outer)
        button_frame.grid(
            row=6,
            column=0,
            columnspan=2,
            pady=(8, 0),
            sticky='e',
        )
        ttk.Button(
            button_frame,
            text='Cancel',
            command=self._cancel,
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(
            button_frame,
            text='Interpret',
            command=self._accept,
        ).grid(row=0, column=1)

        self.window.bind('<Return>', lambda _event: self._accept())
        self.window.bind('<Escape>', lambda _event: self._cancel())
        self._on_mode_changed()
        self._fit_window_to_contents()
        self.window.grab_set()
        self.first_entry.focus_set()
        self.window.wait_window()

    def _fit_window_to_contents(self) -> None:
        """Resize and reposition the dialog after mode-dependent text changes."""
        self.window.update_idletasks()
        requested_width = max(1, self.window.winfo_reqwidth())
        requested_height = max(1, self.window.winfo_reqheight())
        screen_width = max(1, self.window.winfo_screenwidth())
        screen_height = max(1, self.window.winfo_screenheight())
        margin = POPUP_SCREEN_MARGIN_PIXELS
        width = min(requested_width, max(320, screen_width - 2 * margin))
        height = min(requested_height, max(240, screen_height - 2 * margin))
        x, y = bounded_popup_position(
            parent_x=self.window.master.winfo_rootx(),
            parent_y=self.window.master.winfo_rooty(),
            parent_width=max(1, self.window.master.winfo_width()),
            parent_height=max(1, self.window.master.winfo_height()),
            popup_width=width,
            popup_height=height,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def _selected_mode(self) -> str:
        """Return the internal graph-mode key."""
        return self.MODE_LABELS.get(self.mode_var.get(), 'histogram')

    def _measurement_options(self) -> dict[str, dict[str, Any]]:
        """Return the measurement choices allowed for the selected graph mode."""
        if self._selected_mode() == 'histogram':
            return HISTOGRAM_MEASUREMENT_OPTIONS
        return LINE_MEASUREMENT_OPTIONS

    def _default_measurement_label(self) -> str:
        """Choose a useful default measurement from the current review metric."""
        if self._selected_mode() == 'histogram':
            return DEFAULT_HISTOGRAM_MEASUREMENT_BY_STAT.get(
                self.stat_key,
                'Heart rate (bpm)',
            )
        return DEFAULT_LINE_MEASUREMENT_BY_STAT.get(
            self.stat_key,
            'Heart rate (bpm)',
        )

    def _on_mode_changed(
        self,
        _event: tk.Event[Any] | None = None,
    ) -> None:
        """Switch measurement choices and bound labels for line versus histogram."""
        histogram_mode = self._selected_mode() == 'histogram'
        options = self._measurement_options()
        default_label = self._default_measurement_label()
        self.measurement_combobox.configure(values=list(options))
        self.measurement_var.set(
            default_label if default_label in options else next(iter(options))
        )

        self.first_label.configure(
            text='Lowest-bin lower edge:' if histogram_mode else 'Top edge value:'
        )
        self.second_label.configure(
            text='Highest-bin upper edge:' if histogram_mode else 'Bottom edge value:'
        )
        self.mode_help_var.set(
            'Enter the outer bin edges and numerical width (default 10); their range must contain a whole number of bins.'
            if histogram_mode
            else 'Choose bpm, activity (%), or activity (h/d), then enter the y-axis bounds.'
        )
        for widget in (self.bin_width_label, self.bin_width_entry):
            if histogram_mode:
                widget.grid()
            else:
                widget.grid_remove()
        self._apply_selected_measurement()
        self.window.after_idle(self._fit_window_to_contents)
        self.first_entry.focus_set()

    def _apply_selected_measurement(
        self,
        _event: tk.Event[Any] | None = None,
    ) -> None:
        """Fill sensible numeric defaults for the selected measurement."""
        options = self._measurement_options()
        measurement = options.get(self.measurement_var.get())
        if measurement is None:
            return

        if self._selected_mode() == 'histogram':
            self.first_var.set(f"{float(measurement['minimum']):g}")
            self.second_var.set(f"{float(measurement['maximum']):g}")
        else:
            self.first_var.set(f"{float(measurement['top_value']):g}")
            self.second_var.set(f"{float(measurement['bottom_value']):g}")

    def _accept(self) -> None:
        """Validate the selected measurement and numeric graph calibration."""
        mode = self._selected_mode()
        options = self._measurement_options()
        measurement_label = self.measurement_var.get()
        measurement = options.get(measurement_label)
        if measurement is None:
            messagebox.showerror(
                WINDOW_TITLE,
                'Select a measurement type.',
                parent=self.window,
            )
            return

        first_text = self.first_var.get().strip()
        second_text = self.second_var.get().strip()
        if not first_text or not second_text:
            messagebox.showerror(
                WINDOW_TITLE,
                'Both calibration values are required.',
                parent=self.window,
            )
            return
        try:
            first_value = float(first_text)
            second_value = float(second_text)
        except ValueError:
            messagebox.showerror(
                WINDOW_TITLE,
                'Calibration values must be valid numbers.',
                parent=self.window,
            )
            return
        if not math.isfinite(first_value) or not math.isfinite(second_value):
            messagebox.showerror(
                WINDOW_TITLE,
                'Calibration values must be finite numbers.',
                parent=self.window,
            )
            return

        common = {
            'mode': mode,
            'measurement_label': measurement_label,
            'unit': str(measurement['unit']),
            'tracker_field': str(measurement['tracker_field']),
            'target': str(measurement['target']),
        }

        if mode == 'histogram':
            if second_value <= first_value:
                messagebox.showerror(
                    WINDOW_TITLE,
                    'The highest-bin upper bound must be greater than the lowest-bin lower bound.',
                    parent=self.window,
                )
                return
            try:
                width_value = float(self.bin_width_var.get().strip())
                histogram_known_bin_count(first_value, second_value, width_value)
            except (ValueError, OverflowError) as error:
                messagebox.showerror(WINDOW_TITLE, f'Invalid histogram bin width/range:\n{error}', parent=self.window)
                return
            self.result = {
                **common,
                'minimum': first_value,
                'maximum': second_value,
                'bin_width': width_value,
            }
        else:
            self.result = {
                **common,
                'top_value': first_value,
                'bottom_value': second_value,
            }

        self.window.grab_release()
        self.window.destroy()

    def _cancel(self) -> None:
        """Close the dialog without returning interpretation settings."""
        self.result = None
        try:
            self.window.grab_release()
        except tk.TclError:
            pass
        self.window.destroy()


class MeyerBotGui:
    """PDF/folder-processing and sequential candidate-review application."""

    def __init__(self, root: tk.Tk) -> None:
        """Create the complete interface and initialize application state."""
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(DEFAULT_WINDOW_SIZE)
        self.root.minsize(MINIMUM_WINDOW_WIDTH, MINIMUM_WINDOW_HEIGHT)
        self.script_dir = Path(__file__).resolve().parent
        self.pipeline_path = self.script_dir / PIPELINE_FILENAME
        self.event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.pipeline_process: subprocess.Popen[str] | None = None
        self.selected_folder: Path | None = None
        self.output_root: Path | None = None
        self.submitted_pdfs: list[Path] = []
        self.total_pdf_count = 0
        self.completed_pdf_count = 0
        self.finished_job_names: set[str] = set()
        self.job_name_to_pdf_name: dict[str, str] = {}
        self.job_name_to_batch_position: dict[str, int] = {}
        self.successful_pdf_count = 0
        self.failed_pdf_count = 0
        self.pipeline_worker_count = 0
        self.pipeline_reserved_cores = 0
        self.pipeline_running = False
        self.pipeline_finished = False
        self.review_waiting_for_processing = False
        self.loaded_output_dirs: set[str] = set()
        self.review_pdfs: list[dict[str, Any]] = []
        self.review_pdf_index = 0
        self.review_stat_index = 0
        self.review_candidate_index = 0
        self.review_decisions: dict[str, dict[str, Any]] = {}
        self.review_complete = False
        self.tracker_rows: dict[str, dict[str, str]] = {}
        self.tracker_manual_fields: dict[str, set[str]] = {}
        self.tracker_key_to_iid: dict[str, str] = {}
        self.tracker_iid_to_key: dict[str, str] = {}
        self.tracker_active_key: str | None = None
        self.tracker_editor_loading = False
        self.highlights_visible = DEFAULT_HIGHLIGHTS_VISIBLE
        self.current_candidate: dict[str, Any] | None = None
        self.current_highlight_regions: list[dict[str, Any]] = []
        self.current_focus_bbox: list[int] | None = None
        self.current_source_image: Image.Image | None = None
        self.current_display_image: ImageTk.PhotoImage | None = None
        self.current_page_path: Path | None = None
        self.current_pdf_pages: list[Path] = []
        self.current_pdf_page_index = 0
        self.manual_search_active = False
        self.manual_review_candidate: dict[str, Any] | None = None
        self.current_highlight_images: list[ImageTk.PhotoImage] = []
        self.current_canvas_message: str | None = None
        self.show_manual_pdf_search_button = False
        self.current_zoom_factor = 1.0
        self.current_fit_scale = 1.0
        self.current_display_scale = 1.0
        self.current_display_width = 0
        self.current_display_height = 0
        self.current_image_origin = (0.0, 0.0)
        self.manual_selection_active = False
        self.manual_selection_start_source: tuple[float, float] | None = None
        self.manual_selection_end_source: tuple[float, float] | None = None
        self.manual_selection_bbox: list[int] | None = None
        self.current_digitization_result: dict[str, Any] | None = None
        self.digitization_edge_drag: str | None = None
        self.histogram_height_drag: int | None = None
        self.statistics_window: tk.Toplevel | None = None
        self.digitization_bounds_dirty = False
        self.interpretation_previous_highlights_visible: bool | None = None
        self.interpretation_overlays_suppressed = False
        self.selection_summary_var = tk.StringVar(value='No input selected')
        self.input_toggle_var = tk.StringVar(value='Hide ▲')
        self.input_panel_collapsed = False
        self.pdf_var = tk.StringVar(value='No PDF selected — 0/0')
        self.progress_text_var = tk.StringVar(value='Ready')
        self.stat_var = tk.StringVar(value='Select PDF(s) or a folder to begin')
        self.candidate_var = tk.StringVar(value='')
        self.status_var = tk.StringVar(value='')
        self.zoom_var = tk.StringVar(value='Fit Page')
        self.page_nav_var = tk.StringVar(value='Page —/—')
        self.tracker_current_var = tk.StringVar(value='No processed report selected')
        self.tracker_id_preview_var = tk.StringVar(value='ID: —')
        self.tracker_initials_var = tk.StringVar(value='')
        self.tracker_birth_year_var = tk.StringVar(value='')
        self.tracker_prefix_var = tk.StringVar(value='')
        self.tracker_suffix_var = tk.StringVar(value='')
        self.tracker_afib_var = tk.StringVar(value='')
        self.tracker_hr_var = tk.StringVar(value='')
        self.tracker_activity_var = tk.StringVar(value='')
        self._build_interface()
        self._bind_shortcuts()
        self.root.after(EVENT_POLL_INTERVAL_MILLISECONDS, self._poll_events)

    def _build_interface(self) -> None:
        """Create all visible controls and configure resize behavior."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.input_frame = ttk.LabelFrame(self.root, text='Input PDFs', padding=(12, 8, 12, 10))
        self.input_frame.grid(row=0, column=0, padx=14, pady=(14, 8), sticky='ew')
        self.input_frame.columnconfigure(0, weight=1)
        input_header_frame = ttk.Frame(self.input_frame)
        input_header_frame.grid(row=0, column=0, sticky='ew')
        input_header_frame.columnconfigure(0, weight=1)
        ttk.Label(input_header_frame, textvariable=self.selection_summary_var, font=('TkDefaultFont', 10, 'bold'), anchor='w').grid(row=0, column=0, sticky='ew')
        self.input_toggle_button = ttk.Button(input_header_frame, textvariable=self.input_toggle_var, command=self._toggle_input_panel, width=9)
        self.input_toggle_button.grid(row=0, column=1, padx=(10, 0), sticky='e')
        self.input_content_frame = ttk.Frame(self.input_frame)
        self.input_content_frame.grid(row=1, column=0, pady=(9, 0), sticky='ew')
        self.input_content_frame.columnconfigure(0, weight=1)
        input_button_frame = ttk.Frame(self.input_content_frame)
        input_button_frame.grid(row=0, column=0, sticky='ew')
        input_button_frame.columnconfigure(0, weight=1)
        input_button_frame.columnconfigure(1, weight=1)
        input_button_frame.columnconfigure(2, weight=1)
        self.choose_pdfs_button = ttk.Button(input_button_frame, text='Choose PDF(s)', command=self._choose_pdfs)
        self.choose_pdfs_button.grid(row=0, column=0, padx=(0, 5), ipady=3, sticky='ew')
        self.choose_folder_button = ttk.Button(input_button_frame, text='Choose Folder', command=self._choose_folder)
        self.choose_folder_button.grid(row=0, column=1, padx=5, ipady=3, sticky='ew')
        self.process_button = ttk.Button(input_button_frame, text='Process Selected', command=self._start_processing)
        self.process_button.grid(row=0, column=2, padx=(5, 0), ipady=3, sticky='ew')
        ttk.Label(self.input_content_frame, text='Files to process:', anchor='w').grid(row=1, column=0, pady=(9, 3), sticky='ew')
        selection_list_frame = ttk.Frame(self.input_content_frame)
        selection_list_frame.grid(row=2, column=0, sticky='ew')
        selection_list_frame.columnconfigure(0, weight=1)
        self.selection_list = tk.Listbox(selection_list_frame, height=4, exportselection=False, selectmode=tk.EXTENDED, takefocus=False)
        self.selection_list.grid(row=0, column=0, sticky='ew')
        selection_scrollbar = ttk.Scrollbar(selection_list_frame, orient='vertical', command=self.selection_list.yview)
        selection_scrollbar.grid(row=0, column=1, sticky='ns')
        self.selection_list.configure(yscrollcommand=selection_scrollbar.set)
        self.selection_list.bind('<Button-1>', lambda _event: 'break')
        self.selection_list.bind('<B1-Motion>', lambda _event: 'break')
        progress_frame = ttk.Frame(self.root, padding=(14, 4, 14, 8))
        progress_frame.grid(row=1, column=0, sticky='ew')
        progress_frame.columnconfigure(0, weight=1)
        ttk.Label(progress_frame, textvariable=self.pdf_var, font=('TkDefaultFont', 14, 'bold'), anchor='center').grid(row=0, column=0, sticky='ew')
        self.progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate', maximum=100.0, value=0.0)
        self.progress_bar.grid(row=1, column=0, pady=(8, 3), sticky='ew')
        ttk.Label(progress_frame, textvariable=self.progress_text_var, anchor='center').grid(row=2, column=0, sticky='ew')
        review_header = ttk.Frame(self.root, padding=(14, 4, 14, 6))
        review_header.grid(row=2, column=0, sticky='ew')
        review_header.columnconfigure(0, weight=1)
        ttk.Label(review_header, textvariable=self.stat_var, font=('TkDefaultFont', 13, 'bold'), anchor='center').grid(row=0, column=0, sticky='ew')
        ttk.Label(review_header, textvariable=self.candidate_var, anchor='center').grid(row=1, column=0, pady=(3, 0), sticky='ew')
        canvas_frame = ttk.Frame(self.root, padding=(14, 4, 14, 6))
        canvas_frame.grid(row=3, column=0, sticky='nsew')
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.columnconfigure(1, weight=0)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, background='#303030', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.manual_pdf_search_button = tk.Button(self.canvas, text='Manual Mode', command=self._open_current_pdf_for_manual_search, font=('TkDefaultFont', 16, 'bold'), padx=28, pady=16, cursor='hand2')
        self.page_nav_frame = ttk.Frame(self.canvas, padding=3, relief='raised')
        self.page_back_button = ttk.Button(self.page_nav_frame, text='◀ Page', command=lambda: self._change_pdf_page(-1), width=8)
        self.page_back_button.grid(row=0, column=0, padx=(0, 3))
        ttk.Label(self.page_nav_frame, textvariable=self.page_nav_var, anchor='center', width=11).grid(row=0, column=1, padx=3)
        self.page_forward_button = ttk.Button(self.page_nav_frame, text='Page ▶', command=lambda: self._change_pdf_page(1), width=8)
        self.page_forward_button.grid(row=0, column=2, padx=(3, 0))
        tracker_frame = ttk.LabelFrame(canvas_frame, text='Data Tracker', padding=9)
        tracker_frame.grid(row=0, column=1, padx=(10, 0), sticky='ns')
        tracker_frame.columnconfigure(0, weight=1)
        tracker_frame.rowconfigure(2, weight=1)
        ttk.Label(tracker_frame, textvariable=self.tracker_current_var, font=('TkDefaultFont', 10, 'bold'), anchor='w', wraplength=390).grid(row=0, column=0, sticky='ew')
        ttk.Label(tracker_frame, text='Review values below; the CSV auto-saves for Excel / Google Sheets.', anchor='w', wraplength=390).grid(row=1, column=0, pady=(2, 7), sticky='ew')
        tree_frame = ttk.Frame(tracker_frame)
        tree_frame.grid(row=2, column=0, sticky='nsew')
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.tracker_tree = ttk.Treeview(tree_frame, columns=('id', 'afib', 'hr', 'activity'), show='tree headings', height=7, selectmode='browse')
        self.tracker_tree.heading('#0', text='Report')
        self.tracker_tree.heading('id', text='ID')
        self.tracker_tree.heading('afib', text='AFib')
        self.tracker_tree.heading('hr', text='HR (bpm)')
        self.tracker_tree.heading('activity', text='Activity (mins/day)')
        self.tracker_tree.column('#0', width=115, minwidth=85, stretch=True)
        self.tracker_tree.column('id', width=115, minwidth=80, stretch=True)
        self.tracker_tree.column('afib', width=58, minwidth=48, stretch=True)
        self.tracker_tree.column('hr', width=48, minwidth=42, stretch=True)
        self.tracker_tree.column('activity', width=66, minwidth=52, stretch=True)
        self.tracker_tree.grid(row=0, column=0, sticky='nsew')
        self.tracker_tree.bind('<<TreeviewSelect>>', self._on_tracker_tree_selected)
        tracker_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tracker_tree.yview)
        tracker_scrollbar.grid(row=0, column=1, sticky='ns')
        self.tracker_tree.configure(yscrollcommand=tracker_scrollbar.set)
        self.tracker_entries: list[ttk.Entry] = []
        id_frame = ttk.LabelFrame(tracker_frame, text='ID = Prefix + Initials + DOB Year + Suffix', padding=7)
        id_frame.grid(row=3, column=0, pady=(8, 0), sticky='ew')
        for column in range(4):
            id_frame.columnconfigure(column, weight=1)
        identity_fields = (('Prefix', self.tracker_prefix_var, 0, 8), ('Initials', self.tracker_initials_var, 1, 9), ('DOB year', self.tracker_birth_year_var, 2, 9), ('Suffix', self.tracker_suffix_var, 3, 8))
        for label_text, variable, column_index, width in identity_fields:
            ttk.Label(id_frame, text=label_text, anchor='center').grid(row=0, column=column_index, padx=3, sticky='ew')
            entry = ttk.Entry(id_frame, textvariable=variable, width=width, justify='center')
            entry.grid(row=1, column=column_index, padx=3, pady=(2, 4), sticky='ew')
            entry.bind('<FocusOut>', self._commit_tracker_edits)
            entry.bind('<Return>', self._commit_tracker_edits)
            self.tracker_entries.append(entry)
        ttk.Label(id_frame, textvariable=self.tracker_id_preview_var, font=('TkDefaultFont', 10, 'bold'), anchor='center').grid(row=2, column=0, columnspan=4, pady=(3, 1), sticky='ew')
        ttk.Label(id_frame, text='Initials and DOB year are auto-filled when detected. Prefix/suffix are optional. Empty parts are omitted.', anchor='center', justify='center', wraplength=390).grid(row=3, column=0, columnspan=4, pady=(1, 0), sticky='ew')
        measurement_frame = ttk.LabelFrame(tracker_frame, text='Measurements — numeric export values', padding=7)
        measurement_frame.grid(row=4, column=0, pady=(8, 0), sticky='ew')
        for column in range(3):
            measurement_frame.columnconfigure(column, weight=1)
        tracker_fields = (('AFib burden', self.tracker_afib_var), ('Heart rate (bpm)', self.tracker_hr_var), ('Activity (mins/day)', self.tracker_activity_var))
        for column_index, (label_text, variable) in enumerate(tracker_fields):
            ttk.Label(measurement_frame, text=label_text, anchor='center').grid(row=0, column=column_index, padx=3, sticky='ew')
            entry = ttk.Entry(measurement_frame, textvariable=variable, width=11, justify='center')
            entry.grid(row=1, column=column_index, padx=3, pady=(2, 4), sticky='ew')
            entry.bind('<FocusOut>', self._commit_tracker_edits)
            entry.bind('<Return>', self._commit_tracker_edits)
            self.tracker_entries.append(entry)
        ttk.Label(measurement_frame, text='CSV measurement cells contain numbers only.', anchor='center').grid(row=2, column=0, columnspan=3, sticky='ew')
        tracker_footer = ttk.Frame(tracker_frame)
        tracker_footer.grid(row=5, column=0, pady=(8, 0), sticky='ew')
        tracker_footer.columnconfigure(0, weight=1)
        ttk.Label(tracker_footer, text=f'Auto-saved: {TRACKER_CSV_FILENAME}', anchor='w').grid(row=0, column=0, sticky='ew')
        self.open_tracker_button = ttk.Button(tracker_footer, text='Open CSV', command=self._open_tracker_csv, width=10)
        self.open_tracker_button.grid(row=0, column=1, padx=(8, 0), sticky='e')
        for variable in (self.tracker_prefix_var, self.tracker_initials_var, self.tracker_birth_year_var, self.tracker_suffix_var):
            variable.trace_add('write', self._update_tracker_id_preview)
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.canvas.bind('<Button-4>', self._on_linux_wheel_up)
        self.canvas.bind('<Button-5>', self._on_linux_wheel_down)
        self.canvas.bind('<ButtonPress-1>', self._start_pan)
        self.canvas.bind('<B1-Motion>', self._pan_image)
        self.canvas.bind('<ButtonRelease-1>', self._finish_pointer_action)
        self.canvas.bind('<Motion>', self._update_canvas_pointer_cursor)
        controls_frame = ttk.Frame(self.root, padding=(14, 8, 14, 8))
        controls_frame.grid(row=4, column=0, sticky='ew')
        controls_frame.columnconfigure(0, weight=2)
        controls_frame.columnconfigure(1, weight=2)
        controls_frame.columnconfigure(2, weight=3)
        view_frame = ttk.LabelFrame(controls_frame, text='View', padding=8)
        view_frame.grid(row=0, column=0, padx=(0, 6), sticky='nsew')
        for column in range(3):
            view_frame.columnconfigure(column, weight=1)
        self.toggle_highlights_button = ttk.Button(view_frame, text='Hide Markings', command=self._toggle_highlights, state='disabled')
        self.toggle_highlights_button.grid(row=0, column=0, columnspan=2, padx=(0, 4), pady=(0, 6), sticky='ew')
        self.open_pdf_button = ttk.Button(view_frame, text='Open report externally', command=self._open_current_pdf_externally, state='disabled')
        self.open_pdf_button.grid(row=0, column=2, padx=(4, 0), pady=(0, 6), sticky='ew')
        self.zoom_out_button = ttk.Button(view_frame, text='−', command=self._zoom_out, state='disabled')
        self.zoom_out_button.grid(row=1, column=0, padx=(0, 4), sticky='ew')
        self.zoom_reset_button = ttk.Button(view_frame, textvariable=self.zoom_var, command=self._reset_zoom, state='disabled')
        self.zoom_reset_button.grid(row=1, column=1, padx=4, sticky='ew')
        self.zoom_in_button = ttk.Button(view_frame, text='+', command=self._zoom_in, state='disabled')
        self.zoom_in_button.grid(row=1, column=2, padx=(4, 0), sticky='ew')
        interpretation_frame = ttk.LabelFrame(controls_frame, text='Graph Tools', padding=8)
        interpretation_frame.grid(row=0, column=1, padx=6, sticky='nsew')
        interpretation_frame.columnconfigure(0, weight=1)
        self.manual_digitize_button = ttk.Button(interpretation_frame, text='Interpret Graph', command=self._start_manual_digitization, state='disabled')
        self.manual_digitize_button.grid(row=0, column=0, ipady=6, sticky='ew')
        self.refresh_digitize_button = ttk.Button(
            interpretation_frame,
            text='Refresh Interpretation',
            command=self._refresh_current_digitization,
            state='disabled',
        )
        self.refresh_digitize_button.grid(row=1, column=0, pady=(6, 0), sticky='ew')
        self.graph_tools_hint_var = tk.StringVar(value='Select a graph to interpret.')
        ttk.Label(
            interpretation_frame,
            textvariable=self.graph_tools_hint_var,
            justify='center',
            wraplength=260,
        ).grid(row=2, column=0, pady=(6, 0), sticky='ew')
        navigation_frame = ttk.LabelFrame(controls_frame, text='Review', padding=8)
        navigation_frame.grid(row=0, column=2, padx=(6, 0), sticky='nsew')
        navigation_frame.columnconfigure(0, weight=1)
        navigation_frame.columnconfigure(1, weight=1)
        navigation_frame.columnconfigure(2, weight=2)
        self.back_button = ttk.Button(navigation_frame, text='Back', command=self._go_back, state='disabled')
        self.back_button.grid(row=0, column=0, padx=(0, 5), ipadx=10, ipady=8, sticky='ew')
        self.next_candidate_button = ttk.Button(navigation_frame, text='Try Next Candidate', command=self._next_candidate, state='disabled')
        self.next_candidate_button.grid(row=0, column=1, padx=5, ipadx=10, ipady=8, sticky='ew')
        self.next_stat_button = ttk.Button(navigation_frame, text='Next Metric', command=self._next_stat, state='disabled')
        self.next_stat_button.grid(row=0, column=2, padx=(5, 0), ipadx=14, ipady=8, sticky='ew')
        ttk.Label(self.root, textvariable=self.status_var, anchor='center', padding=(14, 0, 14, 12)).grid(row=5, column=0, sticky='ew')
        self._show_canvas_message('Choose PDF(s) or a folder containing PDF scans.')

    def _set_input_panel_collapsed(self, collapsed: bool) -> None:
        """Show or hide the detailed input controls while keeping the header."""
        self.input_panel_collapsed = bool(collapsed)
        if self.input_panel_collapsed:
            self.input_content_frame.grid_remove()
            self.input_toggle_var.set('Show ▼')
        else:
            self.input_content_frame.grid()
            self.input_toggle_var.set('Hide ▲')

    def _toggle_input_panel(self) -> None:
        """Toggle the detailed PDF-selection/options area."""
        self._set_input_panel_collapsed(not self.input_panel_collapsed)

    def _bind_shortcuts(self) -> None:
        """Bind review and zoom shortcuts without replacing visible controls."""
        self.root.bind('<Left>', lambda _event: self._go_back())
        self.root.bind('<Right>', lambda _event: self._next_candidate())
        self.root.bind('<Return>', lambda _event: self._next_stat())
        self.root.bind('<plus>', lambda _event: self._zoom_in())
        self.root.bind('<equal>', lambda _event: self._zoom_in())
        self.root.bind('<minus>', lambda _event: self._zoom_out())
        self.root.bind('<Key-0>', lambda _event: self._reset_zoom())
        self.root.bind('<Key-h>', lambda _event: self._toggle_highlights())
        self.root.bind('<Key-d>', lambda _event: self._start_manual_digitization())

    def _refresh_input_selection_display(self) -> None:
        """Show the exact current input set and keep every row highlighted."""
        self.selection_list.delete(0, tk.END)
        if self.selected_folder is not None:
            self.selection_list.insert(tk.END, f'Folder: {self.selected_folder}')
            for pdf_path in self.submitted_pdfs:
                self.selection_list.insert(tk.END, f'    PDF: {pdf_path.name}')
            self.selection_summary_var.set(f'Selected folder — {len(self.submitted_pdfs)} PDF(s)')
        elif self.submitted_pdfs:
            for pdf_path in self.submitted_pdfs:
                self.selection_list.insert(tk.END, f'PDF: {pdf_path.name} — {pdf_path.parent}')
            self.selection_summary_var.set(f'Selected PDF files — {len(self.submitted_pdfs)} file(s)')
        else:
            self.selection_summary_var.set('No input selected')
        if self.selection_list.size() > 0:
            self.selection_list.selection_set(0, tk.END)
            self.selection_list.see(0)

    def _selection_output_parent(self, pdf_paths: list[Path]) -> Path:
        """Choose a predictable result location for either selection mode.

        Folder mode writes inside the chosen folder. Direct-file mode writes
        beside the first selected PDF; this is also the natural case when all
        selected PDFs came from the same directory."""
        if self.selected_folder is not None:
            return self.selected_folder
        return pdf_paths[0].parent

    def _apply_input_selection(self, pdf_paths: list[Path], selected_folder: Path | None) -> None:
        """Replace the current input set and update all pre-processing feedback."""
        self.selected_folder = selected_folder
        self.submitted_pdfs = list(pdf_paths)
        self.total_pdf_count = len(pdf_paths)
        self.completed_pdf_count = 0
        self.finished_job_names.clear()
        self.job_name_to_pdf_name = {}
        self._clear_review_state()
        self.progress_bar['value'] = 0.0
        self._refresh_input_selection_display()
        if not pdf_paths:
            self.pdf_var.set('No PDF selected — 0/0')
            self.progress_text_var.set('Ready — 0 PDFs')
            self.candidate_var.set('')
            if selected_folder is not None:
                self.stat_var.set('Selected folder contains no PDF files')
                self.status_var.set(str(selected_folder))
                self._show_canvas_message('The selected folder contains no PDF files.\nChoose another folder or choose PDF files directly.')
            else:
                self.stat_var.set('Select PDF(s) or a folder to begin')
                self.status_var.set('')
                self._show_canvas_message('Choose PDF(s) or a folder containing PDF scans.')
            return
        self.pdf_var.set(f'{pdf_paths[0].name} — PDF 1/{len(pdf_paths)}')
        self.progress_text_var.set(f'Ready to process 0/{len(pdf_paths)} PDFs — 0%')
        self.stat_var.set('Ready to process selected PDFs')
        self.candidate_var.set('')
        output_root = (self._selection_output_parent(pdf_paths) / OUTPUT_DIRECTORY_NAME).resolve()
        self.status_var.set(f'Output will be saved in {output_root}')
        self._show_canvas_message(f'{len(pdf_paths)} PDF file(s) selected and ready.')

    def _choose_pdfs(self) -> None:
        """Choose one or more PDF files directly.

        Canceling the native picker intentionally leaves the existing input
        selection unchanged."""
        selected_texts = filedialog.askopenfilenames(title='Choose PDF scan(s)', filetypes=[('PDF files', '*.pdf')])
        if not selected_texts:
            return
        pdf_paths: list[Path] = []
        seen_paths: set[Path] = set()
        for selected_text in selected_texts:
            pdf_path = Path(selected_text).expanduser().resolve()
            if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf' and (pdf_path not in seen_paths):
                pdf_paths.append(pdf_path)
                seen_paths.add(pdf_path)
        if not pdf_paths:
            messagebox.showwarning(WINDOW_TITLE, 'No PDF files were selected.')
            return
        self._apply_input_selection(pdf_paths=pdf_paths, selected_folder=None)

    def _choose_folder(self) -> None:
        """Choose one folder and select every PDF directly inside it.

        Canceling the native picker intentionally leaves the existing input
        selection unchanged."""
        selected_text = filedialog.askdirectory(title='Choose a folder containing PDF scans')
        if not selected_text:
            return
        selected_folder = Path(selected_text).expanduser().resolve()
        pdf_paths = list_folder_pdfs(selected_folder)
        self._apply_input_selection(pdf_paths=pdf_paths, selected_folder=selected_folder)

    def _start_processing(self) -> None:
        """Start the existing batch pipeline on the exact displayed PDF set."""
        if not self.submitted_pdfs:
            if self.selected_folder is not None:
                feedback = 'The selected folder does not contain any PDF files.\n\nChoose another folder or choose one or more PDFs directly.'
            else:
                feedback = "You haven't selected any PDFs yet.\n\nChoose PDF(s) or choose a folder before processing."
            self.stat_var.set('No PDFs selected')
            self.status_var.set('Choose PDF(s) or a folder before processing.')
            self._show_canvas_message('No PDFs are selected.')
            messagebox.showwarning(WINDOW_TITLE, feedback)
            return
        pdf_paths = list(self.submitted_pdfs)
        missing_paths = [pdf_path for pdf_path in pdf_paths if not pdf_path.exists() or not pdf_path.is_file() or pdf_path.suffix.lower() != '.pdf']
        if missing_paths:
            missing_preview = '\n'.join((str(path) for path in missing_paths[:5]))
            if len(missing_paths) > 5:
                missing_preview += f'\n...and {len(missing_paths) - 5} more'
            messagebox.showerror(WINDOW_TITLE, f'One or more selected PDFs are no longer available:\n\n{missing_preview}\n\nChoose the input again before processing.')
            return
        if not running_frozen() and (not self.pipeline_path.exists()):
            messagebox.showerror(WINDOW_TITLE, f'Required pipeline file is missing:\n{self.pipeline_path}')
            return
        self.total_pdf_count = len(pdf_paths)
        self.completed_pdf_count = 0
        self.successful_pdf_count = 0
        self.failed_pdf_count = 0
        self.pipeline_worker_count = 0
        self.pipeline_reserved_cores = 0
        self.pipeline_running = True
        self.pipeline_finished = False
        self.review_waiting_for_processing = True
        self.finished_job_names.clear()
        self.loaded_output_dirs.clear()
        self.output_root = (self._selection_output_parent(pdf_paths) / OUTPUT_DIRECTORY_NAME).resolve()
        job_names = make_unique_job_names(pdf_paths)
        self.job_name_to_pdf_name = {job_name: pdf_path.name for job_name, pdf_path in zip(job_names, pdf_paths, strict=True)}
        self.job_name_to_batch_position = {job_name: index for index, job_name in enumerate(job_names, start=1)}
        self._clear_review_state()
        self.review_waiting_for_processing = True
        self._set_input_panel_collapsed(True)
        self._set_processing_controls()
        self.progress_bar['value'] = 0.0
        self.pdf_var.set(f'{pdf_paths[0].name} — PDF 1/{len(pdf_paths)}')
        self.progress_text_var.set(f'Processing 0/{len(pdf_paths)} PDFs — 0%')
        self.stat_var.set('Processing scans')
        self.candidate_var.set('')
        self.status_var.set('Starting pipeline...')
        self._show_processing_waiting_screen(caught_up=False)
        command = build_pipeline_command(pipeline_path=self.pipeline_path, pdf_paths=pdf_paths, output_root=self.output_root)
        threading.Thread(target=self._pipeline_worker, args=(command,), daemon=True).start()

    def _set_processing_controls(self) -> None:
        """Disable controls that could change input or start a second batch.

        The Show/Hide input-panel toggle deliberately remains enabled so the
        selected file list can still be inspected while processing runs."""
        self.choose_pdfs_button.configure(state='disabled')
        self.choose_folder_button.configure(state='disabled')
        self.process_button.configure(state='disabled')
        self.toggle_highlights_button.configure(state='disabled')
        self.open_pdf_button.configure(state='disabled')
        self.manual_digitize_button.configure(state='disabled')
        self.refresh_digitize_button.configure(state='disabled')
        self.back_button.configure(state='disabled')
        self._set_zoom_controls_enabled(False)
        self.next_candidate_button.configure(state='disabled')
        self.next_stat_button.configure(state='disabled')

    def _set_idle_controls(self) -> None:
        """Re-enable input controls after processing or review completion.

        Process Selected intentionally remains enabled with no input so pressing
        it can explain what is missing instead of failing silently."""
        self.choose_pdfs_button.configure(state='normal')
        self.choose_folder_button.configure(state='normal')
        self.process_button.configure(state='normal')

    def _pipeline_worker(self, command: list[str]) -> None:
        """Run the pipeline and forward each output line to the GUI thread."""
        try:
            environment = os.environ.copy()
            environment['PYTHONUNBUFFERED'] = '1'
            process = subprocess.Popen(command, cwd=self.script_dir, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1, creationflags=hidden_subprocess_creation_flags())
            self.pipeline_process = process
            if process.stdout is not None:
                for line in process.stdout:
                    self.event_queue.put(('pipeline_line', line.rstrip()))
            self.event_queue.put(('pipeline_finished', process.wait()))
        except Exception as error:
            self.event_queue.put(('pipeline_error', str(error)))
        finally:
            self.pipeline_process = None

    def _poll_events(self) -> None:
        """Handle worker-thread events without calling Tk from that thread."""
        try:
            while True:
                event_name, payload = self.event_queue.get_nowait()
                if event_name == 'pipeline_line':
                    self._handle_pipeline_line(str(payload))
                elif event_name == 'pipeline_finished':
                    self._handle_pipeline_finished(int(payload))
                elif event_name == 'pipeline_error':
                    self._handle_pipeline_error(str(payload))
        except queue.Empty:
            pass
        finally:
            self.root.after(EVENT_POLL_INTERVAL_MILLISECONDS, self._poll_events)

    def _update_processing_progress(self) -> None:
        """Keep PDF-level processing progress accurate while review happens."""
        total = max(1, self.total_pdf_count)
        finished = min(self.total_pdf_count, self.completed_pdf_count)
        percent = 100.0 * finished / total
        remaining = max(0, self.total_pdf_count - finished)
        active = min(self.pipeline_worker_count, remaining)
        queued = max(0, remaining - active)
        self.progress_bar['value'] = percent
        if self.pipeline_finished:
            self.progress_text_var.set(f'Processing complete — {finished}/{self.total_pdf_count} finished — {len(self.review_pdfs)} ready, {self.failed_pdf_count} failed')
            return
        parts = [f'{finished}/{self.total_pdf_count} finished — {percent:.0f}%', f'{len(self.review_pdfs)} ready']
        if self.failed_pdf_count:
            parts.append(f'{self.failed_pdf_count} failed')
        if active:
            parts.append(f'{active} processing')
        if queued:
            parts.append(f'{queued} queued')
        self.progress_text_var.set(' • '.join(parts))

    def _show_processing_waiting_screen(self, caught_up: bool) -> None:
        """Show a stable waiting state until another immutable PDF result is ready."""
        self.review_waiting_for_processing = True
        self._disable_candidate_controls()
        self.open_pdf_button.configure(state='disabled')
        self.next_stat_button.configure(state='disabled')
        self._update_back_button_state()
        self._update_processing_progress()
        if caught_up:
            self.pdf_var.set(f'Caught up — {self.completed_pdf_count}/{self.total_pdf_count} processing results finished')
            self.stat_var.set('Waiting for the next processed report')
            self.status_var.set('You are caught up. Background processing continues; review resumes automatically when another report is ready.')
            self._show_canvas_message("You're caught up with processing.\n\nThe next completed report will appear here automatically.\nYou can use Back to revisit earlier review decisions.")
        else:
            self.pdf_var.set(f'Waiting — 0/{self.total_pdf_count} reports ready')
            self.stat_var.set('Waiting for the first processed report')
            self.status_var.set('Processing is running in the background. Review starts automatically as soon as the first report is ready.')
            self._show_canvas_message('Processing reports...\n\nReview will begin automatically when the first report is ready.')

    def _resume_review_if_ready(self) -> None:
        """Resume from a waiting screen without disturbing an active review."""
        if not self.review_waiting_for_processing:
            return
        if not 0 <= self.review_pdf_index < len(self.review_pdfs):
            return
        self.review_waiting_for_processing = False
        self.review_complete = False
        self.status_var.set('A processed report is ready. Review continues while remaining reports process in the background.')
        self._show_current_review_item()

    def _handle_job_finished_event(self, event: dict[str, Any]) -> None:
        """Record one completed/failed PDF and expose successes immediately."""
        job_name = str(event.get('job_name', '')).strip()
        if not job_name or job_name in self.finished_job_names:
            return
        self.finished_job_names.add(job_name)
        self.completed_pdf_count = min(self.total_pdf_count, self.completed_pdf_count + 1)
        status = str(event.get('status', '')).casefold()
        if status == 'success':
            self.successful_pdf_count += 1
            batch_position = int(event.get('batch_position', self.job_name_to_batch_position.get(job_name, 0)) or 0)
            batch_item = {'status': 'success', 'job_name': job_name, 'source_pdf': event.get('source_pdf', ''), 'output_dir': event.get('output_dir', str(self.output_root / job_name) if self.output_root is not None else '')}
            try:
                self._load_single_review_pdf(batch_item=batch_item, batch_position=batch_position)
            except Exception as error:
                self.status_var.set(f'{job_name} finished but its review result could not be loaded yet: {error}')
        else:
            self.failed_pdf_count += 1
        self._update_processing_progress()
        self._resume_review_if_ready()

    def _handle_pipeline_event(self, event: dict[str, Any]) -> None:
        """Handle one structured pipeline progress event."""
        event_name = str(event.get('event', ''))
        if event_name == 'batch_started':
            self.pipeline_worker_count = max(1, int(event.get('worker_count', 1)))
            self.pipeline_reserved_cores = max(0, int(event.get('reserved_logical_cores', 0)))
            self._update_processing_progress()
            if self.review_waiting_for_processing and (not self.review_pdfs):
                self.status_var.set(f'Using {self.pipeline_worker_count} background worker(s); reserving about {self.pipeline_reserved_cores} logical core(s) for review and normal system use.')
            return
        if event_name == 'job_finished':
            self._handle_job_finished_event(event)
            return
        if event_name == 'batch_finished':
            self.successful_pdf_count = int(event.get('successful_count', self.successful_pdf_count))
            self.failed_pdf_count = int(event.get('failed_count', self.failed_pdf_count))
            self.completed_pdf_count = min(self.total_pdf_count, self.successful_pdf_count + self.failed_pdf_count)
            self._update_processing_progress()

    def _handle_pipeline_line(self, line: str) -> None:
        """Consume structured events, with legacy text completion as a fallback."""
        if not line:
            return
        if line.startswith(PIPELINE_EVENT_PREFIX):
            try:
                event = json.loads(line[len(PIPELINE_EVENT_PREFIX):])
            except json.JSONDecodeError:
                return
            if isinstance(event, dict):
                self._handle_pipeline_event(event)
            return
        completion_match = COMPLETION_PATTERN.match(line)
        if completion_match is not None:
            job_name = completion_match.group('job_name')
            status = 'success' if completion_match.group('status') == 'Completed' else 'failed'
            self._handle_job_finished_event({'event': 'job_finished', 'status': status, 'job_name': job_name, 'batch_position': self.job_name_to_batch_position.get(job_name, 0), 'source_pdf': '', 'output_dir': str(self.output_root / job_name) if self.output_root is not None and status == 'success' else ''})
            return
        if not self.review_pdfs and (not self.review_waiting_for_processing):
            self.status_var.set(line)

    def _reconcile_batch_results(self, batch_results: dict[str, Any]) -> None:
        """Load any successful result whose streaming event was missed."""
        raw_results = batch_results.get('results', [])
        if not isinstance(raw_results, list):
            raise ValueError('batch_results.json has no valid results list.')
        for batch_position, batch_item in enumerate(raw_results, start=1):
            if not isinstance(batch_item, dict):
                continue
            job_name = str(batch_item.get('job_name', '')).strip()
            if batch_item.get('status') == 'success':
                self._load_single_review_pdf(batch_item=batch_item, batch_position=batch_position)
            if job_name:
                self.finished_job_names.add(job_name)
        self.successful_pdf_count = int(batch_results.get('successful_count', len(self.review_pdfs)))
        self.failed_pdf_count = int(batch_results.get('failed_count', max(0, self.total_pdf_count - self.successful_pdf_count)))
        self.completed_pdf_count = min(self.total_pdf_count, self.successful_pdf_count + self.failed_pdf_count)

    def _handle_pipeline_finished(self, return_code: int) -> None:
        """Finalize processing without interrupting an in-progress review."""
        self.pipeline_running = False
        self.pipeline_finished = True
        if self.output_root is None:
            self._handle_pipeline_error('The output directory was not initialized.')
            return
        batch_results_path = self.output_root / BATCH_RESULTS_FILENAME
        if not batch_results_path.exists():
            self._handle_pipeline_error(f'The pipeline finished without creating:\n{batch_results_path}')
            return
        try:
            batch_results = load_json(batch_results_path)
            self._reconcile_batch_results(batch_results)
        except Exception as error:
            self._handle_pipeline_error(f'Could not load pipeline results: {error}')
            return
        self._update_processing_progress()
        if not self.review_pdfs:
            self._set_idle_controls()
            self.stat_var.set('No successful PDFs to review')
            self.candidate_var.set('')
            self.status_var.set(f'Pipeline exit code: {return_code}')
            self._show_canvas_message('No candidate results are available.')
            messagebox.showerror(WINDOW_TITLE, 'Processing completed, but no PDF produced a reviewable result.')
            return
        if self.review_waiting_for_processing:
            if self.review_pdf_index < len(self.review_pdfs):
                self._resume_review_if_ready()
            else:
                self._finish_review()
            return
        if self.review_pdf_index >= len(self.review_pdfs):
            self._finish_review()
            return
        if self.failed_pdf_count:
            self.status_var.set(f'Processing is complete; {self.failed_pdf_count} PDF(s) failed. Continue reviewing the successful reports.')

    def _handle_pipeline_error(self, error_text: str) -> None:
        """Stop background processing while preserving any reviewable results."""
        self.pipeline_running = False
        self.pipeline_finished = True
        if self.review_pdfs:
            self.status_var.set(f'Background processing stopped: {error_text}. Already completed reports remain reviewable.')
            self._update_processing_progress()
            if self.review_waiting_for_processing:
                if self.review_pdf_index < len(self.review_pdfs):
                    self._resume_review_if_ready()
                else:
                    self._finish_review()
            return
        self._set_idle_controls()
        self.stat_var.set('Processing failed')
        self.candidate_var.set('')
        self.status_var.set(error_text)
        self._show_canvas_message('The pipeline could not be completed.')
        messagebox.showerror(WINDOW_TITLE, error_text)

    @staticmethod
    def _tracker_value_text(value: Any, unit: str='') -> str:
        """Return a plain numeric tracker value with no comparator or unit text.

        Numeric inputs use the normal rounding rules. String inputs
        preserve the first listed numeric token, so values such as "< 0.0%"
        and "<0.01" become "0.0" and "0.01" rather than being rounded away."""
        if value is None or isinstance(value, bool):
            return ''
        if isinstance(value, (int, float)):
            numeric = float(value)
            if not math.isfinite(numeric):
                return ''
            return format_measurement_value(numeric, unit)
        text_value = str(value).strip()
        if not text_value:
            return ''
        match = TRACKER_NUMBER_PATTERN.search(text_value)
        if match is None:
            return ''
        numeric_text = match.group(0).replace(',', '.')
        try:
            numeric = float(numeric_text)
        except ValueError:
            return ''
        if not math.isfinite(numeric):
            return ''
        if '.' in numeric_text:
            whole, fraction = numeric_text.split('.', 1)
            whole = whole or '0'
            return f'{whole}.{fraction}'
        return numeric_text

    def _tracker_key_for_review_pdf(self, review_pdf: dict[str, Any]) -> str:
        """Return the stable tracker key used for one processed PDF."""
        return str(review_pdf['pdf_output_dir'])

    def _append_tracker_tree_row(self, review_pdf: dict[str, Any]) -> None:
        """Append one newly processed PDF without disturbing active tracker edits."""
        tracker_key = self._tracker_key_for_review_pdf(review_pdf)
        if tracker_key in self.tracker_key_to_iid:
            return
        row = self.tracker_rows.get(tracker_key, {'ID': '', 'AFib burden': '', 'HR': '', 'Activity': ''})
        item_id = f'tracker_{len(self.tracker_key_to_iid)}'
        self.tracker_key_to_iid[tracker_key] = item_id
        self.tracker_iid_to_key[item_id] = tracker_key
        self.tracker_tree.insert('', tk.END, iid=item_id, text=str(review_pdf['source_pdf_name']), values=(row.get('ID', ''), row.get('AFib burden', ''), row.get('HR', ''), row.get('Activity', '')))
        if self.tracker_active_key is None:
            self._select_tracker_key(tracker_key)

    def _update_tracker_tree_row(self, tracker_key: str) -> None:
        """Refresh one visible tracker row after an auto-fill or manual edit."""
        item_id = self.tracker_key_to_iid.get(tracker_key)
        row = self.tracker_rows.get(tracker_key)
        if item_id is None or row is None:
            return
        self.tracker_tree.item(item_id, values=(row.get('ID', ''), row.get('AFib burden', ''), row.get('HR', ''), row.get('Activity', '')))

    def _review_pdf_for_tracker_key(self, tracker_key: str) -> dict[str, Any] | None:
        """Return the review PDF associated with one tracker row."""
        for review_pdf in self.review_pdfs:
            if self._tracker_key_for_review_pdf(review_pdf) == tracker_key:
                return review_pdf
        return None

    def _review_pdf_index_for_tracker_key(self, tracker_key: str) -> int | None:
        """Return the review index associated with one tracker row."""
        for index, review_pdf in enumerate(self.review_pdfs):
            if self._tracker_key_for_review_pdf(review_pdf) == tracker_key:
                return index
        return None

    def _update_tracker_id_preview(self, *_args: Any) -> None:
        """Show the exact ID produced by the four visible ID components."""
        composed = compose_tracker_id(initials=self.tracker_initials_var.get(), birth_year=self.tracker_birth_year_var.get(), prefix=self.tracker_prefix_var.get(), suffix=self.tracker_suffix_var.get())
        self.tracker_id_preview_var.set(f'ID: {composed}' if composed else 'ID: —')

    def _open_tracker_csv(self) -> None:
        """Open the auto-saved tracker CSV in the system spreadsheet application."""
        if self.output_root is None:
            return
        tracker_path = self.output_root / TRACKER_CSV_FILENAME
        if not tracker_path.is_file():
            messagebox.showinfo(WINDOW_TITLE, 'The tracker CSV will appear after the first report is ready.')
            return
        try:
            open_with_default_application(tracker_path)
        except Exception as error:
            messagebox.showerror(WINDOW_TITLE, f'Could not open the tracker CSV:\n{error}')

    def _load_tracker_editor(self, tracker_key: str | None) -> None:
        """Load one row into the editable fields without marking manual edits."""
        self.tracker_editor_loading = True
        try:
            self.tracker_active_key = tracker_key
            if tracker_key is None or tracker_key not in self.tracker_rows:
                self.tracker_current_var.set('No processed report selected')
                self.tracker_initials_var.set('')
                self.tracker_birth_year_var.set('')
                self.tracker_prefix_var.set('')
                self.tracker_suffix_var.set('')
                self.tracker_afib_var.set('')
                self.tracker_hr_var.set('')
                self.tracker_activity_var.set('')
                self._update_tracker_id_preview()
                return
            row = self.tracker_rows[tracker_key]
            review_pdf = self._review_pdf_for_tracker_key(tracker_key)
            self.tracker_current_var.set(f"Selected report: {review_pdf['source_pdf_name']}" if review_pdf is not None else 'Selected result')
            self.tracker_initials_var.set(row.get('ID Initials', ''))
            self.tracker_birth_year_var.set(row.get('Birth Year', ''))
            self.tracker_prefix_var.set(row.get('ID Prefix', ''))
            self.tracker_suffix_var.set(row.get('ID Suffix', ''))
            self.tracker_afib_var.set(row.get('AFib burden', ''))
            self.tracker_hr_var.set(row.get('HR', ''))
            self.tracker_activity_var.set(row.get('Activity', ''))
            self._update_tracker_id_preview()
        finally:
            self.tracker_editor_loading = False

    def _select_tracker_key(self, tracker_key: str) -> None:
        """Highlight one tracker row and load it into the edit fields."""
        item_id = self.tracker_key_to_iid.get(tracker_key)
        if item_id is None:
            self._load_tracker_editor(tracker_key)
            return
        self.tracker_tree.selection_set(item_id)
        self.tracker_tree.focus(item_id)
        self.tracker_tree.see(item_id)
        self._load_tracker_editor(tracker_key)

    def _select_tracker_for_review_pdf(self, review_pdf: dict[str, Any]) -> None:
        """Keep the tracker highlight synchronized with review navigation."""
        tracker_key = self._tracker_key_for_review_pdf(review_pdf)
        if tracker_key in self.tracker_rows:
            self._select_tracker_key(tracker_key)

    def _on_tracker_tree_selected(self, _event: tk.Event[Any] | None=None) -> None:
        """Select a report and jump to its first metric only when the PDF changes."""
        selection = self.tracker_tree.selection()
        if not selection:
            return

        tracker_key = self.tracker_iid_to_key.get(str(selection[0]))
        if tracker_key is None:
            return

        # Preserve any edits from the row that was active before the click.
        self._commit_tracker_edits()

        current_pdf = self._current_review_pdf()
        current_key = (
            self._tracker_key_for_review_pdf(current_pdf)
            if current_pdf is not None
            else None
        )

        # Clicking the current PDF is purely a tracker/editor selection. It must
        # not reset the metric, candidate, visible page, zoom, or Manual Mode.
        if tracker_key == current_key:
            self._load_tracker_editor(tracker_key)
            return

        target_index = self._review_pdf_index_for_tracker_key(tracker_key)
        if target_index is None:
            self._load_tracker_editor(tracker_key)
            return

        if self.manual_selection_active:
            self._cancel_manual_selection(redraw=False)

        self.review_complete = False
        self.review_waiting_for_processing = False
        self.review_pdf_index = target_index
        self.review_stat_index = 0
        self.review_candidate_index = 0
        self.manual_search_active = False
        self.manual_review_candidate = None
        self._load_tracker_editor(tracker_key)
        self.status_var.set('Opened selected report at its first metric.')
        self._save_review_results(review_complete=False)
        self._show_current_review_item()

    def _commit_tracker_edits(self, _event: tk.Event[Any] | None=None) -> str | None:
        """Store tracker edits and compose the CSV ID from its visible components.

        Initials and birth year can be auto-filled independently. Prefix and
        suffix are never auto-filled; they exist only as user-entered ID parts."""
        if self.tracker_editor_loading or self.tracker_active_key is None:
            return None
        tracker_key = self.tracker_active_key
        row = self.tracker_rows.get(tracker_key)
        if row is None:
            return None
        component_values = {'ID Initials': ''.join((character for character in self.tracker_initials_var.get().strip().upper() if character.isalpha())), 'Birth Year': self.tracker_birth_year_var.get().strip(), 'ID Prefix': self.tracker_prefix_var.get().strip(), 'ID Suffix': self.tracker_suffix_var.get().strip()}
        measurement_values = {'AFib burden': self._tracker_value_text(self.tracker_afib_var.get()), 'HR': self._tracker_value_text(self.tracker_hr_var.get()), 'Activity': self._tracker_value_text(self.tracker_activity_var.get())}
        manual_fields = self.tracker_manual_fields.setdefault(tracker_key, set())
        changed = False
        for field_name, edited_value in component_values.items():
            if row.get(field_name, '') != edited_value:
                row[field_name] = edited_value
                manual_fields.add(field_name)
                changed = True
        for field_name, edited_value in measurement_values.items():
            if row.get(field_name, '') != edited_value:
                row[field_name] = edited_value
                manual_fields.add(field_name)
                changed = True
        composed_id = compose_tracker_id(initials=row.get('ID Initials', ''), birth_year=row.get('Birth Year', ''), prefix=row.get('ID Prefix', ''), suffix=row.get('ID Suffix', ''))
        if row.get('ID', '') != composed_id:
            row['ID'] = composed_id
            changed = True
        self._update_tracker_id_preview()
        if changed:
            self._update_tracker_tree_row(tracker_key)
            self._save_tracker_csv()
        if _event is not None and getattr(_event, 'keysym', '') == 'Return':
            return 'break'
        return None

    def _set_tracker_auto_value(self, tracker_key: str, field_name: str, value: Any, unit: str='') -> None:
        """Auto-fill one tracker value unless the reviewer has overridden it."""
        if field_name not in TRACKER_CSV_COLUMNS:
            return
        row = self.tracker_rows.get(tracker_key)
        if row is None:
            return
        if field_name in self.tracker_manual_fields.get(tracker_key, set()):
            return
        value_text = self._tracker_value_text(value, unit=unit)
        if row.get(field_name, '') == value_text:
            return
        row[field_name] = value_text
        self._update_tracker_tree_row(tracker_key)
        if self.tracker_active_key == tracker_key:
            self._load_tracker_editor(tracker_key)

    def _set_tracker_explicit_value(self, review_pdf: dict[str, Any], field_name: str, value: Any, unit: str='') -> None:
        """Set one tracker field from an explicit result-popup action.

        Because the reviewer deliberately pressed Export Value, this
        becomes a manual field and is not silently replaced by later auto-fill."""
        if field_name not in TRACKER_CSV_COLUMNS:
            return
        tracker_key = self._tracker_key_for_review_pdf(review_pdf)
        row = self.tracker_rows.get(tracker_key)
        if row is None:
            return
        row[field_name] = self._tracker_value_text(value, unit=unit)
        self.tracker_manual_fields.setdefault(tracker_key, set()).add(field_name)
        self._update_tracker_tree_row(tracker_key)
        if self.tracker_active_key == tracker_key:
            self._load_tracker_editor(tracker_key)
        self._save_tracker_csv()

    def _selected_demographic_candidate(self, review_pdf: dict[str, Any], stat_key: str) -> dict[str, Any] | None:
        """Return the reviewed demographic choice, or the OCR-selected default when
        that demographic has not yet been reviewed."""
        tracker_key = self._tracker_key_for_review_pdf(review_pdf)
        stats = self.review_decisions.get(tracker_key, {}).get('stats', {})
        if stat_key in stats:
            selected = stats[stat_key].get('selected_candidate')
            return selected if isinstance(selected, dict) else None
        machine_selected = review_pdf.get('machine_selected_demographics', {})
        selected = machine_selected.get(stat_key) if isinstance(machine_selected, dict) else None
        return selected if isinstance(selected, dict) else None

    def _refresh_tracker_id_from_demographics(self, review_pdf: dict[str, Any]) -> None:
        """Auto-fill initials/year independently, then compose the CSV ID."""
        tracker_key = self._tracker_key_for_review_pdf(review_pdf)
        row = self.tracker_rows.get(tracker_key)
        if row is None:
            return
        name_candidate = self._selected_demographic_candidate(review_pdf, 'patient_name')
        dob_candidate = self._selected_demographic_candidate(review_pdf, 'date_of_birth')
        initials, birth_year = patient_identity_parts_from_candidates(name_candidate=name_candidate, dob_candidate=dob_candidate)
        manual_fields = self.tracker_manual_fields.get(tracker_key, set())
        changed = False
        if 'ID Initials' not in manual_fields:
            if row.get('ID Initials', '') != initials:
                row['ID Initials'] = initials
                changed = True
        if 'Birth Year' not in manual_fields:
            if row.get('Birth Year', '') != birth_year:
                row['Birth Year'] = birth_year
                changed = True
        composed_id = compose_tracker_id(initials=row.get('ID Initials', ''), birth_year=row.get('Birth Year', ''), prefix=row.get('ID Prefix', ''), suffix=row.get('ID Suffix', ''))
        if row.get('ID', '') != composed_id:
            row['ID'] = composed_id
            changed = True
        if changed:
            self._update_tracker_tree_row(tracker_key)
            if self.tracker_active_key == tracker_key:
                self._load_tracker_editor(tracker_key)
            self._save_tracker_csv()

    def _recompose_tracker_id(self, tracker_key: str) -> bool:
        """Compose the CSV ID from the row's separately stored identity parts."""
        row = self.tracker_rows.get(tracker_key)
        if row is None:
            return False
        composed = compose_tracker_id(initials=row.get('ID Initials', ''), birth_year=row.get('Birth Year', ''), prefix=row.get('ID Prefix', ''), suffix=row.get('ID Suffix', ''))
        if row.get('ID', '') == composed:
            return False
        row['ID'] = composed
        return True

    def _preview_autofill_from_ocr_candidate(self, review_pdf: dict[str, Any], stat_key: str, candidate: dict[str, Any] | None) -> None:
        """Mirror the currently visible OCR candidate into the tracker immediately.

        This is a preview/working value, not a manual override:
        - candidate 1 shows candidate 1's value;
        - Next Candidate immediately changes to candidate 2, etc.;
        - exhausting the candidate list clears that target's automatic value."""
        if stat_key not in {'patient_name', 'date_of_birth', 'afib_burden'}:
            return
        tracker_key = self._tracker_key_for_review_pdf(review_pdf)
        row = self.tracker_rows.get(tracker_key)
        if row is None:
            return
        manual_fields = self.tracker_manual_fields.get(tracker_key, set())
        changed = False
        if stat_key == 'patient_name':
            if 'ID Initials' not in manual_fields:
                initials = name_initials_from_candidate(candidate)
                if row.get('ID Initials', '') != initials:
                    row['ID Initials'] = initials
                    changed = True
            changed = self._recompose_tracker_id(tracker_key) or changed
        elif stat_key == 'date_of_birth':
            if 'Birth Year' not in manual_fields:
                birth_year = birth_year_from_candidate(candidate)
                if row.get('Birth Year', '') != birth_year:
                    row['Birth Year'] = birth_year
                    changed = True
            changed = self._recompose_tracker_id(tracker_key) or changed
        elif stat_key == 'afib_burden':
            if 'AFib burden' not in manual_fields:
                value_text = ''
                if isinstance(candidate, dict):
                    listed_value = candidate.get('matched_text') or candidate.get('display_value') or candidate.get('value_percent') or ''
                    value_text = self._tracker_value_text(listed_value)
                if row.get('AFib burden', '') != value_text:
                    row['AFib burden'] = value_text
                    changed = True
        if not changed:
            return
        self._update_tracker_tree_row(tracker_key)
        if self.tracker_active_key == tracker_key:
            self._load_tracker_editor(tracker_key)
        self._save_tracker_csv()

    def _identity_requirement_met(self, review_pdf: dict[str, Any]) -> bool:
        """Return True when Initials or Birth Year has a non-empty tracker value."""
        tracker_key = self._tracker_key_for_review_pdf(review_pdf)
        row = self.tracker_rows.get(tracker_key, {})
        return bool(str(row.get('ID Initials', '')).strip() or str(row.get('Birth Year', '')).strip())

    def _tracker_value_from_digitization(
        self,
        result: dict[str, Any],
    ) -> tuple[str, float, str] | None:
        """Route one graph result to HR or Activity strictly from its selected measurement unit."""
        statistics = result.get('statistics', {})
        if not isinstance(statistics, dict):
            return None
        try:
            average = float(statistics['average'])
        except (KeyError, TypeError, ValueError):
            return None

        selection = result.get('measurement_selection', {})
        tracker_field = (
            str(selection.get('tracker_field', '')).strip()
            if isinstance(selection, dict)
            else ''
        )
        unit = str(statistics.get('unit', '')).strip()
        normalized_unit = normalize_measurement_unit(unit)

        if not tracker_field:
            if normalized_unit == 'bpm':
                tracker_field = 'HR'
            elif normalized_unit in {
                ACTIVITY_UNIT_PERCENT,
                ACTIVITY_UNIT_HOURS_PER_DAY,
                ACTIVITY_UNIT_MINUTES_PER_DAY,
            }:
                tracker_field = 'Activity'

        if tracker_field == 'HR':
            return ('HR', average, 'bpm')

        if tracker_field == 'Activity':
            if normalized_unit in {
                ACTIVITY_UNIT_PERCENT,
                ACTIVITY_UNIT_HOURS_PER_DAY,
                ACTIVITY_UNIT_MINUTES_PER_DAY,
            }:
                value = convert_activity_value(
                    average,
                    source_unit=normalized_unit,
                    target_unit=ACTIVITY_UNIT_MINUTES_PER_DAY,
                )
                return ('Activity', value, ACTIVITY_UNIT_MINUTES_PER_DAY)
            return ('Activity', average, unit)

        return None

    def _autofill_tracker_from_digitization(
        self,
        review_pdf: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Immediately populate the tracker field selected by the graph measurement unit."""
        routed = self._tracker_value_from_digitization(result)
        if routed is None:
            return
        field_name, value, unit = routed
        tracker_key = self._tracker_key_for_review_pdf(review_pdf)
        self._set_tracker_auto_value(
            tracker_key,
            field_name,
            value,
            unit=unit,
        )
        self._save_tracker_csv()

    def _autofill_tracker_from_review_choice(self, review_pdf: dict[str, Any], stat_key: str, selected_candidate: dict[str, Any] | None) -> None:
        """Update the appropriate tracker field from the accepted review choice."""
        tracker_key = self._tracker_key_for_review_pdf(review_pdf)
        manual_digitization = (
            selected_candidate.get('manual_digitization')
            if isinstance(selected_candidate, dict)
            else None
        )
        if isinstance(manual_digitization, dict):
            routed = self._tracker_value_from_digitization(manual_digitization)
            if routed is not None:
                field_name, value, unit = routed
                self._set_tracker_auto_value(
                    tracker_key,
                    field_name,
                    value,
                    unit=unit,
                )
            return

        if stat_key in {'patient_name', 'date_of_birth'}:
            self._refresh_tracker_id_from_demographics(review_pdf)
            return
        if stat_key == 'afib_burden':
            value: Any = ''
            if selected_candidate is not None:
                value = selected_candidate.get('matched_text') or selected_candidate.get('display_value') or selected_candidate.get('value_percent') or ''
            self._set_tracker_auto_value(tracker_key, 'AFib burden', value)
            return
        tracker_field = {'afib_burden': 'AFib burden', 'average_heart_rate': 'HR', 'average_daily_activity': 'Activity'}.get(stat_key)
        if tracker_field is None:
            return
        self._set_tracker_auto_value(tracker_key, tracker_field, '')

    def _save_tracker_csv(self) -> None:
        """Atomically write an Excel/Google-Sheets-friendly tracker CSV.

        ID remains text. AFib burden, HR, and Activity are written as bare
        decimal numbers only: no percent signs, comparators, units, or prose.
        UTF-8 with BOM improves direct Excel opening while remaining Sheets-safe."""
        if self.output_root is None:
            return
        output_path = self.output_root / TRACKER_CSV_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f'.{output_path.name}.tmp')
        try:
            with temporary_path.open('w', encoding='utf-8-sig', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(TRACKER_CSV_COLUMNS), lineterminator='\r\n')
                writer.writeheader()
                for review_pdf in sorted(self.review_pdfs, key=lambda item: int(item.get('batch_position', 0))):
                    tracker_key = self._tracker_key_for_review_pdf(review_pdf)
                    row = self.tracker_rows.get(tracker_key, {})
                    output_row = {'ID': str(row.get('ID', '')).strip()}
                    for column in TRACKER_NUMERIC_COLUMNS:
                        numeric_text = self._tracker_value_text(row.get(column, ''))
                        output_row[column] = numeric_text
                    writer.writerow(output_row)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load_single_review_pdf(self, batch_item: dict[str, Any], batch_position: int) -> bool:
        """Load one immutable completed PDF result without resetting review state."""
        if batch_item.get('status') != 'success':
            return False
        pdf_output_dir = Path(str(batch_item['output_dir'])).expanduser().resolve()
        output_key = str(pdf_output_dir)
        if output_key in self.loaded_output_dirs:
            return False
        pdf_results_path = pdf_output_dir / PDF_RESULTS_FILENAME
        if not pdf_results_path.exists():
            raise FileNotFoundError(f'Completed result is missing: {pdf_results_path}')
        pdf_results = load_json(pdf_results_path)
        source_pdf_name = str(pdf_results.get('source_pdf_name', Path(str(batch_item.get('source_pdf', 'report.pdf'))).name))
        patient_name_target = pdf_results.get('patient_name', {})
        date_of_birth_target = pdf_results.get('date_of_birth', {})
        selected_patient_name = patient_name_target.get('selected') if isinstance(patient_name_target, dict) else None
        selected_date_of_birth = date_of_birth_target.get('selected') if isinstance(date_of_birth_target, dict) else None
        initial_initials, initial_birth_year = patient_identity_parts_from_candidates(name_candidate=selected_patient_name if isinstance(selected_patient_name, dict) else None, dob_candidate=selected_date_of_birth if isinstance(selected_date_of_birth, dict) else None)
        initial_patient_id = compose_tracker_id(initials=initial_initials, birth_year=initial_birth_year)
        afib_target = pdf_results.get('afib_burden', {})
        selected_afib = afib_target.get('selected') if isinstance(afib_target, dict) else None
        afib_tracker_value = ''
        if isinstance(selected_afib, dict):
            listed_value = selected_afib.get('matched_text') or selected_afib.get('display_value') or selected_afib.get('value_percent') or ''
            afib_tracker_value = self._tracker_value_text(listed_value)
        stat_candidates: dict[str, list[dict[str, Any]]] = {}
        for stat_key, _display_name in STAT_ORDER:
            target_data = pdf_results.get(stat_key, {})
            raw_candidates = target_data.get('candidates', [])
            if not isinstance(raw_candidates, list):
                raw_candidates = []
            valid_candidates: list[dict[str, Any]] = []
            for candidate in raw_candidates:
                if not isinstance(candidate, dict):
                    continue
                page_path = resolve_candidate_page(pdf_output_dir=pdf_output_dir, candidate=candidate)
                focus_bbox = valid_bbox(candidate.get('review_crop_bbox'))
                highlights = candidate_highlights(candidate)
                if page_path is None or focus_bbox is None:
                    continue
                candidate_copy = dict(candidate)
                candidate_copy['_resolved_page_image'] = str(page_path)
                candidate_copy['_focus_bbox'] = focus_bbox
                candidate_copy['_highlight_regions'] = highlights
                valid_candidates.append(candidate_copy)
            stat_candidates[stat_key] = unique_page_candidates(valid_candidates)
        deskewed_pages = resolve_deskewed_pages(
            pdf_output_dir=pdf_output_dir,
            pdf_results=pdf_results,
        )
        review_pdf = {'source_pdf': str(batch_item.get('source_pdf', pdf_results.get('source_pdf', ''))), 'source_pdf_name': source_pdf_name, 'batch_position': batch_position if batch_position > 0 else len(self.review_pdfs) + 1, 'pdf_output_dir': output_key, 'deskewed_pages': [str(path) for path in deskewed_pages], 'stat_candidates': stat_candidates, 'machine_selected_demographics': {'patient_name': selected_patient_name if isinstance(selected_patient_name, dict) else None, 'date_of_birth': selected_date_of_birth if isinstance(selected_date_of_birth, dict) else None}}
        self.review_pdfs.append(review_pdf)
        self.loaded_output_dirs.add(output_key)
        tracker_key = output_key
        self.review_decisions[tracker_key] = {'source_pdf': review_pdf['source_pdf'], 'source_pdf_name': source_pdf_name, 'batch_position': review_pdf['batch_position'], 'pdf_output_dir': output_key, 'stats': {}}
        self.tracker_rows[tracker_key] = {'ID': initial_patient_id, 'ID Initials': initial_initials, 'Birth Year': initial_birth_year, 'ID Prefix': '', 'ID Suffix': '', 'AFib burden': afib_tracker_value, 'HR': '', 'Activity': ''}
        self.tracker_manual_fields[tracker_key] = set()
        self._append_tracker_tree_row(review_pdf)
        self._save_tracker_csv()
        self._save_review_results(review_complete=False)
        return True

    def _current_review_pdf(self) -> dict[str, Any] | None:
        """Return the current PDF review record when one exists."""
        if not 0 <= self.review_pdf_index < len(self.review_pdfs):
            return None
        return self.review_pdfs[self.review_pdf_index]

    def _current_stat(self) -> tuple[str, str] | None:
        """Return the current stat key and display name."""
        if not 0 <= self.review_stat_index < len(STAT_ORDER):
            return None
        return STAT_ORDER[self.review_stat_index]

    def _current_candidates(self) -> list[dict[str, Any]]:
        """Return the current PDF and stat's candidate list."""
        review_pdf = self._current_review_pdf()
        stat_data = self._current_stat()
        if review_pdf is None or stat_data is None:
            return []
        return review_pdf['stat_candidates'].get(stat_data[0], [])

    def _visible_candidate(self) -> dict[str, Any] | None:
        """Return the candidate at the current review position."""
        candidates = self._current_candidates()
        if not 0 <= self.review_candidate_index < len(candidates):
            return None
        return candidates[self.review_candidate_index]

    def _current_pdf_path(self) -> Path | None:
        """Resolve the original PDF for the current review item."""
        review_pdf = self._current_review_pdf()
        if review_pdf is None:
            return None
        raw_path = str(review_pdf.get('source_pdf', '')).strip()
        if raw_path:
            source_path = Path(raw_path).expanduser()
            if source_path.exists() and source_path.is_file():
                return source_path.resolve()
        if self.selected_folder is not None:
            fallback = self.selected_folder / review_pdf['source_pdf_name']
            if fallback.exists() and fallback.is_file():
                return fallback.resolve()
        return None

    def _candidate_navigation_key(self, candidate: dict[str, Any]) -> str:
        """Return a stable public-data key for locating a prior selection."""
        public_candidate = {key: value for key, value in candidate.items() if not key.startswith('_') and key != 'manual_digitization'}
        return json.dumps(public_candidate, sort_keys=True, separators=(',', ':'), default=str)

    def _remove_decision_for_position(self, pdf_index: int, stat_index: int) -> dict[str, Any] | None:
        """Remove and return the saved decision for one reopened statistic."""
        if not 0 <= pdf_index < len(self.review_pdfs):
            return None
        if not 0 <= stat_index < len(STAT_ORDER):
            return None
        review_pdf = self.review_pdfs[pdf_index]
        decision_key = str(review_pdf['pdf_output_dir'])
        stat_key = STAT_ORDER[stat_index][0]
        stats = self.review_decisions.get(decision_key, {}).get('stats', {})
        return stats.pop(stat_key, None)

    def _candidate_index_from_decision(self, candidates: list[dict[str, Any]], decision: dict[str, Any] | None) -> int:
        """Restore a reopened statistic to its selected or exhausted position."""
        if not candidates:
            return 0
        if not isinstance(decision, dict):
            return 0
        if decision.get('status') == 'not_found':
            return len(candidates)
        selected_candidate = decision.get('selected_candidate')
        if not isinstance(selected_candidate, dict):
            return 0
        selected_key = self._candidate_navigation_key(selected_candidate)
        for index, candidate in enumerate(candidates):
            if self._candidate_navigation_key(candidate) == selected_key:
                return index
        return 0

    def _can_go_back(self) -> bool:
        """Return True when an earlier candidate or statistic can be reopened."""
        if self.review_complete:
            return bool(self.review_pdfs)
        if not self.review_pdfs:
            return False
        if self.review_candidate_index > 0:
            return True
        return self.review_pdf_index > 0 or self.review_stat_index > 0

    def _update_back_button_state(self) -> None:
        """Keep the Back control synchronized with the current review position."""
        if self.manual_selection_active:
            self.back_button.configure(state='disabled')
            return
        self.back_button.configure(state='normal' if self._can_go_back() else 'disabled')

    def _go_back(self) -> None:
        """Move to the previous candidate, statistic, or PDF as appropriate."""
        if str(self.back_button['state']) == 'disabled':
            return
        if self.manual_selection_active:
            return
        if self.review_waiting_for_processing:
            self.review_waiting_for_processing = False
        if self.review_complete:
            self.review_complete = False
            self.review_pdf_index = len(self.review_pdfs) - 1
            self.review_stat_index = len(STAT_ORDER) - 1
            prior_decision = self._remove_decision_for_position(self.review_pdf_index, self.review_stat_index)
            self.review_candidate_index = self._candidate_index_from_decision(self._current_candidates(), prior_decision)
            self._save_review_results(review_complete=False)
            self.status_var.set('Reopened the final statistic. Its prior decision was removed.')
            self._show_current_review_item()
            return
        candidates = self._current_candidates()
        if self.review_candidate_index > 0:
            self.review_candidate_index -= 1
            self.status_var.set('Moved to the previous candidate.')
            self._show_current_review_item()
            return
        previous_pdf_index = self.review_pdf_index
        previous_stat_index = self.review_stat_index - 1
        if previous_stat_index < 0:
            previous_pdf_index -= 1
            previous_stat_index = len(STAT_ORDER) - 1
        if previous_pdf_index < 0:
            self._update_back_button_state()
            return
        self.review_pdf_index = previous_pdf_index
        self.review_stat_index = previous_stat_index
        prior_decision = self._remove_decision_for_position(previous_pdf_index, previous_stat_index)
        self.review_candidate_index = self._candidate_index_from_decision(self._current_candidates(), prior_decision)
        self._save_review_results(review_complete=False)
        self.status_var.set('Reopened the previous statistic. Its prior decision was removed.')
        self._show_current_review_item()

    @staticmethod
    def _review_candidate_value_text(stat_key: str, candidate: dict[str, Any]) -> str:
        """Return a concise OCR value for the candidate header when available."""
        if stat_key in {'patient_name', 'date_of_birth', 'afib_burden'}:
            value = candidate.get('display_value')
            if value not in (None, ''):
                return str(value).strip()
        return ''

    def _show_current_review_item(self) -> None:
        """Display the current PDF, stat, candidate, page, and initial viewport."""
        review_pdf = self._current_review_pdf()
        stat_data = self._current_stat()
        if review_pdf is None or stat_data is None:
            if self.pipeline_running:
                self._show_processing_waiting_screen(caught_up=True)
            else:
                self._finish_review()
            return
        self.review_waiting_for_processing = False
        self.manual_search_active = False
        self.manual_review_candidate = None
        stat_key, stat_display_name = stat_data
        candidates = self._current_candidates()
        batch_position = int(review_pdf['batch_position'])
        self._select_tracker_for_review_pdf(review_pdf)
        self.pdf_var.set(f"{review_pdf['source_pdf_name']} — PDF {batch_position}/{self.total_pdf_count}")
        self.stat_var.set(stat_display_name)
        self.next_stat_button.configure(state='normal')
        self._update_back_button_state()
        self.open_pdf_button.configure(state='normal' if self._current_pdf_path() is not None else 'disabled')
        if not candidates:
            self._preview_autofill_from_ocr_candidate(review_pdf=review_pdf, stat_key=stat_key, candidate=None)
            self._disable_candidate_controls()
            self.candidate_var.set('No candidates retained')
            self.next_stat_button.configure(text='Next Metric')
            self._show_canvas_message(f'No {stat_display_name.lower()} candidates were retained.\nUse Manual Mode, or continue as not found.', show_open_pdf=True)
            return
        if self.review_candidate_index >= len(candidates):
            self._preview_autofill_from_ocr_candidate(review_pdf=review_pdf, stat_key=stat_key, candidate=None)
            self._disable_candidate_controls()
            self.candidate_var.set(f'No more candidates — reviewed {len(candidates)}/{len(candidates)}')
            self.next_stat_button.configure(text='Next Metric')
            self._show_canvas_message('No more automated candidates.\nUse Manual Mode, or press Next Metric to record this metric as not found.', show_open_pdf=True)
            return
        candidate = candidates[self.review_candidate_index]
        self._preview_autofill_from_ocr_candidate(review_pdf=review_pdf, stat_key=stat_key, candidate=candidate)
        self.current_candidate = candidate
        self.current_focus_bbox = list(candidate['_focus_bbox'])
        self.current_highlight_regions = list(candidate.get('_highlight_regions', []))
        self.current_digitization_result = candidate.get('_manual_digitization')
        self.manual_selection_bbox = list(self.current_digitization_result['plot_bbox']) if isinstance(self.current_digitization_result, dict) and isinstance(self.current_digitization_result.get('plot_bbox'), list) else None
        candidate_number = self.review_candidate_index + 1
        value_text = self._review_candidate_value_text(stat_key, candidate)
        value_segment = f' — {value_text}' if value_text else ''
        hit_count = int(candidate.get('ocr_hit_count', 0))
        hits_segment = f' — {hit_count} OCR hits' if hit_count > 1 else ''
        self.candidate_var.set(
            f'Candidate {candidate_number}/{len(candidates)}{value_segment}{hits_segment}'
        )
        self.toggle_highlights_button.configure(text='Hide Markings' if self.highlights_visible else 'Mark Metric', state='normal' if self.current_highlight_regions or isinstance(self.current_digitization_result, dict) else 'disabled')
        self.manual_digitize_button.configure(text='Interpret Graph', state='normal')
        self.next_candidate_button.configure(state='normal')
        self.next_stat_button.configure(text='Next Metric')
        self._load_candidate_page(page_path=Path(str(candidate['_resolved_page_image'])), focus_bbox=self.current_focus_bbox)

    def _disable_candidate_controls(self) -> None:
        """Disable controls that require a currently visible candidate."""
        self.current_candidate = None
        self.current_focus_bbox = None
        self.current_highlight_regions = []
        self.current_digitization_result = None
        self.manual_selection_bbox = None
        self._cancel_manual_selection(redraw=False)
        self.toggle_highlights_button.configure(state='disabled')
        self.manual_digitize_button.configure(state='disabled')
        self.refresh_digitize_button.configure(state='disabled')
        self._set_zoom_controls_enabled(False)
        self.next_candidate_button.configure(state='disabled')

    def _toggle_highlights(self) -> None:
        """Toggle overlays while preserving the current viewport and preference."""
        if str(self.toggle_highlights_button['state']) == 'disabled':
            return
        self.highlights_visible = not self.highlights_visible
        self.toggle_highlights_button.configure(text='Hide Markings' if self.highlights_visible else 'Mark Metric')
        self._redraw_overlays()

    def _open_current_pdf_externally(self) -> None:
        """Open the current original PDF in the operating system's PDF viewer."""
        pdf_path = self._current_pdf_path()
        if pdf_path is None:
            messagebox.showerror(WINDOW_TITLE, 'The original PDF could not be located.')
            self.open_pdf_button.configure(state='disabled')
            return
        try:
            open_with_default_application(pdf_path)
        except Exception as error:
            messagebox.showerror(WINDOW_TITLE, f'Could not open the PDF:\n{error}')

    def _current_deskewed_pages(self) -> list[Path]:
        """Return every currently available deskewed page in reading order."""
        review_pdf = self._current_review_pdf()
        if review_pdf is None:
            return []
        pages: list[Path] = []
        for raw_path in review_pdf.get('deskewed_pages', []):
            page_path = Path(str(raw_path)).expanduser()
            if page_path.is_file():
                pages.append(page_path.resolve())
        unique = {str(path): path for path in pages}
        return sorted(unique.values(), key=deskewed_page_sort_key)

    def _set_page_sequence(self, current_page: Path) -> None:
        """Set the page sequence without changing candidate or metric review position."""
        current_page = current_page.resolve()
        pages = self._current_deskewed_pages()
        if current_page not in pages and current_page.is_file():
            pages.append(current_page)
            pages.sort(key=deskewed_page_sort_key)
        self.current_pdf_pages = pages
        try:
            self.current_pdf_page_index = pages.index(current_page)
        except ValueError:
            self.current_pdf_page_index = 0

    def _update_page_navigation_controls(self) -> None:
        """Float page-back/page-forward controls at the lower-left of the PDF canvas."""
        if self.current_source_image is None or not self.current_pdf_pages:
            self.page_nav_frame.place_forget()
            self.page_nav_var.set('Page —/—')
            return
        count = len(self.current_pdf_pages)
        self.current_pdf_page_index = max(
            0,
            min(self.current_pdf_page_index, count - 1),
        )
        index = self.current_pdf_page_index
        self.page_nav_var.set(f'Page {index + 1}/{count}')
        locked = self.manual_selection_active
        self.page_back_button.configure(
            state='disabled' if locked or index <= 0 else 'normal'
        )
        self.page_forward_button.configure(
            state='disabled' if locked or index >= count - 1 else 'normal'
        )
        self.page_nav_frame.place(x=10, rely=1.0, y=-10, anchor='sw')

    def _current_page_matches_candidate(self, candidate: dict[str, Any] | None) -> bool:
        """Return whether the visible page belongs to one review candidate."""
        if candidate is None or self.current_page_path is None:
            return False
        raw_path = candidate.get('_resolved_page_image')
        if not raw_path:
            return False
        return Path(str(raw_path)).resolve() == self.current_page_path.resolve()

    def _sync_overlays_to_visible_page(self) -> None:
        """Show review/manual overlays only on the deskewed page that owns them."""
        visible_candidate = self._visible_candidate()
        if self._current_page_matches_candidate(visible_candidate):
            self.current_highlight_regions = list(
                visible_candidate.get('_highlight_regions', [])
            )
            self.current_digitization_result = visible_candidate.get('_manual_digitization')
            self.manual_selection_bbox = (
                list(self.current_digitization_result['plot_bbox'])
                if isinstance(self.current_digitization_result, dict)
                and isinstance(self.current_digitization_result.get('plot_bbox'), list)
                else None
            )
            return

        manual_candidate = self.manual_review_candidate
        if self._current_page_matches_candidate(manual_candidate):
            self.current_highlight_regions = []
            self.current_digitization_result = manual_candidate.get('_manual_digitization')
            self.manual_selection_bbox = (
                list(self.current_digitization_result['plot_bbox'])
                if isinstance(self.current_digitization_result, dict)
                and isinstance(self.current_digitization_result.get('plot_bbox'), list)
                else None
            )
            return

        self.current_highlight_regions = []
        self.current_digitization_result = None
        self.manual_selection_bbox = None

    def _load_pdf_page(
        self,
        page_path: Path,
        focus_bbox: list[int] | None = None,
    ) -> None:
        """Load one deskewed page for candidate review or whole-PDF browsing."""
        self._close_statistics_window()
        page_path = page_path.expanduser().resolve()
        if self.current_source_image is not None:
            self.current_source_image.close()
            self.current_source_image = None
        try:
            with Image.open(page_path) as opened_image:
                self.current_source_image = opened_image.convert('RGB')
        except Exception as error:
            self.current_source_image = None
            self.current_display_image = None
            self.current_page_path = None
            self._set_zoom_controls_enabled(False)
            self._update_page_navigation_controls()
            self._show_canvas_message(f'Could not open deskewed page:\n{error}')
            return

        self.current_page_path = page_path
        if page_path in self.current_pdf_pages:
            self.current_pdf_page_index = self.current_pdf_pages.index(page_path)
        self.current_canvas_message = None
        self.show_manual_pdf_search_button = False
        self._sync_overlays_to_visible_page()
        self._set_zoom_controls_enabled(True)
        self.root.update_idletasks()

        if focus_bbox is None:
            self.current_zoom_factor = 1.0
            self._render_current_image(
                anchor_source_fraction=(0.5, 0.5),
                anchor_canvas_position=None,
            )
        else:
            self._zoom_to_source_bbox(focus_bbox)

        stat_data = self._current_stat()
        stat_key = stat_data[0] if stat_data is not None else ''
        self.manual_digitize_button.configure(
            state='normal' if self.current_source_image is not None else 'disabled'
        )
        self.refresh_digitize_button.configure(
            state='normal'
            if isinstance(self.current_digitization_result, dict)
            else 'disabled'
        )
        self.digitization_bounds_dirty = False
        self._update_digitization_controls()
        self._update_page_navigation_controls()

    def _update_digitization_controls(self) -> None:
        """Show actions for the graph mode on the visible page."""
        result = self.current_digitization_result
        if not isinstance(result, dict):
            self.graph_tools_hint_var.set('Select a graph to interpret.')
            self.refresh_digitize_button.configure(text='Refresh Interpretation')
        elif result.get('interpretation_mode') == 'histogram':
            self.graph_tools_hint_var.set('Drag a white column-top handle to correct a histogram bar. Review Statistics to export the updated value.')
            self.refresh_digitize_button.configure(text='Review Statistics')
        else:
            self.graph_tools_hint_var.set('Drag any orange edge within the original line graph selection, then press Refresh Interpretation.')
            self.refresh_digitize_button.configure(text='Refresh Interpretation *' if self.digitization_bounds_dirty else 'Refresh Interpretation')

    def _open_current_pdf_for_manual_search(self) -> None:
        """Open every available deskewed page inside for Manual Mode."""
        pages = self._current_deskewed_pages()
        if not pages:
            messagebox.showerror(
                WINDOW_TITLE,
                'No deskewed pages are available for Manual Mode.\n\n'
                'Manual Mode requires the processing pipeline to retain '
                'the complete deskewed page set.',
            )
            return

        self.manual_search_active = True
        self.manual_review_candidate = None
        self.current_pdf_pages = pages
        self.current_pdf_page_index = 0
        self.current_candidate = None
        self.current_highlight_regions = []
        self.current_digitization_result = None
        self.manual_selection_bbox = None
        self._load_pdf_page(pages[0], focus_bbox=None)
        self.candidate_var.set(f'Manual Mode — Page 1/{len(pages)}')
        self.next_candidate_button.configure(state='disabled')

    def _change_pdf_page(self, step: int) -> None:
        """Move one deskewed page without changing candidate or metric review position."""
        if self.manual_selection_active or not self.current_pdf_pages:
            return
        next_index = self.current_pdf_page_index + int(step)
        if not 0 <= next_index < len(self.current_pdf_pages):
            return
        self.current_pdf_page_index = next_index
        self._load_pdf_page(self.current_pdf_pages[next_index], focus_bbox=None)
        if self.manual_search_active:
            self.candidate_var.set(
                f'Manual Mode — Page {next_index + 1}/{len(self.current_pdf_pages)}'
            )
        self.status_var.set(
            f'Viewing deskewed page {next_index + 1}/{len(self.current_pdf_pages)}. '
            'Candidate review position is unchanged.'
        )

    def _next_candidate(self) -> None:
        """Advance once through candidates without wrapping."""
        if str(self.next_candidate_button['state']) == 'disabled':
            return
        if not self._current_candidates():
            return
        self.review_candidate_index += 1
        self._show_current_review_item()

    def _next_stat(self) -> None:
        """Record the visible candidate, then advance stat or PDF automatically."""
        if str(self.next_stat_button['state']) == 'disabled':
            return
        self._commit_tracker_edits()
        review_pdf = self._current_review_pdf()
        stat_data = self._current_stat()
        if review_pdf is None or stat_data is None:
            return
        stat_key, stat_display_name = stat_data
        selected_candidate: dict[str, Any] | None = None
        visible_candidate = (
            self.manual_review_candidate
            if isinstance(self.manual_review_candidate, dict)
            else self._visible_candidate()
        )
        if visible_candidate is not None:
            selected_candidate = dict(visible_candidate)
            manual_digitization = selected_candidate.pop('_manual_digitization', None)
            for private_field in ('_resolved_page_image', '_focus_bbox', '_highlight_regions'):
                selected_candidate.pop(private_field, None)
            if manual_digitization is not None:
                selected_candidate['manual_digitization'] = manual_digitization
        decision_key = str(review_pdf['pdf_output_dir'])
        self.review_decisions[decision_key]['stats'][stat_key] = {'display_name': stat_display_name, 'status': 'selected' if selected_candidate is not None else 'not_found', 'selected_candidate': selected_candidate}
        self._autofill_tracker_from_review_choice(review_pdf=review_pdf, stat_key=stat_key, selected_candidate=selected_candidate)
        self._save_review_results(review_complete=False)
        leaving_current_pdf = self.review_stat_index == len(STAT_ORDER) - 1
        if leaving_current_pdf and (not self._identity_requirement_met(review_pdf)):
            self._select_tracker_for_review_pdf(review_pdf)
            messagebox.showwarning(WINDOW_TITLE, 'Initials or Birth year is required before finishing this PDF. Fill at least one of those two tracker fields; all other tracker fields are optional.')
            self.status_var.set('Enter Initials or Birth year before finishing this PDF.')
            return
        self.review_stat_index += 1
        self.review_candidate_index = 0
        if self.review_stat_index >= len(STAT_ORDER):
            self.review_pdf_index += 1
            self.review_stat_index = 0
        if self.review_pdf_index >= len(self.review_pdfs):
            self._finish_review()
            return
        self._show_current_review_item()

    def _finish_review(self) -> None:
        """Finish only after processing has ended; otherwise wait for more results."""
        if self.pipeline_running:
            self._save_review_results(review_complete=False)
            self._show_processing_waiting_screen(caught_up=True)
            return
        self.review_waiting_for_processing = False
        self.review_complete = True
        self._save_review_results(review_complete=True)
        self.toggle_highlights_button.configure(state='disabled')
        self.open_pdf_button.configure(state='disabled')
        self.manual_digitize_button.configure(state='disabled')
        self.refresh_digitize_button.configure(state='disabled')
        self.back_button.configure(state='disabled')
        self._set_zoom_controls_enabled(False)
        self.next_candidate_button.configure(state='disabled')
        self.next_stat_button.configure(state='disabled')
        self._update_back_button_state()
        self.stat_var.set('Review complete')
        self.candidate_var.set('')
        self.status_var.set(f'Saved review and {self.output_root / TRACKER_CSV_FILENAME}. Use Back to reopen the final statistic.' if self.output_root is not None else 'Review decisions saved. Use Back to reopen the final statistic.')
        self._show_canvas_message('All available PDFs and stats have been reviewed.\nPress Back to revisit the final statistic.')
        self._set_idle_controls()
        messagebox.showinfo(WINDOW_TITLE, 'Candidate review is complete.')

    def _save_review_results(self, review_complete: bool) -> None:
        """Write all decisions recorded so far to the batch output directory."""
        if self.output_root is None:
            return
        reviewed_stat_count = sum((len(pdf_decision['stats']) for pdf_decision in self.review_decisions.values()))
        total_stat_count = len(self.review_pdfs) * len(STAT_ORDER)
        results = {'pipeline_version': 'Mullenizer v2', 'updated_at_utc': utc_now_text(), 'source_folder': str(self.selected_folder) if self.selected_folder is not None else None, 'input_mode': 'folder' if self.selected_folder is not None else 'selected_pdfs', 'output_root': str(self.output_root), 'submitted_pdf_count': self.total_pdf_count, 'successful_pdf_count': len(self.review_pdfs), 'processing_complete': self.pipeline_finished, 'failed_pdf_count': self.failed_pdf_count, 'review_complete': review_complete, 'reviewed_stat_count': reviewed_stat_count, 'total_reviewable_stat_count': total_stat_count, 'highlight_preference': self.highlights_visible, 'pdfs': sorted(self.review_decisions.values(), key=lambda item: int(item.get('batch_position', 0)))}
        save_json(data=results, output_path=self.output_root / REVIEW_RESULTS_FILENAME)
        self._save_tracker_csv()

    def _load_candidate_page(self, page_path: Path, focus_bbox: list[int]) -> None:
        """Load a candidate page while keeping PDF page browsing available."""
        self._set_page_sequence(page_path)
        self._load_pdf_page(page_path, focus_bbox=focus_bbox)


    def _zoom_to_source_bbox(self, bbox: list[int]) -> None:
        """Set zoom so one source-page box nearly fills the canvas."""
        if self.current_source_image is None:
            return
        image_width, image_height = self.current_source_image.size
        x1 = max(0, min(image_width, bbox[0]))
        y1 = max(0, min(image_height, bbox[1]))
        x2 = max(x1 + 1, min(image_width, bbox[2]))
        y2 = max(y1 + 1, min(image_height, bbox[3]))
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        full_fit_scale = min(canvas_width / float(image_width), canvas_height / float(image_height))
        available_width = canvas_width * (1.0 - 2.0 * CANDIDATE_VIEW_MARGIN_FRACTION)
        available_height = canvas_height * (1.0 - 2.0 * CANDIDATE_VIEW_MARGIN_FRACTION)
        target_scale = min(available_width / float(x2 - x1), available_height / float(y2 - y1))
        self.current_zoom_factor = max(MINIMUM_ZOOM_FACTOR, min(MAXIMUM_ZOOM_FACTOR, target_scale / max(full_fit_scale, 1e-09)))
        center_fraction = ((x1 + x2) / 2.0 / float(image_width), (y1 + y2) / 2.0 / float(image_height))
        self._render_current_image(anchor_source_fraction=center_fraction, anchor_canvas_position=None)

    def _set_zoom_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable all zoom controls as one coherent group."""
        state = 'normal' if enabled else 'disabled'
        self.zoom_out_button.configure(state=state)
        self.zoom_reset_button.configure(state=state)
        self.zoom_in_button.configure(state=state)
        if not enabled:
            self.zoom_var.set('Fit Page')

    def _zoom_in(self) -> None:
        """Increase zoom around the center of the visible canvas."""
        if self.current_source_image is None:
            return
        self._set_zoom_factor(min(MAXIMUM_ZOOM_FACTOR, self.current_zoom_factor * ZOOM_STEP))

    def _zoom_out(self) -> None:
        """Decrease zoom around the center of the visible canvas."""
        if self.current_source_image is None:
            return
        self._set_zoom_factor(max(MINIMUM_ZOOM_FACTOR, self.current_zoom_factor / ZOOM_STEP))

    def _reset_zoom(self) -> None:
        """Fit the complete deskewed page in the canvas."""
        if self.current_source_image is None:
            return
        self.current_zoom_factor = 1.0
        self._render_current_image(anchor_source_fraction=(0.5, 0.5), anchor_canvas_position=None)

    def _set_zoom_factor(self, new_zoom_factor: float, canvas_x: float | None=None, canvas_y: float | None=None) -> None:
        """Apply zoom while keeping a chosen canvas point over the same source."""
        if self.current_source_image is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        canvas_x = canvas_width / 2.0 if canvas_x is None else canvas_x
        canvas_y = canvas_height / 2.0 if canvas_y is None else canvas_y
        source_fraction = self._source_fraction_at_canvas_point(canvas_x=canvas_x, canvas_y=canvas_y)
        self.current_zoom_factor = max(MINIMUM_ZOOM_FACTOR, min(MAXIMUM_ZOOM_FACTOR, new_zoom_factor))
        self._render_current_image(anchor_source_fraction=source_fraction, anchor_canvas_position=(canvas_x, canvas_y))

    def _source_fraction_at_canvas_point(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        """Map one visible canvas point into normalized source coordinates."""
        if self.current_display_width <= 0 or self.current_display_height <= 0:
            return (0.5, 0.5)
        content_x = self.canvas.canvasx(canvas_x)
        content_y = self.canvas.canvasy(canvas_y)
        origin_x, origin_y = self.current_image_origin
        fraction_x = (content_x - origin_x) / float(self.current_display_width)
        fraction_y = (content_y - origin_y) / float(self.current_display_height)
        return (max(0.0, min(1.0, fraction_x)), max(0.0, min(1.0, fraction_y)))

    def _on_mouse_wheel(self, event: tk.Event[Any]) -> str:
        """Zoom at the mouse pointer on Windows and macOS."""
        if self.current_source_image is None or event.delta == 0:
            return 'break'
        multiplier = ZOOM_STEP if event.delta > 0 else 1.0 / ZOOM_STEP
        self._set_zoom_factor(self.current_zoom_factor * multiplier, canvas_x=float(event.x), canvas_y=float(event.y))
        return 'break'

    def _on_linux_wheel_up(self, event: tk.Event[Any]) -> str:
        """Zoom in at the mouse pointer on Linux/X11."""
        if self.current_source_image is None:
            return 'break'
        self._set_zoom_factor(self.current_zoom_factor * ZOOM_STEP, canvas_x=float(event.x), canvas_y=float(event.y))
        return 'break'

    def _on_linux_wheel_down(self, event: tk.Event[Any]) -> str:
        """Zoom out at the mouse pointer on Linux/X11."""
        if self.current_source_image is None:
            return 'break'
        self._set_zoom_factor(self.current_zoom_factor / ZOOM_STEP, canvas_x=float(event.x), canvas_y=float(event.y))
        return 'break'

    def _canvas_point_to_source(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        """Map a visible canvas coordinate into source-page pixel coordinates."""
        if self.current_source_image is None or self.current_display_scale <= 0.0:
            return (0.0, 0.0)
        content_x = self.canvas.canvasx(canvas_x)
        content_y = self.canvas.canvasy(canvas_y)
        origin_x, origin_y = self.current_image_origin
        source_x = (content_x - origin_x) / self.current_display_scale
        source_y = (content_y - origin_y) / self.current_display_scale
        image_width, image_height = self.current_source_image.size
        return (max(0.0, min(float(image_width), source_x)), max(0.0, min(float(image_height), source_y)))

    def _source_bbox_to_canvas(self, bbox: list[int | float]) -> tuple[float, float, float, float]:
        """Map one source-page rectangle into scrollable canvas coordinates."""
        origin_x, origin_y = self.current_image_origin
        return (origin_x + float(bbox[0]) * self.current_display_scale, origin_y + float(bbox[1]) * self.current_display_scale, origin_x + float(bbox[2]) * self.current_display_scale, origin_y + float(bbox[3]) * self.current_display_scale)

    def _start_manual_digitization(self) -> None:
        """Enter or cancel plot-bound selection without changing the viewport."""
        if self.manual_selection_active:
            self._cancel_manual_selection(redraw=True, restore_highlights=True)
            return
        stat_data = self._current_stat()
        if stat_data is None:
            return
        _stat_key, _stat_display_name = stat_data
        if self.current_source_image is None or self.current_page_path is None:
            return
        self.interpretation_previous_highlights_visible = self.highlights_visible
        self.highlights_visible = False
        self.interpretation_overlays_suppressed = True
        self._redraw_overlays()
        self.manual_selection_active = True
        self.manual_selection_start_source = None
        self.manual_selection_end_source = None
        self.canvas.delete('manual_selection')
        self.canvas.configure(cursor='crosshair')
        self.manual_digitize_button.configure(text='Cancel Interpretation', state='normal')
        self.toggle_highlights_button.configure(text='Highlights Hidden', state='disabled')
        self.open_pdf_button.configure(state='disabled')
        self._set_zoom_controls_enabled(True)
        self.back_button.configure(state='disabled')
        self.next_candidate_button.configure(state='disabled')
        self.next_stat_button.configure(state='disabled')
        self._update_page_navigation_controls()
        self.status_var.set('Drag around the INNER plotting area only. Zoom buttons, mouse wheel, and keyboard zoom remain available. Exclude the title, axis labels, tick labels, and legend. After selection, enter line calibration or the histogram minimum/maximum range.')

    def _restore_interpretation_overlays(self, redraw: bool) -> None:
        """Restore the prior highlight preference after interpretation ends."""
        previous_visibility = self.interpretation_previous_highlights_visible
        if previous_visibility is not None:
            self.highlights_visible = previous_visibility
        self.interpretation_previous_highlights_visible = None
        self.interpretation_overlays_suppressed = False
        self.toggle_highlights_button.configure(text='Hide Markings' if self.highlights_visible else 'Mark Metric')
        if redraw:
            self._redraw_overlays()

    def _cancel_manual_selection(self, redraw: bool, restore_highlights: bool=True) -> None:
        """Leave selection mode and restore normal candidate controls."""
        was_active = self.manual_selection_active
        self.manual_selection_active = False
        self.manual_selection_start_source = None
        self.manual_selection_end_source = None
        self.canvas.configure(cursor='')
        self.canvas.delete('manual_selection')
        if hasattr(self, 'manual_digitize_button'):
            self.manual_digitize_button.configure(text='Interpret Graph')
        if not was_active:
            if restore_highlights:
                self._restore_interpretation_overlays(redraw=redraw)
            return
        stat_data = self._current_stat()
        stat_key = stat_data[0] if stat_data is not None else ''
        self.manual_digitize_button.configure(
            state='normal'
            if self.current_source_image is not None
            and self.current_page_path is not None
            else 'disabled'
        )
        self.toggle_highlights_button.configure(state='normal' if self.current_highlight_regions or isinstance(self.current_digitization_result, dict) else 'disabled')
        self.open_pdf_button.configure(state='normal' if self._current_pdf_path() is not None else 'disabled')
        self._set_zoom_controls_enabled(self.current_source_image is not None)
        self.next_candidate_button.configure(state='normal' if self._visible_candidate() is not None else 'disabled')
        self.next_stat_button.configure(state='normal')
        self._update_back_button_state()
        self._update_page_navigation_controls()
        if restore_highlights:
            self._restore_interpretation_overlays(redraw=redraw)
        elif redraw:
            self._redraw_overlays()

    def _draw_manual_selection_rectangle(self, end_source: tuple[float, float]) -> None:
        """Draw the active source-coordinate rectangle at the current zoom."""
        if self.manual_selection_start_source is None:
            return
        self.manual_selection_end_source = end_source
        x1, y1 = self.manual_selection_start_source
        x2, y2 = end_source
        canvas_bbox = self._source_bbox_to_canvas([x1, y1, x2, y2])
        self.canvas.delete('manual_selection')
        self.canvas.create_rectangle(*canvas_bbox, outline=MANUAL_SELECTION_OUTLINE, width=MANUAL_SELECTION_WIDTH, dash=MANUAL_SELECTION_DASH, tags=('manual_selection',))

    def _digitization_edge_at_canvas_point(
        self,
        canvas_x: float,
        canvas_y: float,
    ) -> str | None:
        """Find a line graph edge; histogram plotting bounds are fixed."""
        if (
            self.manual_selection_active
            or not self.highlights_visible
            or not isinstance(self.current_digitization_result, dict)
            or self.current_digitization_result.get('interpretation_mode') != 'line'
            or self.manual_selection_bbox is None
        ):
            return None

        x1, y1, x2, y2 = self._source_bbox_to_canvas(self.manual_selection_bbox)
        content_x = float(self.canvas.canvasx(canvas_x))
        content_y = float(self.canvas.canvasy(canvas_y))
        if not x1 - DIGITIZATION_HANDLE_HIT_PIXELS <= content_x <= x2 + DIGITIZATION_HANDLE_HIT_PIXELS:
            return None
        if not y1 - DIGITIZATION_HANDLE_HIT_PIXELS <= content_y <= y2 + DIGITIZATION_HANDLE_HIT_PIXELS:
            return None
        if abs(content_x - x1) <= DIGITIZATION_HANDLE_HIT_PIXELS:
            return 'left'
        if abs(content_x - x2) <= DIGITIZATION_HANDLE_HIT_PIXELS:
            return 'right'
        if abs(content_y - y1) <= DIGITIZATION_HANDLE_HIT_PIXELS:
            return 'top'
        if abs(content_y - y2) <= DIGITIZATION_HANDLE_HIT_PIXELS:
            return 'bottom'
        return None

    def _drag_digitization_edge(self, source_x: float, source_y: float) -> None:
        """Adjust one line graph edge within the original plot selection."""
        if (
            self.digitization_edge_drag not in {'left', 'right', 'top', 'bottom'}
            or self.manual_selection_bbox is None
            or not isinstance(self.current_digitization_result, dict)
            or self.current_digitization_result.get('interpretation_mode') != 'line'
        ):
            return

        original = valid_bbox(self.current_digitization_result.get('original_plot_bbox')) or valid_bbox(self.current_digitization_result.get('plot_bbox'))
        if original is None:
            return

        pending = list(self.manual_selection_bbox)
        minimum_width = MINIMUM_MANUAL_SELECTION_WIDTH_PIXELS
        if self.digitization_edge_drag == 'left':
            pending[0] = int(round(max(
                original[0],
                min(source_x, pending[2] - minimum_width),
            )))
        else:
            if self.digitization_edge_drag == 'right':
                pending[2] = int(round(min(original[2], max(source_x, pending[0] + minimum_width))))
            elif self.digitization_edge_drag == 'top':
                pending[1] = int(round(max(original[1], min(source_y, pending[3] - MINIMUM_MANUAL_SELECTION_HEIGHT_PIXELS))))
            else:
                pending[3] = int(round(min(original[3], max(source_y, pending[1] + MINIMUM_MANUAL_SELECTION_HEIGHT_PIXELS))))

        if pending == self.manual_selection_bbox:
            return

        self.manual_selection_bbox = pending
        self.digitization_bounds_dirty = pending != valid_bbox(self.current_digitization_result.get('plot_bbox'))
        self.refresh_digitize_button.configure(
            text='Refresh Interpretation *' if self.digitization_bounds_dirty else 'Refresh Interpretation',
            state='normal',
        )
        self._redraw_overlays()

    def _histogram_bar_at_canvas_point(self, canvas_x: float, canvas_y: float) -> int | None:
        """Find the top handle of an editable histogram column."""
        result = self.current_digitization_result
        if self.manual_selection_active or not self.highlights_visible or not isinstance(result, dict) or result.get('interpretation_mode') != 'histogram':
            return None
        x = float(self.canvas.canvasx(canvas_x))
        y = float(self.canvas.canvasy(canvas_y))
        nearest: tuple[float, int] | None = None
        for column in result.get('histogram_columns', []):
            bbox = valid_bbox(column.get('bbox'))
            if bbox is None:
                continue
            x1, y1, x2, _y2 = self._source_bbox_to_canvas(bbox)
            center_x = (x1 + x2) / 2.0
            if abs(x - center_x) <= max(7.0, (x2 - x1) / 2.0) and abs(y - y1) <= 9.0:
                distance = abs(x - center_x) + abs(y - y1)
                if nearest is None or distance < nearest[0]:
                    nearest = distance, int(column['index'])
        return nearest[1] if nearest else None

    def _drag_histogram_height(self, canvas_y: float) -> None:
        """Preview updated column statistics while moving its top handle."""
        result = self.current_digitization_result
        if self.histogram_height_drag is None or not isinstance(result, dict):
            return
        _source_x, source_y = self._canvas_point_to_source(canvas_x=0.0, canvas_y=canvas_y)
        column = result['histogram_columns'][self.histogram_height_drag - 1]
        height = max(0.0, float(column['bbox'][3]) - round(source_y))
        try:
            updated = adjust_histogram_column_height(result, self.histogram_height_drag, height)
        except ValueError:
            return
        self.current_digitization_result = updated
        self._attach_refreshed_digitization(updated)
        self._redraw_overlays()

    def _attach_refreshed_digitization(
        self,
        result: dict[str, Any],
    ) -> None:
        """Replace the graph result on whichever candidate owns the visible page."""
        for candidate in (self.manual_review_candidate, self._visible_candidate()):
            if self._current_page_matches_candidate(candidate):
                candidate['_manual_digitization'] = result
                return

    def _refresh_current_digitization(self) -> None:
        """Reinterpret line plots against their original axis calibration."""
        old_result = self.current_digitization_result
        review_pdf = self._current_review_pdf()
        page_path = self.current_page_path
        if (
            not isinstance(old_result, dict)
            or review_pdf is None
            or page_path is None
        ):
            return

        if old_result.get('interpretation_mode') == 'histogram':
            self._show_digitization_statistics(old_result)
            return

        old_bbox = valid_bbox(old_result.get('plot_bbox'))
        new_bbox = valid_bbox(self.manual_selection_bbox)
        if old_bbox is None or new_bbox is None:
            return

        mode = str(old_result.get('interpretation_mode', 'line'))
        try:
            if mode != 'line':
                raise ValueError(f'Unsupported graph mode: {mode}')
            reference_bbox = valid_bbox(old_result.get('original_plot_bbox')) or old_bbox
            y_axis = old_result.get('calibration_y_axis') or old_result['y_axis']
            original_top = float(y_axis['top_value'])
            original_bottom = float(y_axis['bottom_value'])
            axis_span = original_bottom - original_top
            reference_height = max(1, reference_bbox[3] - reference_bbox[1] - 1)
            new_top = original_top + (new_bbox[1] - reference_bbox[1]) / reference_height * axis_span
            new_bottom = original_top + (new_bbox[3] - 1 - reference_bbox[1]) / reference_height * axis_span
            refreshed = digitize_line_graph(
                image_path=page_path,
                plot_bbox=new_bbox,
                y_top=new_top,
                y_bottom=new_bottom,
                unit=str(y_axis.get('unit', '')),
                target=str(old_result.get('target', 'graph')),
            )
            refreshed['original_plot_bbox'] = list(reference_bbox)
            refreshed['sampled_y_axis'] = dict(refreshed['y_axis'])
            refreshed['calibration_y_axis'] = dict(y_axis)
            refreshed['y_axis'] = dict(y_axis)
        except Exception as error:
            messagebox.showerror(
                WINDOW_TITLE,
                f'Could not refresh the interpretation:\n{error}',
            )
            return

        for key in (
            'measurement_selection',
            'page_number',
            'source_page_image',
            'result_path',
        ):
            if key in old_result:
                refreshed[key] = old_result[key]

        raw_result_path = str(refreshed.get('result_path', '')).strip()
        if raw_result_path:
            result_path = (
                Path(str(review_pdf['pdf_output_dir']))
                / Path(raw_result_path)
            )
            save_digitization_json(refreshed, result_path)

        self._attach_refreshed_digitization(refreshed)
        self.current_digitization_result = refreshed
        self.manual_selection_bbox = list(refreshed['plot_bbox'])
        self.digitization_bounds_dirty = False
        self.refresh_digitize_button.configure(
            text='Refresh Interpretation',
            state='normal',
        )
        self._update_digitization_controls()
        self.highlights_visible = True
        self.toggle_highlights_button.configure(
            text='Hide Markings',
            state='normal',
        )
        self._redraw_overlays()

        statistics = refreshed['statistics']
        unit = str(statistics.get('unit', '')).strip()
        suffix = f' {unit}' if unit else ''
        average_text = format_measurement_value(
            float(statistics['average']),
            unit,
        )
        self.status_var.set(
            f'Refreshed interpretation: average {average_text}{suffix}.'
        )
        self._show_digitization_statistics(refreshed)

    def _draw_persistent_manual_selection_bbox(self, bbox: list[int | float]) -> None:
        """Keep the completed plot selection visible while calibration is entered."""
        canvas_bbox = self._source_bbox_to_canvas(bbox)
        self.canvas.delete('manual_selection')
        self.canvas.create_rectangle(
            *canvas_bbox,
            outline=MANUAL_SELECTION_OUTLINE,
            width=MANUAL_SELECTION_WIDTH,
            dash=MANUAL_SELECTION_DASH,
            tags=('manual_selection',),
        )
        self.canvas.tag_raise('manual_selection')

    def _update_canvas_pointer_cursor(
        self,
        event: tk.Event[Any],
    ) -> None:
        """Show a horizontal-resize cursor when either orange plot edge can be dragged."""
        if self.manual_selection_active:
            self.canvas.configure(cursor='crosshair')
            return
        if self.digitization_edge_drag is not None:
            self.canvas.configure(cursor='sb_h_double_arrow' if self.digitization_edge_drag in {'left', 'right'} else 'sb_v_double_arrow')
            return
        if self.histogram_height_drag is not None or self._histogram_bar_at_canvas_point(float(event.x), float(event.y)) is not None:
            self.canvas.configure(cursor='sb_v_double_arrow')
            return
        edge = self._digitization_edge_at_canvas_point(
            float(event.x),
            float(event.y),
        )
        self.canvas.configure(cursor=(
            'sb_h_double_arrow' if edge in {'left', 'right'}
            else 'sb_v_double_arrow' if edge in {'top', 'bottom'} else ''
        ))


    def _start_pan(self, event: tk.Event[Any]) -> None:
        """Start plot selection, orange-bound adjustment, or normal page panning."""
        if self.current_source_image is None:
            return
        if self.manual_selection_active:
            self.manual_selection_start_source = self._canvas_point_to_source(canvas_x=float(event.x), canvas_y=float(event.y))
            self._draw_manual_selection_rectangle(self.manual_selection_start_source)
            return
        column_index = self._histogram_bar_at_canvas_point(float(event.x), float(event.y))
        if column_index is not None:
            self.histogram_height_drag = column_index
            self.canvas.configure(cursor='sb_v_double_arrow')
            self._close_statistics_window()
            return
        edge = self._digitization_edge_at_canvas_point(
            float(event.x),
            float(event.y),
        )
        if edge is not None:
            self.digitization_edge_drag = edge
            self.canvas.configure(cursor='sb_h_double_arrow' if edge in {'left', 'right'} else 'sb_v_double_arrow')
            self._close_statistics_window()
            return
        self.canvas.scan_mark(event.x, event.y)

    def _pan_image(self, event: tk.Event[Any]) -> None:
        """Update plot selection, orange-bound adjustment, or page panning."""
        if self.current_source_image is None:
            return
        if self.manual_selection_active:
            if self.manual_selection_start_source is None:
                return
            self._draw_manual_selection_rectangle(self._canvas_point_to_source(canvas_x=float(event.x), canvas_y=float(event.y)))
            return
        if self.histogram_height_drag is not None:
            self._drag_histogram_height(float(event.y))
            return
        if self.digitization_edge_drag is not None:
            source_x, source_y = self._canvas_point_to_source(
                canvas_x=float(event.x),
                canvas_y=float(event.y),
            )
            self._drag_digitization_edge(source_x, source_y)
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _finish_pointer_action(self, event: tk.Event[Any]) -> None:
        """Finish an orange-bound drag or a manual plot-selection drag."""
        if self.histogram_height_drag is not None:
            self._drag_histogram_height(float(event.y))
            self.histogram_height_drag = None
            self.canvas.configure(cursor='')
            result = self.current_digitization_result
            review_pdf = self._current_review_pdf()
            if isinstance(result, dict) and review_pdf is not None and result.get('result_path'):
                save_digitization_json(
                    result,
                    Path(str(review_pdf['pdf_output_dir'])) / Path(str(result['result_path'])),
                )
                self.refresh_digitize_button.configure(text='Review Statistics', state='normal')
                self.status_var.set('Column height saved. Press Review Statistics to inspect or export the recalculated values.')
            return
        if self.digitization_edge_drag is not None:
            self.digitization_edge_drag = None
            self.canvas.configure(cursor='')
            if self.digitization_bounds_dirty:
                self.status_var.set(
                    'Line graph bounds adjusted. Press Refresh Interpretation to recalculate using the original axis scale.'
                )
            return
        if not self.manual_selection_active:
            return
        if self.manual_selection_start_source is None:
            return
        end_source = self._canvas_point_to_source(canvas_x=float(event.x), canvas_y=float(event.y))
        start_x, start_y = self.manual_selection_start_source
        end_x, end_y = end_source
        source_bbox = [int(round(min(start_x, end_x))), int(round(min(start_y, end_y))), int(round(max(start_x, end_x))), int(round(max(start_y, end_y)))]
        if source_bbox[2] - source_bbox[0] < MINIMUM_MANUAL_SELECTION_WIDTH_PIXELS or source_bbox[3] - source_bbox[1] < MINIMUM_MANUAL_SELECTION_HEIGHT_PIXELS:
            self.status_var.set('Selection is too small. Drag a larger rectangle around the complete inner plotting area.')
            self.manual_selection_start_source = None
            self.manual_selection_end_source = None
            self.canvas.delete('manual_selection')
            return
        self.manual_selection_bbox = source_bbox
        self._cancel_manual_selection(redraw=False, restore_highlights=False)
        self._draw_persistent_manual_selection_bbox(source_bbox)
        self.root.update_idletasks()
        self._run_manual_digitization(source_bbox)

    def _run_manual_digitization(self, plot_bbox: list[int]) -> None:
        """Interpret a graph on the currently visible deskewed page and save it."""
        visible_candidate = self._visible_candidate()
        stat_data = self._current_stat()
        review_pdf = self._current_review_pdf()
        page_path = self.current_page_path
        if stat_data is None or review_pdf is None or page_path is None:
            self._restore_interpretation_overlays(redraw=True)
            return

        stat_key, stat_display_name = stat_data

        self.manual_selection_bbox = list(plot_bbox)
        self._draw_persistent_manual_selection_bbox(plot_bbox)
        self.root.update_idletasks()
        dialog = GraphInterpretationDialog(
            parent=self.root,
            stat_display_name=stat_display_name,
            stat_key=stat_key,
        )
        if dialog.result is None:
            self.manual_selection_bbox = None
            self.status_var.set('Graph interpretation cancelled.')
            self._restore_interpretation_overlays(redraw=True)
            return

        settings = dialog.result
        interpretation_mode = str(settings['mode'])
        self.status_var.set('Interpreting the manually bounded graph...')
        self.root.update_idletasks()
        try:
            if interpretation_mode == 'histogram':
                result = digitize_histogram(
                    image_path=page_path,
                    plot_bbox=plot_bbox,
                    target=str(settings['target']),
                    minimum=float(settings['minimum']),
                    maximum=float(settings['maximum']),
                    unit=str(settings['unit']),
                    bin_width=float(settings['bin_width']),
                )
            else:
                result = digitize_line_graph(
                    image_path=page_path,
                    plot_bbox=plot_bbox,
                    y_top=float(settings['top_value']),
                    y_bottom=float(settings['bottom_value']),
                    unit=str(settings['unit']),
                    target=str(settings['target']),
                )
        except Exception as error:
            self.manual_selection_bbox = None
            self.status_var.set('Graph interpretation failed.')
            self._restore_interpretation_overlays(redraw=True)
            messagebox.showerror(
                WINDOW_TITLE,
                f'Could not interpret the selected graph:\n{error}',
            )
            return

        result['measurement_selection'] = {
            'label': str(settings['measurement_label']),
            'unit': str(settings['unit']),
            'tracker_field': str(settings['tracker_field']),
            'target': str(settings['target']),
        }
        if interpretation_mode == 'line':
            result['original_plot_bbox'] = list(result['plot_bbox'])
            result['calibration_y_axis'] = dict(result['y_axis'])
        page_number = self.current_pdf_page_index + 1
        result['page_number'] = page_number
        result['source_page_image'] = str(page_path)

        pdf_output_dir = Path(str(review_pdf['pdf_output_dir']))
        attach_to_visible_candidate = self._current_page_matches_candidate(
            visible_candidate
        )
        if attach_to_visible_candidate and visible_candidate is not None:
            candidate_number = self.review_candidate_index + 1
            filename = (
                f"{settings['target']}_page_{page_number:04d}_"
                f'candidate_{candidate_number:02d}.json'
            )
            target_candidate = visible_candidate
        else:
            filename = f"{settings['target']}_page_{page_number:04d}_manual.json"
            try:
                stored_page_path = page_path.relative_to(pdf_output_dir).as_posix()
            except ValueError:
                stored_page_path = str(page_path)
            target_candidate = {
                'candidate_type': 'manual_search',
                'page_number': page_number,
                'image_path': stored_page_path,
                'review_crop_bbox': list(plot_bbox),
                'score': None,
                '_resolved_page_image': str(page_path),
                '_focus_bbox': list(plot_bbox),
                '_highlight_regions': [],
            }
            self.manual_review_candidate = target_candidate
            self.manual_search_active = True
            self.candidate_var.set(
                f'Manual selection — Page {page_number}/{len(self.current_pdf_pages)}'
            )

        result_relative_path = Path('manual_digitizations') / filename
        result_path = pdf_output_dir / result_relative_path
        result['result_path'] = result_relative_path.as_posix()
        save_digitization_json(data=result, output_path=result_path)
        target_candidate['_manual_digitization'] = result
        self.current_digitization_result = result
        self.manual_selection_bbox = list(result['plot_bbox'])
        self.digitization_bounds_dirty = False
        self.refresh_digitize_button.configure(
            text='Refresh Interpretation',
            state='normal',
        )
        self._update_digitization_controls()
        self.next_stat_button.configure(text='Next Metric', state='normal')

        statistics = result['statistics']
        if interpretation_mode == 'histogram':
            unit = str(statistics.get('unit', '')).strip()
            suffix = f' {unit}' if unit else ''
            average_text = format_measurement_value(float(statistics['average']), unit)
            median_text = format_measurement_value(float(statistics['median']), unit)
            self.status_var.set(
                f'Histogram average {average_text}{suffix}; '
                f'median {median_text}{suffix}; '
                f"confidence {result['confidence'] * 100:.0f}%."
            )
            self._restore_interpretation_overlays(redraw=False)
            self.highlights_visible = True
            self.toggle_highlights_button.configure(
                text='Hide Markings',
                state='normal',
            )
            self._redraw_overlays()
        else:
            unit = str(statistics.get('unit', '')).strip()
            unit_text = f' {unit}' if unit else ''
            average_text = format_measurement_value(float(statistics['average']), unit)
            self.status_var.set(
                f'Interpreted average: {average_text}{unit_text}; '
                f"confidence {result['confidence'] * 100:.0f}%."
            )
            self._restore_interpretation_overlays(redraw=True)

        self._show_digitization_statistics(result)


    def _persist_digitization_result(self, result: dict[str, Any]) -> None:
        """Save edits made in the result popup back to the digitization JSON."""
        review_pdf = self._current_review_pdf()
        candidate = self._visible_candidate()
        if review_pdf is None:
            return
        raw_result_path = str(result.get('result_path', '')).strip()
        if raw_result_path:
            result_path = Path(str(review_pdf['pdf_output_dir'])) / Path(raw_result_path)
            save_digitization_json(data=result, output_path=result_path)
        if candidate is not None:
            candidate['_manual_digitization'] = result
        self.current_digitization_result = result

    def _close_statistics_window(self) -> None:
        """Dismiss a statistics view whose values may be superseded by edits."""
        window = self.statistics_window
        self.statistics_window = None
        if window is not None and window.winfo_exists():
            window.destroy()

    def _show_digitization_statistics(self, result: dict[str, Any]) -> None:
        """Show graph statistics and the unit-selected tracker destination."""
        statistics = result['statistics']
        mode = str(result.get('interpretation_mode', 'line'))
        stat_data = self._current_stat()
        review_pdf = self._current_review_pdf()
        if stat_data is None or review_pdf is None:
            return

        self._close_statistics_window()

        _stat_key, stat_display_name = stat_data
        unit = str(statistics.get('unit', '')).strip()
        suffix = f' {unit}' if unit else ''
        window = tk.Toplevel(self.root)
        self.statistics_window = window
        window.title(f'{stat_display_name} interpretation')
        window.transient(self.root)
        window.resizable(False, False)
        window.protocol('WM_DELETE_WINDOW', self._close_statistics_window)

        outer = ttk.Frame(window, padding=14)
        outer.grid(row=0, column=0, sticky='nsew')
        outer.columnconfigure(0, weight=1)

        ttk.Label(
            outer,
            text='Interpretation result',
            font=('TkDefaultFont', 11, 'bold'),
        ).grid(row=0, column=0, sticky='w')

        stats_frame = ttk.LabelFrame(outer, text='Statistics', padding=9)
        stats_frame.grid(row=1, column=0, pady=(8, 0), sticky='ew')
        stats_frame.columnconfigure(1, weight=1)

        rows = [
            ('Average', statistics['average']),
        ]
        if mode == 'histogram':
            rows.append(('Median (50th)', statistics['median']))
        rows.extend([
            ('Minimum', statistics['minimum']),
            ('Maximum', statistics['maximum']),
            ('25th percentile', statistics['percentile_25']),
            ('75th percentile', statistics['percentile_75']),
        ])

        for row_index, (label_text, value) in enumerate(rows):
            value_text = format_measurement_value(float(value), unit)
            ttk.Label(
                stats_frame,
                text=f'{label_text}:',
            ).grid(
                row=row_index,
                column=0,
                padx=(0, 12),
                pady=2,
                sticky='e',
            )
            ttk.Label(
                stats_frame,
                text=f'{value_text}{suffix}',
            ).grid(
                row=row_index,
                column=1,
                pady=2,
                sticky='w',
            )

        routed = self._tracker_value_from_digitization(result)
        export_status_var = tk.StringVar(value='')
        export_frame = ttk.Frame(outer)
        export_frame.grid(row=2, column=0, pady=(9, 0), sticky='ew')
        export_frame.columnconfigure(0, weight=1)

        if routed is not None:
            field_name, value, tracker_unit = routed
            display_value = format_measurement_value(value, tracker_unit)
            ttk.Label(
                export_frame,
                text=(
                    f'Export target: {field_name} — '
                    f'{display_value} {tracker_unit}'.strip()
                ),
                font=('TkDefaultFont', 10, 'bold'),
            ).grid(row=0, column=0, sticky='w')

            def export_value() -> None:
                """Export the interpreted value to the unit-selected tracker field."""
                self._set_tracker_explicit_value(
                    review_pdf=review_pdf,
                    field_name=field_name,
                    value=value,
                    unit=tracker_unit,
                )
                export_status_var.set(f'Exported to {field_name}.')

            ttk.Button(
                export_frame,
                text='Export Value',
                command=export_value,
            ).grid(row=0, column=1, padx=(12, 0), sticky='e')
        else:
            ttk.Label(
                export_frame,
                text='No tracker destination selected.',
            ).grid(row=0, column=0, sticky='w')

        ttk.Label(
            outer,
            textvariable=export_status_var,
            anchor='w',
        ).grid(row=3, column=0, pady=(4, 0), sticky='ew')

        ttk.Label(
            outer,
            text=('Drag a white handle at the top of a column in the main window to correct its height. This statistics window closes when you edit; press Review Statistics to see updated values.' if mode == 'histogram' else 'Drag any orange edge in the main window to trim the line graph. The original axis calibration stays fixed; refresh to recalculate.'),
            wraplength=480,
            justify='left',
        ).grid(row=4, column=0, pady=(7, 0), sticky='w')

        ttk.Button(
            outer,
            text='Close',
            command=self._close_statistics_window,
        ).grid(row=5, column=0, pady=(10, 0), sticky='e')

        window.bind('<Escape>', lambda _event: self._close_statistics_window())
        place_popup_right_of_center(
            window,
            self.root,
            x_offset=POPUP_RIGHT_OFFSET_PIXELS + 40,
        )


    def _on_canvas_resize(self, _event: tk.Event[Any]) -> None:
        """Redraw the current page or message when the canvas changes size."""
        if self.current_source_image is not None:
            source_fraction = self._source_fraction_at_canvas_point(canvas_x=max(1, self.canvas.winfo_width()) / 2.0, canvas_y=max(1, self.canvas.winfo_height()) / 2.0)
            self._render_current_image(anchor_source_fraction=source_fraction, anchor_canvas_position=None)
        elif self.current_canvas_message is not None:
            self._render_canvas_message()

    def _redraw_overlays(self) -> None:
        """Redraw canvas overlays without resampling or moving the page image."""
        if self.current_source_image is None:
            return
        for tag in OVERLAY_CANVAS_TAGS:
            self.canvas.delete(tag)
        self.current_highlight_images.clear()
        self._draw_highlights()
        self._draw_digitization_overlay()
        if self.manual_selection_active and self.manual_selection_start_source is not None and (self.manual_selection_end_source is not None):
            self._draw_manual_selection_rectangle(self.manual_selection_end_source)

    def _render_current_image(self, anchor_source_fraction: tuple[float, float], anchor_canvas_position: tuple[float, float] | None) -> None:
        """Render the full page and optional source-coordinate highlights."""
        if self.current_source_image is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_width, image_height = self.current_source_image.size
        self.current_fit_scale = min(canvas_width / float(image_width), canvas_height / float(image_height))
        self.current_display_scale = self.current_fit_scale * self.current_zoom_factor
        output_width = max(1, int(round(image_width * self.current_display_scale)))
        output_height = max(1, int(round(image_height * self.current_display_scale)))
        resampling_namespace = getattr(Image, 'Resampling', Image)
        resized_image = self.current_source_image.resize((output_width, output_height), resampling_namespace.LANCZOS)
        try:
            self.current_display_image = ImageTk.PhotoImage(resized_image)
        finally:
            resized_image.close()
        self.current_display_width = output_width
        self.current_display_height = output_height
        content_width = max(canvas_width, output_width)
        content_height = max(canvas_height, output_height)
        origin_x = (content_width - output_width) / 2.0
        origin_y = (content_height - output_height) / 2.0
        self.current_image_origin = (origin_x, origin_y)
        self.canvas.delete('all')
        self.current_highlight_images.clear()
        self.canvas.create_image(origin_x, origin_y, image=self.current_display_image, anchor='nw', tags=('page_image',))
        self._draw_highlights()
        self._draw_digitization_overlay()
        if self.manual_selection_active and self.manual_selection_start_source is not None and (self.manual_selection_end_source is not None):
            self._draw_manual_selection_rectangle(self.manual_selection_end_source)
        self.canvas.configure(scrollregion=(0, 0, content_width, content_height))
        fraction_x, fraction_y = anchor_source_fraction
        target_content_x = origin_x + fraction_x * output_width
        target_content_y = origin_y + fraction_y * output_height
        if anchor_canvas_position is None:
            target_canvas_x = canvas_width / 2.0
            target_canvas_y = canvas_height / 2.0
        else:
            target_canvas_x, target_canvas_y = anchor_canvas_position
        desired_left = target_content_x - target_canvas_x
        desired_top = target_content_y - target_canvas_y
        maximum_left = max(0.0, content_width - canvas_width)
        maximum_top = max(0.0, content_height - canvas_height)
        desired_left = max(0.0, min(maximum_left, desired_left))
        desired_top = max(0.0, min(maximum_top, desired_top))
        if content_width > 0:
            self.canvas.xview_moveto(desired_left / content_width)
        if content_height > 0:
            self.canvas.yview_moveto(desired_top / content_height)
        if abs(self.current_zoom_factor - 1.0) < 1e-06:
            self.zoom_var.set('Fit Page')
        else:
            self.zoom_var.set(f'{self.current_zoom_factor * 100:.0f}%')
        self._update_page_navigation_controls()

    def _draw_highlights(self) -> None:
        """Draw padded translucent highlighter boxes over detected text only."""
        if not self.highlights_visible or not self.current_highlight_regions or self.current_source_image is None:
            return
        if not self._current_page_matches_candidate(self._visible_candidate()):
            return
        image_width, image_height = self.current_source_image.size
        origin_x, origin_y = self.current_image_origin
        for region in self.current_highlight_regions:
            role = str(region.get('role', ''))
            if role not in TEXT_HIGHLIGHT_ROLES:
                continue
            bbox = valid_bbox(region.get('bbox'))
            if bbox is None:
                continue
            bbox = padded_text_bbox(bbox, image_width, image_height)
            x1 = origin_x + bbox[0] * self.current_display_scale
            y1 = origin_y + bbox[1] * self.current_display_scale
            x2 = origin_x + bbox[2] * self.current_display_scale
            y2 = origin_y + bbox[3] * self.current_display_scale
            width = max(1, int(round(x2 - x1)))
            height = max(1, int(round(y2 - y1)))
            overlay = Image.new('RGBA', (width, height), HIGHLIGHT_STYLES[role])
            try:
                photo = ImageTk.PhotoImage(overlay)
            finally:
                overlay.close()
            self.current_highlight_images.append(photo)
            self.canvas.create_image(x1, y1, image=photo, anchor='nw', tags=('candidate_highlight',))

    def _draw_translucent_source_box(self, bbox: list[int], rgba: tuple[int, int, int, int], tag: str) -> tuple[float, float, float, float]:
        """Draw one translucent filled box in source-image coordinates."""
        canvas_bbox = self._source_bbox_to_canvas(bbox)
        x1, y1, x2, y2 = canvas_bbox
        width = max(1, int(round(x2 - x1)))
        height = max(1, int(round(y2 - y1)))
        overlay = Image.new('RGBA', (width, height), rgba)
        try:
            photo = ImageTk.PhotoImage(overlay)
        finally:
            overlay.close()
        self.current_highlight_images.append(photo)
        self.canvas.create_image(x1, y1, image=photo, anchor='nw', tags=(tag,))
        return canvas_bbox

    def _draw_digitization_overlay(self) -> None:
        """Draw line traces or clearly marked detected histogram bars."""
        if not self.highlights_visible or self.interpretation_overlays_suppressed:
            return
        if (
            not self._current_page_matches_candidate(self._visible_candidate())
            and not self._current_page_matches_candidate(self.manual_review_candidate)
        ):
            return
        result = self.current_digitization_result
        if not isinstance(result, dict):
            return
        plot_bbox = (
            valid_bbox(self.manual_selection_bbox)
            or valid_bbox(result.get('plot_bbox'))
        )
        histogram_mode = result.get('interpretation_mode') == 'histogram'
        if plot_bbox is not None:
            canvas_bbox = self._source_bbox_to_canvas(plot_bbox)
            self.canvas.create_rectangle(
                *canvas_bbox,
                outline=HISTOGRAM_DETECTED_BAR_COLOR if histogram_mode else MANUAL_SELECTION_OUTLINE,
                width=HISTOGRAM_DETECTED_BAR_WIDTH if histogram_mode else MANUAL_SELECTION_WIDTH,
                tags=('digitization_bounds',),
                **({} if histogram_mode else {'dash': DIGITIZED_BOUNDS_DASH}),
            )
            if not histogram_mode:
                left, top, right, bottom = canvas_bbox
                middle_y = (top + bottom) / 2.0
                middle_x = (left + right) / 2.0
                handle_centers = (
                    (left, middle_y), (right, middle_y),
                    (middle_x, top), (middle_x, bottom),
                )
                for center_x, center_y in handle_centers:
                    self.canvas.create_rectangle(
                        center_x - DIGITIZATION_HANDLE_HALF_WIDTH_PIXELS,
                        center_y - DIGITIZATION_HANDLE_HALF_HEIGHT_PIXELS,
                        center_x + DIGITIZATION_HANDLE_HALF_WIDTH_PIXELS,
                        center_y + DIGITIZATION_HANDLE_HALF_HEIGHT_PIXELS,
                        outline=MANUAL_SELECTION_OUTLINE,
                        fill='white',
                        width=2,
                        tags=('digitization_handles',),
                    )
        if result.get('interpretation_mode') == 'histogram':
            raw_columns = result.get('histogram_columns', [])
            statistics = result.get('statistics', {})
            try:
                mean_column_index = int(statistics.get('average_column_index'))
            except (TypeError, ValueError):
                mean_column_index = None
            try:
                median_column_index = int(statistics.get('median_column_index'))
            except (TypeError, ValueError):
                median_column_index = None

            if isinstance(raw_columns, list):
                for column in raw_columns:
                    if not isinstance(column, dict):
                        continue
                    bar_bbox = valid_bbox(column.get('bbox'))
                    if bar_bbox is None:
                        continue
                    try:
                        weight = float(column.get('weight', 0.0))
                    except (TypeError, ValueError):
                        weight = 0.0
                    if weight > 0.0:
                        canvas_bbox = self._draw_translucent_source_box(
                            bar_bbox,
                            HISTOGRAM_DETECTED_BAR_FILL,
                            'histogram_detected_bars',
                        )
                    else:
                        canvas_bbox = self._source_bbox_to_canvas(bar_bbox)
                    self.canvas.create_rectangle(
                        *canvas_bbox,
                        outline=HISTOGRAM_DETECTED_BAR_COLOR,
                        width=HISTOGRAM_DETECTED_BAR_WIDTH,
                        tags=('histogram_detected_bars',),
                    )
                    x1, y1, x2, _y2 = canvas_bbox
                    center_x = (x1 + x2) / 2.0
                    self.canvas.create_oval(
                        center_x - 5, y1 - 5, center_x + 5, y1 + 5,
                        outline=HISTOGRAM_DETECTED_BAR_COLOR,
                        fill='white', width=2, tags=('histogram_bar_handles',),
                    )
                    try:
                        column_index = int(column.get('index'))
                    except (TypeError, ValueError):
                        column_index = None
                    if column_index is not None:
                        x1, y1, x2, _y2 = canvas_bbox
                        label_y = y1 + 10 if weight > 0.0 else y1 - 8
                        self.canvas.create_text(
                            (x1 + x2) / 2.0,
                            label_y,
                            text=str(column_index),
                            fill=HISTOGRAM_DETECTED_BAR_COLOR,
                            font=HISTOGRAM_BAR_LABEL_FONT,
                            tags=('histogram_detected_bars',),
                        )

            average_bbox = valid_bbox(result.get('average_column_bbox'))
            if average_bbox is not None:
                canvas_bbox = self._draw_translucent_source_box(
                    average_bbox,
                    HISTOGRAM_AVERAGE_COLUMN_FILL,
                    'histogram_average_column',
                )
                self.canvas.create_rectangle(
                    *canvas_bbox,
                    outline=HISTOGRAM_AVERAGE_COLUMN_COLOR,
                    width=HISTOGRAM_AVERAGE_COLUMN_WIDTH,
                    tags=('histogram_average_column',),
                )
                x1, y1, x2, _y2 = canvas_bbox
                mean_text = (
                    f'MEAN {mean_column_index}'
                    if mean_column_index is not None
                    else 'MEAN'
                )
                self.canvas.create_text(
                    (x1 + x2) / 2.0,
                    y1 + 11,
                    text=mean_text,
                    fill=HISTOGRAM_AVERAGE_COLUMN_COLOR,
                    font=HISTOGRAM_BAR_LABEL_FONT,
                    tags=('histogram_average_column',),
                )

            median_bbox = valid_bbox(result.get('median_column_bbox'))
            if median_bbox is not None:
                canvas_bbox = self._draw_translucent_source_box(
                    median_bbox,
                    HISTOGRAM_MEDIAN_COLUMN_FILL,
                    'histogram_median_column',
                )
                self.canvas.create_rectangle(
                    *canvas_bbox,
                    outline=HISTOGRAM_MEDIAN_COLUMN_COLOR,
                    width=HISTOGRAM_MEDIAN_COLUMN_WIDTH,
                    tags=('histogram_median_column',),
                )
                x1, y1, x2, _y2 = canvas_bbox
                median_text = (
                    f'50% {median_column_index}'
                    if median_column_index is not None
                    else '50%'
                )
                self.canvas.create_text(
                    (x1 + x2) / 2.0,
                    y1 + 24,
                    text=median_text,
                    fill=HISTOGRAM_MEDIAN_COLUMN_COLOR,
                    font=HISTOGRAM_BAR_LABEL_FONT,
                    tags=('histogram_median_column',),
                )
            self.canvas.tag_raise('digitization_bounds')
            self.canvas.tag_raise('digitization_handles')
            self.canvas.tag_raise('histogram_bar_handles')
            return
        raw_trace_points = result.get('trace_points')
        if not isinstance(raw_trace_points, list):
            return
        canvas_points: list[float] = []
        origin_x, origin_y = self.current_image_origin
        for point in raw_trace_points:
            if not isinstance(point, list) or len(point) != 2 or (not all((isinstance(value, (int, float)) for value in point))):
                continue
            source_x, source_y = point
            canvas_points.extend([origin_x + float(source_x) * self.current_display_scale, origin_y + float(source_y) * self.current_display_scale])
        if len(canvas_points) >= 4:
            self.canvas.create_line(*canvas_points, fill=DIGITIZED_TRACE_COLOR, width=DIGITIZED_TRACE_WIDTH, smooth=False, tags=('digitized_trace',))
        self.canvas.tag_raise('digitization_bounds')
        self.canvas.tag_raise('digitization_handles')

    def _show_canvas_message(self, message: str, show_open_pdf: bool=False) -> None:
        """Clear the page and show centered instructions and optional PDF action."""
        self._close_statistics_window()
        if self.current_source_image is not None:
            self.current_source_image.close()
        self.current_source_image = None
        self.current_display_image = None
        self.current_page_path = None
        self.current_pdf_pages = []
        self.current_pdf_page_index = 0
        self.page_nav_frame.place_forget()
        self.page_nav_var.set('Page —/—')
        self.current_display_width = 0
        self.current_display_height = 0
        self.current_canvas_message = message
        self.show_manual_pdf_search_button = bool(show_open_pdf)
        self._set_zoom_controls_enabled(False)
        self._render_canvas_message()

    def _render_canvas_message(self) -> None:
        """Redraw the current canvas message and manual-search action."""
        if self.current_canvas_message is None:
            return
        self.canvas.delete('all')
        self.canvas.configure(scrollregion=(0, 0, 0, 0))
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        if self.show_manual_pdf_search_button:
            message_y = max(55, canvas_height // 2 - 70)
            button_y = min(canvas_height - 55, canvas_height // 2 + 45)
        else:
            message_y = canvas_height // 2
            button_y = canvas_height // 2
        self.canvas.create_text(canvas_width // 2, message_y, text=self.current_canvas_message, fill='white', font=('TkDefaultFont', 16), justify='center', width=max(200, canvas_width - 80))
        if self.show_manual_pdf_search_button:
            pages_available = bool(self._current_deskewed_pages())
            self.manual_pdf_search_button.configure(
                state='normal' if pages_available else 'disabled'
            )
            self.canvas.create_window(canvas_width // 2, button_y, window=self.manual_pdf_search_button, anchor='center', tags=('manual_pdf_search_button',))

    def _clear_review_state(self) -> None:
        """Reset review indexes without resetting highlight preference."""
        self.review_pdfs = []
        self.review_pdf_index = 0
        self.review_stat_index = 0
        self.review_candidate_index = 0
        self.review_decisions = {}
        self.review_complete = False
        self.tracker_rows = {}
        self.tracker_manual_fields = {}
        self.tracker_key_to_iid = {}
        self.tracker_iid_to_key = {}
        self.tracker_active_key = None
        self.loaded_output_dirs.clear()
        self.review_waiting_for_processing = False
        if hasattr(self, 'tracker_tree'):
            for item_id in self.tracker_tree.get_children():
                self.tracker_tree.delete(item_id)
            self._load_tracker_editor(None)
        self.current_candidate = None
        self.current_page_path = None
        self.current_pdf_pages = []
        self.current_pdf_page_index = 0
        self.manual_search_active = False
        self.manual_review_candidate = None
        self.page_nav_frame.place_forget()
        self.page_nav_var.set('Page —/—')
        self.current_highlight_regions = []
        self.current_focus_bbox = None
        self.current_digitization_result = None
        self.manual_selection_bbox = None
        self.digitization_edge_drag = None
        self.histogram_height_drag = None
        self.digitization_bounds_dirty = False
        self.manual_selection_active = False
        self.manual_selection_start_source = None
        self.manual_selection_end_source = None
        self.interpretation_previous_highlights_visible = None
        self.interpretation_overlays_suppressed = False
        self.show_manual_pdf_search_button = False
        self.canvas.configure(cursor='')
        self.toggle_highlights_button.configure(text='Hide Markings' if self.highlights_visible else 'Mark Metric', state='disabled')
        self.open_pdf_button.configure(state='disabled')
        self.manual_digitize_button.configure(state='disabled')
        self.refresh_digitize_button.configure(state='disabled')
        self.back_button.configure(state='disabled')
        self._set_zoom_controls_enabled(False)
        self.next_candidate_button.configure(state='disabled')
        self.next_stat_button.configure(state='disabled')

def main() -> None:
    """Build the Tkinter GUI and enter its event loop.

    The GUI orchestrates PDF processing, candidate review, graph interpretation
    and editing/export of tracker values.
    """
    root = tk.Tk()
    MeyerBotGui(root)
    root.mainloop()
if __name__ == '__main__':
    main()
