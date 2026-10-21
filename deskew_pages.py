"""Deskew pages with one method: Canny -> long Hough lines -> one angle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


#File/output settings.
DEFAULT_OUTPUT_DIR = "output/deskewed_pages"
DEFAULT_DEBUG_DIR = "output/deskew_debug"
METADATA_FILENAME = "deskew_results.json"
SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
ANALYSIS_MAX_DIMENSION_PIXELS = 1800
MAX_INTERMEDIATE_FILENAME_CHARS = 96
DESKEW_HASH_CHARS = 16
#For ensuring intermediate filenames are safe length.
DESKEW_LONGEST_SUFFIX = "_deskew_debug.png"

#Detection/acceptance settings.
CANNY_LOW_THRESHOLD = 40
CANNY_HIGH_THRESHOLD = 120
HOUGH_VOTE_THRESHOLD = 70
HOUGH_MIN_LINE_LENGTH_FRACTION = 0.16
HOUGH_MAX_LINE_GAP_FRACTION = 0.018
MAX_HOUGH_CANDIDATES = 160
MAX_HOUGH_EDGE_DENSITY = 0.20
CARDINAL_AXIS_WINDOW_DEGREES = 8.0
IMAGE_BORDER_MARGIN_FRACTION = 0.008
ANGLE_CLUSTER_RADIUS_DEGREES = 0.40
MIN_DOMINANT_SUPPORT_RATIO = 0.34
MIN_DOMINANT_SUPPORT = 0.24
MIN_DOMINANT_CANDIDATES = 2
MAX_DOMINANT_SPREAD_DEGREES = 0.30
MIN_APPLIED_ROTATION_DEGREES = 0.10
MAX_ABS_DESKEW_DEGREES = 8.0
DEFAULT_SAVE_DEBUG_IMAGES = False


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    """Return a filename sort key that orders numeric page suffixes naturally."""
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name.casefold()))


def list_image_paths(input_dir: Path) -> list[Path]:
    """Return supported images from a directory in natural filename order."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    return sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=natural_sort_key,
    )


def safe_source_stem(image_path: Path) -> str:
    """Create a Windows-safe, length-limited filename stem.

    Append a short source-name hash when truncation is necessary.
    """
    original = image_path.stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("._-") or "page"
    limit = MAX_INTERMEDIATE_FILENAME_CHARS - len(DESKEW_LONGEST_SUFFIX)
    if len(safe) <= limit:
        return safe
    digest = hashlib.sha256(original.encode()).hexdigest()[:DESKEW_HASH_CHARS]
    return f"{safe[:limit - DESKEW_HASH_CHARS - 1]}_{digest}"


def read_grayscale(image_path: Path) -> np.ndarray:
    """Load a page image in grayscale or raise an error if unreadable."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def resize_for_analysis(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Return an image scaled to the analysis limit and its scale factor."""
    height, width = image.shape[:2]
    scale = min(1.0, ANALYSIS_MAX_DIMENSION_PIXELS / max(height, width))
    if scale == 1.0:
        return image, 1.0
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def normalize_line_angle(angle_degrees: float) -> float:
    """Normalize an undirected line angle to the range (-90, 90] degrees."""
    angle = ((angle_degrees + 90.0) % 180.0) - 90.0
    return 90.0 if math.isclose(angle, -90.0) else angle


def line_skew(x1: int, y1: int, x2: int, y2: int) -> float | None:
    """Return a line’s deviation from a page axis, or None if too diagonal."""
    angle = normalize_line_angle(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    if abs(angle) <= CARDINAL_AXIS_WINDOW_DEGREES:
        return angle
    if abs(abs(angle) - 90.0) <= CARDINAL_AXIS_WINDOW_DEGREES:
        return angle - math.copysign(90.0, angle)
    return None


def is_border_line(line: tuple[int, int, int, int], width: int, height: int) -> bool:
    """Return whether both endpoints lie close to the same image border."""
    x1, y1, x2, y2 = line
    mx = max(2, round(width * IMAGE_BORDER_MARGIN_FRACTION))
    my = max(2, round(height * IMAGE_BORDER_MARGIN_FRACTION))
    return (
        (x1 <= mx and x2 <= mx) 
        or (x1 >= width - mx and x2 >= width - mx)
        or (y1 <= my and y2 <= my) 
        or (y1 >= height - my and y2 >= height - my)
    )


def hough_candidates(image: np.ndarray) -> list[dict[str, Any]]:
    """Detect and rank long, near-horizontal/vertical deskew line segments."""
    edges = cv2.Canny(image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    # Skip noisy pages that would produce unreliable Hough line candidates.
    if np.count_nonzero(edges) / max(1, edges.size) > MAX_HOUGH_EDGE_DENSITY:
        return []

    height, width = edges.shape
    short_side, long_side = min(height, width), max(height, width)
    # Search for long straight segments with a half-degree angular resolution.
    lines = cv2.HoughLinesP(
        edges, 1, np.deg2rad(0.5), HOUGH_VOTE_THRESHOLD,
        minLineLength=max(40, round(short_side * HOUGH_MIN_LINE_LENGTH_FRACTION)),
        maxLineGap=max(3, round(long_side * HOUGH_MAX_LINE_GAP_FRACTION)),
    )
    if lines is None:
        return []

    candidates = []
    for raw in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, raw)
        line = (x1, y1, x2, y2)
        skew = None if is_border_line(line, width, height) else line_skew(*line)
        if skew is None or abs(skew) > MAX_ABS_DESKEW_DEGREES:
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        candidates.append({
            "angle_degrees": float(skew),
            "weight": length / long_side,
            "line": list(line),
        })

    candidates.sort(key=lambda item: float(item["weight"]), reverse=True)
    return candidates[:MAX_HOUGH_CANDIDATES]


def select_angle(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate skew from a median-centered cluster of candidate line angles.

    Return the angle, confidence, spread, acceptance flag and candidate counts.
    """
    if not candidates:
        return {
            "angle": 0.0, "confidence": 0.0, "spread": 0.0,
            "accepted": False, "candidate_count": 0, "dominant_count": 0,
        }

    angles = np.array([item["angle_degrees"] for item in candidates], dtype=float)
    weights = np.array([item["weight"] for item in candidates], dtype=float)

    # Cluster angles around the median to exclude outlier line orientations.
    median = np.median(angles)
    mask = np.abs(angles - median) <= ANGLE_CLUSTER_RADIUS_DEGREES

    # If the median falls between separate clusters, use the nearest observed angle.
    if not mask.any():
        center = angles[np.argmin(np.abs(angles - median))]
        mask = np.abs(angles - center) <= ANGLE_CLUSTER_RADIUS_DEGREES

    cluster_angles, cluster_weights = angles[mask], weights[mask]

    # Estimate skew using length-weighted angles and measure their disagreement.
    angle = float(np.average(cluster_angles, weights=cluster_weights))
    spread = float(np.sqrt(np.average((cluster_angles - angle) ** 2, weights=cluster_weights)))

    # Confidence represents the fraction of total candidate weight supporting the cluster.
    support = float(cluster_weights.sum())
    confidence = support / max(float(weights.sum()), 1e-9)
    dominant_count = int(mask.sum())

    # Accept only clusters with sufficient support, consistency, and plausible skew.
    accepted = (
        dominant_count >= MIN_DOMINANT_CANDIDATES
        and support >= MIN_DOMINANT_SUPPORT
        and confidence >= MIN_DOMINANT_SUPPORT_RATIO
        and spread <= MAX_DOMINANT_SPREAD_DEGREES
        and abs(angle) <= MAX_ABS_DESKEW_DEGREES
    )

    return {
        "angle": angle, "confidence": confidence, "spread": spread,
        "accepted": accepted, "candidate_count": len(candidates),
        "dominant_count": dominant_count,
    }


def estimate_page_skew(image: np.ndarray) -> dict[str, Any]:
    """Collect Hough candidates and return skew diagnostics and analysis image."""
    analysis_image, scale = resize_for_analysis(image)
    candidates = hough_candidates(analysis_image)
    result = select_angle(candidates)
    result["analysis_scale"] = scale

    return {"analysis_image": analysis_image, "candidates": candidates, "result": result}


def rotate_without_clipping(
    image: np.ndarray,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the image on a larger white canvas and return its affine matrix."""
    # Avoid unnecessary interpolation when no rotation is needed.
    if math.isclose(angle_degrees, 0.0, abs_tol=1e-12):
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        return image, matrix

    height, width = image.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # Calculate the bounding dimensions of the rotated image to prevent clipping.
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    output_width = math.ceil(height * sine + width * cosine)
    output_height = math.ceil(height * cosine + width * sine)

    # Shift the rotation's translation components to center the image on the new canvas.
    matrix[0, 2] += output_width / 2.0 - center[0]
    matrix[1, 2] += output_height / 2.0 - center[1]

    # Apply the transformation using linear interpolation and a white background.
    rotated = cv2.warpAffine(
        image, matrix, (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    return rotated, matrix


def save_png(image: np.ndarray, output_path: Path) -> None:
    """Save a PNG with moderate compression, raising an error on failure."""
    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Could not write image: {output_path}")


def draw_debug(
    image: np.ndarray,
    candidates: list[dict[str, Any]],
    result: dict[str, Any],
) -> np.ndarray:
    """Draw Hough candidates and skew results on a diagnostic color image."""
    debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Highlight candidate lines used to estimate page skew.
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["line"]
        cv2.line(debug, (x1, y1), (x2, y2), (0, 200, 0), 2)

    text = (
        f"angle={result['angle']:.3f}  confidence={result['confidence']:.3f}  "
        f"accepted={result['accepted']}"
    )
    cv2.putText(
        debug, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2, cv2.LINE_AA,
    )

    return debug


def deskew_one_image(
    image_path: Path,
    output_dir: Path,
    debug_dir: Path,
    save_debug_images: bool,
) -> dict[str, Any]:
    """Deskew one page and return output paths and rotation diagnostics.

    Only rotate accepted angles exceeding the minimum; optionally save an overlay.
    """
    image = read_grayscale(image_path)
    skew = estimate_page_skew(image)
    result = skew["result"]

    # Apply rotation only when sufficiently confident and large enough to matter.
    angle = float(result["angle"])
    rotation = angle if result["accepted"] and abs(angle) >= MIN_APPLIED_ROTATION_DEGREES else 0.0

    stem = safe_source_stem(image_path)
    output_path = output_dir / f"{stem}_deskewed.png"

    # Preserve the original PNG when no rotation is necessary.
    if rotation == 0.0 and image_path.suffix.lower() == ".png":
        shutil.copy2(image_path, output_path)
    else:
        corrected, _ = rotate_without_clipping(image, rotation)
        save_png(corrected, output_path)

    # Optionally save detected Hough lines and skew diagnostics for inspection.
    debug_path = None
    if save_debug_images:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"{stem}_deskew_debug.png"
        save_png(draw_debug(skew["analysis_image"], skew["candidates"], result), debug_path)

    return {
        "source_image_path": str(image_path),
        "deskewed_image_path": str(output_path),
        "debug_image_path": str(debug_path) if debug_path else None,
        "estimated_skew_degrees": round(angle, 6),
        "applied_rotation_degrees": round(rotation, 6),
        "angle_confidence": round(float(result["confidence"]), 6),
        "angle_accepted": bool(result["accepted"]),
        "candidate_count": int(result["candidate_count"]),
    }


def deskew_image_folder(
    input_dir: Path,
    output_dir: Path,
    debug_dir: Path,
    save_debug_images: bool,
) -> dict[str, Any]:
    """Deskew supported images in natural order and save a results manifest."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    debug_dir = debug_dir.resolve()

    # Prevent overwriting source images by requiring a separate output directory.
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different.")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_image_paths(input_dir)
    if not image_paths:
        raise ValueError(f"No supported page images found in: {input_dir}")

    pages = []
    for index, image_path in enumerate(image_paths, 1):
        result = deskew_one_image(image_path, output_dir, debug_dir, save_debug_images)
        pages.append(result)
        print(
            f"Deskewed {index}/{len(image_paths)}: {image_path.name} | "
            f"{result['applied_rotation_degrees']:.3f} deg",
            flush=True,
        )

    # Save a manifest of all processed pages and their deskew results.
    results = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "page_count": len(pages),
        "debug_images_saved": save_debug_images,
        "pages": pages,
    }
    (output_dir / METADATA_FILENAME).write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    """Parse deskew CLI options, process a folder and print output locations."""
    parser = argparse.ArgumentParser(description="Deskew rendered PDF page images.")
    parser.add_argument("input_dir", help="Directory containing page images.")
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Deskewed-page directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--debug-dir", default=DEFAULT_DEBUG_DIR,
        help=f"Debug-image directory. Default: {DEFAULT_DEBUG_DIR}",
    )
    parser.add_argument(
        "--debug", action="store_true", default=DEFAULT_SAVE_DEBUG_IMAGES,
        help="Save Hough-line debug overlays.",
    )
    args = parser.parse_args()

    results = deskew_image_folder(
        Path(args.input_dir).expanduser(),
        Path(args.out_dir).expanduser(),
        Path(args.debug_dir).expanduser(),
        args.debug,
    )

    print(f"Deskewed pages: {results['page_count']}")
    print(f"Output: {results['output_dir']}")


if __name__ == "__main__":
    main()
