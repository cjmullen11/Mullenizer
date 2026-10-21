"""
Interpret manually selected line graphs and histograms.

Line graphs:
1. Enhance the image and isolate dark marks from gridlines.
2. Track the graph's continuous line across the plotting area.
3. Smooth the detected path and map it to the supplied axis calibration.

Histograms:
1. Determine the number of bins from the supplied range and bin width.
2. Identify bin positions and heights while suppressing gridlines and noise.
3. Preserve empty and small bins.
4. Calculate weighted statistics, including mean and quantiles.

Both modes return results compatible with GUI and support
manual review and correction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


#Shared geometry/image settings.
MINIMUM_PLOT_WIDTH_PIXELS = 60
MINIMUM_PLOT_HEIGHT_PIXELS = 45
INNER_BORDER_FRACTION = 0.012
MINIMUM_INNER_BORDER_PIXELS = 2
MINIMUM_DARK_THRESHOLD = 35
MAXIMUM_DARK_THRESHOLD = 185
DARK_THRESHOLD_PERCENTILE = 18.0
DARK_PERCENTILE_MARGIN = 18.0
LOW_COVERAGE_WARNING_THRESHOLD = 0.45
LOW_CONFIDENCE_WARNING_THRESHOLD = 0.42

# Line-mode grid removal and tracking settings.
GRID_HORIZONTAL_LENGTH_FRACTION = 0.82
GRID_VERTICAL_LENGTH_FRACTION = 0.82
BROKEN_GRID_HORIZONTAL_GAP_FRACTION = 0.012
BROKEN_GRID_VERTICAL_GAP_FRACTION = 0.012
BROKEN_GRID_HORIZONTAL_LENGTH_FRACTION = 0.78
BROKEN_GRID_VERTICAL_LENGTH_FRACTION = 0.78
BROKEN_GRID_MINIMUM_GAP_KERNEL_PIXELS = 5
GRID_ROW_SUPPORT_FRACTION = 0.97
GRID_COLUMN_SUPPORT_FRACTION = 0.97
GRID_SUPPORT_BAND_RADIUS_PIXELS = 2
MINIMUM_GRID_KERNEL_PIXELS = 24
GRID_MASK_DILATION_PIXELS = 2
MAXIMUM_TRACKED_COLUMNS = 1400
MAXIMUM_CANDIDATES_PER_COLUMN = 10
MINIMUM_RUN_DARKNESS = 0.12
TRACKING_BEAM_WIDTH = 18
CONTINUITY_PENALTY = 2.8
LARGE_JUMP_PENALTY = 5.0
LARGE_JUMP_FRACTION = 0.10
GHOST_COLUMN_PENALTY = 0.34
OBSERVED_COLUMN_BONUS = 0.12
SMOOTHING_WINDOW = 3
VERTICAL_EXTREMUM_MIN_RUN_PIXELS = 5
VERTICAL_EXTREMUM_NEIGHBOR_TOLERANCE_FRACTION = 0.08
VERTICAL_EXTREMUM_MIN_GAIN_PIXELS = 3
DARK_RIDGE_MAX_INTENSITY_DELTA = 24
DARK_RIDGE_MINIMUM_RELATIVE_DARKNESS = 0.72
DARK_RIDGE_PEAK_WEIGHT = 1.35
DARK_RIDGE_MEAN_WEIGHT = 0.35
DARK_RIDGE_THICKNESS_PENALTY = 0.025
GRID_TRACE_RESCUE_DILATION_PIXELS = 4
VERTICAL_TRANSITION_SUPPORT_BONUS = 0.88
MAXIMUM_OUTPUT_TRACE_POINTS = 2500
TRACE_DARKEST_PERCENTILE = 5.0
TRACE_DARK_THRESHOLD_MARGIN = 24.0
LINE_BOTTOM_AXIS_SEARCH_FRACTION = 0.10
LINE_BOTTOM_AXIS_MINIMUM_SPAN_FRACTION = 0.50

HISTOGRAM_REGULAR_MAXIMUM_BINS = 80
DEFAULT_HISTOGRAM_BIN_WIDTH = 10.0
# The reviewer supplies the numerical range and bin width. When calibrated,
# these determine the exact slot count; the image never gets to change it.
HISTOGRAM_KNOWN_BIN_COUNT_RELATIVE_TOLERANCE = 1e-7
HISTOGRAM_KNOWN_BIN_INNER_FRACTION = 0.13
HISTOGRAM_KNOWN_GRID_BASELINE_SEARCH_FRACTION = 0.12
HISTOGRAM_REGULAR_MINIMUM_BIN_PIXELS = 5
HISTOGRAM_REGULAR_INNER_FRACTION = 0.16
HISTOGRAM_REGULAR_ROW_SUPPORT = 0.10
HISTOGRAM_REGULAR_TICK_SEARCH_PIXELS = 14
HISTOGRAM_SPACED_BAR_BASELINE_TOLERANCE_FRACTION = 0.025
HISTOGRAM_SPACED_BAR_WIDTH_TOLERANCE_FRACTION = 0.30
HISTOGRAM_SPACED_BAR_PITCH_TOLERANCE_FRACTION = 0.18
HISTOGRAM_SPACED_BAR_MINIMUM_WIDTH_PIXELS = 3
HISTOGRAM_BROKEN_GRID_GAP_FRACTION = 0.012
HISTOGRAM_BROKEN_GRID_MINIMUM_LENGTH_FRACTION = 0.42
HISTOGRAM_BAR_COMPONENT_MINIMUM_WIDTH_FRACTION = 0.24
HISTOGRAM_BAR_COMPONENT_BASELINE_TOLERANCE_PIXELS = 3
HISTOGRAM_BAR_HORIZONTAL_RUN_FRACTION = 0.010
HISTOGRAM_BAR_INTERVAL_OVERLAP_FRACTION = 0.65
HISTOGRAM_AXIS_SEARCH_HEIGHT_FRACTION = 0.045
HISTOGRAM_AXIS_MINIMUM_SEARCH_PIXELS = 28
HISTOGRAM_AXIS_MAXIMUM_SEARCH_PIXELS = 90
HISTOGRAM_AXIS_TEXT_DILATION_FRACTION = 0.008

#Histogram rectangle settings.
HISTOGRAM_MINIMUM_BAR_WIDTH_FRACTION = 0.010
HISTOGRAM_MINIMUM_BAR_HEIGHT_FRACTION = 0.040
HISTOGRAM_MAXIMUM_BAR_WIDTH_FRACTION = 0.32
HISTOGRAM_MAXIMUM_BAR_HEIGHT_FRACTION = 0.96
HISTOGRAM_GRID_HORIZONTAL_LENGTH_FRACTION = 0.55
HISTOGRAM_GRID_VERTICAL_LENGTH_FRACTION = 0.82
HISTOGRAM_FILLED_MINIMUM_DENSITY = 0.24
HISTOGRAM_OUTLINE_MINIMUM_SIDE_SUPPORT = 0.62
HISTOGRAM_OUTLINE_MINIMUM_TOP_SUPPORT = 0.58
HISTOGRAM_OUTLINE_MINIMUM_BOTTOM_SUPPORT = 0.50
HISTOGRAM_OUTLINE_MAXIMUM_EXTENSION_SUPPORT = 0.38
HISTOGRAM_EDGE_SEARCH_FRACTION = 0.055
HISTOGRAM_BASELINE_TOLERANCE_FRACTION = 0.030
HISTOGRAM_MINIMUM_WIDTH_RATIO_TO_MEDIAN = 0.45
HISTOGRAM_MAXIMUM_WIDTH_RATIO_TO_MEDIAN = 2.20
HISTOGRAM_MINIMUM_COLUMNS = 2
HISTOGRAM_SHADE_PERCENTILE = 75.0
HISTOGRAM_SHADE_THRESHOLD_MARGIN = 12.0
HISTOGRAM_SHADE_MAX_THRESHOLD = 220
HISTOGRAM_STACK_X_OVERLAP_FRACTION = 0.68
HISTOGRAM_STACK_MAX_VERTICAL_GAP_FRACTION = 0.035
HISTOGRAM_PROFILE_BASELINE_START_FRACTION = 0.50
HISTOGRAM_PROFILE_HORIZONTAL_RULE_FRACTION = 0.45
HISTOGRAM_PROFILE_MEDIAN_WINDOW = 7
HISTOGRAM_PROFILE_MINIMUM_WIDTH_FRACTION = 0.010
HISTOGRAM_PROFILE_MINIMUM_HEIGHT_FRACTION = 0.004
HISTOGRAM_PROFILE_CHANGE_FRACTION = 0.018
HISTOGRAM_PROFILE_WIDE_SEGMENT_SPLIT_RATIO = 1.60
HISTOGRAM_PROFILE_EMPTY_GAP_RATIO = 1.45
HISTOGRAM_PROFILE_MAXIMUM_INFERRED_EMPTY_BINS = 64

#Histogram tick-grid settings.
HISTOGRAM_TICK_BASELINE_MINIMUM_COVERAGE = 0.45
HISTOGRAM_TICK_MINIMUM_COUNT = 5
HISTOGRAM_TICK_MAXIMUM_PITCH_DEVIATION_FRACTION = 0.28
HISTOGRAM_TICK_BELOW_BASELINE_SEARCH_PIXELS = 7
HISTOGRAM_TICK_MINIMUM_BELOW_BASELINE_ROWS = 2
HISTOGRAM_TICK_MINIMUM_BAR_HEIGHT_PIXELS = 1
HISTOGRAM_TICK_MINIMUM_TOP_SUPPORT = 0.50
HISTOGRAM_TICK_MINIMUM_SIDE_SUPPORT = 0.42
HISTOGRAM_TICK_LONG_HORIZONTAL_SUPPORT_LIMIT = 0.50
HISTOGRAM_BIN_TOP_MINIMUM_SUPPORT = 0.14
HISTOGRAM_EDGE_EMPTY_INFERENCE_TOLERANCE = 0.32


def read_grayscale_image(image_path: Path) -> np.ndarray:
    """Load a grayscale image, validating its existence and readability."""
    image_path = image_path.expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    return image


def require_finite_number(name: str, value: int | float) -> float:
    """Convert a numeric value to float, rejecting NaN and infinity."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number.")
    return numeric


def normalize_bbox(
    bbox: list[int | float],
    image_width: int,
    image_height: int,
) -> list[int]:
    """Normalize a plot rectangle to integer image coordinates.

    Orders the corners, restricts the rectangle to the image boundaries,
    and rejects regions smaller than the minimum plot dimensions.
    """
    if len(bbox) != 4:
        raise ValueError("Plot bounding box must contain x1, y1, x2, y2.")

    # Validate and round coordinates, then ensure corners are correctly ordered.
    x1, y1, x2, y2 = [
        int(round(require_finite_number("Plot bounding-box coordinate", value)))
        for value in bbox
    ]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    # Restrict the rectangle to the image boundaries.
    x1, x2 = max(0, x1), min(image_width, x2)
    y1, y2 = max(0, y1), min(image_height, y2)
    width, height = x2 - x1, y2 - y1

    if width < MINIMUM_PLOT_WIDTH_PIXELS:
        raise ValueError(
            f"Plot rectangle is too narrow ({width}px); "
            f"minimum is {MINIMUM_PLOT_WIDTH_PIXELS}px."
        )
    if height < MINIMUM_PLOT_HEIGHT_PIXELS:
        raise ValueError(
            f"Plot rectangle is too short ({height}px); "
            f"minimum is {MINIMUM_PLOT_HEIGHT_PIXELS}px."
        )

    return [x1, y1, x2, y2]


def temporary_sibling_path(path: Path) -> Path:
    """Return a temporary path beside the destination for atomic file writes."""
    return path.with_name(f".{path.name}.tmp")


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Atomically save a dictionary as formatted UTF-8 JSON.

    Writes to a temporary file before replacing the destination,
    ensuring incomplete writes do not corrupt existing results.
    """
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_sibling_path(output_path)

    try:
        temporary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def choose_dark_threshold(crop: np.ndarray) -> int:
    """Choose a conservative dark-pixel threshold using Otsu and percentile methods.

    Uses the lower estimate to favor dark graph markings over faint gridlines.
    Restricts the result to the configured threshold limits.
    """
    nonwhite = crop[crop < 250]
    if nonwhite.size == 0:
        return MAXIMUM_DARK_THRESHOLD

    percentile = float(np.percentile(nonwhite, DARK_THRESHOLD_PERCENTILE))
    otsu, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Favor the more conservative threshold, within the configured limits.
    threshold = min(otsu, percentile + DARK_PERCENTILE_MARGIN)
    return int(round(np.clip(threshold, MINIMUM_DARK_THRESHOLD, MAXIMUM_DARK_THRESHOLD)))


def choose_trace_threshold(crop: np.ndarray) -> int:
    """Choose a threshold favoring the darkest plotted line.

    Refines the general dark-pixel threshold using the darkest nonwhite
    pixels to reduce interference from lighter lines and gridlines.
    """
    general_threshold = choose_dark_threshold(crop)
    nonwhite = crop[crop < 245]

    if nonwhite.size == 0:
        return general_threshold

    # Use the darkest pixels plus a margin, without exceeding the general threshold.
    darkest = float(np.percentile(nonwhite, TRACE_DARKEST_PERCENTILE))
    threshold = min(general_threshold, darkest + TRACE_DARK_THRESHOLD_MARGIN)

    return int(np.clip(threshold, MINIMUM_DARK_THRESHOLD, MAXIMUM_DARK_THRESHOLD))


def long_rule_mask(dark_mask: np.ndarray) -> np.ndarray:
    """Identify continuous and broken horizontal/vertical gridlines.

    Uses morphological operations and pixel-density checks to identify
    long rules. Returns a binary mask for excluding them from graph tracing.
    """
    height, width = dark_mask.shape

    # Morphological opening isolates long, continuous horizontal and vertical lines.
    horizontal_length = max(MINIMUM_GRID_KERNEL_PIXELS, round(width * GRID_HORIZONTAL_LENGTH_FRACTION))
    vertical_length = max(MINIMUM_GRID_KERNEL_PIXELS, round(height * GRID_VERTICAL_LENGTH_FRACTION))

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length))

    continuous_horizontal = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, horizontal_kernel)
    continuous_vertical = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, vertical_kernel)

    # Closing reconnects short gaps in dashed or scan-broken lines.
    horizontal_gap = max(
        BROKEN_GRID_MINIMUM_GAP_KERNEL_PIXELS,
        round(width * BROKEN_GRID_HORIZONTAL_GAP_FRACTION),
    )
    vertical_gap = max(
        BROKEN_GRID_MINIMUM_GAP_KERNEL_PIXELS,
        round(height * BROKEN_GRID_VERTICAL_GAP_FRACTION),
    )

    # Ensure odd kernel dimensions for symmetric gap filling.
    horizontal_gap |= 1
    vertical_gap |= 1

    closed_horizontal = cv2.morphologyEx(
        dark_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_gap, 1)),
    )
    closed_vertical = cv2.morphologyEx(
        dark_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_gap)),
    )

    # Opening retains only sufficiently long lines after reconnecting their gaps.
    broken_horizontal_length = max(
        MINIMUM_GRID_KERNEL_PIXELS,
        round(width * BROKEN_GRID_HORIZONTAL_LENGTH_FRACTION),
    )
    broken_vertical_length = max(
        MINIMUM_GRID_KERNEL_PIXELS,
        round(height * BROKEN_GRID_VERTICAL_LENGTH_FRACTION),
    )

    broken_horizontal = cv2.morphologyEx(
        closed_horizontal, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (broken_horizontal_length, 1)),
    )
    broken_vertical = cv2.morphologyEx(
        closed_vertical, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, broken_vertical_length)),
    )

    # Combine continuous and broken rules into one mask.
    mask = continuous_horizontal | continuous_vertical | broken_horizontal | broken_vertical

    # Additionally identify rows or columns that are almost entirely dark.
    row_support = np.count_nonzero(dark_mask, axis=1) / float(max(1, width))
    column_support = np.count_nonzero(dark_mask, axis=0) / float(max(1, height))

    for y in np.flatnonzero(row_support >= GRID_ROW_SUPPORT_FRACTION):
        mask[
            max(0, int(y) - GRID_SUPPORT_BAND_RADIUS_PIXELS):
            min(height, int(y) + GRID_SUPPORT_BAND_RADIUS_PIXELS + 1),
            :,
        ] = 255

    for x in np.flatnonzero(column_support >= GRID_COLUMN_SUPPORT_FRACTION):
        mask[
            :,
            max(0, int(x) - GRID_SUPPORT_BAND_RADIUS_PIXELS):
            min(width, int(x) + GRID_SUPPORT_BAND_RADIUS_PIXELS + 1),
        ] = 255

    # Expand detected lines slightly to cover their full thickness.
    if GRID_MASK_DILATION_PIXELS:
        size = 2 * GRID_MASK_DILATION_PIXELS + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))

    return mask

def bottom_axis_mask(dark_mask: np.ndarray) -> np.ndarray:
    """Identify the bottom horizontal x-axis of a line graph.

    Searches the bottom portion of the plot for rows with sufficient
    dark-pixel coverage. Returns a mask of the lowest qualifying line.
    """
    height, width = dark_mask.shape

    # Find sufficiently dark rows within the bottom portion of the plot.
    start = max(0, round(height * (1.0 - LINE_BOTTOM_AXIS_SEARCH_FRACTION)))
    support = np.count_nonzero(dark_mask[start:, :], axis=1) / max(1, width)
    rows = np.flatnonzero(support >= LINE_BOTTOM_AXIS_MINIMUM_SPAN_FRACTION)

    result = np.zeros_like(dark_mask)
    if rows.size == 0:
        return result

    # Select the lowest continuous group of qualifying rows and pad its thickness.
    groups = contiguous_groups(start + rows)
    top, bottom = groups[-1]
    top = max(start, top - 2)
    bottom = min(height - 1, bottom + 2)

    result[top:bottom + 1, :] = 255
    return result


def prepare_trace_evidence(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare line-tracing evidence by removing gridlines and restoring crossings.

    Returns the original image, a binary mask of usable trace pixels,
    and the grayscale threshold used for dark-pixel detection.
    """
    threshold = choose_trace_threshold(crop)
    dark_mask = cv2.inRange(crop, 0, threshold)

    # Remove long gridlines and the bottom axis from the trace evidence.
    baseline_mask = bottom_axis_mask(dark_mask)
    grid_mask = long_rule_mask(dark_mask) | baseline_mask
    usable = dark_mask & ~grid_mask

    nonwhite = crop[crop < 250]
    if nonwhite.size:
        # Identify especially dark pixels, which are more likely to belong to the trace.
        darkest_reference = float(np.percentile(nonwhite, 12.0))
        strong_threshold = int(np.clip(
            darkest_reference + DARK_RIDGE_MAX_INTENSITY_DELTA,
            MINIMUM_DARK_THRESHOLD,
            threshold,
        ))
        strong_dark = cv2.inRange(crop, 0, strong_threshold)

        # Recover dark pixels removed with gridlines when adjacent to surviving trace pixels.
        trace_seed = strong_dark & ~grid_mask
        rescue_size = 2 * GRID_TRACE_RESCUE_DILATION_PIXELS + 1
        rescue_zone = cv2.dilate(
            trace_seed,
            np.ones((rescue_size, rescue_size), np.uint8),
        )
        usable |= strong_dark & grid_mask & rescue_zone & ~baseline_mask

    # Exclude plot boundaries to prevent axes and selection edges from influencing tracing.
    border = max(
        MINIMUM_INNER_BORDER_PIXELS,
        round(min(crop.shape[:2]) * INNER_BORDER_FRACTION),
    )
    usable[:border, :] = usable[-border:, :] = 0
    usable[:, :border] = usable[:, -border:] = 0

    return crop, usable, threshold

def choose_sample_columns(width: int) -> np.ndarray:
    """Return evenly spaced x-coordinates for line tracing.

    Includes the plot boundaries and limits sampling to
    MAXIMUM_TRACKED_COLUMNS columns.
    """
    if width <= 0:
        raise ValueError("Plot width must be positive.")

    count = min(width, MAXIMUM_TRACKED_COLUMNS)
    return np.rint(np.linspace(0, width - 1, count)).astype(np.int32)


def column_candidates(
    enhanced: np.ndarray,
    usable_mask: np.ndarray,
    x: int,
) -> list[tuple[float, float]]:
    """Identify the strongest trace positions at a given x-coordinate.

    Returns up to MAXIMUM_CANDIDATES_PER_COLUMN candidates as (y, score)
    pairs. Searches adjacent columns only when the requested column is empty.
    """
    width = enhanced.shape[1]
    x1, x2 = x, min(width, x + 1)
    indices = np.flatnonzero(np.any(usable_mask[:, x1:x2], axis=1))

    # Search neighboring columns if grid removal or sampling left this column empty.
    if indices.size == 0:
        x1, x2 = max(0, x - 1), min(width, x + 2)
        indices = np.flatnonzero(np.any(usable_mask[:, x1:x2], axis=1))

    if indices.size == 0:
        return []

    candidates = []

    # Treat each continuous vertical group of dark pixels as a separate candidate.
    for y1, y2 in contiguous_groups(indices):
        intensities = np.min(enhanced[y1:y2 + 1, x1:x2], axis=1).astype(np.float32)
        darkness = (255.0 - intensities) / 255.0
        peak_darkness = float(np.max(darkness))

        if peak_darkness < MINIMUM_RUN_DARKNESS:
            continue

        # Isolate the darkest ridge within the group, excluding lighter surrounding pixels.
        minimum_ridge_darkness = max(
            MINIMUM_RUN_DARKNESS,
            peak_darkness * DARK_RIDGE_MINIMUM_RELATIVE_DARKNESS,
        )
        minimum_ridge_intensity = float(np.min(intensities)) + DARK_RIDGE_MAX_INTENSITY_DELTA

        ridge_mask = (
            (darkness >= minimum_ridge_darkness)
            & (intensities <= minimum_ridge_intensity)
        )

        if not np.any(ridge_mask):
            ridge_mask[int(np.argmax(darkness))] = True

        ridge_darkness = darkness[ridge_mask]
        ridge_rows = np.arange(y1, y2 + 1, dtype=np.float32)[ridge_mask]

        # Weight darker pixels more heavily to estimate the ridge's vertical center.
        weights = np.maximum(ridge_darkness, 1e-4) ** 3
        center = float(np.average(ridge_rows, weights=weights))

        # Favor candidates with strong peak darkness and consistently dark pixels.
        score = float(
            peak_darkness * DARK_RIDGE_PEAK_WEIGHT
            + np.mean(ridge_darkness) * DARK_RIDGE_MEAN_WEIGHT
        )
        candidates.append((center, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:MAXIMUM_CANDIDATES_PER_COLUMN]

def vertical_transition_support(
    usable_mask: np.ndarray,
    x: int,
    previous_y: np.ndarray,
    next_y: float,
) -> np.ndarray:
    """Measure dark-pixel support for vertical transitions between trace positions.

    Returns the fraction of rows containing dark pixels between each previous
    y-position and the proposed next position, using a three-column window.
    """
    height, width = usable_mask.shape
    x1, x2 = max(0, x - 1), min(width, x + 2)

    # Identify rows containing dark pixels near the proposed transition.
    profile = np.any(usable_mask[:, x1:x2], axis=1).astype(np.int32)

    # Cumulative sums allow counting dark rows across multiple intervals.
    cumulative = np.r_[0, np.cumsum(profile)]

    # Clamp all positions to valid image rows.
    y2 = int(np.clip(round(next_y), 0, height - 1))
    y1 = np.clip(np.rint(previous_y).astype(np.int32), 0, height - 1)

    # Calculate the fraction of dark rows along each proposed vertical transition.
    low = np.minimum(y1, y2)
    high = np.maximum(y1, y2)
    counts = cumulative[high + 1] - cumulative[low]
    lengths = high - low + 1

    return counts / lengths


def track_continuous_path(
    enhanced: np.ndarray,
    usable_mask: np.ndarray,
    sample_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Track the strongest continuous dark path across sampled columns.

    Use a bounded beam search, allowing inferred gaps and supported steep jumps.
    Return y-positions, observation flags and quality diagnostics.
    """
    height = enhanced.shape[0]
    # Beam state: accumulated score, y-position, direct-observation flag and predecessor index.
    history: list[list[tuple[float, float, bool, int | None]]] = []
    first: int | None = None

    for x in sample_columns:
        actual = column_candidates(
            enhanced,
            usable_mask,
            int(x),
        )

        if first is None:
            if not actual:
                history.append([])
                continue

            states = [
                (
                    score + OBSERVED_COLUMN_BONUS,
                    y,
                    True,
                    None,
                )
                for y, score in actual
            ]
            states.sort(reverse=True)
            history.append(
                states[:TRACKING_BEAM_WIDTH]
            )
            first = len(history) - 1
            continue

        previous = history[-1]
        # Add observed ridge positions, then ghost positions to bridge missing trace pixels.
        options: dict[int, tuple[float, float, bool]] = {
            int(round(y)): (
                y,
                score + OBSERVED_COLUMN_BONUS,
                True,
            )
            for y, score in actual
        }

        for _score, y, _observed, _prior in previous:
            options.setdefault(
                int(round(y)),
                (
                    y,
                    -GHOST_COLUMN_PENALTY,
                    False,
                ),
            )

        previous_y = np.asarray(
            [state[1] for state in previous],
            dtype=float,
        )
        previous_score = np.asarray(
            [state[0] for state in previous],
            dtype=float,
        )
        states = []

        # Penalize large jumps unless a dark vertical segment supports the transition.
        for y, local_score, observed in options.values():
            jump = (
                np.abs(previous_y - y)
                / max(1, height)
            )
            penalties = (
                CONTINUITY_PENALTY * jump
                + LARGE_JUMP_PENALTY
                * np.maximum(
                    0.0,
                    jump - LARGE_JUMP_FRACTION,
                )
            )

            if observed and np.any(
                jump > LARGE_JUMP_FRACTION * 0.35
            ):
                support = vertical_transition_support(
                    usable_mask,
                    int(x),
                    previous_y,
                    y,
                )
                penalties *= (
                    1.0
                    - VERTICAL_TRANSITION_SUPPORT_BONUS
                    * np.clip(support, 0.0, 1.0)
                )

            scores = (
                previous_score
                + local_score
                - penalties
            )
            prior = int(np.argmax(scores))
            states.append(
                (
                    float(scores[prior]),
                    float(y),
                    observed,
                    prior,
                )
            )

        states.sort(reverse=True)
        history.append(
            states[:TRACKING_BEAM_WIDTH]
        )

    if first is None:
        raise RuntimeError(
            "No dark continuous trace could be found inside the selected plot."
        )

    # Backtrack through the best surviving beam state to recover the full path.
    final_index = int(
        np.argmax(
            [state[0] for state in history[-1]]
        )
    )
    tracked = np.full(
        len(sample_columns),
        np.nan,
        dtype=float,
    )
    observed = np.zeros(
        len(sample_columns),
        dtype=bool,
    )

    for column in range(
        len(history) - 1,
        first - 1,
        -1,
    ):
        state = history[column][final_index]
        tracked[column] = state[1]
        observed[column] = state[2]

        if state[3] is None:
            break

        final_index = int(state[3])

    tracked[:first] = tracked[first]

    if not np.all(np.isfinite(tracked)):
        raise RuntimeError(
            "The trace tracker produced invalid coordinates."
        )

    coverage = float(np.mean(observed))
    mean_jump = (
        float(np.mean(np.abs(np.diff(tracked))))
        if tracked.size > 1
        else 0.0
    )
    continuity = max(
        0.0,
        1.0
        - mean_jump
        / max(1.0, height * 0.06),
    )
    return tracked, observed, {
        "observed_column_fraction": coverage,
        "mean_sampled_jump_pixels": mean_jump,
        "continuity_score": continuity,
    }

def refine_vertical_extrema(
    values: np.ndarray,
    usable_mask: np.ndarray,
    sample_columns: np.ndarray,
) -> np.ndarray:
    """Restore tall peaks and valleys flattened by line tracking.

    Prefer vertical-run endpoints only when neighboring trace positions agree.
    """
    if values.size < 3:
        return values.copy()

    refined = values.copy()
    height = usable_mask.shape[0]
    neighbor_tolerance = (
        height
        * VERTICAL_EXTREMUM_NEIGHBOR_TOLERANCE_FRACTION
    )

    for index in range(1, values.size - 1):
        previous_y = float(values[index - 1])
        next_y = float(values[index + 1])

        # A sharp isolated peak/valley is plausible only when its neighboring levels agree.
        if abs(previous_y - next_y) > neighbor_tolerance:
            continue

        x = int(sample_columns[index])
        rows = np.flatnonzero(usable_mask[:, x] > 0)

        if rows.size < VERTICAL_EXTREMUM_MIN_RUN_PIXELS:
            continue

        breaks = np.flatnonzero(np.diff(rows) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, rows.size - 1]
        current_y = float(values[index])

        groups = [
            (
                int(rows[start]),
                int(rows[end]),
            )
            for start, end in zip(starts, ends)
        ]
        top, bottom = min(
            groups,
            key=lambda group: min(
                abs(group[0] - current_y),
                abs(group[1] - current_y),
                0.0
                if group[0] <= current_y <= group[1]
                else float("inf"),
            ),
        )

        if (
            bottom - top + 1
            < VERTICAL_EXTREMUM_MIN_RUN_PIXELS
        ):
            continue

        reference = (previous_y + next_y) / 2.0
        # Pick the endpoint farthest from the mean neighboring height.
        endpoint = (
            float(top)
            if abs(top - reference)
            >= abs(bottom - reference)
            else float(bottom)
        )

        if (
            abs(endpoint - reference)
            >= abs(current_y - reference)
            + VERTICAL_EXTREMUM_MIN_GAIN_PIXELS
        ):
            refined[index] = endpoint

    return refined


def median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a centered median filter."""
    if window <= 1 or values.size < 3:
        return values.copy()
    window = min(window | 1, values.size if values.size % 2 else values.size - 1)
    if window <= 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.median(
        np.lib.stride_tricks.sliding_window_view(padded, window),
        axis=1,
    )


def pixel_y_to_value(
    y_positions: np.ndarray,
    plot_height: int,
    y_top: float,
    y_bottom: float,
) -> np.ndarray:
    """Map plot-relative y pixels to calibrated values."""
    return y_top + y_positions / max(1, plot_height - 1) * (y_bottom - y_top)


def normalized_measurement_unit(unit: str) -> str:
    """Normalize units used rounding rules."""
    normalized = str(unit).strip().lower().replace(" ", "")
    if normalized in {"bpm", "beats/min", "beatsperminute"}:
        return "bpm"
    if normalized in {"%", "percent", "percentage", "pct"}:
        return "%"
    if normalized in {
        "h/d", "hr/d", "hrs/d", "hour/day", "hours/day",
        "hourperday", "hoursperday",
    }:
        return "h/d"
    if normalized in {
        "min/d", "mins/d", "minute/day", "minutes/day",
        "minuteperday", "minutesperday",
    }:
        return "min/d"
    return normalized


def round_measurement_value(value: int | float, unit: str) -> float | int:
    """Apply standard output precision."""
    numeric = require_finite_number("Measurement value", value)
    normalized = normalized_measurement_unit(unit)
    if normalized in {"%", "bpm", "min/d"}:
        return int(round(numeric))
    if normalized == "h/d":
        return round(numeric, 2)
    return round(numeric, 4)


def calculate_statistics(values: np.ndarray, unit: str) -> dict[str, Any]:
    """Calculate compact line-trace statistics."""
    if values.size == 0:
        raise ValueError("Cannot calculate statistics from an empty trace.")
    p5, p10, p25, median, p75, p90, p95 = np.percentile(
        values, [5, 10, 25, 50, 75, 90, 95]
    )
    robust = values[(values >= p5) & (values <= p95)]
    if robust.size == 0:
        robust = values

    round_value = lambda value: round_measurement_value(float(value), unit)
    return {
        "unit": unit,
        "average": round_value(np.mean(values)),
        "robust_average_5_95": round_value(np.mean(robust)),
        "median": round_value(median),
        "minimum": round_value(np.min(values)),
        "maximum": round_value(np.max(values)),
        "standard_deviation": round_value(np.std(values)),
        "percentile_10": round_value(p10),
        "percentile_25": round_value(p25),
        "percentile_75": round_value(p75),
        "percentile_90": round_value(p90),
        "sample_count": int(values.size),
    }


def thin_trace_points(
    source_x: np.ndarray,
    source_y: np.ndarray,
) -> list[list[float]]:
    """Limit source-page trace points used by the GUI overlay."""
    if source_x.size > MAXIMUM_OUTPUT_TRACE_POINTS:
        indices = np.unique(
            np.rint(
                np.linspace(0, source_x.size - 1, MAXIMUM_OUTPUT_TRACE_POINTS)
            ).astype(np.int32)
        )
        source_x, source_y = source_x[indices], source_y[indices]
    return np.column_stack((np.round(source_x, 3), np.round(source_y, 3))).tolist()


def histogram_masks(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Build dark-outline and lighter-fill masks for histogram detection."""
    threshold = choose_dark_threshold(crop)
    dark = cv2.inRange(crop, 0, threshold)

    nonwhite = crop[crop < 245]
    shade_threshold = threshold

    if nonwhite.size:
        shade_threshold = int(
            np.clip(
                max(
                    threshold,
                    float(
                        np.percentile(
                            nonwhite,
                            HISTOGRAM_SHADE_PERCENTILE,
                        )
                    )
                    + HISTOGRAM_SHADE_THRESHOLD_MARGIN,
                ),
                threshold,
                HISTOGRAM_SHADE_MAX_THRESHOLD,
            )
        )

    shaded = cv2.inRange(
        crop,
        0,
        shade_threshold,
    )
    height, width = shaded.shape
    horizontal_length = max(
        MINIMUM_GRID_KERNEL_PIXELS,
        round(
            width
            * HISTOGRAM_GRID_HORIZONTAL_LENGTH_FRACTION
        ),
    )
    vertical_length = max(
        MINIMUM_GRID_KERNEL_PIXELS,
        round(
            height
            * HISTOGRAM_GRID_VERTICAL_LENGTH_FRACTION
        ),
    )
    rules = cv2.bitwise_or(
        cv2.morphologyEx(
            shaded,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (horizontal_length, 1),
            ),
        ),
        cv2.morphologyEx(
            shaded,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, vertical_length),
            ),
        ),
    )
    rules = cv2.dilate(
        rules,
        np.ones((3, 3), np.uint8),
    )
    bars = cv2.bitwise_and(
        shaded,
        cv2.bitwise_not(rules),
    )
    close_width = max(
        3,
        round(width * 0.008),
    )
    close_height = max(
        3,
        round(height * 0.010),
    )
    bars = cv2.morphologyEx(
        bars,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (close_width, close_height),
        ),
    )
    border = max(
        MINIMUM_INNER_BORDER_PIXELS,
        round(
            min(crop.shape)
            * INNER_BORDER_FRACTION
        ),
    )
    bars[:border, :] = bars[-border:, :] = 0
    bars[:, :border] = bars[:, -border:] = 0
    return bars, dark, threshold

def band_support(
    mask: np.ndarray,
    horizontal: bool,
    coordinate: int,
    start: int,
    end: int,
    radius: int = 1,
) -> float:
    """Measure dark-pixel coverage along a horizontal or vertical band."""
    height, width = mask.shape
    if horizontal:
        y1, y2 = max(0, coordinate - radius), min(height, coordinate + radius + 1)
        x1, x2 = max(0, start), min(width, end)
        region = mask[y1:y2, x1:x2]
        return float(np.mean(np.any(region > 0, axis=0))) if region.size else 0.0
    x1, x2 = max(0, coordinate - radius), min(width, coordinate + radius + 1)
    y1, y2 = max(0, start), min(height, end)
    region = mask[y1:y2, x1:x2]
    return float(np.mean(np.any(region > 0, axis=1))) if region.size else 0.0


def best_edge(
    mask: np.ndarray,
    horizontal: bool,
    expected: int,
    start: int,
    end: int,
    search_radius: int,
) -> tuple[float, int]:
    """Find the highest-support edge near an expected bar boundary."""
    best = (0.0, int(expected))
    limit = mask.shape[0 if horizontal else 1]
    for coordinate in range(max(0, expected - search_radius), min(limit - 1, expected + search_radius) + 1):
        support = band_support(mask, horizontal, coordinate, start, end)
        if support > best[0]:
            best = (support, coordinate)
    return best


def edge_extension(
    mask: np.ndarray,
    bbox: list[int],
    top_y: int,
    left_x: int,
    right_x: int,
) -> tuple[float, float]:
    """Measure whether a bar outline continues beyond its supposed corner."""
    x1, y1, x2, _y2 = bbox
    width, height = max(1, x2 - x1), max(1, bbox[3] - y1)
    x_extension = max(3, round(width * 0.55))
    y_extension = max(3, round(height * 0.55))

    top_support = max(
        band_support(mask, True, top_y, max(0, x1 - x_extension), x1),
        band_support(mask, True, top_y, x2, min(mask.shape[1], x2 + x_extension)),
    )
    side_support = max(
        band_support(mask, False, left_x, max(0, y1 - y_extension), y1),
        band_support(mask, False, right_x, max(0, y1 - y_extension), y1),
    )
    return top_support, side_support


def outline_evidence(dark_mask: np.ndarray, bbox: list[int]) -> dict[str, Any]:
    """Check whether a candidate rectangle has supported bar outlines."""
    x1, y1, x2, y2 = map(int, bbox)
    search = max(2, round(min(x2 - x1, y2 - y1) * HISTOGRAM_EDGE_SEARCH_FRACTION))
    top, top_y = best_edge(dark_mask, True, y1, x1, x2, search)
    bottom, bottom_y = best_edge(dark_mask, True, y2 - 1, x1, x2, search)
    left, left_x = best_edge(dark_mask, False, x1, y1, y2, search)
    right, right_x = best_edge(dark_mask, False, x2 - 1, y1, y2, search)
    refined = [
        min(left_x, right_x),
        min(top_y, bottom_y),
        max(left_x, right_x) + 1,
        max(top_y, bottom_y) + 1,
    ]
    top_extension, side_extension = edge_extension(
        dark_mask, refined, top_y, left_x, right_x
    )
    complete = (
        left >= HISTOGRAM_OUTLINE_MINIMUM_SIDE_SUPPORT
        and right >= HISTOGRAM_OUTLINE_MINIMUM_SIDE_SUPPORT
        and top >= HISTOGRAM_OUTLINE_MINIMUM_TOP_SUPPORT
        and bottom >= HISTOGRAM_OUTLINE_MINIMUM_BOTTOM_SUPPORT
        and top_extension <= HISTOGRAM_OUTLINE_MAXIMUM_EXTENSION_SUPPORT
        and side_extension <= HISTOGRAM_OUTLINE_MAXIMUM_EXTENSION_SUPPORT
    )
    return {
        "complete": complete,
        "bbox": refined,
        "left_edge_support": left,
        "right_edge_support": right,
        "top_edge_support": top,
        "bottom_edge_support": bottom,
        "top_edge_extension_support": top_extension,
        "side_edge_extension_support": side_extension,
    }


def merge_stacked_histogram_candidates(
    candidates: list[dict[str, Any]],
    plot_height: int,
) -> list[dict[str, Any]]:
    """Merge connected or vertically stacked segments belonging to one bar."""
    if len(candidates) < 2:
        return candidates

    maximum_gap = max(
        3,
        round(
            plot_height
            * HISTOGRAM_STACK_MAX_VERTICAL_GAP_FRACTION
        ),
    )
    remaining = sorted(
        candidates,
        key=lambda item: (
            item["bbox"][0],
            item["bbox"][1],
        ),
    )
    merged: list[dict[str, Any]] = []

    while remaining:
        group = [remaining.pop(0)]
        changed = True

        while changed:
            changed = False
            group_x1 = min(
                item["bbox"][0]
                for item in group
            )
            group_x2 = max(
                item["bbox"][2]
                for item in group
            )
            group_y1 = min(
                item["bbox"][1]
                for item in group
            )
            group_y2 = max(
                item["bbox"][3]
                for item in group
            )
            group_width = max(
                1,
                group_x2 - group_x1,
            )

            for candidate in list(remaining):
                x1, y1, x2, y2 = candidate["bbox"]
                overlap = max(
                    0,
                    min(group_x2, x2)
                    - max(group_x1, x1),
                )
                x_overlap = (
                    overlap
                    / max(
                        1,
                        min(
                            group_width,
                            x2 - x1,
                        ),
                    )
                )
                vertical_gap = max(
                    0,
                    y1 - group_y2,
                    group_y1 - y2,
                )

                if (
                    x_overlap
                    >= HISTOGRAM_STACK_X_OVERLAP_FRACTION
                    and vertical_gap <= maximum_gap
                ):
                    group.append(candidate)
                    remaining.remove(candidate)
                    changed = True

        if len(group) == 1:
            merged.append(group[0])
            continue

        x1 = min(item["bbox"][0] for item in group)
        y1 = min(item["bbox"][1] for item in group)
        x2 = max(item["bbox"][2] for item in group)
        y2 = max(item["bbox"][3] for item in group)
        best = max(
            group,
            key=lambda item: float(
                item.get("rectangularity", 0.0)
            ),
        )
        combined = dict(best)
        combined["bbox"] = [x1, y1, x2, y2]
        combined["width_pixels"] = int(x2 - x1)
        combined["height_pixels"] = float(y2 - y1)
        combined["density"] = float(
            max(
                item.get("density", 0.0)
                for item in group
            )
        )
        combined["rectangularity"] = float(
            max(
                item.get("rectangularity", 0.0)
                for item in group
            )
        )
        combined["filled_like"] = any(
            item.get("filled_like", False)
            for item in group
        )
        combined["outline_like"] = any(
            item.get("outline_like", False)
            for item in group
        )
        combined["complete_outline"] = any(
            item.get("complete_outline", False)
            for item in group
        )
        merged.append(combined)

    return merged


def rectangle_histogram_columns(
    bar_mask: np.ndarray,
    dark_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Detect filled/outlined bars and estimate their shared baseline."""
    height, width = bar_mask.shape
    min_w = max(3, round(width * HISTOGRAM_MINIMUM_BAR_WIDTH_FRACTION))
    max_w = max(min_w, round(width * HISTOGRAM_MAXIMUM_BAR_WIDTH_FRACTION))
    min_h = max(5, round(height * HISTOGRAM_MINIMUM_BAR_HEIGHT_FRACTION))
    max_h = max(min_h, round(height * HISTOGRAM_MAXIMUM_BAR_HEIGHT_FRACTION))

    contours, _ = cv2.findContours(bar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        region = bar_mask[y:y + h, x:x + w]
        density = float(np.count_nonzero(region)) / max(1, region.size)
        outline = outline_evidence(dark_mask, [x, y, x + w, y + h])
        filled = density >= HISTOGRAM_FILLED_MINIMUM_DENSITY
        if not filled and not outline["complete"]:
            continue

        bbox = list(outline["bbox"]) if outline["complete"] else [x, y, x + w, y + h]
        w2, h2 = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if not (min_w <= w2 <= max_w and min_h <= h2 <= max_h):
            continue
        outline_support = np.mean([
            outline["left_edge_support"], outline["right_edge_support"],
            outline["top_edge_support"], outline["bottom_edge_support"],
        ]) if outline["complete"] else 0.0
        candidates.append({
            "bbox": bbox,
            "width_pixels": int(w2),
            "height_pixels": float(h2),
            "density": density,
            "left_edge_support": float(outline["left_edge_support"]),
            "right_edge_support": float(outline["right_edge_support"]),
            "top_edge_support": float(outline["top_edge_support"]),
            "bottom_edge_support": float(outline["bottom_edge_support"]),
            "top_edge_extension_support": float(outline["top_edge_extension_support"]),
            "side_edge_extension_support": float(outline["side_edge_extension_support"]),
            "filled_like": bool(filled),
            "outline_like": bool(outline["complete"]),
            "complete_outline": bool(outline["complete"]),
            "rectangularity": float(max(min(1.0, density / HISTOGRAM_FILLED_MINIMUM_DENSITY), outline_support)),
        })

    candidates = merge_stacked_histogram_candidates(
        candidates,
        height,
    )

    if not candidates:
        return [], {
            "raw_rectangle_candidate_count": 0.0,
            "baseline_candidate_count": 0.0,
            "retained_column_count": 0.0,
            "detected_bar_count": 0.0,
            "inferred_empty_bin_count": 0.0,
            "baseline_y_pixels": 0.0,
            "baseline_spread_pixels": 0.0,
            "baseline_coverage": 0.0,
            "tick_count": 0.0,
            "tick_pitch_pixels": 0.0,
            "detection_method_tick_grid": 0.0,
        }

    tolerance = max(3, round(height * HISTOGRAM_BASELINE_TOLERANCE_FRACTION))
    best_members = max(
        (
            [other for other in candidates if abs(other["bbox"][3] - candidate["bbox"][3]) <= tolerance]
            for candidate in candidates
        ),
        key=lambda members: sum(0.5 + member["rectangularity"] for member in members),
    )
    baseline = float(np.median([candidate["bbox"][3] for candidate in best_members]))

    if len(best_members) > 2:
        median_width = float(np.median([candidate["width_pixels"] for candidate in best_members]))
        best_members = [
            candidate for candidate in best_members
            if HISTOGRAM_MINIMUM_WIDTH_RATIO_TO_MEDIAN
            <= candidate["width_pixels"] / max(1.0, median_width)
            <= HISTOGRAM_MAXIMUM_WIDTH_RATIO_TO_MEDIAN
        ]

    best_members.sort(key=lambda candidate: candidate["bbox"][0])
    columns = []
    for index, candidate in enumerate(best_members, 1):
        x1, y1, x2, _ = candidate["bbox"]
        bottom = int(round(min(height, max(y1 + 1, baseline))))
        bar_height = float(max(1, bottom - y1))
        columns.append({
            **candidate,
            "index": index,
            "bbox": [x1, y1, x2, bottom],
            "weight": bar_height,
            "height_pixels": bar_height,
            "tick_grid_recovered": False,
            "inferred_empty_bin": False,
        })

    spread = float(np.std([candidate["bbox"][3] for candidate in best_members])) if best_members else 0.0
    return columns, {
        "raw_rectangle_candidate_count": float(len(candidates)),
        "baseline_candidate_count": float(len(best_members)),
        "retained_column_count": float(len(columns)),
        "detected_bar_count": float(len(columns)),
        "inferred_empty_bin_count": 0.0,
        "baseline_y_pixels": baseline,
        "baseline_spread_pixels": spread,
        "baseline_coverage": 0.0,
        "tick_count": 0.0,
        "tick_pitch_pixels": 0.0,
        "detection_method_tick_grid": 0.0,
    }


def contiguous_groups(indices: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted integer indices into inclusive consecutive ranges."""
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, indices.size - 1]
    return [(int(indices[a]), int(indices[b])) for a, b in zip(starts, ends)]


def histogram_interval_height(
    bar_mask: np.ndarray,
    x1: int,
    x2: int,
    baseline_y: int,
) -> tuple[float, int | None, float]:
    """Measure a bin’s height and row support above the shared baseline."""
    x1 = max(0, int(x1))
    x2 = min(bar_mask.shape[1], int(x2))
    baseline_y = max(1, min(int(baseline_y), bar_mask.shape[0] - 1))
    if x2 <= x1:
        return (0.0, None, 0.0)

    region = bar_mask[:baseline_y, x1:x2]
    if region.size == 0:
        return (0.0, None, 0.0)

    row_support = np.mean(region > 0, axis=1)
    supported_rows = np.flatnonzero(
        row_support >= HISTOGRAM_BIN_TOP_MINIMUM_SUPPORT
    )
    if supported_rows.size == 0:
        return (0.0, None, 0.0)

    top = int(supported_rows[0])
    height = float(max(0, baseline_y - top))
    strongest_support = float(np.max(row_support[supported_rows]))
    return (height, top, strongest_support)


def tick_histogram_columns(
    dark_mask: np.ndarray,
    bar_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]] | None:
    """Infer regular bins from axis ticks, retaining even empty and tiny bins."""
    height, width = dark_mask.shape
    horizontal_rules = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    MINIMUM_GRID_KERNEL_PIXELS,
                    round(width * HISTOGRAM_GRID_HORIZONTAL_LENGTH_FRACTION),
                ),
                1,
            ),
        ),
    )
    local_dark = cv2.bitwise_and(
        dark_mask,
        cv2.bitwise_not(horizontal_rules),
    )
    evidence_mask = (
        cv2.bitwise_or(bar_mask, local_dark)
        if bar_mask is not None
        else local_dark
    )
    row_counts = np.count_nonzero(dark_mask, axis=1)
    start_y = round(height * 0.35)
    if start_y >= height:
        return None

    baseline_y = int(start_y + np.argmax(row_counts[start_y:]))
    baseline_coverage = row_counts[baseline_y] / max(1, width)
    if baseline_coverage < HISTOGRAM_TICK_BASELINE_MINIMUM_COVERAGE:
        return None

    broad = max(
        round(width * HISTOGRAM_TICK_BASELINE_MINIMUM_COVERAGE),
        round(row_counts[baseline_y] * 0.72),
    )
    while baseline_y + 1 < height and row_counts[baseline_y + 1] >= broad:
        baseline_y += 1

    below = dark_mask[
        min(height, baseline_y + 1):
        min(height, baseline_y + 1 + HISTOGRAM_TICK_BELOW_BASELINE_SEARCH_PIXELS),
        :,
    ]
    if below.shape[0] < HISTOGRAM_TICK_MINIMUM_BELOW_BASELINE_ROWS:
        return None

    tick_columns = np.flatnonzero(
        np.count_nonzero(below > 0, axis=0)
        >= HISTOGRAM_TICK_MINIMUM_BELOW_BASELINE_ROWS
    )
    centers = [(a + b) / 2.0 for a, b in contiguous_groups(tick_columns)]
    if len(centers) < HISTOGRAM_TICK_MINIMUM_COUNT:
        return None

    differences = np.diff(np.asarray(centers, dtype=float))
    positive = differences[differences > 2]
    if positive.size < HISTOGRAM_TICK_MINIMUM_COUNT - 1:
        return None

    pitch = float(np.median(positive))
    if pitch <= 2:
        return None

    regular = (
        np.abs(positive - pitch) / pitch
        <= HISTOGRAM_TICK_MAXIMUM_PITCH_DEVIATION_FRACTION
    )
    if float(np.mean(regular)) < 0.70:
        return None

    columns: list[dict[str, Any]] = []
    detected_count = 0

    for left, right in zip(centers[:-1], centers[1:]):
        interval_width = right - left
        if not (
            pitch * (1 - HISTOGRAM_TICK_MAXIMUM_PITCH_DEVIATION_FRACTION)
            <= interval_width
            <= pitch * (1 + HISTOGRAM_TICK_MAXIMUM_PITCH_DEVIATION_FRACTION)
        ):
            continue

        x1 = int(round(left))
        x2 = int(round(right))
        bar_height, top, top_support = histogram_interval_height(
            evidence_mask,
            x1,
            x2 + 1,
            baseline_y,
        )
        detected = (
            top is not None
            and bar_height >= HISTOGRAM_TICK_MINIMUM_BAR_HEIGHT_PIXELS
        )
        if detected:
            detected_count += 1
            top_y = int(top)
            region = evidence_mask[top_y:baseline_y + 1, x1:x2 + 1]
            density = float(np.count_nonzero(region)) / max(1, region.size)
        else:
            top_y = max(0, baseline_y - 1)
            bar_height = 0.0
            density = 0.0

        columns.append({
            "index": len(columns) + 1,
            "bbox": [x1, top_y, x2 + 1, baseline_y + 1],
            "width_pixels": max(1, x2 - x1),
            "height_pixels": float(bar_height),
            "density": density,
            "left_edge_support": 0.0,
            "right_edge_support": 0.0,
            "top_edge_support": float(top_support if detected else 0.0),
            "bottom_edge_support": float(baseline_coverage),
            "top_edge_extension_support": 0.0,
            "side_edge_extension_support": 0.0,
            "filled_like": bool(density >= HISTOGRAM_FILLED_MINIMUM_DENSITY),
            "outline_like": bool(detected),
            "complete_outline": False,
            "rectangularity": float(top_support if detected else 0.0),
            "weight": float(bar_height),
            "tick_grid_recovered": True,
            "profile_recovered": False,
            "inferred_empty_bin": not detected,
        })

    if not columns or detected_count == 0:
        return None

    return columns, {
        "raw_rectangle_candidate_count": 0.0,
        "baseline_candidate_count": float(detected_count),
        "retained_column_count": float(len(columns)),
        "detected_bar_count": float(detected_count),
        "inferred_empty_bin_count": float(len(columns) - detected_count),
        "baseline_y_pixels": float(baseline_y + 1),
        "baseline_spread_pixels": 0.0,
        "baseline_coverage": float(baseline_coverage),
        "tick_count": float(len(centers)),
        "tick_pitch_pixels": pitch,
        "detection_method_tick_grid": 1.0,
        "detection_method_profile": 0.0,
    }


def histogram_profile_columns(
    crop: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]] | None:
    """Infer regular adjacent bins from the image’s height profile.

    Retain empty slots between visible bins when their pitch can be inferred.
    """
    threshold = choose_dark_threshold(crop)
    dark_mask = cv2.inRange(crop, 0, threshold)
    nonwhite = crop[crop < 245]
    shade_threshold = threshold

    if nonwhite.size:
        shade_threshold = int(
            np.clip(
                max(
                    threshold,
                    float(
                        np.percentile(
                            nonwhite,
                            HISTOGRAM_SHADE_PERCENTILE,
                        )
                    )
                    + HISTOGRAM_SHADE_THRESHOLD_MARGIN,
                ),
                threshold,
                HISTOGRAM_SHADE_MAX_THRESHOLD,
            )
        )

    shade_mask = cv2.inRange(
        crop,
        0,
        shade_threshold,
    )
    height, width = shade_mask.shape
    baseline_start = min(
        height - 1,
        round(
            height
            * HISTOGRAM_PROFILE_BASELINE_START_FRACTION
        ),
    )
    row_counts = np.count_nonzero(
        shade_mask,
        axis=1,
    )
    lower_counts = row_counts[
        baseline_start:
    ]
    maximum_lower_support = int(
        np.max(
            lower_counts,
            initial=0,
        )
    )

    if maximum_lower_support > 0:
        baseline_candidates = np.flatnonzero(
            lower_counts
            >= maximum_lower_support * 0.90
        )
        baseline_y = int(
            baseline_start
            + baseline_candidates[-1]
        )
    else:
        baseline_y = height - 1

    baseline_coverage = (
        float(row_counts[baseline_y])
        / max(1, width)
    )

    horizontal_length = max(
        MINIMUM_GRID_KERNEL_PIXELS,
        round(
            width
            * HISTOGRAM_PROFILE_HORIZONTAL_RULE_FRACTION
        ),
    )
    horizontal_rules = cv2.morphologyEx(
        shade_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (horizontal_length, 1),
        ),
    )
    horizontal_rules = cv2.dilate(
        horizontal_rules,
        np.ones((3, 1), np.uint8),
    )
    usable = cv2.bitwise_and(
        shade_mask,
        cv2.bitwise_not(horizontal_rules),
    )
    usable[
        max(0, baseline_y - 2):,
        :,
    ] = 0

    heights = np.zeros(
        width,
        dtype=np.float64,
    )

    for x in range(width):
        rows = np.flatnonzero(
            usable[:baseline_y, x] > 0
        )

        if rows.size:
            heights[x] = (
                baseline_y
                - int(rows[0])
            )

    heights = median_smooth(
        heights,
        HISTOGRAM_PROFILE_MEDIAN_WINDOW,
    )

    minimum_height = max(
        4,
        round(
            height
            * HISTOGRAM_PROFILE_MINIMUM_HEIGHT_FRACTION
        ),
    )
    minimum_width = max(
        4,
        round(
            width
            * HISTOGRAM_PROFILE_MINIMUM_WIDTH_FRACTION
        ),
    )
    active_indices = np.flatnonzero(
        heights >= minimum_height
    )

    if active_indices.size == 0:
        return None

    runs = contiguous_groups(active_indices)
    change_threshold = max(
        3,
        round(
            height
            * HISTOGRAM_PROFILE_CHANGE_FRACTION
        ),
    )
    segments: list[
        tuple[int, int, float]
    ] = []

    for run_start, run_end in runs:
        run_end += 1

        if (
            run_end - run_start
            < minimum_width
        ):
            continue

        local = heights[
            run_start:run_end
        ]
        change_points = (
            np.flatnonzero(
                np.abs(
                    np.diff(local)
                )
                >= change_threshold
            )
            + 1
        )
        boundaries = [run_start]
        last_boundary = run_start

        for offset in change_points:
            boundary = (
                run_start
                + int(offset)
            )

            if (
                boundary - last_boundary
                >= minimum_width
            ):
                boundaries.append(boundary)
                last_boundary = boundary

        if (
            run_end - boundaries[-1]
            < minimum_width
            and len(boundaries) > 1
        ):
            boundaries.pop()

        boundaries.append(run_end)

        for left, right in zip(
            boundaries[:-1],
            boundaries[1:],
        ):
            if (
                right - left
                < minimum_width
            ):
                continue

            bar_height = float(
                np.median(
                    heights[left:right]
                )
            )

            if bar_height >= minimum_height:
                segments.append(
                    (
                        left,
                        right,
                        bar_height,
                    )
                )

    if len(segments) < HISTOGRAM_MINIMUM_COLUMNS:
        return None

    widths = sorted(
        right - left
        for left, right, _height
        in segments
    )
    lower_half = widths[
        : max(
            1,
            (len(widths) + 1) // 2,
        )
    ]
    width_pitch = max(
        float(np.median(lower_half)),
        float(minimum_width),
    )
    centers = np.asarray(
        [(left + right) / 2.0 for left, right, _height in segments],
        dtype=float,
    )
    center_gaps = np.diff(centers)
    positive_center_gaps = center_gaps[center_gaps > minimum_width]
    if positive_center_gaps.size:
        center_pitch = float(np.percentile(positive_center_gaps, 25.0))
        pitch = (
            max(width_pitch, center_pitch)
            if center_pitch <= width_pitch * 2.2
            else width_pitch
        )
    else:
        pitch = width_pitch
    split_segments: list[
        tuple[int, int, float]
    ] = []

    for left, right, bar_height in segments:
        width_pixels = right - left
        split_count = 1

        if (
            len(segments) >= 3
            and width_pixels
            >= pitch
            * HISTOGRAM_PROFILE_WIDE_SEGMENT_SPLIT_RATIO
        ):
            split_count = max(
                1,
                int(
                    round(
                        width_pixels
                        / pitch
                    )
                ),
            )

        boundaries = np.linspace(
            left,
            right,
            split_count + 1,
        )

        for index in range(split_count):
            part_left = int(
                round(boundaries[index])
            )
            part_right = int(
                round(
                    boundaries[index + 1]
                )
            )

            if part_right <= part_left:
                continue

            part_height = float(
                np.median(
                    heights[
                        part_left:part_right
                    ]
                )
            )
            split_segments.append(
                (
                    part_left,
                    part_right,
                    max(
                        float(minimum_height),
                        part_height,
                    ),
                )
            )

    split_segments.sort(
        key=lambda item: item[0]
    )
    split_centers = np.asarray(
        [(left + right) / 2.0 for left, right, _height in split_segments],
        dtype=float,
    )
    split_gaps = np.diff(split_centers)
    positive_split_gaps = split_gaps[split_gaps > minimum_width]

    if positive_split_gaps.size:
        low_gap = float(np.percentile(positive_split_gaps, 25.0))
        if low_gap <= pitch * 1.55:
            pitch = max(pitch, low_gap)

    first_center = float(split_centers[0])
    leading_float = (first_center - pitch / 2.0) / max(1.0, pitch)
    leading_count = 0
    if leading_float >= 0.5:
        rounded = max(0, int(round(leading_float)))
        residual = abs(leading_float - rounded)
        if residual <= HISTOGRAM_EDGE_EMPTY_INFERENCE_TOLERANCE:
            leading_count = min(
                rounded,
                HISTOGRAM_PROFILE_MAXIMUM_INFERRED_EMPTY_BINS,
            )

    slot_to_segment: dict[int, tuple[int, int, float]] = {}
    for segment, center in zip(split_segments, split_centers):
        slot = leading_count + int(round((float(center) - first_center) / max(1.0, pitch)))
        previous = slot_to_segment.get(slot)
        if previous is None or segment[2] > previous[2]:
            slot_to_segment[slot] = segment

    last_detected_slot = max(slot_to_segment)
    last_center = first_center + (last_detected_slot - leading_count) * pitch
    trailing_float = (
        width - (last_center + pitch / 2.0)
    ) / max(1.0, pitch)
    trailing_count = 0
    if trailing_float >= 0.5:
        rounded = max(0, int(round(trailing_float)))
        residual = abs(trailing_float - rounded)
        if residual <= HISTOGRAM_EDGE_EMPTY_INFERENCE_TOLERANCE:
            trailing_count = min(
                rounded,
                HISTOGRAM_PROFILE_MAXIMUM_INFERRED_EMPTY_BINS,
            )

    total_slots = last_detected_slot + 1 + trailing_count
    total_slots = min(
        total_slots,
        last_detected_slot + 1 + HISTOGRAM_PROFILE_MAXIMUM_INFERRED_EMPTY_BINS,
    )
    columns: list[dict[str, Any]] = []
    inferred_empty_count = 0
    origin_center = first_center - leading_count * pitch

    # Keep empty bins explicitly so the sequence of numeric slots stays regular.
    def append_column(
        slot_index: int,
        bar_height: float,
        inferred_empty: bool,
    ) -> None:
        """Append one regular profile-grid bin, including explicit empty bins."""
        nonlocal inferred_empty_count
        center = origin_center + slot_index * pitch
        left = int(round(center - pitch / 2.0))
        right = int(round(center + pitch / 2.0))
        left = max(0, min(width - 1, left))
        right = max(left + 1, min(width, right))
        bar_height = max(0.0, float(bar_height))
        top = int(round(baseline_y - bar_height))
        top = max(0, min(baseline_y, top))
        region = shade_mask[top:baseline_y + 1, left:right]
        density = (
            float(np.count_nonzero(region)) / max(1, region.size)
            if not inferred_empty
            else 0.0
        )
        local_profile = heights[left:right]
        profile_spread = (
            float(np.std(local_profile))
            if local_profile.size and not inferred_empty
            else 0.0
        )
        rectangularity = (
            0.0
            if inferred_empty
            else max(
                0.0,
                min(
                    1.0,
                    1.0 - profile_spread / max(1.0, bar_height * 0.35),
                ),
            )
        )
        if inferred_empty:
            inferred_empty_count += 1

        columns.append({
            "index": slot_index + 1,
            "bbox": [left, top, right, baseline_y + 1],
            "width_pixels": int(right - left),
            "height_pixels": 0.0 if inferred_empty else bar_height,
            "density": density,
            "left_edge_support": 0.0,
            "right_edge_support": 0.0,
            "top_edge_support": 0.0,
            "bottom_edge_support": float(baseline_coverage),
            "top_edge_extension_support": 0.0,
            "side_edge_extension_support": 0.0,
            "filled_like": bool(
                (not inferred_empty)
                and density >= HISTOGRAM_FILLED_MINIMUM_DENSITY
            ),
            "outline_like": False,
            "complete_outline": False,
            "rectangularity": rectangularity,
            "weight": 0.0 if inferred_empty else bar_height,
            "tick_grid_recovered": False,
            "profile_recovered": True,
            "inferred_empty_bin": inferred_empty,
        })

    for slot_index in range(total_slots):
        segment = slot_to_segment.get(slot_index)
        if segment is None:
            append_column(slot_index, 0.0, True)
        else:
            append_column(slot_index, float(segment[2]), False)

    detected_count = sum(
        not column["inferred_empty_bin"]
        for column in columns
    )

    if detected_count < HISTOGRAM_MINIMUM_COLUMNS:
        return None

    return (
        columns,
        {
            "raw_rectangle_candidate_count": 0.0,
            "baseline_candidate_count": float(
                detected_count
            ),
            "retained_column_count": float(
                len(columns)
            ),
            "detected_bar_count": float(
                detected_count
            ),
            "inferred_empty_bin_count": float(
                inferred_empty_count
            ),
            "baseline_y_pixels": float(
                baseline_y + 1
            ),
            "baseline_spread_pixels": 0.0,
            "baseline_coverage": float(
                baseline_coverage
            ),
            "tick_count": 0.0,
            "tick_pitch_pixels": float(pitch),
            "detection_method_tick_grid": 0.0,
            "detection_method_profile": 1.0,
        },
    )





def regular_position_pitch(
    positions: np.ndarray,
    minimum_pitch: float,
) -> float | None:
    """Find the dominant interval between adjacent sorted positions."""
    if positions.size < 3:
        return None

    gaps = np.diff(np.sort(positions.astype(float)))
    gaps = gaps[gaps >= minimum_pitch]
    if gaps.size < 2:
        return None

    best_members: np.ndarray | None = None
    for candidate in gaps:
        members = np.flatnonzero(
            np.abs(gaps - candidate) / max(1.0, float(candidate)) <= 0.20
        )
        if best_members is None or members.size > best_members.size:
            best_members = members

    if best_members is None or best_members.size < 2:
        return None

    pitch = float(np.median(gaps[best_members]))
    residual = np.abs(gaps / pitch - np.rint(gaps / pitch))
    if float(np.mean(residual <= 0.20)) < 0.60:
        return None
    return pitch


def histogram_axis_pitch_hint(
    image: np.ndarray,
    plot_bbox: list[int],
) -> float | None:
    """Estimate bin pitch from the spacing of axis-label components."""
    x1, _y1, x2, y2 = plot_bbox
    plot_width = x2 - x1
    search_height = int(
        np.clip(
            round(image.shape[0] * HISTOGRAM_AXIS_SEARCH_HEIGHT_FRACTION),
            HISTOGRAM_AXIS_MINIMUM_SEARCH_PIXELS,
            HISTOGRAM_AXIS_MAXIMUM_SEARCH_PIXELS,
        )
    )
    bottom = min(image.shape[0], y2 + search_height)
    if bottom <= y2 + 4 or plot_width < 40:
        return None

    band = image[y2:bottom, x1:x2]
    threshold = choose_dark_threshold(band)
    dark = cv2.inRange(band, 0, threshold)

    # Ignore the immediate baseline and merge digits belonging to one tick label.
    dark[: min(4, dark.shape[0]), :] = 0
    dilation = max(
        7,
        round(plot_width * HISTOGRAM_AXIS_TEXT_DILATION_FRACTION),
    )
    merged = cv2.dilate(
        dark,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, 2)),
    )

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (merged > 0).astype(np.uint8),
        connectivity=8,
    )
    centers: list[float] = []
    for component in range(1, count):
        left = int(stats[component, cv2.CC_STAT_LEFT])
        top = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])

        if height < 4 or height > search_height * 0.70:
            continue
        if width < 2 or width > plot_width * 0.16:
            continue
        if top > search_height * 0.65:
            continue
        centers.append(float(centroids[component][0]))

    if len(centers) < 3:
        return None

    return regular_position_pitch(
        np.asarray(sorted(centers), dtype=float),
        minimum_pitch=max(8.0, plot_width * 0.04),
    )


def histogram_horizontal_rule_mask(mask: np.ndarray) -> np.ndarray:
    """Identify continuous/broken grid rules without erasing bar tops."""
    height, width = mask.shape
    continuous = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(12, round(width * 0.50)), 1),
        ),
    )

    gap = max(
        5,
        round(width * HISTOGRAM_BROKEN_GRID_GAP_FRACTION),
    )
    gap |= 1
    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)),
    )
    broken = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    12,
                    round(
                        width
                        * HISTOGRAM_BROKEN_GRID_MINIMUM_LENGTH_FRACTION
                    ),
                ),
                1,
            ),
        ),
    )
    return cv2.bitwise_or(continuous, broken)


def measure_histogram_bar(
    usable: np.ndarray,
    shaded: np.ndarray,
    baseline: int,
    left: int,
    right: int,
    reject_full_width_one_pixel: bool = True,
) -> tuple[int, float, float, float] | None:
    """Measure one baseline-connected bar while ignoring gridline artifacts."""
    left = max(0, int(left))
    right = min(usable.shape[1], int(right))
    if right <= left:
        return None

    region = usable[:baseline, left:right]
    if region.size == 0 or not np.any(region):
        return None

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (region > 0).astype(np.uint8),
        connectivity=8,
    )
    region_width = right - left
    minimum_component_width = max(
        2,
        round(
            region_width
            * HISTOGRAM_BAR_COMPONENT_MINIMUM_WIDTH_FRACTION
        ),
    )
    bottom_target = region.shape[0] - 1
    candidates: list[tuple[int, int, int]] = []

    for component in range(1, component_count):
        component_left = int(stats[component, cv2.CC_STAT_LEFT])
        component_top = int(stats[component, cv2.CC_STAT_TOP])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        component_area = int(stats[component, cv2.CC_STAT_AREA])
        component_bottom = component_top + component_height - 1

        if component_width < minimum_component_width:
            continue
        if (
            bottom_target - component_bottom
            > HISTOGRAM_BAR_COMPONENT_BASELINE_TOLERANCE_PIXELS
        ):
            continue

        candidates.append(
            (component_width, component_area, component)
        )

    if not candidates:
        return None

    _component_width, _component_area, component = max(candidates)
    selected = labels == component

    top_positions: list[int] = []
    for column in range(selected.shape[1]):
        rows = np.flatnonzero(selected[:, column])
        if rows.size:
            top_positions.append(int(rows[0]))

    if len(top_positions) < minimum_component_width:
        return None

    # A narrow vertical gridline can merge into the bar and extend far upward.
    # The median across the bar width ignores those few extreme columns.
    top = int(round(float(np.median(top_positions))))
    bar_height = float(max(1, baseline - top))

    selected_width = max(1, int(stats[component, cv2.CC_STAT_WIDTH]))
    if (
        reject_full_width_one_pixel
        and bar_height <= 1.0
        and selected_width >= region_width * 0.90
    ):
        # A one-pixel full-slot strip is indistinguishable from the top row of
        # a thick x-axis. Treat it as baseline, not a data bar.
        return None
    selected_area = max(1, int(stats[component, cv2.CC_STAT_AREA]))
    density = selected_area / max(
        1.0,
        float(selected_width * max(1, baseline - top)),
    )

    top_band = selected[
        max(0, top - 1):
        min(selected.shape[0], top + 2),
        :,
    ]
    top_support = float(np.mean(np.any(top_band, axis=0))) if top_band.size else 0.0

    return top, bar_height, density, top_support


def histogram_regular_phase(
    centers: np.ndarray,
    pitch: float,
) -> float:
    """Estimate the alignment of repeating bar centers within one pitch."""
    if centers.size == 0 or pitch <= 0:
        return pitch / 2.0
    angles = (centers % pitch) / pitch * (2.0 * math.pi)
    vector = np.mean(np.exp(1j * angles))
    if abs(vector) < 1e-06:
        return float(np.median(centers % pitch))
    angle = float(np.angle(vector)) % (2.0 * math.pi)
    return angle / (2.0 * math.pi) * pitch


def histogram_columns_from_pitch(
    usable: np.ndarray,
    shaded: np.ndarray,
    baseline: int,
    baseline_coverage: float,
    pitch: float,
    phase: float,
    median_bar_width: float,
) -> tuple[list[dict[str, Any]], dict[str, float]] | None:
    """Measure a regular series of bar slots at the selected pitch."""
    width = shaded.shape[1]
    total_slots = int(round(width / max(1.0, pitch)))
    if not HISTOGRAM_MINIMUM_COLUMNS <= total_slots <= HISTOGRAM_REGULAR_MAXIMUM_BINS:
        return None

    first_center = float(phase)
    while first_center < -pitch / 2.0:
        first_center += pitch
    while first_center > pitch:
        first_center -= pitch

    columns: list[dict[str, Any]] = []
    detected_count = 0
    inferred_empty_count = 0

    for slot in range(total_slots):
        center = first_center + slot * pitch
        slot_left = max(0, int(round(center - pitch / 2.0)))
        slot_right = min(width, int(round(center + pitch / 2.0)))
        if slot_right <= slot_left:
            continue

        measurement = measure_histogram_bar(
            usable,
            shaded,
            baseline,
            slot_left,
            slot_right,
        )
        if measurement is None:
            top = max(0, baseline - 1)
            measured_height = 0.0
            density = 0.0
            top_support = 0.0
            inferred_empty_count += 1
        else:
            top, measured_height, density, top_support = measurement
            detected_count += 1

        columns.append({
            "index": slot + 1,
            "bbox": [slot_left, top, slot_right, baseline + 1],
            "width_pixels": int(max(1, slot_right - slot_left)),
            "height_pixels": measured_height,
            "density": density,
            "left_edge_support": 0.0,
            "right_edge_support": 0.0,
            "top_edge_support": top_support,
            "bottom_edge_support": baseline_coverage,
            "top_edge_extension_support": 0.0,
            "side_edge_extension_support": 0.0,
            "filled_like": bool(density >= HISTOGRAM_FILLED_MINIMUM_DENSITY),
            "outline_like": bool(measured_height > 0),
            "complete_outline": False,
            "rectangularity": top_support,
            "weight": measured_height,
            "tick_grid_recovered": False,
            "profile_recovered": False,
            "spaced_bar_recovered": True,
            "inferred_empty_bin": bool(measured_height <= 0),
        })

    if detected_count == 0 or len(columns) != total_slots:
        return None

    return columns, {
        "raw_rectangle_candidate_count": 0.0,
        "baseline_candidate_count": float(detected_count),
        "retained_column_count": float(len(columns)),
        "detected_bar_count": float(detected_count),
        "inferred_empty_bin_count": float(inferred_empty_count),
        "baseline_y_pixels": float(baseline + 1),
        "baseline_spread_pixels": 0.0,
        "baseline_coverage": baseline_coverage,
        "tick_count": 0.0,
        "tick_pitch_pixels": pitch,
        "bar_width_pixels": median_bar_width,
        "bar_gap_pixels": max(0.0, pitch - median_bar_width),
        "detection_method_tick_grid": 0.0,
        "detection_method_profile": 0.0,
        "detection_method_spaced_bars": 1.0,
    }


def spaced_histogram_columns(
    shaded: np.ndarray,
    baseline: int,
    baseline_coverage: float,
    pitch_hint: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]] | None:
    """Recover equally spaced bars, allowing gaps and imperfect margins."""
    height, width = shaded.shape
    horizontal_grid = histogram_horizontal_rule_mask(shaded)
    usable = cv2.bitwise_and(
        shaded,
        cv2.bitwise_not(horizontal_grid),
    )

    # Require a horizontal run. This keeps full bar tops/fills while rejecting
    # narrow vertical gridlines and diagonal hatch strokes as independent bars.
    minimum_run = max(
        HISTOGRAM_SPACED_BAR_MINIMUM_WIDTH_PIXELS,
        round(width * HISTOGRAM_BAR_HORIZONTAL_RUN_FRACTION),
    )
    horizontal_bar_marks = cv2.morphologyEx(
        usable[:baseline + 1, :],
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (minimum_run, 1)),
    )

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (horizontal_bar_marks > 0).astype(np.uint8),
        connectivity=8,
    )
    raw_intervals: list[tuple[int, int]] = []
    for component in range(1, count):
        left = int(stats[component, cv2.CC_STAT_LEFT])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        if component_width < minimum_run:
            continue
        if component_width > width * HISTOGRAM_MAXIMUM_BAR_WIDTH_FRACTION:
            continue
        raw_intervals.append((left, left + component_width))

    if not raw_intervals:
        return None

    # Outline bars can contribute separate top/bottom horizontal components.
    # Collapse highly overlapping intervals before estimating bar width.
    raw_intervals.sort()
    bars: list[tuple[int, int]] = []
    for interval in raw_intervals:
        left, right = interval
        matched_index: int | None = None
        for index, previous in enumerate(bars):
            overlap = max(
                0,
                min(right, previous[1]) - max(left, previous[0]),
            )
            minimum_width = max(
                1,
                min(right - left, previous[1] - previous[0]),
            )
            if overlap / minimum_width >= HISTOGRAM_BAR_INTERVAL_OVERLAP_FRACTION:
                matched_index = index
                break
        if matched_index is None:
            bars.append(interval)
        else:
            previous = bars[matched_index]
            bars[matched_index] = (
                min(previous[0], left),
                max(previous[1], right),
            )

    if not bars:
        return None

    widths = np.asarray(
        [right - left for left, right in bars],
        dtype=float,
    )

    # Choose the strongest equal-width cluster rather than allowing hatch
    # fragments or one unusual shaded component to determine the bin width.
    best_members: np.ndarray | None = None
    best_width_support = -1.0
    for candidate_width in widths:
        if candidate_width <= 0:
            continue
        deviation = np.abs(widths - candidate_width) / candidate_width
        members = np.flatnonzero(
            deviation <= HISTOGRAM_SPACED_BAR_WIDTH_TOLERANCE_FRACTION
        )
        width_support = (
            float(members.size)
            * float(np.median(widths[members]))
            if members.size
            else 0.0
        )
        if width_support > best_width_support:
            best_members = members
            best_width_support = width_support

    if best_members is None or best_members.size == 0:
        return None

    median_width = float(np.median(widths[best_members]))
    bars = [
        bar
        for bar, bar_width in zip(bars, widths)
        if abs(float(bar_width) - median_width) / max(1.0, median_width)
        <= HISTOGRAM_SPACED_BAR_WIDTH_TOLERANCE_FRACTION
    ]
    bars.sort()
    if not bars:
        return None

    centers = np.asarray(
        [(left + right) / 2.0 for left, right in bars],
        dtype=float,
    )

    valid_pitch_hint = (
        float(pitch_hint)
        if pitch_hint is not None
        and math.isfinite(float(pitch_hint))
        and float(pitch_hint) >= median_width * 0.90
        else None
    )

    if valid_pitch_hint is not None:
        phase = histogram_regular_phase(centers, valid_pitch_hint)
        hinted = histogram_columns_from_pitch(
            usable,
            shaded,
            baseline,
            baseline_coverage,
            valid_pitch_hint,
            phase,
            median_width,
        )
        if hinted is not None:
            columns, diagnostics = hinted
            diagnostics["raw_rectangle_candidate_count"] = float(len(raw_intervals))
            diagnostics["axis_pitch_hint_pixels"] = valid_pitch_hint
            return columns, diagnostics

    pitch = (
        regular_position_pitch(
            centers,
            minimum_pitch=max(8.0, median_width * 0.90),
        )
        if centers.size >= 3
        else None
    )
    if pitch is None and centers.size >= 2:
        gaps = np.diff(centers)
        positive_gaps = gaps[gaps >= median_width * 0.90]
        if positive_gaps.size:
            pitch = float(np.min(positive_gaps))

    if pitch is None:
        return None

    inferred_phase = histogram_regular_phase(centers, pitch)
    inferred = histogram_columns_from_pitch(
        usable,
        shaded,
        baseline,
        baseline_coverage,
        pitch,
        inferred_phase,
        median_width,
    )
    if inferred is not None:
        columns, diagnostics = inferred
        diagnostics["raw_rectangle_candidate_count"] = float(len(raw_intervals))
        return columns, diagnostics

    # The selected left/right edges may land anywhere within the outer white
    # gaps. Rounding edge distance to the regular pitch tolerates that error.
    first_center = float(centers[0])
    leading_count = max(
        0,
        int(round((first_center - pitch / 2.0) / pitch)),
    )
    origin_center = first_center - leading_count * pitch

    slot_to_bar: dict[int, tuple[int, int]] = {}
    for bar, center in zip(bars, centers):
        slot = int(
            round(
                (float(center) - origin_center)
                / max(1.0, pitch)
            )
        )
        expected_center = origin_center + slot * pitch
        if (
            abs(float(center) - expected_center)
            > pitch * HISTOGRAM_SPACED_BAR_PITCH_TOLERANCE_FRACTION
        ):
            continue
        previous = slot_to_bar.get(slot)
        if (
            previous is None
            or (bar[1] - bar[0]) > (previous[1] - previous[0])
        ):
            slot_to_bar[slot] = bar

    if not slot_to_bar:
        return None

    last_slot = max(slot_to_bar)
    last_center = origin_center + last_slot * pitch
    trailing_count = max(
        0,
        int(
            round(
                (
                    width
                    - (last_center + pitch / 2.0)
                )
                / max(1.0, pitch)
            )
        ),
    )
    total_slots = last_slot + trailing_count + 1

    if not (
        HISTOGRAM_MINIMUM_COLUMNS
        <= total_slots
        <= HISTOGRAM_REGULAR_MAXIMUM_BINS
    ):
        return None

    columns: list[dict[str, Any]] = []
    detected_count = 0
    inferred_empty_count = 0

    for slot in range(total_slots):
        slot_center = origin_center + slot * pitch
        slot_left = max(
            0,
            int(round(slot_center - pitch / 2.0)),
        )
        slot_right = min(
            width,
            int(round(slot_center + pitch / 2.0)),
        )
        bar = slot_to_bar.get(slot)

        if bar is None:
            top = max(0, baseline - 1)
            measured_height = 0.0
            density = 0.0
            top_support = 0.0
            inferred_empty_count += 1
        else:
            bar_left = max(slot_left, bar[0])
            bar_right = min(slot_right, bar[1])
            measurement = measure_histogram_bar(
                usable,
                shaded,
                baseline,
                bar_left,
                bar_right,
                reject_full_width_one_pixel=False,
            )
            if measurement is not None:
                top, measured_height, density, top_support = measurement
                detected_count += 1
            else:
                top = max(0, baseline - 1)
                measured_height = 0.0
                density = 0.0
                top_support = 0.0
                inferred_empty_count += 1

        columns.append({
            "index": slot + 1,
            "bbox": [slot_left, top, slot_right, baseline + 1],
            "width_pixels": int(max(1, slot_right - slot_left)),
            "height_pixels": measured_height,
            "density": density,
            "left_edge_support": 0.0,
            "right_edge_support": 0.0,
            "top_edge_support": top_support,
            "bottom_edge_support": baseline_coverage,
            "top_edge_extension_support": 0.0,
            "side_edge_extension_support": 0.0,
            "filled_like": bool(
                density >= HISTOGRAM_FILLED_MINIMUM_DENSITY
            ),
            "outline_like": bool(measured_height > 0),
            "complete_outline": False,
            "rectangularity": top_support,
            "weight": measured_height,
            "tick_grid_recovered": False,
            "profile_recovered": False,
            "spaced_bar_recovered": True,
            "inferred_empty_bin": bool(measured_height <= 0),
        })

    if detected_count == 0:
        return None

    return columns, {
        "raw_rectangle_candidate_count": float(len(raw_intervals)),
        "baseline_candidate_count": float(detected_count),
        "retained_column_count": float(len(columns)),
        "detected_bar_count": float(detected_count),
        "inferred_empty_bin_count": float(inferred_empty_count),
        "baseline_y_pixels": float(baseline + 1),
        "baseline_spread_pixels": 0.0,
        "baseline_coverage": baseline_coverage,
        "tick_count": 0.0,
        "tick_pitch_pixels": pitch,
        "bar_width_pixels": median_width,
        "bar_gap_pixels": max(0.0, pitch - median_width),
        "axis_pitch_hint_pixels": (
            float(pitch_hint)
            if pitch_hint is not None
            else 0.0
        ),
        "detection_method_tick_grid": 0.0,
        "detection_method_profile": 0.0,
        "detection_method_spaced_bars": 1.0,
    }



def regular_histogram_columns(
    crop: np.ndarray,
    axis_pitch_hint: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]] | None:
    """Infer a regular bin lattice and measure occupied/empty bin heights."""
    threshold = choose_dark_threshold(crop)
    dark = cv2.inRange(crop, 0, threshold)
    nonwhite = crop[crop < 245]
    shade_threshold = threshold
    if nonwhite.size:
        shade_threshold = int(
            np.clip(
                max(
                    threshold,
                    float(np.percentile(nonwhite, HISTOGRAM_SHADE_PERCENTILE))
                    + max(HISTOGRAM_SHADE_THRESHOLD_MARGIN, 30.0),
                ),
                threshold,
                HISTOGRAM_SHADE_MAX_THRESHOLD,
            )
        )
    shaded = cv2.inRange(crop, 0, shade_threshold)
    height, width = shaded.shape

    # Lowest strong horizontal rule in the lower half is the histogram baseline.
    row_counts = np.count_nonzero(shaded, axis=1)
    start = max(0, height // 2)
    lower = row_counts[start:]
    if lower.size == 0 or np.max(lower, initial=0) == 0:
        return None
    strong = np.flatnonzero(lower >= np.max(lower) * 0.80)
    strong_rows = start + strong
    groups = contiguous_groups(strong_rows)
    baseline_group = groups[-1]
    baseline = int(baseline_group[0])
    baseline_bottom = int(baseline_group[1])
    baseline_coverage = float(row_counts[baseline]) / max(1, width)
    if baseline_coverage < 0.20:
        return None

    # Prefer x-axis ticks because they explicitly preserve blank bins.
    below = dark[
        min(height, baseline_bottom + 1):
        min(height, baseline_bottom + 1 + HISTOGRAM_REGULAR_TICK_SEARCH_PIXELS),
        :,
    ]
    tick_centers: list[float] = []
    if below.shape[0] >= 2:
        tick_x = np.flatnonzero(np.count_nonzero(below, axis=0) >= 2)
        tick_centers = [
            (left + right) / 2.0
            for left, right in contiguous_groups(tick_x)
        ]
        if len(tick_centers) >= 3:
            gaps = np.diff(np.asarray(tick_centers, dtype=float))
            pitch = float(np.median(gaps))
            regular = gaps[
                np.abs(gaps - pitch) / max(1.0, pitch)
                <= HISTOGRAM_TICK_MAXIMUM_PITCH_DEVIATION_FRACTION
            ]
            if regular.size < max(2, len(gaps) * 0.65):
                tick_centers = []

    if tick_centers:
        boundaries = np.asarray(tick_centers, dtype=float)
    else:
        spaced_result = spaced_histogram_columns(
            shaded,
            baseline,
            baseline_coverage,
            pitch_hint=axis_pitch_hint,
        )
        if spaced_result is not None:
            return spaced_result

        # Remove long horizontal rules, then use vertical bar/grid edges to infer
        # the regular number of bins across the complete manual selection.
        horizontal = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(12, round(width * 0.45)), 1),
            ),
        )
        local = cv2.bitwise_and(dark, cv2.bitwise_not(horizontal))
        vertical = cv2.morphologyEx(
            local[:baseline + 1, :],
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, max(3, round(height * 0.04))),
            ),
        )
        support = np.count_nonzero(vertical, axis=0)
        edge_x = np.flatnonzero(support >= max(3, round(height * 0.04)))
        positions = np.asarray(
            [(left + right) / 2.0 for left, right in contiguous_groups(edge_x)],
            dtype=float,
        )
        positions = positions[(positions > 1) & (positions < width - 2)]
        if positions.size < 2:
            return None

        best: tuple[float, int] | None = None
        maximum_bins = min(
            HISTOGRAM_REGULAR_MAXIMUM_BINS,
            max(2, width // HISTOGRAM_REGULAR_MINIMUM_BIN_PIXELS),
        )
        for bin_count in range(2, maximum_bins + 1):
            pitch = width / float(bin_count)
            residual = np.abs(positions / pitch - np.rint(positions / pitch))
            score = float(np.median(residual)) + bin_count * 0.0008
            if best is None or score < best[0]:
                best = (score, bin_count)

        if best is None or best[0] > 0.22:
            return None
        boundaries = np.linspace(0.0, float(width), best[1] + 1)

    if boundaries.size < 3:
        return None

    # Long horizontal rules are gridlines. Measure bar tops only from the
    # remaining shade evidence inside each bin's interior.
    horizontal = histogram_horizontal_rule_mask(shaded)
    usable = cv2.bitwise_and(
        shaded,
        cv2.bitwise_not(horizontal),
    )
    columns: list[dict[str, Any]] = []
    detected_count = 0

    for index, (left_f, right_f) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        left = max(0, int(round(left_f)))
        right = min(width, int(round(right_f)))
        if right <= left:
            continue
        bin_width = right - left
        trim = min(
            max(1, round(bin_width * HISTOGRAM_REGULAR_INNER_FRACTION)),
            max(1, bin_width // 3),
        )
        inner_left = min(right - 1, left + trim)
        inner_right = max(inner_left + 1, right - trim)
        measurement = measure_histogram_bar(
            usable,
            shaded,
            baseline,
            left,
            right,
        )

        if measurement is not None:
            top, bar_height, density, top_support = measurement
            detected_count += 1
        else:
            top = max(0, baseline - 1)
            bar_height = 0.0
            density = 0.0
            top_support = 0.0

        columns.append({
            "index": index,
            "bbox": [left, top, right, baseline + 1],
            "width_pixels": int(bin_width),
            "height_pixels": bar_height,
            "density": density,
            "left_edge_support": 0.0,
            "right_edge_support": 0.0,
            "top_edge_support": top_support,
            "bottom_edge_support": baseline_coverage,
            "top_edge_extension_support": 0.0,
            "side_edge_extension_support": 0.0,
            "filled_like": bool(density >= HISTOGRAM_FILLED_MINIMUM_DENSITY),
            "outline_like": bool(bar_height > 0),
            "complete_outline": False,
            "rectangularity": top_support,
            "weight": bar_height,
            "tick_grid_recovered": bool(tick_centers),
            "profile_recovered": not bool(tick_centers),
            "inferred_empty_bin": bool(bar_height <= 0),
        })

    if len(columns) < HISTOGRAM_MINIMUM_COLUMNS or detected_count == 0:
        return None

    widths = np.asarray([column["width_pixels"] for column in columns], dtype=float)
    pitch = float(np.median(widths))
    return columns, {
        "raw_rectangle_candidate_count": 0.0,
        "baseline_candidate_count": float(detected_count),
        "retained_column_count": float(len(columns)),
        "detected_bar_count": float(detected_count),
        "inferred_empty_bin_count": float(len(columns) - detected_count),
        "baseline_y_pixels": float(baseline + 1),
        "baseline_spread_pixels": 0.0,
        "baseline_coverage": baseline_coverage,
        "tick_count": float(len(tick_centers)),
        "tick_pitch_pixels": pitch,
        "detection_method_tick_grid": float(bool(tick_centers)),
        "detection_method_profile": float(not bool(tick_centers)),
    }


def histogram_known_bin_count(minimum: float, maximum: float, bin_width: float) -> int:
    """Validate that range and bin width imply an exact supported bin count."""
    minimum = require_finite_number("Histogram minimum", minimum)
    maximum = require_finite_number("Histogram maximum", maximum)
    bin_width = require_finite_number("Histogram bin width", bin_width)
    if maximum <= minimum:
        raise ValueError("Histogram maximum must be greater than histogram minimum.")
    if bin_width <= 0:
        raise ValueError("Histogram bin width must be greater than zero.")
    # Calibration fixes the bin count; image evidence must not alter it.
    count_float = (maximum - minimum) / bin_width
    if not np.isfinite(count_float):
        raise ValueError("Histogram bounds and bin width yield an invalid bin count.")
    count = round(count_float)
    if not math.isclose(
        count_float, count,
        rel_tol=HISTOGRAM_KNOWN_BIN_COUNT_RELATIVE_TOLERANCE,
        abs_tol=HISTOGRAM_KNOWN_BIN_COUNT_RELATIVE_TOLERANCE,
    ):
        raise ValueError(
            f"Histogram range {maximum - minimum:g} is not an integer multiple "
            f"of bin width {bin_width:g}. Adjust the bounds or bin width."
        )
    if not HISTOGRAM_MINIMUM_COLUMNS <= count <= HISTOGRAM_REGULAR_MAXIMUM_BINS:
        raise ValueError(
            f"Histogram must have between {HISTOGRAM_MINIMUM_COLUMNS} and "
            f"{HISTOGRAM_REGULAR_MAXIMUM_BINS} bins; these settings imply {count}."
        )
    return count


def known_grid_histogram_columns(
    crop: np.ndarray, expected_bin_count: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Measure histogram heights on the user-calibrated fixed bin grid.

    Keep the bin count fixed, suppress grid rules and preserve empty bins.
    """
    height, width = crop.shape
    # A fixed slot grid still needs enough source pixels per bin to be measurable.
    if width / expected_bin_count < HISTOGRAM_REGULAR_MINIMUM_BIN_PIXELS:
        raise ValueError(
            f"The selected plot is too narrow for {expected_bin_count} bins "
            "(fewer than five pixels per bin). Zoom or select a higher-resolution page."
        )

    threshold = choose_dark_threshold(crop)
    nonwhite = crop[crop < 245]
    shade_threshold = threshold
    if nonwhite.size:
        shade_threshold = int(np.clip(
            max(threshold, float(np.percentile(nonwhite, HISTOGRAM_SHADE_PERCENTILE))
                + max(HISTOGRAM_SHADE_THRESHOLD_MARGIN, 30.0)),
            threshold, HISTOGRAM_SHADE_MAX_THRESHOLD,
        ))
    shaded = cv2.inRange(crop, 0, shade_threshold)
    dark = cv2.inRange(crop, 0, threshold)

    def remove_gridlines(mask: np.ndarray) -> np.ndarray:
        """Delete full-span rules but retain bar tops and extended filled rows.

        Morphological closing alone merges neighboring filled bars near their
        baseline, making entire *valid* bar sections look like gridlines. A
        genuine thin rule has little same-x vertical support three rows away;
        filled regions and bar side edges do have such support.
        """
        # Use *continuous* horizontal runs here: closing small gaps before
        # finding rules would join several adjacent, equal-height bar tops
        # into a fake gridline (and erase 1-2 px high bars altogether).
        horizontal = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, round(width * 0.55)), 1)),
        )
        support = np.count_nonzero(horizontal, axis=1)
        horizontal[support < round(width * 0.55), :] = 0
        keep = np.zeros_like(mask)
        step = min(3, max(1, height // 25))
        keep[:-step, :] |= mask[step:, :]
        keep[step:, :] |= mask[:-step, :]
        removal = cv2.bitwise_and(horizontal, cv2.bitwise_not(keep))
        return cv2.bitwise_and(mask, cv2.bitwise_not(removal))

    usable = remove_gridlines(shaded)
    hard_usable = remove_gridlines(dark)

    # The selection bottom is the common baseline. A prominent x-axis within
    # the lowest 12% may be a few pixels above the selection edge; identify it
    # only within this narrow tolerance, not via unrelated lower plot gridlines.
    search_start = max(height // 2, height - max(5, round(height * HISTOGRAM_KNOWN_GRID_BASELINE_SEARCH_FRACTION)))
    row_support = np.count_nonzero(shaded, axis=1) / max(1, width)
    baseline_candidates = np.flatnonzero(row_support[search_start:] >= 0.60)
    baseline = (search_start + int(baseline_candidates[-1])) if baseline_candidates.size else height - 1
    baseline = min(height - 1, baseline)
    baseline_coverage = float(row_support[baseline])

    boundaries = np.linspace(0.0, float(width), expected_bin_count + 1)
    columns: list[dict[str, Any]] = []
    for index in range(expected_bin_count):
        left = int(round(boundaries[index]))
        right = int(round(boundaries[index + 1]))
        bin_pixels = right - left
        trim = min(max(1, round(bin_pixels * HISTOGRAM_KNOWN_BIN_INNER_FRACTION)), max(1, bin_pixels // 3))
        inner_left = min(right - 1, left + trim)
        inner_right = max(inner_left + 1, right - trim)
        # Prefer interior evidence, rejecting adjacent-bin edges and vertical
        # grid lines. For outline-only bars, try the complete slot second.
        # Dark outlines carry more specific information than weak gridlines;
        # try them first, then accept lighter gray bars when no outline exists.
        measurement = None
        for evidence in (hard_usable, usable):
            inset = measure_histogram_bar(
                evidence, shaded, baseline, inner_left, inner_right,
            )
            complete = measure_histogram_bar(
                evidence, shaded, baseline, left, right,
            )
            # Narrow diagonal hatch segments can touch the baseline but do not
            # span the bin. Prefer broad full-bin top support when the inset
            # would mistake one of these fragments for the complete bar.
            candidates = [
                item for item in (inset, complete)
                if item is not None and (item[3] >= 0.20 or item[2] >= 0.30)
            ]
            if candidates:
                measurement = max(candidates, key=lambda item: (
                    item[3], item[1], item[2],
                ))
                break
        if measurement is None:
            top, bar_height, density, top_support = max(0, baseline - 1), 0.0, 0.0, 0.0
        else:
            top, bar_height, density, top_support = measurement
        columns.append({
            "index": index + 1,
            "bbox": [left, int(top), right, baseline + 1],
            "width_pixels": bin_pixels,
            "height_pixels": float(bar_height),
            "density": float(density),
            "left_edge_support": 0.0,
            "right_edge_support": 0.0,
            "top_edge_support": float(top_support),
            "bottom_edge_support": baseline_coverage,
            "top_edge_extension_support": 0.0,
            "side_edge_extension_support": 0.0,
            "filled_like": bool(density >= HISTOGRAM_FILLED_MINIMUM_DENSITY),
            "outline_like": bool(bar_height > 0),
            "complete_outline": False,
            "rectangularity": float(top_support),
            "weight": float(bar_height),
            "inferred_empty_bin": bool(bar_height <= 0),
            "tick_grid_recovered": False,
            "profile_recovered": False,
            "known_grid_recovered": True,
        })

    detected = sum(column["height_pixels"] > 0 for column in columns)
    return columns, {
        "raw_rectangle_candidate_count": 0.0,
        "baseline_candidate_count": float(detected),
        "retained_column_count": float(expected_bin_count),
        "detected_bar_count": float(detected),
        "inferred_empty_bin_count": float(expected_bin_count - detected),
        "baseline_y_pixels": float(baseline + 1),
        "baseline_spread_pixels": 0.0,
        "baseline_coverage": baseline_coverage,
        "tick_count": 0.0,
        "tick_pitch_pixels": width / expected_bin_count,
        "detection_method_tick_grid": 0.0,
        "detection_method_profile": 0.0,
        "detection_method_known_grid": 1.0,
    }


def detect_histogram_columns(
    bar_mask: np.ndarray,
    dark_mask: np.ndarray,
    crop: np.ndarray | None = None,
    axis_pitch_hint: float | None = None,
    expected_bin_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Use calibrated fixed slots when available, otherwise infer bins from image evidence."""
    if expected_bin_count is not None:
        if crop is None:
            raise ValueError("Fixed-grid histogram detection requires the image crop.")
        return known_grid_histogram_columns(crop, expected_bin_count)
    if crop is not None:
        regular = regular_histogram_columns(
            crop,
            axis_pitch_hint=axis_pitch_hint,
        )
        if regular is not None:
            return regular

    tick_result = tick_histogram_columns(
        dark_mask,
        bar_mask,
    )

    if tick_result is not None:
        tick_columns, _tick_diagnostics = (
            tick_result
        )
        detected_tick_bars = sum(
            float(column.get("weight", 0.0))
            > 0.0
            for column in tick_columns
        )

        if (
            detected_tick_bars
            >= HISTOGRAM_MINIMUM_COLUMNS
        ):
            return tick_result

    if crop is not None:
        profile_result = (
            histogram_profile_columns(crop)
        )

        if profile_result is not None:
            return profile_result

    if tick_result is not None:
        return tick_result

    return rectangle_histogram_columns(
        bar_mask,
        dark_mask,
    )

def histogram_bin_edges(
    minimum: float,
    maximum: float,
    column_count: int,
) -> np.ndarray:
    """Compute equally spaced bin edges from the supplied numerical bounds."""
    minimum = require_finite_number("Histogram minimum", minimum)
    maximum = require_finite_number("Histogram maximum", maximum)
    if maximum <= minimum:
        raise ValueError("Histogram maximum must be greater than histogram minimum.")
    if column_count <= 0:
        raise ValueError("Histogram must contain at least one column.")
    return np.linspace(minimum, maximum, column_count + 1, dtype=float)


def histogram_quantile_value(
    weights: np.ndarray,
    edges: np.ndarray,
    quantile: float,
) -> tuple[float, int, float]:
    """Interpolate a histogram quantile from cumulative bin heights."""
    if weights.ndim != 1 or edges.ndim != 1 or edges.size != weights.size + 1:
        raise ValueError("Histogram weights and bin edges are inconsistent.")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Histogram columns have no measurable height.")

    q = float(np.clip(quantile, 0.0, 1.0))
    if q <= 0.0:
        return (float(edges[0]), 1, 0.0)
    if q >= 1.0:
        return (float(edges[-1]), int(weights.size), 1.0)

    cumulative = np.cumsum(weights)
    target = q * total
    index = min(
        int(np.searchsorted(cumulative, target, side="left")),
        weights.size - 1,
    )
    before = float(cumulative[index - 1]) if index else 0.0
    weight = float(weights[index])

    if weight <= 0:
        fraction = 0.5
    else:
        fraction = float(np.clip((target - before) / weight, 0.0, 1.0))

    value = float(edges[index] + fraction * (edges[index + 1] - edges[index]))
    return (value, index + 1, fraction)


def adjust_histogram_column_height(
    result: dict[str, Any], column_index: int, height_pixels: float,
) -> dict[str, Any]:
    """Update one manually edited bin and recompute histogram statistics."""
    if result.get('interpretation_mode') != 'histogram':
        raise ValueError('Column height adjustment requires a histogram result.')
    columns = [dict(column) for column in result['histogram_columns']]
    if not 1 <= column_index <= len(columns):
        raise IndexError('Histogram column index is out of range.')
    if not np.isfinite(height_pixels):
        raise ValueError('Histogram column height must be finite.')
    plot_top = int(result['plot_bbox'][1])
    column = columns[column_index - 1]
    left, _top, right, bottom = column['bbox']
    height = float(np.clip(height_pixels, 0, bottom - plot_top))
    if 'automatic_height_pixels' not in column:
        column['automatic_height_pixels'] = column['height_pixels']
    column['height_pixels'] = round(height, 4)
    column['weight'] = round(height, 4)
    column['bbox'] = [left, max(plot_top, bottom - max(1, int(round(height)))), right, bottom]
    column['is_detected_bar'] = height > 0
    column['manually_adjusted'] = True

    weights = np.asarray([float(item['weight']) for item in columns], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError('At least one histogram column must have a positive height.')
    x_axis = result['x_axis']
    edges = histogram_bin_edges(float(x_axis['minimum']), float(x_axis['maximum']), len(columns))
    centers = (edges[:-1] + edges[1:]) / 2.0
    average = float(np.average(centers, weights=weights))
    average_index = int(np.clip(np.searchsorted(edges, average, side='right'), 1, len(columns)))
    p25, p25_index, p25_fraction = histogram_quantile_value(weights, edges, 0.25)
    median, median_index, median_fraction = histogram_quantile_value(weights, edges, 0.50)
    p75, p75_index, p75_fraction = histogram_quantile_value(weights, edges, 0.75)
    unit = str(x_axis.get('unit', ''))
    calibrated = bool(x_axis.get('calibrated', bool(unit) or float(x_axis['minimum']) != 0.5))
    display = (lambda value: round_measurement_value(value, unit)) if calibrated else (lambda value: round(value, 6))

    updated = dict(result)
    updated['histogram_columns'] = columns
    updated['statistics'] = {
        **result['statistics'],
        'average': display(average), 'average_raw': round(average, 8),
        'median': display(median), 'median_raw': round(median, 8),
        'percentile_50': display(median),
        'percentile_25': display(p25), 'percentile_25_raw': round(p25, 8),
        'percentile_75': display(p75), 'percentile_75_raw': round(p75, 8),
        'average_column_index': average_index,
        'median_column_index': median_index,
        'median_within_column_fraction': round(median_fraction, 8),
        'percentile_25_column_index': p25_index,
        'percentile_25_within_column_fraction': round(p25_fraction, 8),
        'percentile_75_column_index': p75_index,
        'percentile_75_within_column_fraction': round(p75_fraction, 8),
        'total_height_pixels': round(total, 4),
    }
    for index, item in enumerate(columns):
        item['probability_fraction'] = round(float(weights[index] / total), 8)
        item['is_mean_column'] = index + 1 == average_index
        item['is_median_column'] = index + 1 == median_index
    updated['average_column_bbox'] = list(columns[average_index - 1]['bbox'])
    updated['median_column_bbox'] = list(columns[median_index - 1]['bbox'])
    updated['detected_bar_bboxes'] = [list(item['bbox']) for item in columns if float(item['weight']) > 0]
    updated['manual_height_adjustments'] = {
        str(item['index']): item['height_pixels']
        for item in columns if item.get('manually_adjusted')
    }
    return updated


def digitize_histogram(
    image_path: Path,
    plot_bbox: list[int | float],
    target: str,
    minimum: float | None = None,
    maximum: float | None = None,
    unit: str = "",
    bin_width: float = DEFAULT_HISTOGRAM_BIN_WIDTH,
) -> dict[str, Any]:
    """Digitize a selected histogram using its numeric range and bin width.

    Return bin geometry, weighted statistics, calibration and review diagnostics.
    """
    target = str(target).strip() or "graph"
    unit = str(unit).strip()
    image = read_grayscale_image(image_path)
    image_height, image_width = image.shape[:2]
    normalized_bbox = normalize_bbox(plot_bbox, image_width, image_height)
    x1, y1, x2, y2 = normalized_bbox
    crop = image[y1:y2, x1:x2]

    calibrated = minimum is not None or maximum is not None
    if calibrated and (minimum is None or maximum is None):
        raise ValueError("Histogram minimum and maximum must be supplied together.")
    expected_bin_count = (
        histogram_known_bin_count(float(minimum), float(maximum), bin_width)
        if calibrated else None
    )

    bar_mask, dark_mask, threshold = histogram_masks(crop)
    axis_pitch_hint = (
        histogram_axis_pitch_hint(image, normalized_bbox)
        if not calibrated else None
    )
    columns, diagnostics = detect_histogram_columns(
        bar_mask, dark_mask, crop,
        axis_pitch_hint=axis_pitch_hint,
        expected_bin_count=expected_bin_count,
    )
    if not columns:
        raise RuntimeError(
            "No ordered histogram columns could be detected inside the selected plot. "
            "Verify that the bounds contain the complete histogram."
        )

    weights = np.asarray(
        [max(0.0, float(column["height_pixels"])) for column in columns],
        dtype=float,
    )
    total_height = float(np.sum(weights))
    if total_height <= 0:
        raise RuntimeError("Detected histogram columns have no measurable height.")

    if calibrated:
        edges = histogram_bin_edges(float(minimum), float(maximum), len(columns))
    else:
        # Backward-compatible index scale for older direct callers.
        edges = np.arange(0.5, len(columns) + 1.5, 1.0, dtype=float)
        unit = ""

    centers = (edges[:-1] + edges[1:]) / 2.0
    average = float(np.average(centers, weights=weights))
    average_column_index = int(
        np.clip(np.searchsorted(edges, average, side="right"), 1, len(columns))
    )

    p25, p25_column, p25_fraction = histogram_quantile_value(weights, edges, 0.25)
    median, median_column, median_fraction = histogram_quantile_value(weights, edges, 0.50)
    p75, p75_column, p75_fraction = histogram_quantile_value(weights, edges, 0.75)

    source_columns = []
    for zero_index, column in enumerate(columns):
        local = column["bbox"]
        source = {
            **column,
            "bbox": [local[0] + x1, local[1] + y1, local[2] + x1, local[3] + y1],
            "weight": round(float(weights[zero_index]), 4),
            "height_pixels": round(float(weights[zero_index]), 4),
            "probability_fraction": round(float(weights[zero_index] / total_height), 8),
            "bin_lower_bound": round(float(edges[zero_index]), 8),
            "bin_upper_bound": round(float(edges[zero_index + 1]), 8),
            "bin_center_value": round(float(centers[zero_index]), 8),
            "density": round(float(column["density"]), 6),
            "left_edge_support": round(float(column["left_edge_support"]), 4),
            "right_edge_support": round(float(column["right_edge_support"]), 4),
            "top_edge_support": round(float(column["top_edge_support"]), 4),
            "bottom_edge_support": round(float(column["bottom_edge_support"]), 4),
            "top_edge_extension_support": round(float(column.get("top_edge_extension_support", 0.0)), 4),
            "side_edge_extension_support": round(float(column.get("side_edge_extension_support", 0.0)), 4),
            "complete_outline": bool(column.get("complete_outline", False)),
            "tick_grid_recovered": bool(column.get("tick_grid_recovered", False)),
            "profile_recovered": bool(column.get("profile_recovered", False)),
            "inferred_empty_bin": bool(column.get("inferred_empty_bin", False)),
            "rectangularity": round(float(column["rectangularity"]), 4),
        }
        source["is_detected_bar"] = bool(source["weight"] > 0)
        source["is_mean_column"] = int(source["index"]) == average_column_index
        source["is_median_column"] = int(source["index"]) == median_column
        source_columns.append(source)

    detected = [column for column in source_columns if float(column["weight"]) > 0]
    widths = np.asarray([column["width_pixels"] for column in columns], dtype=float)
    width_consistency = max(
        0.0,
        1.0 - float(np.std(widths)) / max(1.0, float(np.mean(widths))),
    )
    rectangularity = (
        float(np.mean([column["rectangularity"] for column in detected]))
        if detected else 0.0
    )
    baseline_spread = float(diagnostics["baseline_spread_pixels"])
    baseline_consistency = max(
        0.0,
        1.0 - baseline_spread / max(1.0, crop.shape[0] * 0.04),
    )
    confidence = (
        min(1.0, len(detected) / 5.0) * 0.25
        + rectangularity * 0.25
        + baseline_consistency * 0.25
        + width_consistency * 0.25
    )
    coverage = sum(column["width_pixels"] for column in columns) / max(1, crop.shape[1])
    mask_support = float(np.count_nonzero(bar_mask)) / max(1, bar_mask.size)

    warnings = []
    if len(detected) < HISTOGRAM_MINIMUM_COLUMNS:
        warnings.append(
            "Only one non-empty histogram column was detected; verify the selected plot bounds."
        )
    if confidence < LOW_CONFIDENCE_WARNING_THRESHOLD:
        warnings.append(
            "Histogram confidence is low; inspect the recovered bin grid before relying on the statistics."
        )

    statistics = {
        "unit": unit,
        "average": round_measurement_value(average, unit) if calibrated else round(average, 6),
        "median": round_measurement_value(median, unit) if calibrated else round(median, 6),
        "percentile_50": round_measurement_value(median, unit) if calibrated else round(median, 6),
        "minimum": round_measurement_value(float(edges[0]), unit) if calibrated else float(edges[0]),
        "maximum": round_measurement_value(float(edges[-1]), unit) if calibrated else float(edges[-1]),
        "percentile_25": round_measurement_value(p25, unit) if calibrated else round(p25, 6),
        "percentile_75": round_measurement_value(p75, unit) if calibrated else round(p75, 6),
        "average_raw": round(average, 8),
        "median_raw": round(median, 8),
        "average_method": "height_weighted_bin_centers",
        "quantile_method": "cumulative_height_uniform_within_bin",
        "percentile_25_raw": round(p25, 8),
        "percentile_75_raw": round(p75, 8),
        "average_column_index": average_column_index,
        "median_column_index": median_column,
        "median_within_column_fraction": round(median_fraction, 8),
        "percentile_25_column_index": p25_column,
        "percentile_25_within_column_fraction": round(p25_fraction, 8),
        "percentile_75_column_index": p75_column,
        "percentile_75_within_column_fraction": round(p75_fraction, 8),
        "sample_count": len(columns),
        "total_height_pixels": round(total_height, 4),
    }

    return {
        "interpreter_version": "Mullenizer human-guided graph interpreter 1.0",
        "interpretation_mode": "histogram",
        "target": target,
        "source_image": str(image_path.expanduser().resolve()),
        "plot_bbox": normalized_bbox,
        "x_axis": {
            "minimum": float(edges[0]),
            "maximum": float(edges[-1]),
            "unit": unit,
            "calibrated": calibrated,
            "bin_count": len(columns),
            "bin_width": round(float(edges[1] - edges[0]), 8),
            "bin_count_source": "supplied_width" if calibrated else "image_detection",
        },
        "statistics": statistics,
        "average_column_bbox": list(source_columns[average_column_index - 1]["bbox"]),
        "median_column_bbox": list(source_columns[median_column - 1]["bbox"]),
        "detected_bar_bboxes": [
            list(column["bbox"]) for column in source_columns if column["is_detected_bar"]
        ],
        "histogram_columns": source_columns,
        "confidence": round(confidence, 4),
        "coverage": round(coverage, 4),
        "warnings": warnings,
        "trace_points": [],
        "diagnostics": {
            "plot_width_pixels": int(crop.shape[1]),
            "plot_height_pixels": int(crop.shape[0]),
            "detected_column_count": len(columns),
            "expected_bin_count": expected_bin_count,
            "known_grid_used": bool(diagnostics.get("detection_method_known_grid", 0)),
            "dark_threshold": int(threshold),
            "usable_dark_pixel_fraction": round(mask_support, 8),
            "column_width_consistency": round(width_consistency, 4),
            "mean_rectangularity": round(rectangularity, 4),
            "baseline_y_pixels": round(float(diagnostics["baseline_y_pixels"]), 3),
            "baseline_spread_pixels": round(baseline_spread, 3),
            "raw_rectangle_candidate_count": int(diagnostics["raw_rectangle_candidate_count"]),
            "detected_bar_count": int(diagnostics.get("detected_bar_count", len(detected))),
            "inferred_empty_bin_count": int(diagnostics.get("inferred_empty_bin_count", 0)),
            "tick_grid_used": bool(diagnostics.get("detection_method_tick_grid", 0)),
            "profile_grid_used": bool(diagnostics.get("detection_method_profile", 0)),
            "tick_count": int(diagnostics.get("tick_count", 0)),
            "tick_pitch_pixels": round(float(diagnostics.get("tick_pitch_pixels", 0.0)), 3),
        },
    }


def digitize_line_graph(
    image_path: Path,
    plot_bbox: list[int | float],
    y_top: float,
    y_bottom: float,
    unit: str,
    target: str,
) -> dict[str, Any]:
    """Trace a bounded line plot and map its pixels to calibrated y-values.

    Return trace geometry, summary statistics and confidence diagnostics.
    """
    y_top = require_finite_number("Top y-axis value", y_top)
    y_bottom = require_finite_number("Bottom y-axis value", y_bottom)
    unit = str(unit).strip()
    target = str(target).strip() or "graph"

    image = read_grayscale_image(image_path)
    image_height, image_width = image.shape[:2]
    normalized_bbox = normalize_bbox(plot_bbox, image_width, image_height)
    x1, y1, x2, y2 = normalized_bbox
    crop = image[y1:y2, x1:x2]

    enhanced, usable_mask, threshold = prepare_trace_evidence(crop)
    sample_columns = choose_sample_columns(crop.shape[1])
    sampled_y, sampled_observed, tracking = track_continuous_path(
        enhanced, usable_mask, sample_columns
    )
    sampled_y = median_smooth(sampled_y, SMOOTHING_WINDOW)
    sampled_y = refine_vertical_extrema(
        sampled_y,
        usable_mask,
        sample_columns,
    )

    # Interpolate the sampled path back to full horizontal resolution before calibration.
    full_x = np.arange(crop.shape[1], dtype=float)
    full_y = np.clip(
        np.interp(full_x, sample_columns.astype(float), sampled_y),
        0,
        crop.shape[0] - 1,
    )
    values = pixel_y_to_value(full_y, crop.shape[0], y_top, y_bottom)
    statistics = calculate_statistics(values, unit)
    trace_points = thin_trace_points(full_x + x1, full_y + y1)

    observed_fraction = float(tracking["observed_column_fraction"])
    continuity = float(tracking["continuity_score"])
    mask_support = float(np.count_nonzero(usable_mask)) / max(1, usable_mask.size)
    confidence = (
        observed_fraction * 0.55
        + continuity * 0.30
        + min(1.0, mask_support / 0.025) * 0.15
    )

    warnings = []
    if observed_fraction < LOW_COVERAGE_WARNING_THRESHOLD:
        warnings.append(
            "The plotted line was directly visible in fewer than half of the "
            "tracked columns; inspect the trace overlay carefully."
        )
    if confidence < LOW_CONFIDENCE_WARNING_THRESHOLD:
        warnings.append(
            "Digitization confidence is low; revise the plot bounds or axis "
            "calibration before relying on the statistics."
        )
    if math.isclose(y_top, y_bottom):
        warnings.append(
            "Top and bottom y-axis values are equal, so all calibrated values will be identical."
        )

    return {
        "interpreter_version": "Mullenizer human-guided graph interpreter 1.0",
        "interpretation_mode": "line",
        "target": target,
        "source_image": str(image_path.expanduser().resolve()),
        "plot_bbox": normalized_bbox,
        "y_axis": {"top_value": y_top, "bottom_value": y_bottom, "unit": unit},
        "statistics": statistics,
        "confidence": round(confidence, 4),
        "coverage": round(observed_fraction, 4),
        "warnings": warnings,
        "trace_points": trace_points,
        "diagnostics": {
            "plot_width_pixels": int(crop.shape[1]),
            "plot_height_pixels": int(crop.shape[0]),
            "tracked_column_count": int(sample_columns.size),
            "directly_observed_column_count": int(np.count_nonzero(sampled_observed)),
            "dark_threshold": int(threshold),
            "usable_dark_pixel_fraction": round(mask_support, 8),
            "mean_sampled_jump_pixels": round(float(tracking["mean_sampled_jump_pixels"]), 4),
            "continuity_score": round(continuity, 4),
        },
    }


def main() -> None:
    """Parse graph-interpreter CLI options and save the digitization result."""
    parser = argparse.ArgumentParser(
        description="Interpret one manually bounded line graph or histogram."
    )
    parser.add_argument("image", help="Path to the retained deskewed page image.")
    parser.add_argument(
        "--bbox", type=float, nargs=4, required=True,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Inner plot rectangle in source-page pixels.",
    )
    parser.add_argument(
        "--mode", choices=("line", "histogram"), default="histogram",
        help="Interpretation mode. Default: histogram",
    )
    parser.add_argument("--y-top", type=float, default=None)
    parser.add_argument("--y-bottom", type=float, default=None)
    parser.add_argument("--hist-min", type=float, default=None)
    parser.add_argument("--hist-max", type=float, default=None)
    parser.add_argument("--bin-width", type=float, default=DEFAULT_HISTOGRAM_BIN_WIDTH,
                        help="Numerical width of each histogram bin (default: 10).")
    parser.add_argument("--unit", default="")
    parser.add_argument("--target", default="graph")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode == "histogram":
        if args.hist_min is None or args.hist_max is None:
            parser.error("--hist-min and --hist-max are required in histogram mode.")
        result = digitize_histogram(
            Path(args.image),
            list(args.bbox),
            args.target,
            minimum=args.hist_min,
            maximum=args.hist_max,
            unit=args.unit,
            bin_width=args.bin_width,
        )
    else:
        if args.y_top is None or args.y_bottom is None:
            parser.error("--y-top and --y-bottom are required in line mode.")
        result = digitize_line_graph(
            Path(args.image), list(args.bbox),
            args.y_top, args.y_bottom, args.unit, args.target,
        )

    output_path = Path(args.out)
    save_json(result, output_path)
    statistics = result["statistics"]
    suffix = f" {statistics['unit']}" if statistics.get("unit") else ""

    print(f"Average: {statistics['average']}{suffix}")
    if result["interpretation_mode"] == "histogram":
        print(f"Median (50th percentile): {statistics['median']}{suffix}")

    print(f"Minimum: {statistics['minimum']}{suffix}")
    print(f"Maximum: {statistics['maximum']}{suffix}")
    print(f"25th percentile: {statistics['percentile_25']}{suffix}")
    print(f"75th percentile: {statistics['percentile_75']}{suffix}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Result: {output_path}")


if __name__ == "__main__":
    main()
