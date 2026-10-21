"""OCR each deskewed page once, extract demographics/AFib, and rank spatially unique heart-rate/activity guide text."""
from __future__ import annotations
import argparse
import difflib
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
DEFAULT_OUTPUT_PATH = 'output/ocr_targets.json'
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
DEFAULT_OCR_SCALE = 1.5
DEFAULT_PAGE_SEGMENTATION_MODE = 11
DEFAULT_OCR_LANGUAGE = 'eng'
DEFAULT_PAGE_TIMEOUT_SECONDS = 120
MIN_WORD_CONFIDENCE = 1.0
MIN_ACCEPTED_AFIB_SCORE = 150.0
AFIB_NEARBY_VERTICAL_PIXELS = 180
AFIB_NEARBY_HORIZONTAL_PAGE_FRACTION = 0.38
AFIB_MAX_NEARBY_LINES = 8
AFIB_REVIEW_PADDING_LEFT_PIXELS = 180
AFIB_REVIEW_PADDING_RIGHT_PIXELS = 420
AFIB_REVIEW_PADDING_TOP_PIXELS = 120
AFIB_REVIEW_PADDING_BOTTOM_PIXELS = 120
DEFAULT_TOP_AFIB_CANDIDATES = 5
DEFAULT_TOP_GRAPH_GUIDES = 5
GRAPH_GUIDE_CONTEXT_VERTICAL_PIXELS = 90
DEFAULT_TOP_DEMOGRAPHIC_CANDIDATES = 5
MIN_ACCEPTED_PATIENT_NAME_SCORE = 135.0
MIN_ACCEPTED_DATE_OF_BIRTH_SCORE = 135.0
DEMOGRAPHIC_NEARBY_VERTICAL_PIXELS = 170
DEMOGRAPHIC_NEARBY_HORIZONTAL_PAGE_FRACTION = 0.85
DEMOGRAPHIC_MAX_NEARBY_LINES = 12
DEMOGRAPHIC_ALIGNED_ROW_MIN_VERTICAL_OVERLAP = 0.45
DEMOGRAPHIC_ALIGNED_ROW_CENTER_TOLERANCE_PIXELS = 70
DEMOGRAPHIC_ALIGNED_ROW_HEIGHT_TOLERANCE_FRACTION = 0.85
DEMOGRAPHIC_ALIGNED_VALUE_RIGHT_MARGIN_PIXELS = 30
PATIENT_NAME_MIN_WORDS = 2
PATIENT_NAME_MAX_WORDS = 5
PATIENT_NAME_MAX_CHARACTERS = 72
PATIENT_NAME_BELOW_MAX_VERTICAL_PIXELS = 420
PATIENT_NAME_BELOW_START_X_TOLERANCE_PAGE_FRACTION = 0.14
PATIENT_NAME_BELOW_MIN_HORIZONTAL_OVERLAP = 0.2
PATIENT_NAME_ROW_FRAGMENT_MAX_GAP_PAGE_FRACTION = 0.24
PATIENT_NAME_ROW_MAX_FRAGMENTS = 5
PATIENT_NAME_REJECT_WORDS = frozenset({'patient', 'name', 'dob', 'birth', 'date', 'age', 'sex', 'gender', 'address', 'phone', 'telephone', 'report', 'summary', 'clinic', 'hospital', 'device', 'monitor', 'rate', 'heart', 'activity', 'burden', 'average', 'mean', 'record', 'medical', 'physician', 'doctor', 'technician', 'interpretation', 'diagnosis', 'indication', 'page', 'study', 'analysis', 'duration', 'hours', 'days', 'implant', 'implanted', 'implantation', 'serial', 'model', 'manufacturer', 'account', 'visit', 'admit', 'admission', 'discharge', 'procedure', 'provider', 'referring', 'ordering', 'location', 'room', 'facility', 'service', 'history', 'status', 'has', 'have', 'had', 'is', 'was', 'will', 'seen', 'presented', 'presents', 'with', 'for', 'and', 'or', 'the', 'this', 'that', 'male', 'female', 'unknown'})
PATIENT_NAME_LOWERCASE_PARTICLES = frozenset({'de', 'del', 'della', 'da', 'di', 'du', 'la', 'le', 'van', 'von', 'der', 'den', 'bin', 'ibn', 'al'})
PATIENT_NAME_LABEL_REJECT_PATTERNS: tuple[re.Pattern[str], ...] = (re.compile('\\bpatient\\s+history\\b', re.IGNORECASE), re.compile('\\bpatient\\s+(?:has|is|was|will)\\b', re.IGNORECASE), re.compile('\\bpatient\\s+(?:seen|was\\s+seen|is\\s+seen)\\b', re.IGNORECASE))
PATIENT_NAME_PATTERNS: tuple[tuple[re.Pattern[str], float, str], ...] = ((re.compile("^\\s*(?:patient'?s?\\s+name|name\\s+of\\s+patient|patient\\s+full\\s+name)\\b", re.IGNORECASE), 130.0, 'patient name'), (re.compile('^\\s*pt[\\s._-]*name\\b', re.IGNORECASE), 118.0, 'Pt name'), (re.compile('^\\s*patient\\s*:\\s*', re.IGNORECASE), 112.0, 'patient colon'), (re.compile('^\\s*patient\\s*:?\\s*$', re.IGNORECASE), 95.0, 'patient'), (re.compile('^\\s*full\\s+name\\b\\s*:?\\s*', re.IGNORECASE), 92.0, 'full name'), (re.compile('^\\s*name\\b\\s*:?\\s*', re.IGNORECASE), 72.0, 'name'))
DATE_OF_BIRTH_PATTERNS: tuple[tuple[re.Pattern[str], float, str], ...] = ((re.compile('\\bdate[\\s._:-]*(?:of|0f|o[fF])[\\s._:-]*(?:birth|b[i1l]rth)\\b', re.IGNORECASE), 132.0, 'date of birth'), (re.compile('\\bbirth\\s*date\\b', re.IGNORECASE), 122.0, 'birth date'), (re.compile('\\bbirthdate\\b', re.IGNORECASE), 122.0, 'birthdate'), (re.compile('\\bbirth[\\s._-]*(?:dt|dte)\\b', re.IGNORECASE), 116.0, 'birth date abbreviation'), (re.compile('\\bb[\\s._-]*(?:date|dt)\\b', re.IGNORECASE), 108.0, 'B-date abbreviation'), (re.compile('\\bd[\\s._:-]*o[\\s._:-]*b\\b', re.IGNORECASE), 124.0, 'DOB'), (re.compile('\\bd[\\s._:-]*[o0][\\s._:-]*[b8]\\b', re.IGNORECASE), 120.0, 'DOB OCR-confusable'), (re.compile('\\bpatient\\s+d[\\s._:-]*[o0][\\s._:-]*[b8]\\b', re.IGNORECASE), 125.0, 'patient DOB'))
# Times immediately following a date disqualify it as a date-of-birth candidate.
# Accept colon/dot-delimited times, HHMM/HHMMSS, and optional AM/PM.
DOB_TRAILING_TIME_PATTERN = re.compile(
    r"^\s*(?:[-Tt,;@]|at\b)?\s*"
    r"(?:(?:[01]?\d|2[0-3])(?:[:./-]|\s+)[0-5]\d(?:(?:[:./-]|\s+)[0-5]\d)?"
    r"|(?:[01]\d|2[0-3])[0-5]\d(?:[0-5]\d)?"
    r"|(?:0?[1-9]|1[0-2])\s*[AaPp]\.?[Mm]\.?)"
    r"(?:\s*[AaPp]\.?[Mm]\.?)?(?!\d)"
)
DOB_DATE_SHAPE_PATTERN = re.compile(
    r"(?<!\w)(?:\d{1,4}[./-]\d{1,2}[./-]\d{2,4}"
    r"|[a-z]{3,12}\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+|[-/]+)\d{2,4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+[a-z]{3,12}\.?(?:,?\s+|[-/]+)\d{2,4})(?!\d)",
    re.IGNORECASE,
)
DOB_STANDALONE_DATE_BASE_SCORE = 82.0
DOB_STANDALONE_CONTEXT_BONUS = 28.0
DOB_SPLIT_LABEL_MAX_COMPONENTS = 3
DOB_SPLIT_LABEL_MAX_GAP_PIXELS = 120
DOB_SPLIT_LABEL_MAX_CENTER_Y_PIXELS = 55
NAME_VALUE_STOP_PATTERN = re.compile('\\b(?:d[\\s._-]*o[\\s._-]*b|date\\s+of\\s+birth|birth\\s*date|birthdate|mrn|medical\\s+record|sex|gender|age|phone|address|implant|serial|model|account|visit|admission|discharge)\\b', re.IGNORECASE)
MONTH_NAME_TO_NUMBER = {'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3, 'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9, 'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12}
PERCENT_PATTERN = re.compile('\n    (?P<comparator>\n        <=|>=|<|>|≤|≥|~|≈\n    )?\n    \\s*\n    (?P<value>\n        \\d{1,3}(?:[.,]\\d{1,3})?\n    )\n    \\s*\n    (?P<unit>\n        %|percent\n    )\n    ', flags=re.IGNORECASE | re.VERBOSE)
HEART_RATE_PATTERNS: tuple[tuple[re.Pattern[str], float, str], ...] = ((re.compile(r'\bV[\s._:/-]*Rate\b', re.IGNORECASE), 270.0, 'V Rate'), (re.compile(r'\bventricular[\s._:/-]*rate\b', re.IGNORECASE), 260.0, 'ventricular rate'), (re.compile(r'\bVent[\s._:/-]*Rate\b', re.IGNORECASE), 265.0, 'Vent Rate'), (re.compile('\\b(?:average|avg\\.?|ave\\.?|mean)\\s+(?:daily\\s+)?heart[\\s._:/-]*rate\\b', re.IGNORECASE), 130.0, 'average heart rate'), (re.compile('\\bheart[\\s._:/-]*rate\\s+(?:average|avg\\.?|mean)\\b', re.IGNORECASE), 125.0, 'heart rate average'), (re.compile('\\b(?:average|avg\\.?|ave\\.?|mean)\\s+H[\\s._:/-]*R\\b'), 118.0, 'average HR'), (re.compile('\\b(?:average|avg\\.?|ave\\.?|mean)\\s+ventricular[\\s._:/-]*rate\\b', re.IGNORECASE), 112.0, 'average ventricular rate'), (re.compile('\\bventricular[\\s._:/-]*activity\\b', re.IGNORECASE), 108.0, 'ventricular activity'), (re.compile('\\bmean[\\s._:/-]*rate\\b', re.IGNORECASE), 102.0, 'mean rate'), (re.compile('\\bheart[\\s._:/-]*rate\\b', re.IGNORECASE), 92.0, 'heart rate'), (re.compile('\\bH[\\s._:/-]*R\\b'), 82.0, 'HR'), (re.compile('\\bh[\\s._:/-]*rate\\b', re.IGNORECASE), 78.0, 'abbreviated heart rate'), (re.compile('\\bpulse[\\s._:/-]*rate\\b', re.IGNORECASE), 76.0, 'pulse rate'), (re.compile('\\b(?:bpm|b\\.?\\s*p\\.?\\s*m\\.?|8pm|bpn|bprn)\\b', re.IGNORECASE), 48.0, 'heart-rate unit'), (re.compile('\\bbeats?\\s*(?:/|per)\\s*(?:min(?:ute)?s?|minute)\\b', re.IGNORECASE), 48.0, 'heart-rate unit'), (re.compile('\\brate\\b', re.IGNORECASE), 20.0, 'rate'))
HEART_RATE_FUZZY_PHRASES: tuple[tuple[str, float, str], ...] = (('v rate', 190.0, 'near-match V Rate'), ('ventricular rate', 185.0, 'near-match ventricular rate'), ('vent rate', 188.0, 'near-match Vent Rate'), ('average heart rate', 112.0, 'near-match average heart rate'), ('avg heart rate', 106.0, 'near-match avg heart rate'), ('mean heart rate', 106.0, 'near-match mean heart rate'), ('ventricular activity', 96.0, 'near-match ventricular activity'), ('heart rate', 84.0, 'near-match heart rate'), ('h rate', 70.0, 'near-match H rate'), ('pulse rate', 68.0, 'near-match pulse rate'), ('beats per minute', 54.0, 'near-match beats per minute'))
HEART_RATE_FUZZY_MINIMUM_RATIO = 0.74
HEART_RATE_FUZZY_MINIMUM_TOKEN_LENGTH = 4
HEART_RATE_EXPLICIT_REASONS = {'V Rate', 'Vent Rate', 'average heart rate', 'heart rate average', 'average HR', 'heart rate', 'HR', 'abbreviated heart rate', 'pulse rate', 'average ventricular rate', 'ventricular rate', 'ventricular activity'}
ACTIVITY_PATTERNS: tuple[tuple[re.Pattern[str], float, str], ...] = ((re.compile('\\b(?:average|avg\\.?|ave\\.?|mean)\\s+(?:daily\\s+)?activity\\b', re.IGNORECASE), 125.0, 'average daily activity'), (re.compile('\\bdaily\\s+(?:average|avg\\.?|ave\\.?|mean)\\s+activity\\b', re.IGNORECASE), 125.0, 'average daily activity'), (re.compile('\\bdaily[\\s._:/-]*activity\\b', re.IGNORECASE), 105.0, 'daily activity'), (re.compile('\\bpatient[\\s._:/-]*activity\\b', re.IGNORECASE), 88.0, 'patient activity'), (re.compile('\\b(?:active\\s+time|time\\s+active)\\b', re.IGNORECASE), 76.0, 'active time'), (re.compile('\\bactivity\\b', re.IGNORECASE), 62.0, 'activity'), (re.compile('\\b(?:hours?|hrs?|h)\\s*(?:/|per)\\s*day\\b', re.IGNORECASE), 40.0, 'activity time unit'), (re.compile('\\b(?:minutes?|mins?|min)\\s*(?:/|per)\\s*day\\b', re.IGNORECASE), 40.0, 'activity time unit'))
ACTIVITY_FUZZY_PHRASES: tuple[tuple[str, float, str], ...] = (('average daily activity', 108.0, 'near-match average daily activity'), ('daily activity', 90.0, 'near-match daily activity'), ('patient activity', 76.0, 'near-match patient activity'), ('active time', 66.0, 'near-match active time'))
ACTIVITY_FUZZY_MINIMUM_RATIO = 0.76
# EDIT HERE to tune how Mullenizer ranks OCR review suggestions for each metric.
# "prefer" and "penalize" map whole-word phrases to points added to the base score.
# "reject" lists hard exclusions (regex, reason) that never reach the review UI.
# Graph-title preferences apply to BOTH graph metrics; demographic/AFib value
# parsing and validity checks remain separate from these ranking preferences.
OCR_RANKING_RULES = {
    "_graph_common": {
        "prefer": {
            "histogram": 120, "graph": 65, "trend": 55,
            "profile": 45, "distribution": 45, "over time": 40,
        },
        "penalize": {
            "average": -40, "avg": -40, "ave": -40, "mean": -50,
            "minimum": -100, "min": -100, "maximum": -100, "max": -100,
            "median": -90, "summary": -40,
        },
        "reject": (
            # Episode measurements are not graph headings. Tolerate OCR misspellings.
            (re.compile(r"\b(?:v|vent|ventricular)[\s._:/-]*r[a-z0-9]{2,5}\b.{0,35}\b[0o]ns[a-z0-9]{2,3}\b", re.I), "V rate at onset"),
            (re.compile(r"\bAT\s*[/\\|_-]?\s*AF\s+Burden\b.*\b(?:\d+(?:[.,]\d+)?\s*)?hours?\s+elevated\s+per\s+day\b", re.I), "AT/AF burden hours elevated"),
            (re.compile(r"\bAT\s*[/\\|_-]?\s*AF\s+Detection\s+(?:and|&)\s+Response\b", re.I), "AT/AF device settings"),
        ),
    },
    "average_heart_rate": {
        "minimum_score": 72.0,
        "prefer": {
            "v rate during at af": 95, "vent rate during at af": 95,
            "ventricular rate during at af": 95, "v rate in at af": 80,
            "vent rate in at af": 80, "ventricular rate in at af": 80,
        },
        "penalize": {"episode": -60, "reading": -45},
        "exact_titles": {  # Exact whole-line headings receive their own bonus.
            "heart rate histogram": 175, "v rate": 140, "vent rate": 135, "ventricular rate": 135,
            "heart rate": 145, "hr": 45, "h r": 45, "pulse rate": 90,
            "v rate during at af": 140, "vent rate during at af": 140,
            "ventricular rate during at af": 140,
            "v rate in at af": 125, "vent rate in at af": 125,
            "ventricular rate in at af": 125,
        },
        "histogram_title_base": 285,  # Allows common OCR errors in HR histogram titles.
        "nearby": (
            (r"\b(?:bpm|b\.?\s*p\.?\s*m\.?)\b", 12, "nearby bpm"),
            (r"\bheart[\s._:/-]*rate\b", 24, "nearby Heart Rate"),
        ),
        "reject": (
            (re.compile(r"\b(?:right\s+)?ventricular\s+pacing\b", re.I), "ventricular pacing"),
            (re.compile(r"\batr(?:\s*at|at)?\b", re.I), "ATR event-rate text"),
            (re.compile(r"\b(?:all\s+)?events?\s+since\b", re.I), "event log"),
            (re.compile(r"\bmagnet\b", re.I), "magnet rate"),
            (re.compile(r"\b(?:lower|upper|maximum|max\.?|minimum|min\.?)\s+(?:tracking\s+|sensor\s+|pacing\s+)?rate\b", re.I), "programmed rate limit"),
            (re.compile(r"\b(?:tracking|sensor|pacing|paced|base|basic|backup|escape|mode\s*switch|therapy|monitor)\s+rate\b", re.I), "device-programming rate"),
            (re.compile(r"\brate\s+(?:response|adaptive|adaptation|hysteresis)\b", re.I), "rate response setting"),
            (re.compile(r"\b(?:atrial|tachy(?:cardia)?|brady(?:cardia)?|VT|VF)\s+rate\b", re.I), "arrhythmia/device rate"),
            (re.compile(r"\brespirat(?:ory|ion)\s+rate\b", re.I), "respiratory rate"),
            (re.compile(r"\bbreath(?:ing)?\s+rate\b", re.I), "breathing rate"),
            (re.compile(r"\bventilat(?:ion|ory)\s+rate\b", re.I), "ventilation rate"),
            (re.compile(r"\b(?:sampling|sample|scan|scanning|data|transfer|transmission|upload|download|refresh)\s+rate\b", re.I), "data/sampling rate"),
            (re.compile(r"\b(?:event|episode|detection|success|failure|error|capture|charge|discharge)\s+rate\b", re.I), "unrelated rate"),
        ),
    },
    "average_daily_activity": {
        "minimum_score": 68.0,
        "prefer": {"daily activity": 35, "patient activity": 25},
        "penalize": {"total": -35, "recorded": -20},
        "exact_titles": {
            "activity": 100, "daily activity": 100,
            "patient activity": 90, "active time": 75,
        },
        "nearby": (
            (r"\b(?:h(?:ours?)?|min(?:utes?)?)\s*(?:/|per)\s*day\b", 12, "nearby activity unit"),
            (r"\bactivity\b", 20, "nearby Activity"),
        ),
        "reject": (
            (re.compile(r"\bactivity\s+(?:threshold|sensor|response|sensitivity|setting)\b", re.I), "activity device setting"),
            (re.compile(r"\b(?:sensor|accelerometer)\s+activity\b", re.I), "activity sensor setting"),
            (re.compile(r"\bventricular[\s._:/-]*activity\b", re.I), "ventricular activity belongs to heart rate"),
        ),
    },
    "afib_burden": {
        "prefer": {"at af burden": 20, "afib burden": 15, "af burden": 10},
        "penalize": {"summary": -5},
        "reject": (),
    },
    "patient_name": {
        "prefer": {"patient name": 15, "full name": 10},
        "penalize": {},
        "reject": (),
    },
    "date_of_birth": {
        "prefer": {"date of birth": 15, "birth date": 10, "dob": 10},
        "penalize": {},
        "reject": (),
    },
}

# Backwards-compatible threshold names; configure their values in the table above.
MIN_ACCEPTED_HEART_RATE_GUIDE_SCORE = OCR_RANKING_RULES["average_heart_rate"]["minimum_score"]
MIN_ACCEPTED_ACTIVITY_GUIDE_SCORE = OCR_RANKING_RULES["average_daily_activity"]["minimum_score"]

def list_image_paths(input_dir: Path) -> list[Path]:
    """Return supported deskewed page images in stable filename order."""
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory does not exist: {input_dir}')
    if not input_dir.is_dir():
        raise ValueError(f'Input path is not a directory: {input_dir}')
    image_paths = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    return sorted(image_paths)

def ensure_output_parent(output_path: Path) -> Path:
    """Create the output file's parent directory and return the path."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def read_grayscale_image(image_path: Path) -> np.ndarray:
    """Read one deskewed page as an 8-bit grayscale image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')
    return image

def expand_review_bbox(bbox: list[int], image_width: int, image_height: int) -> list[int]:
    """Frame an OCR hit with enough page context for comfortable review."""
    x1 = max(0, bbox[0] - AFIB_REVIEW_PADDING_LEFT_PIXELS)
    y1 = max(0, bbox[1] - AFIB_REVIEW_PADDING_TOP_PIXELS)
    x2 = min(image_width, bbox[2] + AFIB_REVIEW_PADDING_RIGHT_PIXELS)
    y2 = min(image_height, bbox[3] + AFIB_REVIEW_PADDING_BOTTOM_PIXELS)

    # Avoid zooming into tiny text boxes even when the original padding is small.
    width = min(image_width, max(x2 - x1, round(image_width * 0.62)))
    height = min(image_height, max(y2 - y1, round(image_height * 0.38)))
    left = max(0, min(round((x1 + x2 - width) / 2), image_width - width))
    top = max(0, min(round((y1 + y2 - height) / 2), image_height - height))
    return [left, top, left + width, top + height]

def prepare_ocr_image(image: Any, scale: float) -> tuple[Any, float]:
    """Resize a temporary OCR copy and report its actual scale."""
    if scale <= 0.0:
        raise ValueError(f'OCR scale must be positive, received: {scale}')
    if abs(scale - 1.0) < 1e-09:
        return (image, 1.0)
    height, width = image.shape[:2]
    output_width = max(1, round(width * scale))
    output_height = max(1, round(height * scale))
    prepared = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    actual_scale = (output_width / float(width) + output_height / float(height)) / 2.0
    return (prepared, actual_scale)

def normalize_text(text: str) -> str:
    """Normalize OCR text for stable matching without changing saved text."""
    normalized = unicodedata.normalize('NFKC', text)
    normalized = normalized.replace('–', '-').replace('—', '-')
    normalized = re.sub('\\s+', ' ', normalized)
    return normalized.strip()

def parse_confidence(raw_confidence: Any) -> float:
    """Convert one pytesseract confidence value into a usable float."""
    try:
        return float(raw_confidence)
    except (TypeError, ValueError):
        return -1.0

def scale_bbox_to_source(left: int, top: int, width: int, height: int, scale: float, source_width: int, source_height: int) -> list[int]:
    """Map one OCR-image bounding box back to source-page coordinates."""
    x1 = int(round(left / scale))
    y1 = int(round(top / scale))
    x2 = int(round((left + width) / scale))
    y2 = int(round((top + height) / scale))
    x1 = max(0, min(source_width, x1))
    y1 = max(0, min(source_height, y1))
    x2 = max(x1, min(source_width, x2))
    y2 = max(y1, min(source_height, y2))
    return [x1, y1, x2, y2]

def union_bboxes(bboxes: list[list[int]]) -> list[int]:
    """Return the smallest rectangle containing all supplied boxes."""
    if not bboxes:
        return [0, 0, 0, 0]
    return [min((bbox[0] for bbox in bboxes)), min((bbox[1] for bbox in bboxes)), max((bbox[2] for bbox in bboxes)), max((bbox[3] for bbox in bboxes))]

def run_page_ocr(image: np.ndarray, language: str, page_segmentation_mode: int, timeout_seconds: int, ocr_scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """OCR a page once and reconstruct recognized words into source-coordinate lines."""
    source_height, source_width = image.shape[:2]
    ocr_image, actual_scale = prepare_ocr_image(image=image, scale=ocr_scale)
    tesseract_config = f'--oem 1 --psm {page_segmentation_mode} -c preserve_interword_spaces=1'
    ocr_data = pytesseract.image_to_data(ocr_image, lang=language, config=tesseract_config, output_type=Output.DICT, timeout=timeout_seconds)
    grouped_words: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    entry_count = len(ocr_data['text'])
    for index in range(entry_count):
        text = normalize_text(str(ocr_data['text'][index]))
        if not text:
            continue
        confidence = parse_confidence(ocr_data['conf'][index])
        if confidence < MIN_WORD_CONFIDENCE:
            continue
        bbox = scale_bbox_to_source(left=int(ocr_data['left'][index]), top=int(ocr_data['top'][index]), width=int(ocr_data['width'][index]), height=int(ocr_data['height'][index]), scale=actual_scale, source_width=source_width, source_height=source_height)
        line_key = (int(ocr_data['block_num'][index]), int(ocr_data['par_num'][index]), int(ocr_data['line_num'][index]))
        grouped_words.setdefault(line_key, []).append({'text': text, 'confidence': confidence, 'bbox': bbox, 'word_number': int(ocr_data['word_num'][index])})
    lines: list[dict[str, Any]] = []
    for line_key, words in grouped_words.items():
        words.sort(key=lambda word: (word['word_number'], word['bbox'][0]))
        line_text = normalize_text(' '.join((word['text'] for word in words)))
        line_bbox = union_bboxes([word['bbox'] for word in words])
        confidence_weight = sum((max(1, len(word['text'])) for word in words))
        weighted_confidence = sum((word['confidence'] * max(1, len(word['text'])) for word in words))
        line_confidence = weighted_confidence / max(1, confidence_weight)
        lines.append({'text': line_text, 'normalized_text': line_text.casefold(), 'confidence': round(line_confidence, 3), 'bbox': line_bbox, 'block_number': line_key[0], 'paragraph_number': line_key[1], 'line_number': line_key[2], 'words': words})
    lines.sort(key=lambda line: (line['bbox'][1], line['bbox'][0]))
    for index, line in enumerate(lines):
        line['page_line_index'] = index
    ocr_metadata = {'source_width_pixels': source_width, 'source_height_pixels': source_height, 'ocr_width_pixels': int(ocr_image.shape[1]), 'ocr_height_pixels': int(ocr_image.shape[0]), 'ocr_scale': round(actual_scale, 8), 'recognized_line_count': len(lines), 'recognized_word_count': sum((len(line['words']) for line in lines)), 'tesseract_config': tesseract_config}
    return (lines, ocr_metadata)

def compact_dob_label_text(text: str) -> str:
    """Normalize DOB labels across spacing, punctuation, and common OCR confusions."""
    compact = re.sub('[^A-Za-z0-9]+', '', normalize_text(text)).casefold()
    if not compact:
        return ''
    if len(compact) <= 24:
        compact = compact.replace('0', 'o').replace('8', 'b').replace('1', 'i').replace('l', 'i')
    return compact

def looks_like_dob_label_text(text: str) -> bool:
    """Return True for exact or OCR-confusable common DOB label text."""
    compact = compact_dob_label_text(text)
    if compact in {'dob', 'patientdob', 'dateofbirth', 'birthdate', 'birthdt', 'birthdte', 'bdate', 'bdt'}:
        return True
    if compact:
        ratio = difflib.SequenceMatcher(None, compact, 'dateofbirth').ratio()
        if ratio >= 0.86:
            return True
    return False

def augment_split_dob_label_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add virtual OCR lines when a short DOB label was split into D / O / B pieces."""
    augmented = list(lines)
    short_lines = [line for line in lines if normalize_text(line['text']).strip() and len(normalize_text(line['text']).strip()) <= 3]
    seen: set[tuple[int, ...]] = set()
    for anchor in short_lines:
        anchor_bbox = anchor['bbox']
        anchor_center_y = (anchor_bbox[1] + anchor_bbox[3]) / 2.0
        same_row = [line for line in short_lines if abs((line['bbox'][1] + line['bbox'][3]) / 2.0 - anchor_center_y) <= DOB_SPLIT_LABEL_MAX_CENTER_Y_PIXELS]
        same_row.sort(key=lambda line: line['bbox'][0])
        try:
            anchor_position = same_row.index(anchor)
        except ValueError:
            continue
        for component_count in range(2, DOB_SPLIT_LABEL_MAX_COMPONENTS + 1):
            end_position = anchor_position + component_count
            if end_position > len(same_row):
                break
            pieces = same_row[anchor_position:end_position]
            valid_spacing = True
            for previous, candidate in zip(pieces[:-1], pieces[1:]):
                gap = candidate['bbox'][0] - previous['bbox'][2]
                if gap < -20 or gap > DOB_SPLIT_LABEL_MAX_GAP_PIXELS:
                    valid_spacing = False
                    break
            if not valid_spacing:
                continue
            combined_text = ' '.join((normalize_text(piece['text']).strip() for piece in pieces))
            if not looks_like_dob_label_text(combined_text):
                continue
            component_indices = tuple((int(piece['page_line_index']) for piece in pieces))
            if component_indices in seen:
                continue
            seen.add(component_indices)
            combined_bbox = union_bboxes([piece['bbox'] for piece in pieces])
            confidence = sum((float(piece['confidence']) for piece in pieces)) / len(pieces)
            augmented.append({'text': combined_text, 'normalized_text': combined_text.casefold(), 'confidence': round(confidence, 3), 'bbox': combined_bbox, 'block_number': -1, 'paragraph_number': -1, 'line_number': -1, 'page_line_index': -(100000 + min(component_indices)), 'words': [word for piece in pieces for word in piece.get('words', [])], 'synthetic_split_dob_label': True})
    return augmented

def contains_obvious_date_shape(text: str) -> bool:
    """Return True when text visibly resembles a complete date, independent of DOB label."""
    normalized = normalize_text(text)
    month_names = '|'.join(sorted(MONTH_NAME_TO_NUMBER, key=len, reverse=True))
    patterns = (f'\\b\\d{{1,4}}[./-]+\\d{{1,2}}[./-]+\\d{{2,4}}\\b', f'\\b(?:{month_names})\\.?(?:\\s+|[-/]+)\\d{{1,2}}(?:st|nd|rd|th)?(?:[,]?\\s+|[-/]+)\\d{{2,4}}\\b', f'\\b\\d{{1,2}}(?:st|nd|rd|th)?(?:\\s+|[-/]+)(?:{month_names})\\.?(?:[,]?\\s+|[-/]+)\\d{{2,4}}\\b')
    return any((re.search(pattern, normalized, flags=re.IGNORECASE) is not None for pattern in patterns))

def find_standalone_dob_date_candidates(lines: list[dict[str, Any]], page_number: int, image_path: Path, patient_name_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain obvious date-shaped lines as low-priority DOB review fallbacks."""
    candidates: list[dict[str, Any]] = []
    name_bboxes = [candidate.get('context_bbox') for candidate in patient_name_candidates if isinstance(candidate.get('context_bbox'), list)]
    for line in lines:
        if not contains_obvious_date_shape(line['text']):
            continue
        parsed = parse_date_of_birth(line['text'])
        if parsed is None:
            continue
        label_score, _, _ = score_label_patterns(line['text'], PATIENT_NAME_PATTERNS + DATE_OF_BIRTH_PATTERNS)
        if label_score > 0.0:
            continue
        context_bonus = 0.0
        if name_bboxes:
            line_bbox = line['bbox']
            line_center_y = (line_bbox[1] + line_bbox[3]) / 2.0
            nearest_vertical_distance = min((abs(line_center_y - (bbox[1] + bbox[3]) / 2.0) for bbox in name_bboxes))
            if nearest_vertical_distance <= 350:
                context_bonus = DOB_STANDALONE_CONTEXT_BONUS
        score = DOB_STANDALONE_DATE_BASE_SCORE + context_bonus + float(line['confidence']) / 20.0
        candidates.append({'candidate_type': 'date_pattern_only', 'value_kind': 'date_of_birth', 'page_number': page_number, 'image_path': str(image_path), 'score': round(score, 3), 'label_type': 'date-shaped fallback', 'label_text': None, 'value_text': line['text'], 'same_ocr_line': False, 'value_relation': 'standalone_date_pattern', 'label_bbox': line['bbox'], 'value_bbox': line['bbox'], 'context_bbox': line['bbox'], 'label_confidence': None, 'value_confidence': line['confidence'], 'review_rank': None, 'review_crop_bbox': None, 'highlight_regions': [], **parsed})
    return candidates

def score_label_patterns(text: str, patterns: tuple[tuple[re.Pattern[str], float, str], ...]) -> tuple[float, str | None, re.Match[str] | None]:
    """Return the strongest label score, description, and regex match."""
    best_score = 0.0
    best_type: str | None = None
    best_match: re.Match[str] | None = None
    for pattern, score, label_type in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        if score > best_score:
            best_score = score
            best_type = label_type
            best_match = match
    if best_score <= 0.0 and patterns is DATE_OF_BIRTH_PATTERNS and looks_like_dob_label_text(text):
        fallback_match = re.search('.+', text)
        return (116.0, 'date of birth OCR-near-match', fallback_match)
    return (best_score, best_type, best_match)

def clean_candidate_value_tail(text: str, match_end: int) -> str:
    """Return text following one OCR label after stripping separators."""
    tail = normalize_text(text[match_end:])
    tail = re.sub('^[\\s:=|#._-]+', '', tail)
    return tail.strip()

def name_word_parts(word: str) -> list[str]:
    """Return initial-bearing pieces from one name word."""
    normalized = word.strip().strip(',')
    normalized = normalized.replace('–', '-').replace('—', '-')
    parts = [part for part in normalized.split('-') if part]
    return parts

def name_piece_is_plausible(piece: str) -> bool:
    """Return True for a full/abbreviated name piece rather than prose."""
    cleaned = piece.strip().strip('.')
    cleaned = cleaned.replace('’', "'")
    if not cleaned:
        return False
    if len(cleaned) == 1:
        return cleaned.isalpha()
    if not re.fullmatch("[A-Za-z]+(?:'[A-Za-z]+)?", cleaned):
        return False
    folded = cleaned.casefold()
    if folded in PATIENT_NAME_REJECT_WORDS:
        return False
    if folded in PATIENT_NAME_LOWERCASE_PARTICLES:
        return True
    return cleaned.isupper() or cleaned[0].isupper()

def initials_from_name_words(words: list[str]) -> str:
    """Return every initial, including each half of hyphenated names."""
    initials: list[str] = []
    for word in words:
        for part in name_word_parts(word):
            cleaned = part.strip().strip('.').replace('’', "'")
            if cleaned and cleaned[0].isalpha():
                initials.append(cleaned[0].upper())
    return ''.join(initials)

def parse_person_name(text: str) -> dict[str, Any] | None:
    """Parse a short, formatting-plausible patient name."""
    value = normalize_text(text)
    value = NAME_VALUE_STOP_PATTERN.split(value, maxsplit=1)[0]
    value = re.sub('^[\\s:=|#._-]+|[\\s:=|#._-]+$', '', value)
    value = re.sub('^(?:mr|mrs|ms|miss|dr|doctor)\\.?\\s+', '', value, flags=re.IGNORECASE).strip()
    if not value or len(value) > PATIENT_NAME_MAX_CHARACTERS or any((character.isdigit() for character in value)) or re.search('[!?;:]', value):
        return None
    comma_parts = [part.strip() for part in value.split(',', 1)]
    if len(comma_parts) == 2 and comma_parts[0] and comma_parts[1]:
        last_words = comma_parts[0].split()
        first_words = comma_parts[1].split()
        words = [*first_words, *last_words]
    else:
        words = value.split()
    if not PATIENT_NAME_MIN_WORDS <= len(words) <= PATIENT_NAME_MAX_WORDS:
        return None
    if not all((name_piece_is_plausible(part) for word in words for part in name_word_parts(word))):
        return None
    initials = initials_from_name_words(words)
    if len(initials) < 2:
        return None
    display_name = ' '.join(words)
    first_name = words[0]
    last_name = words[-1]
    return {'display_value': display_name, 'name_words': words, 'initials': initials, 'first_name': first_name, 'last_name': last_name, 'first_initial': initials[0], 'last_initial': initials[-1]}

def valid_birth_date(year: int, month: int, day: int) -> tuple[int, int, int] | None:
    """Validate a plausible full date of birth."""
    current_year = datetime.now().year
    if not 1900 <= year <= current_year:
        return None
    try:
        datetime(year, month, day)
    except ValueError:
        return None
    return (year, month, day)

def normalize_two_digit_birth_year(year: int) -> int:
    """Map a two-digit DOB year into the most plausible century."""
    current_year = datetime.now().year
    current_two_digits = current_year % 100
    if not 0 <= year <= 99:
        return year
    return 2000 + year if year <= current_two_digits else 1900 + year

def parsed_birth_date_result(year: int, month: int, day: int, matched_text: str) -> dict[str, Any] | None:
    """Validate and serialize one parsed birth date."""
    normalized_year = normalize_two_digit_birth_year(int(year))
    parsed = valid_birth_date(normalized_year, int(month), int(day))
    if parsed is None:
        return None
    valid_year, valid_month, valid_day = parsed
    return {'display_value': f'{valid_month:02d}/{valid_day:02d}/{valid_year:04d}', 'birth_year': valid_year, 'birth_month': valid_month, 'birth_day': valid_day, 'matched_text': matched_text}

def birth_date_has_time(text: str) -> bool:
    """Recognize a clock time following a date in OCR text."""
    value = normalize_text(text)
    return any(
        DOB_TRAILING_TIME_PATTERN.match(value[match.end():])
        for match in DOB_DATE_SHAPE_PATTERN.finditer(value)
    )


def parse_date_of_birth(text: str) -> dict[str, Any] | None:
    """Parse common DOB formats, including abbreviated months and two-digit years."""
    value = normalize_text(text)
    value = re.sub('^[\\s:=|#._-]+|[\\s:=|#._-]+$', '', value)
    if birth_date_has_time(value):
        return None
    year_token = '\\d{2}|\\d{4}'
    month_names = '|'.join(sorted(MONTH_NAME_TO_NUMBER, key=len, reverse=True))
    numeric_match = re.search(f'\\b(?P<month>\\d{{1,2}})[./-]+(?P<day>\\d{{1,2}})[./-]+(?P<year>{year_token})\\b', value)
    if numeric_match is not None:
        result = parsed_birth_date_result(year=int(numeric_match.group('year')), month=int(numeric_match.group('month')), day=int(numeric_match.group('day')), matched_text=numeric_match.group(0))
        if result is not None:
            return result
    iso_match = re.search(f'\\b(?P<year>\\d{{4}})[./-]+(?P<month>\\d{{1,2}})[./-]+(?P<day>\\d{{1,2}})\\b', value)
    if iso_match is not None:
        result = parsed_birth_date_result(year=int(iso_match.group('year')), month=int(iso_match.group('month')), day=int(iso_match.group('day')), matched_text=iso_match.group(0))
        if result is not None:
            return result
    month_first = re.search(f'\\b(?P<month>{month_names})\\.?(?:\\s+|[-/]+)(?P<day>\\d{{1,2}})(?:st|nd|rd|th)?(?:[,]?\\s+|[-/]+)(?P<year>{year_token})\\b', value, flags=re.IGNORECASE)
    if month_first is not None:
        result = parsed_birth_date_result(year=int(month_first.group('year')), month=MONTH_NAME_TO_NUMBER[month_first.group('month').casefold()], day=int(month_first.group('day')), matched_text=month_first.group(0))
        if result is not None:
            return result
    day_first = re.search(f'\\b(?P<day>\\d{{1,2}})(?:st|nd|rd|th)?(?:\\s+|[-/]+)(?P<month>{month_names})\\.?(?:[,]?\\s+|[-/]+)(?P<year>{year_token})\\b', value, flags=re.IGNORECASE)
    if day_first is not None:
        result = parsed_birth_date_result(year=int(day_first.group('year')), month=MONTH_NAME_TO_NUMBER[day_first.group('month').casefold()], day=int(day_first.group('day')), matched_text=day_first.group(0))
        if result is not None:
            return result
    return None

def patient_name_label_is_rejected(text: str) -> bool:
    """Reject near-miss Patient phrases that are prose rather than a name field."""
    normalized = normalize_text(text)
    return any((pattern.search(normalized) is not None for pattern in PATIENT_NAME_LABEL_REJECT_PATTERNS))

def intervening_text_between(anchor_line: dict[str, Any], candidate_line: dict[str, Any], all_lines: list[dict[str, Any]], page_width: int) -> bool:
    """Return True when OCR text occupies the direct label/value corridor."""
    anchor_bbox = anchor_line['bbox']
    candidate_bbox = candidate_line['bbox']
    if candidate_bbox[1] <= anchor_bbox[3]:
        return False
    corridor_left = max(0, min(anchor_bbox[0], candidate_bbox[0]) - int(round(page_width * 0.03)))
    corridor_right = min(page_width, max(anchor_bbox[2], candidate_bbox[2]) + int(round(page_width * 0.03)))
    for other_line in all_lines:
        if other_line['page_line_index'] in {anchor_line['page_line_index'], candidate_line['page_line_index']}:
            continue
        other_bbox = other_line['bbox']
        other_center_y = (other_bbox[1] + other_bbox[3]) / 2.0
        if not anchor_bbox[3] < other_center_y < candidate_bbox[1]:
            continue
        horizontal_overlap = max(0, min(corridor_right, other_bbox[2]) - max(corridor_left, other_bbox[0]))
        if horizontal_overlap <= 0:
            continue
        overlap_fraction = horizontal_overlap / float(max(1, min(corridor_right - corridor_left, other_bbox[2] - other_bbox[0])))
        if overlap_fraction >= 0.2:
            return True
    return False

def patient_name_lines_share_row(label_line: dict[str, Any], candidate_line: dict[str, Any]) -> bool:
    """Return True when two OCR lines occupy the same visual text row."""
    label_bbox = label_line['bbox']
    candidate_bbox = candidate_line['bbox']
    label_height = max(1, label_bbox[3] - label_bbox[1])
    candidate_height = max(1, candidate_bbox[3] - candidate_bbox[1])
    overlap = max(0, min(label_bbox[3], candidate_bbox[3]) - max(label_bbox[1], candidate_bbox[1]))
    overlap_fraction = overlap / float(min(label_height, candidate_height))
    label_center = (label_bbox[1] + label_bbox[3]) / 2.0
    candidate_center = (candidate_bbox[1] + candidate_bbox[3]) / 2.0
    center_tolerance = max(DEMOGRAPHIC_ALIGNED_ROW_CENTER_TOLERANCE_PIXELS, max(label_height, candidate_height) * DEMOGRAPHIC_ALIGNED_ROW_HEIGHT_TOLERANCE_FRACTION)
    return overlap_fraction >= DEMOGRAPHIC_ALIGNED_ROW_MIN_VERTICAL_OVERLAP or abs(label_center - candidate_center) <= center_tolerance

def patient_name_fragment_is_boundary(text: str) -> bool:
    """Stop row merging at another field label or numeric field value."""
    normalized = normalize_text(text)
    if not normalized or any((character.isdigit() for character in normalized)):
        return True
    label_score, _, _ = score_label_patterns(normalized, PATIENT_NAME_PATTERNS + DATE_OF_BIRTH_PATTERNS)
    return label_score > 0.0 or patient_name_label_is_rejected(normalized)

def augment_patient_name_value_lines(lines: list[dict[str, Any]], page_width: int) -> list[dict[str, Any]]:
    """Join split name fragments on the same row to the right of a name label."""
    augmented = list(lines)
    next_virtual_index = -200000
    maximum_fragment_gap = max(80, int(round(page_width * PATIENT_NAME_ROW_FRAGMENT_MAX_GAP_PAGE_FRACTION)))
    for label_line in lines:
        label_score, _, _ = score_label_patterns(label_line['text'], PATIENT_NAME_PATTERNS)
        if label_score <= 0.0 or patient_name_label_is_rejected(label_line['text']):
            continue
        label_bbox = label_line['bbox']
        right_lines = [line for line in lines if line['page_line_index'] != label_line['page_line_index'] and line['bbox'][0] >= label_bbox[2] - DEMOGRAPHIC_ALIGNED_VALUE_RIGHT_MARGIN_PIXELS and patient_name_lines_share_row(label_line, line)]
        right_lines.sort(key=lambda line: line['bbox'][0])
        if not right_lines:
            continue
        fragments: list[dict[str, Any]] = []
        previous_right = label_bbox[2]
        for line in right_lines[:PATIENT_NAME_ROW_MAX_FRAGMENTS]:
            if patient_name_fragment_is_boundary(line['text']):
                break
            gap = max(0, line['bbox'][0] - previous_right)
            if fragments and gap > maximum_fragment_gap:
                break
            fragments.append(line)
            previous_right = line['bbox'][2]
        if not fragments:
            continue
        selected_count = 0
        for count in range(len(fragments), 0, -1):
            combined_text = normalize_text(' '.join((line['text'] for line in fragments[:count])))
            if parse_person_name(combined_text) is not None:
                selected_count = count
                break
        if selected_count == 0:
            continue
        selected = fragments[:selected_count]
        if selected_count == 1 and parse_person_name(selected[0]['text']) is not None:
            continue
        combined_text = normalize_text(' '.join((line['text'] for line in selected)))
        combined_bbox = union_bboxes([line['bbox'] for line in selected])
        confidence_weight = sum((max(1, len(line['text'])) for line in selected))
        confidence = sum((float(line['confidence']) * max(1, len(line['text'])) for line in selected)) / max(1, confidence_weight)
        augmented.append({'text': combined_text, 'normalized_text': combined_text.casefold(), 'confidence': round(confidence, 3), 'bbox': combined_bbox, 'block_number': -1, 'paragraph_number': -1, 'line_number': -1, 'page_line_index': next_virtual_index, 'words': [word for line in selected for word in line.get('words', [])], 'synthetic_patient_name_value': True})
        next_virtual_index -= 1
    return augmented

def find_demographic_nearby_lines(anchor_line: dict[str, Any], all_lines: list[dict[str, Any]], page_width: int) -> list[tuple[dict[str, Any], float, str]]:
    """Return demographic value lines only when formatting supports the pairing."""
    anchor_bbox = anchor_line['bbox']
    anchor_center_y = (anchor_bbox[1] + anchor_bbox[3]) / 2.0
    anchor_height = max(1, anchor_bbox[3] - anchor_bbox[1])
    below_x_tolerance = max(80, int(round(page_width * PATIENT_NAME_BELOW_START_X_TOLERANCE_PAGE_FRACTION)))
    nearby: list[tuple[dict[str, Any], float, str, int]] = []
    for candidate_line in all_lines:
        candidate_bbox = candidate_line['bbox']
        candidate_height = max(1, candidate_bbox[3] - candidate_bbox[1])
        candidate_center_y = (candidate_bbox[1] + candidate_bbox[3]) / 2.0
        same_line = candidate_line['page_line_index'] == anchor_line['page_line_index']
        overlap_height = max(0, min(anchor_bbox[3], candidate_bbox[3]) - max(anchor_bbox[1], candidate_bbox[1]))
        vertical_overlap_fraction = overlap_height / float(max(1, min(anchor_height, candidate_height)))
        center_y_distance = abs(anchor_center_y - candidate_center_y)
        row_center_tolerance = max(DEMOGRAPHIC_ALIGNED_ROW_CENTER_TOLERANCE_PIXELS, max(anchor_height, candidate_height) * DEMOGRAPHIC_ALIGNED_ROW_HEIGHT_TOLERANCE_FRACTION)
        if same_line:
            relation = 'same_ocr_line'
            priority = 0
            penalty = 0.0
        else:
            rightward = candidate_bbox[0] >= anchor_bbox[2] - DEMOGRAPHIC_ALIGNED_VALUE_RIGHT_MARGIN_PIXELS
            same_visual_row = vertical_overlap_fraction >= DEMOGRAPHIC_ALIGNED_ROW_MIN_VERTICAL_OVERLAP or center_y_distance <= row_center_tolerance
            if rightward and same_visual_row:
                relation = 'directly_right'
                priority = 1
                horizontal_gap = max(0, candidate_bbox[0] - anchor_bbox[2])
                penalty = center_y_distance * 0.16 + horizontal_gap / max(1.0, page_width) * 4.0
            else:
                vertical_gap_below = candidate_bbox[1] - anchor_bbox[3]
                horizontal_overlap = horizontal_overlap_fraction(anchor_bbox, candidate_bbox)
                starts_same_column = abs(candidate_bbox[0] - anchor_bbox[0]) <= below_x_tolerance
                directly_below = 0 <= vertical_gap_below <= PATIENT_NAME_BELOW_MAX_VERTICAL_PIXELS and (starts_same_column or horizontal_overlap >= PATIENT_NAME_BELOW_MIN_HORIZONTAL_OVERLAP)
                if not directly_below:
                    continue
                if intervening_text_between(anchor_line=anchor_line, candidate_line=candidate_line, all_lines=all_lines, page_width=page_width):
                    continue
                relation = 'directly_below'
                priority = 2
                penalty = vertical_gap_below * 0.07 + abs(candidate_bbox[0] - anchor_bbox[0]) / max(1.0, page_width) * 8.0
        nearby.append((candidate_line, float(penalty), relation, priority))
    nearby.sort(key=lambda item: (item[3], item[1], item[0]['page_line_index']))
    return [(line, penalty, relation) for line, penalty, relation, _priority in nearby[:DEMOGRAPHIC_MAX_NEARBY_LINES]]

def find_labeled_value_candidates(lines: list[dict[str, Any]], page_number: int, image_path: Path, page_width: int, label_patterns: tuple[tuple[re.Pattern[str], float, str], ...], parser: Any, value_kind: str) -> list[dict[str, Any]]:
    """Locate plausible demographic field labels and their values.

    Keep reviewable fallbacks when OCR cannot confidently parse a value.
    """
    candidates: list[dict[str, Any]] = []
    for label_line in lines:
        label_score, label_type, label_match = score_label_patterns(label_line['text'], label_patterns)
        if label_score <= 0.0 or label_type is None or label_match is None:
            continue
        if value_kind == 'patient_name' and patient_name_label_is_rejected(label_line['text']):
            continue
        # A timestamped date must not reappear as a label-only fallback.
        if value_kind == 'date_of_birth' and birth_date_has_time(label_line['text']):
            continue
        same_line_tail = clean_candidate_value_tail(label_line['text'], label_match.end())
        same_line_value = parser(same_line_tail) if same_line_tail else None
        value_found = False
        nearby_lines = find_demographic_nearby_lines(anchor_line=label_line, all_lines=lines, page_width=page_width)
        for value_line, distance_penalty, value_relation in nearby_lines:
            same_line = value_line['page_line_index'] == label_line['page_line_index']
            if same_line:
                parsed_value = same_line_value
            else:
                other_label_score, _, _ = score_label_patterns(value_line['text'], PATIENT_NAME_PATTERNS + DATE_OF_BIRTH_PATTERNS)
                if other_label_score > 0.0:
                    continue
                parsed_value = parser(value_line['text'])
            if parsed_value is None:
                continue
            value_found = True
            if same_line:
                location_bonus = 86.0
            elif value_relation == 'directly_right':
                location_bonus = 92.0
            elif value_relation == 'directly_below':
                location_bonus = 76.0
            else:
                location_bonus = 0.0
            confidence_bonus = (label_line['confidence'] + value_line['confidence']) / 20.0
            score = label_score + location_bonus + confidence_bonus - distance_penalty
            context_bbox = union_bboxes([label_line['bbox'], value_line['bbox']])
            candidate = {'candidate_type': 'label_and_value', 'value_kind': value_kind, 'page_number': page_number, 'image_path': str(image_path), 'score': round(score, 3), 'label_type': label_type, 'label_text': label_line['text'], 'value_text': value_line['text'], 'same_ocr_line': same_line, 'value_relation': value_relation, 'label_bbox': label_line['bbox'], 'value_bbox': value_line['bbox'], 'context_bbox': context_bbox, 'label_confidence': label_line['confidence'], 'value_confidence': value_line['confidence'], 'review_rank': None, 'review_crop_bbox': None, 'highlight_regions': [], **parsed_value}
            candidates.append(candidate)
        if value_found:
            continue
        # Reject DOB labels whose value appears on a separate line with a time.
        if value_kind == 'date_of_birth' and any(
            birth_date_has_time(line['text']) for line, _, _ in nearby_lines
        ):
            continue
        candidates.append({'candidate_type': 'label_only', 'value_kind': value_kind, 'page_number': page_number, 'image_path': str(image_path), 'display_value': None, 'score': round(label_score + label_line['confidence'] / 20.0, 3), 'label_type': label_type, 'label_text': label_line['text'], 'value_text': None, 'same_ocr_line': False, 'value_relation': None, 'label_bbox': label_line['bbox'], 'value_bbox': None, 'context_bbox': label_line['bbox'], 'label_confidence': label_line['confidence'], 'value_confidence': None, 'review_rank': None, 'review_crop_bbox': None, 'highlight_regions': []})
    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        reference_bbox = candidate['value_bbox'] if candidate['value_bbox'] is not None else candidate['label_bbox']
        key = (candidate['page_number'], candidate.get('display_value'), tuple(reference_bbox))
        previous = deduplicated.get(key)
        if previous is None or candidate['score'] > previous['score']:
            deduplicated[key] = candidate
    return sorted(deduplicated.values(), key=lambda candidate: candidate['score'], reverse=True)

def candidate_text_key(candidate: dict[str, Any], label_role: str) -> str:
    """Normalize OCR candidate text for duplicate-label rejection."""
    if label_role == "date_of_birth_label" and candidate.get("display_value"):
        # Equivalent date formats represent the same DOB, including across pages.
        return "dob:" + str(candidate["display_value"])

    fields = ("text",) if label_role == "ocr_guide" else ("label_text", "value_text")
    return "text:" + normalized_match_text(" ".join(
        str(candidate.get(field) or "") for field in fields
    ))


def group_ocr_candidates_by_page(
    candidates: list[dict[str, Any]],
    maximum_count: int,
    label_role: str,
    value_role: str | None = None,
) -> list[dict[str, Any]]:
    """Retain the shortest descending-score OCR-hit prefix spanning up to five pages.

    Group retained hits by page; exclude lower-ranked hits beyond the cutoff.
    """
    if maximum_count < 0:
        raise ValueError("Maximum OCR candidates cannot be negative; 0 retains all.")

    # Scan OCR hits in score order, stopping at the first hit on the final page.
    # Repeated pages still contribute highlights but do not consume review slots.
    pages: dict[int, list[dict[str, Any]]] = {}
    seen_text: set[str] = set()
    for hit in sorted(candidates, key=lambda item: item["score"], reverse=True):
        # Keep only the best-ranked occurrence of identical text (or the same DOB).
        key = candidate_text_key(hit, label_role)
        if key in seen_text:
            continue
        seen_text.add(key)
        pages.setdefault(int(hit["page_number"]), []).append(hit)
        if maximum_count and len(pages) == maximum_count:
            break

    retained: list[dict[str, Any]] = []
    for rank, hits in enumerate(pages.values(), start=1):
        candidate = dict(hits[0])  # Highest-scoring hit determines the page's rank.
        regions: list[dict[str, Any]] = []
        ocr_hits: list[dict[str, Any]] = []
        seen_regions: set[tuple[str, tuple[int, ...]]] = set()
        seen_hits: set[tuple[tuple[int, ...], str]] = set()

        for hit in hits:
            label_bbox = hit.get("bbox") if label_role == "ocr_guide" else hit.get("label_bbox")
            value_bbox = hit.get("value_bbox") if value_role else None

            # Avoid drawing the same OCR location twice when multiple patterns match.
            for role, bbox in ((label_role, label_bbox), (value_role, value_bbox)):
                if role is None or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                key = (role, tuple(int(value) for value in bbox))
                if key not in seen_regions:
                    regions.append({"role": role, "bbox": list(bbox)})
                    seen_regions.add(key)

            if not isinstance(label_bbox, list) or len(label_bbox) != 4:
                continue
            hit_text = str(hit.get("text") or hit.get("matched_text") or hit.get("display_value") or "")
            key = (tuple(int(value) for value in label_bbox), hit_text)
            if key not in seen_hits:
                ocr_hits.append({"bbox": list(label_bbox), "text": hit_text, "score": hit["score"]})
                seen_hits.add(key)

        if not regions:
            continue

        # Frame only the retained hits, not every matching heading on the page.
        image_height, image_width = read_grayscale_image(Path(candidate["image_path"])).shape
        bounds = union_bboxes([region["bbox"] for region in regions])
        focus = expand_review_bbox(bounds, image_width, image_height)
        if label_role == "ocr_guide":
            # Show page-width context and substantial space beneath graph headings.
            minimum_height = round(image_height * 0.65)
            top = max(0, min(focus[1] - round(image_height * 0.12), image_height - minimum_height))
            focus = [0, top, image_width, max(focus[3], top + minimum_height)]
        candidate.update({
            "review_rank": rank,
            "review_crop_bbox": focus,
            "highlight_regions": regions,
            "ocr_hits": ocr_hits,
            "ocr_hit_count": len(ocr_hits),
        })
        retained.append(candidate)

    return retained


SETTINGS_ADJACENT_WORDS = frozenset({
    'setting', 'settings', 'program', 'programmed', 'programming',
    'parameter', 'parameters', 'configuration', 'configured',
    'threshold', 'thresholds', 'limit', 'limits', 'alert', 'alerts',
    'notification', 'notifications', 'sensitivity', 'zone', 'zones',
})


def has_settings_adjacent_words(
    text: str,
    anchors: set[str] | frozenset[str],
    maximum_distance: int = 4,
) -> bool:
    """Reject metric-looking text when settings/programming words are nearby."""
    normalized = normalize_text(text).casefold()
    words = re.sub(r'[^a-z0-9]+', ' ', normalized).split()
    setting_positions = [
        index for index, word in enumerate(words)
        if word in SETTINGS_ADJACENT_WORDS
    ]
    anchor_positions = [
        index for index, word in enumerate(words)
        if word in anchors
    ]
    return any(
        abs(setting_index - anchor_index) <= maximum_distance
        for setting_index in setting_positions
        for anchor_index in anchor_positions
    )


def score_afib_label(text: str) -> tuple[float, str | None]:
    """Score explicit AF/AT-AF burden labels while rejecting settings labels."""
    if has_settings_adjacent_words(
        text,
        frozenset({'burden', 'afib', 'af', 'ataf', 'fibrillation'}),
    ):
        return (0.0, None)
    normalized = normalize_text(text).casefold()
    compact = re.sub('[^a-z0-9]+', '', normalized)
    has_burden = 'burden' in normalized or 'burden' in compact
    if has_burden and 'atrialfibrillation' in compact:
        return (125.0, 'atrial fibrillation burden')
    if has_burden and 'ataf' in compact:
        return (120.0, 'AT/AF burden')
    if has_burden and 'afib' in compact:
        return (115.0, 'AFib burden')
    if has_burden and re.search('\\ba[\\s-]*fib\\b', normalized):
        return (110.0, 'A-Fib burden')
    if has_burden and re.search('\\baf\\b', normalized):
        return (105.0, 'AF burden')
    at_af = '\\bat\\s*(?:/|\\\\|-)\\s*af\\b'
    if re.search(f'\\btime\\s+(?:in|of)\\s+(?:{at_af})', normalized):
        return (110.0, 'time in AT/AF')
    if re.search(f'(?:{at_af})\\s*(?:time|burden|%)', normalized):
        return (105.0, 'AT/AF metric')
    if re.search(at_af, normalized):
        return (100.0, 'AT/AF')
    if re.search('\\bat\\s*%', normalized):
        return (100.0, 'AT%')
    if re.search('\\baf\\s*%', normalized):
        return (100.0, 'AF%')
    if re.search('\\btime\\s+(?:in|of)\\s+(?:af|a[\\s-]*fib|atrial\\s+fibrillation)\\b', normalized):
        return (95.0, 'time in AF')
    if re.search('\\b(?:atrial\\s+tachycardia|atrial\\s+tachyarrhythmia)\\s*(?:/|\\\\|-)\\s*atrial\\s+fibrillation\\b', normalized):
        return (100.0, 'atrial tachycardia/fibrillation')
    return (0.0, None)

def extract_percentages(text: str) -> list[dict[str, Any]]:
    """Extract valid explicit percentages from one OCR text line."""
    percentages: list[dict[str, Any]] = []
    for match in PERCENT_PATTERN.finditer(text):
        numeric_text = match.group('value').replace(',', '.')
        try:
            numeric_value = float(numeric_text)
        except ValueError:
            continue
        if not 0.0 <= numeric_value <= 100.0:
            continue
        comparator = match.group('comparator') or ''
        comparator = comparator.replace('≤', '<=').replace('≥', '>=')
        display_value = f'{comparator}{numeric_text}%'
        percentages.append({'value_percent': numeric_value, 'comparator': comparator or None, 'display_value': display_value, 'matched_text': match.group(0).strip(), 'match_start': match.start(), 'match_end': match.end()})
    return percentages

def horizontal_overlap_fraction(first_bbox: list[int], second_bbox: list[int]) -> float:
    """Measure horizontal overlap relative to the narrower bounding box."""
    overlap = max(0, min(first_bbox[2], second_bbox[2]) - max(first_bbox[0], second_bbox[0]))
    first_width = max(1, first_bbox[2] - first_bbox[0])
    second_width = max(1, second_bbox[2] - second_bbox[0])
    return overlap / float(min(first_width, second_width))

def vertical_gap(first_bbox: list[int], second_bbox: list[int]) -> int:
    """Return zero for vertically overlapping boxes, otherwise their gap."""
    if first_bbox[3] < second_bbox[1]:
        return second_bbox[1] - first_bbox[3]
    if second_bbox[3] < first_bbox[1]:
        return first_bbox[1] - second_bbox[3]
    return 0

def center_x(bbox: list[int]) -> float:
    """Return the horizontal center of one bounding box."""
    return (bbox[0] + bbox[2]) / 2.0

def find_nearby_lines(anchor_line: dict[str, Any], all_lines: list[dict[str, Any]], page_width: int) -> list[tuple[dict[str, Any], float]]:
    """Return spatially plausible lines near an AFib burden label."""
    anchor_bbox = anchor_line['bbox']
    maximum_center_distance = page_width * AFIB_NEARBY_HORIZONTAL_PAGE_FRACTION
    nearby: list[tuple[dict[str, Any], float]] = []
    for candidate_line in all_lines:
        candidate_bbox = candidate_line['bbox']
        gap = vertical_gap(anchor_bbox, candidate_bbox)
        overlap = horizontal_overlap_fraction(anchor_bbox, candidate_bbox)
        horizontal_distance = abs(center_x(anchor_bbox) - center_x(candidate_bbox))
        is_same_line = candidate_line['page_line_index'] == anchor_line['page_line_index']
        is_spatially_near = gap <= AFIB_NEARBY_VERTICAL_PIXELS and (overlap > 0.0 or horizontal_distance <= maximum_center_distance)
        if not is_same_line and (not is_spatially_near):
            continue
        distance_penalty = gap * 0.12 + horizontal_distance / max(1.0, page_width) * 35.0
        if is_same_line:
            distance_penalty = 0.0
        nearby.append((candidate_line, distance_penalty))
    nearby.sort(key=lambda item: (item[1], item[0]['page_line_index']))
    return nearby[:AFIB_MAX_NEARBY_LINES]

def find_afib_candidates(lines: list[dict[str, Any]], page_number: int, image_path: Path, page_width: int) -> list[dict[str, Any]]:
    """Find explicit or label-only AFib burden candidates on one page."""
    candidates: list[dict[str, Any]] = []
    for label_line in lines:
        label_score, label_type = score_afib_label(label_line['text'])
        if label_score <= 0.0 or label_type is None:
            continue
        nearby_lines = find_nearby_lines(anchor_line=label_line, all_lines=lines, page_width=page_width)
        percentage_found = False
        for value_line, distance_penalty in nearby_lines:
            percentages = extract_percentages(value_line['text'])
            for percentage in percentages:
                percentage_found = True
                same_line = value_line['page_line_index'] == label_line['page_line_index']
                location_bonus = 80.0 if same_line else 48.0
                confidence_bonus = (label_line['confidence'] + value_line['confidence']) / 20.0
                score = label_score + location_bonus + confidence_bonus - distance_penalty
                context_bbox = union_bboxes([label_line['bbox'], value_line['bbox']])
                candidates.append({'candidate_type': 'label_and_percentage', 'page_number': page_number, 'image_path': str(image_path), 'value_percent': percentage['value_percent'], 'comparator': percentage['comparator'], 'display_value': percentage['display_value'], 'score': round(score, 3), 'label_type': label_type, 'label_text': label_line['text'], 'value_text': value_line['text'], 'same_ocr_line': same_line, 'label_bbox': label_line['bbox'], 'value_bbox': value_line['bbox'], 'context_bbox': context_bbox, 'label_confidence': label_line['confidence'], 'value_confidence': value_line['confidence'], 'review_rank': None, 'review_crop_bbox': None, 'highlight_regions': []})
        if percentage_found:
            continue
        label_only_score = label_score + label_line['confidence'] / 20.0
        candidates.append({'candidate_type': 'label_only', 'page_number': page_number, 'image_path': str(image_path), 'value_percent': None, 'comparator': None, 'display_value': None, 'score': round(label_only_score, 3), 'label_type': label_type, 'label_text': label_line['text'], 'value_text': None, 'same_ocr_line': False, 'label_bbox': label_line['bbox'], 'value_bbox': None, 'context_bbox': label_line['bbox'], 'label_confidence': label_line['confidence'], 'value_confidence': None, 'review_rank': None, 'review_crop_bbox': None, 'highlight_regions': []})
    deduplicated: dict[tuple[int, float | None, str | None, tuple[int, int, int, int]], dict[str, Any]] = {}
    for candidate in candidates:
        reference_bbox = candidate['value_bbox'] if candidate['value_bbox'] is not None else candidate['label_bbox']
        key = (candidate['page_number'], candidate['value_percent'], candidate['comparator'], tuple(reference_bbox))
        previous = deduplicated.get(key)
        if previous is None or candidate['score'] > previous['score']:
            deduplicated[key] = candidate
    ordered = sorted(deduplicated.values(), key=lambda candidate: candidate['score'], reverse=True)
    return ordered

def normalized_match_text(text: str) -> str:
    """Normalize OCR text for tolerant phrase comparison."""
    normalized = normalize_text(text).casefold()
    normalized = re.sub('[^a-z0-9]+', ' ', normalized)
    normalized = re.sub('\\s+', ' ', normalized)
    return normalized.strip()

def phrase_windows(words: list[str], phrase_word_count: int) -> list[str]:
    """Return similarly sized word windows for one fuzzy phrase comparison."""
    if not words:
        return []
    minimum_size = phrase_word_count
    maximum_size = min(len(words), phrase_word_count + 1)
    windows: list[str] = []
    for size in range(minimum_size, maximum_size + 1):
        for start_index in range(0, len(words) - size + 1):
            windows.append(' '.join(words[start_index:start_index + size]))
    return windows

def fuzzy_window_has_anchor(window: str, phrase: str) -> bool:
    """Require key words to survive fuzzy matching instead of disappearing semantically."""
    window_words = window.split()
    phrase_words = phrase.split()

    for anchor, minimum_length in (
        ("heart", 3),
        ("activity", 5),
        ("pulse", 3),
        ("ventricular", 6),
    ):
        if anchor not in phrase_words:
            continue

        if not any(
            len(word) >= minimum_length
            and difflib.SequenceMatcher(
                None,
                word,
                anchor,
                autojunk=False,
            ).ratio() >= 0.60
            for word in window_words
        ):
            return False

    return True


def score_fuzzy_phrase_match(text: str, phrases: tuple[tuple[str, float, str], ...], minimum_ratio: float) -> tuple[float, str | None]:
    """Score conservative OCR near-misses while preserving semantic anchor words."""
    normalized = normalized_match_text(text)
    if len(normalized) < HEART_RATE_FUZZY_MINIMUM_TOKEN_LENGTH:
        return (0.0, None)

    words = normalized.split()
    best_score = 0.0
    best_reason = None

    for phrase, phrase_score, reason in phrases:
        phrase_words = phrase.split()
        for window in phrase_windows(words, len(phrase_words)):
            if not fuzzy_window_has_anchor(window, phrase):
                continue

            ratio = difflib.SequenceMatcher(
                None,
                window,
                phrase,
                autojunk=False,
            ).ratio()
            if ratio < minimum_ratio:
                continue

            score = phrase_score * ratio
            if score > best_score:
                best_score = score
                best_reason = reason

    return (best_score, best_reason)

def rejection_reasons(text: str, patterns: tuple[tuple[re.Pattern[str], str], ...]) -> list[str]:
    """Return known false-match reasons for one OCR guide line."""
    return [reason for pattern, reason in patterns if pattern.search(text)]

def is_ventricular_activity_near_match(text: str) -> bool:
    """Recognize OCR-near-misses of ventricular activity for HR routing."""
    normalized = normalized_match_text(text)
    words = normalized.split()
    if len(words) < 2:
        return False

    for index in range(len(words) - 1):
        ventricular_ratio = difflib.SequenceMatcher(
            None,
            words[index],
            'ventricular',
            autojunk=False,
        ).ratio()
        activity_ratio = difflib.SequenceMatcher(
            None,
            words[index + 1],
            'activity',
            autojunk=False,
        ).ratio()
        if ventricular_ratio >= 0.68 and activity_ratio >= 0.68:
            return True

    return False

def is_heart_rate_histogram_guide(text: str) -> bool:
    """Recognize HR histogram headings, including common OCR letter substitutions."""
    words = normalized_match_text(text).split()
    histogram_positions = [
        index for index, word in enumerate(words)
        if 7 <= len(word) <= 12 and
        difflib.SequenceMatcher(None, word, 'histogram', autojunk=False).ratio() >= 0.76
    ]
    for position in histogram_positions:
        nearby = words[max(0, position - 4):position + 5]
        if 'hr' in nearby or any(
            nearby[index:index + 2] == ['h', 'r']
            for index in range(len(nearby) - 1)
        ):
            return True
        for index in range(len(nearby) - 1):
            first, second = nearby[index:index + 2]
            if (
                difflib.SequenceMatcher(None, first, 'heart', autojunk=False).ratio() >= 0.73
                or difflib.SequenceMatcher(None, first, 'ventricular', autojunk=False).ratio() >= 0.72
                or first in {'vent', 'v'}
            ) and difflib.SequenceMatcher(None, second, 'rate', autojunk=False).ratio() >= 0.72:
                return True
    return False

def ranking_adjustment(text: str, target: str) -> tuple[float, list[str]] | None:
    """Apply metric-specific phrase boosts, penalties and hard exclusions."""
    rules = [OCR_RANKING_RULES[target]]
    if "exact_titles" in OCR_RANKING_RULES[target]:
        rules.insert(0, OCR_RANKING_RULES["_graph_common"])

    normalized = f" {normalized_match_text(text)} "
    adjustment = 0.0
    reasons = []
    for rule_set in rules:
        if rejection_reasons(text, rule_set["reject"]):
            return None
        for category in ("prefer", "penalize"):
            for phrase, points in rule_set[category].items():
                if f" {phrase} " in normalized:
                    # In activity labels, "min/day" is a time unit, not "minimum".
                    if phrase == "min" and re.search(r"\bmin\s*(?:/|per)\s*(?:d|day)\b", text, re.IGNORECASE):
                        continue
                    adjustment += points
                    reasons.append(f"{points:+g} {phrase}")

    bonus = OCR_RANKING_RULES[target].get("exact_titles", {}).get(normalized.strip())
    if bonus is not None:
        adjustment += bonus
        reasons.append(f"+{bonus:g} standalone graph heading")
    return adjustment, reasons


def apply_metric_candidate_ranking(candidate: dict[str, Any], target: str) -> bool:
    """Apply shared ranking rules to demographic and AFib candidates."""
    text = str(candidate.get("label_text") or "")
    adjusted = ranking_adjustment(text, target)
    if adjusted is None:
        return False
    points, reasons = adjusted
    candidate["score"] = round(float(candidate["score"]) + points, 3)
    if reasons:
        candidate["ranking_reasons"] = reasons
    return True


def score_graph_guide_line(line: dict[str, Any], target_patterns: tuple[tuple[re.Pattern[str], float, str], ...]) -> dict[str, Any] | None:
    """Score a potential graph heading using its strongest phrase and modifiers."""
    text = line["text"]
    heart_rate = target_patterns is HEART_RATE_PATTERNS
    target = "average_heart_rate" if heart_rate else "average_daily_activity"

    # Take the strongest phrase, not the sum of overlapping matches such as
    # "average ventricular rate" + "ventricular rate" + "rate".
    matches = [(points, reason) for pattern, points, reason in target_patterns if pattern.search(text)]
    best_score, best_reason = max(matches, default=(0.0, ""), key=lambda item: item[0])
    reasons = [best_reason] if best_reason else []

    if heart_rate:
        if is_heart_rate_histogram_guide(text):
            best_score = max(best_score, OCR_RANKING_RULES[target]["histogram_title_base"])
            reasons.append("heart-rate histogram heading")
        if has_settings_adjacent_words(text, frozenset({'rate', 'hr', 'bpm', 'heart', 'ventricular'})):
            return None
        if is_ventricular_activity_near_match(text):
            best_score = max(best_score, 96.0)
            reasons.append('near-match ventricular activity')
        if best_score <= 0.0 or best_reason not in HEART_RATE_EXPLICIT_REASONS:
            fuzzy_score, fuzzy_reason = score_fuzzy_phrase_match(
                text, HEART_RATE_FUZZY_PHRASES, HEART_RATE_FUZZY_MINIMUM_RATIO,
            )
            if fuzzy_score > best_score and fuzzy_reason:
                best_score = fuzzy_score
                reasons.append(fuzzy_reason)
    else:
        if has_settings_adjacent_words(text, frozenset({'activity'})) or is_ventricular_activity_near_match(text):
            return None
        if best_score <= 0.0 or best_reason == 'activity time unit':
            fuzzy_score, fuzzy_reason = score_fuzzy_phrase_match(
                text, ACTIVITY_FUZZY_PHRASES, ACTIVITY_FUZZY_MINIMUM_RATIO,
            )
            if fuzzy_score > best_score and fuzzy_reason:
                best_score = fuzzy_score
                reasons.append(fuzzy_reason)

    # Graph-looking words must not turn generic "rate" or "min/day" units
    # into metric candidates without a recognizable heart-rate/activity phrase.
    if best_score < (70.0 if heart_rate else 60.0):
        return None
    adjusted = ranking_adjustment(text, target)
    if adjusted is None:
        return None
    preference_points, preference_reasons = adjusted
    score = best_score + preference_points + line['confidence'] / 10.0
    return {
        'score': round(score, 3), 'text': text, 'bbox': line['bbox'],
        'ocr_confidence': line['confidence'],
        'matched_reasons': list(dict.fromkeys(reasons + preference_reasons)),
    }

def graph_guide_context_bonus(
    candidate: dict[str, Any],
    lines: list[dict[str, Any]],
    target_patterns: tuple[tuple[re.Pattern[str], float, str], ...],
) -> tuple[float, list[str]]:
    """Reward supporting nearby labels without creating extra candidates."""
    bbox = candidate["bbox"]
    center_y = (bbox[1] + bbox[3]) / 2.0
    guide_height = max(1, bbox[3] - bbox[1])
    maximum_vertical_distance = max(
        GRAPH_GUIDE_CONTEXT_VERTICAL_PIXELS,
        guide_height * 6,
    )

    target = "average_heart_rate" if target_patterns is HEART_RATE_PATTERNS else "average_daily_activity"
    support_patterns = OCR_RANKING_RULES[target]["nearby"]

    bonus = 0.0
    reasons: list[str] = []

    for line in lines:
        if line["bbox"] == candidate["bbox"] and line["text"] == candidate["text"]:
            continue

        line_bbox = line["bbox"]
        line_center_y = (line_bbox[1] + line_bbox[3]) / 2.0
        if abs(line_center_y - center_y) > maximum_vertical_distance:
            continue

        for pattern, score, reason in support_patterns:
            if re.search(pattern, line["text"], re.IGNORECASE) and reason not in reasons:
                bonus += score
                reasons.append(reason)

    return (min(bonus, 48.0), reasons)


def rank_graph_guides(lines: list[dict[str, Any]], target_patterns: tuple[tuple[re.Pattern[str], float, str], ...], page_number: int, image_path: Path) -> list[dict[str, Any]]:
    """Rank metric heading candidates and incorporate nearby context."""
    candidates: list[dict[str, Any]] = []
    target = "average_heart_rate" if target_patterns is HEART_RATE_PATTERNS else "average_daily_activity"
    minimum_score = OCR_RANKING_RULES[target]["minimum_score"]

    for line in lines:
        candidate = score_graph_guide_line(
            line=line,
            target_patterns=target_patterns,
        )
        if candidate is None:
            continue

        bonus, reasons = graph_guide_context_bonus(
            candidate,
            lines,
            target_patterns,
        )
        candidate["score"] = round(float(candidate["score"]) + bonus, 3)
        candidate["matched_reasons"] = list(
            dict.fromkeys(candidate["matched_reasons"] + reasons)
        )

        if candidate["score"] < minimum_score:
            continue

        candidate["page_number"] = page_number
        candidate["image_path"] = str(image_path)
        candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return candidates


def process_page(image_path: Path, page_number: int, language: str, page_segmentation_mode: int, timeout_seconds: int, ocr_scale: float) -> dict[str, Any]:
    """OCR one deskewed page and return all target-related results."""
    page_start = time.perf_counter()
    image = read_grayscale_image(image_path)
    lines, ocr_metadata = run_page_ocr(image=image, language=language, page_segmentation_mode=page_segmentation_mode, timeout_seconds=timeout_seconds, ocr_scale=ocr_scale)
    name_search_lines = augment_patient_name_value_lines(lines=lines, page_width=ocr_metadata['source_width_pixels'])
    patient_name_candidates = find_labeled_value_candidates(lines=name_search_lines, page_number=page_number, image_path=image_path, page_width=ocr_metadata['source_width_pixels'], label_patterns=PATIENT_NAME_PATTERNS, parser=parse_person_name, value_kind='patient_name')
    dob_search_lines = augment_split_dob_label_lines(lines)
    date_of_birth_candidates = find_labeled_value_candidates(lines=dob_search_lines, page_number=page_number, image_path=image_path, page_width=ocr_metadata['source_width_pixels'], label_patterns=DATE_OF_BIRTH_PATTERNS, parser=parse_date_of_birth, value_kind='date_of_birth')
    date_of_birth_candidates.extend(find_standalone_dob_date_candidates(lines=lines, page_number=page_number, image_path=image_path, patient_name_candidates=patient_name_candidates))
    date_of_birth_candidates.sort(key=lambda candidate: candidate['score'], reverse=True)
    afib_candidates = find_afib_candidates(lines=lines, page_number=page_number, image_path=image_path, page_width=ocr_metadata['source_width_pixels'])
    heart_rate_guides = rank_graph_guides(lines=lines, target_patterns=HEART_RATE_PATTERNS, page_number=page_number, image_path=image_path)
    activity_guides = rank_graph_guides(lines=lines, target_patterns=ACTIVITY_PATTERNS, page_number=page_number, image_path=image_path)
    for target, candidates in (
        ("patient_name", patient_name_candidates),
        ("date_of_birth", date_of_birth_candidates),
        ("afib_burden", afib_candidates),
    ):
        candidates[:] = [item for item in candidates if apply_metric_candidate_ranking(item, target)]
        candidates.sort(key=lambda item: item["score"], reverse=True)

    elapsed_seconds = time.perf_counter() - page_start
    return {'page_number': page_number, 'image_path': str(image_path), 'ocr_seconds': round(elapsed_seconds, 6), 'ocr_metadata': ocr_metadata, 'patient_name_candidates': patient_name_candidates, 'date_of_birth_candidates': date_of_birth_candidates, 'afib_burden_candidates': afib_candidates, 'average_heart_rate_guides': heart_rate_guides, 'average_daily_activity_guides': activity_guides}

def analyze_page_folder(input_dir: Path, language: str, page_segmentation_mode: int, timeout_seconds: int, ocr_scale: float, top_graph_guides: int, top_afib_candidates: int, top_demographic_candidates: int=DEFAULT_TOP_DEMOGRAPHIC_CANDIDATES) -> dict[str, Any]:
    """OCR the deskewed pages and combine ranked candidates by metric."""
    image_paths = list_image_paths(input_dir)
    if not image_paths:
        raise ValueError(f'No supported page images found in: {input_dir}')
    batch_start = time.perf_counter()
    page_results: list[dict[str, Any]] = []
    all_patient_name_candidates: list[dict[str, Any]] = []
    all_date_of_birth_candidates: list[dict[str, Any]] = []
    all_afib_candidates: list[dict[str, Any]] = []
    all_heart_rate_guides: list[dict[str, Any]] = []
    all_activity_guides: list[dict[str, Any]] = []
    for page_number, image_path in enumerate(image_paths, start=1):
        page_result = process_page(image_path=image_path, page_number=page_number, language=language, page_segmentation_mode=page_segmentation_mode, timeout_seconds=timeout_seconds, ocr_scale=ocr_scale)
        page_results.append(page_result)
        all_patient_name_candidates.extend(page_result['patient_name_candidates'])
        all_date_of_birth_candidates.extend(page_result['date_of_birth_candidates'])
        all_afib_candidates.extend(page_result['afib_burden_candidates'])
        all_heart_rate_guides.extend(page_result['average_heart_rate_guides'])
        all_activity_guides.extend(page_result['average_daily_activity_guides'])
        print(f"OCR page {page_number}/{len(image_paths)}: {image_path.name} | lines={page_result['ocr_metadata']['recognized_line_count']} | name={len(page_result['patient_name_candidates'])} | DOB={len(page_result['date_of_birth_candidates'])} | AFib={len(page_result['afib_burden_candidates'])} | HR={len(page_result['average_heart_rate_guides'])} | activity={len(page_result['average_daily_activity_guides'])} | {page_result['ocr_seconds']:.3f}s")
    for candidates in (all_patient_name_candidates, all_date_of_birth_candidates, all_afib_candidates):
        candidates.sort(key=lambda candidate: candidate['score'], reverse=True)
    selected_patient_name = next((candidate for candidate in all_patient_name_candidates if candidate.get('display_value') and candidate['score'] >= MIN_ACCEPTED_PATIENT_NAME_SCORE), None)
    selected_date_of_birth = next((candidate for candidate in all_date_of_birth_candidates if candidate.get('birth_year') is not None and candidate['score'] >= MIN_ACCEPTED_DATE_OF_BIRTH_SCORE), None)
    selected_afib_candidate = next((candidate for candidate in all_afib_candidates if candidate['value_percent'] is not None and candidate['score'] >= MIN_ACCEPTED_AFIB_SCORE), None)
    retained_patient_names = group_ocr_candidates_by_page(all_patient_name_candidates, top_demographic_candidates, "patient_name_label", "patient_name_value")
    retained_dates_of_birth = group_ocr_candidates_by_page(all_date_of_birth_candidates, top_demographic_candidates, "date_of_birth_label", "date_of_birth_value")
    retained_afib_candidates = group_ocr_candidates_by_page(all_afib_candidates, top_afib_candidates, "afib_label", "afib_value")
    heart_rate_guides = group_ocr_candidates_by_page(all_heart_rate_guides, top_graph_guides, "ocr_guide")
    activity_guides = group_ocr_candidates_by_page(all_activity_guides, top_graph_guides, "ocr_guide")
    total_seconds = time.perf_counter() - batch_start
    return {'input_dir': str(input_dir), 'page_count': len(page_results), 'ocr_language': language, 'page_segmentation_mode': page_segmentation_mode, 'ocr_scale': ocr_scale, 'total_ocr_seconds': round(total_seconds, 6), 'patient_name': {'status': 'found' if selected_patient_name is not None else 'not_found', 'selected': selected_patient_name, 'detected_candidate_count': len(all_patient_name_candidates), 'candidate_count': len(retained_patient_names), 'candidate_limit': top_demographic_candidates, 'candidates': retained_patient_names}, 'date_of_birth': {'status': 'found' if selected_date_of_birth is not None else 'not_found', 'selected': selected_date_of_birth, 'detected_candidate_count': len(all_date_of_birth_candidates), 'candidate_count': len(retained_dates_of_birth), 'candidate_limit': top_demographic_candidates, 'candidates': retained_dates_of_birth}, 'afib_burden': {'status': 'found' if selected_afib_candidate is not None else 'not_found', 'selected': selected_afib_candidate, 'detected_candidate_count': len(all_afib_candidates), 'candidate_count': len(retained_afib_candidates), 'candidate_limit': top_afib_candidates, 'candidates': retained_afib_candidates}, 'average_heart_rate': {'status': 'guides_found' if heart_rate_guides else 'not_found', 'guides': heart_rate_guides}, 'average_daily_activity': {'status': 'guides_found' if activity_guides else 'not_found', 'guides': activity_guides}, 'pages': page_results}

def running_frozen() -> bool:
    """Return True when OCR is running inside the packaged Mullenizer executable."""
    return bool(getattr(sys, 'frozen', False))

def configure_tesseract(explicit_command: str | None) -> tuple[Path, Path | None]:
    """Configure Tesseract appropriately for packaged and source execution."""
    frozen = running_frozen()
    selected: Path | None = None
    tessdata_dir: Path | None = None
    if explicit_command:
        selected = Path(explicit_command).expanduser()
        if not selected.is_absolute():
            selected = Path.cwd() / selected
        selected = selected.resolve()
    elif frozen:
        selected = (Path(sys.executable).resolve().parent / 'tesseract' / 'tesseract.exe').resolve()
    else:
        system_command = shutil.which('tesseract')
        if system_command:
            selected = Path(system_command).resolve()
    if selected is None or not selected.is_file():
        if frozen:
            expected = Path(sys.executable).resolve().parent / 'tesseract' / 'tesseract.exe'
            raise RuntimeError(f'Bundled Tesseract OCR executable was not found. Expected: {expected}')
        raise RuntimeError('Tesseract OCR executable was not found on the system PATH. Install Tesseract and verify that `tesseract --version` works.')
    pytesseract.pytesseract.tesseract_cmd = str(selected)
    if frozen:
        tessdata_dir = selected.parent / 'tessdata'
        english_data = tessdata_dir / 'eng.traineddata'
        if not english_data.is_file():
            raise RuntimeError(f'Bundled Tesseract English language data was not found. Expected: {english_data}')
        os.environ['TESSDATA_PREFIX'] = str(tessdata_dir)
    return (selected, tessdata_dir)

def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save OCR target results as readable UTF-8 JSON."""
    output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def main() -> None:
    """Parse OCR CLI options, analyze pages and save the ranked result JSON."""
    parser = argparse.ArgumentParser(description='OCR deskewed Mullenizer pages for AFib burden and graph guides.')
    parser.add_argument('input_dir', help='Directory containing deskewed page images.')
    parser.add_argument('--out', default=DEFAULT_OUTPUT_PATH, help=f'Output JSON path. Default: {DEFAULT_OUTPUT_PATH}')
    parser.add_argument('--top-afib-candidates', type=int, default=DEFAULT_TOP_AFIB_CANDIDATES, help=f'Maximum AFib candidates to retain; 0 retains all. Default: {DEFAULT_TOP_AFIB_CANDIDATES}')
    parser.add_argument('--top-demographic-candidates', type=int, default=DEFAULT_TOP_DEMOGRAPHIC_CANDIDATES, help=f'Maximum patient-name and DOB candidates to retain; 0 retains all. Default: {DEFAULT_TOP_DEMOGRAPHIC_CANDIDATES}')
    parser.add_argument('--lang', default=DEFAULT_OCR_LANGUAGE, help=f'Tesseract OCR language. Default: {DEFAULT_OCR_LANGUAGE}')
    parser.add_argument('--psm', type=int, default=DEFAULT_PAGE_SEGMENTATION_MODE, help=f'Tesseract page-segmentation mode. Default: {DEFAULT_PAGE_SEGMENTATION_MODE}')
    parser.add_argument('--ocr-scale', type=float, default=DEFAULT_OCR_SCALE, help=f'Temporary scale applied before OCR; boxes are mapped back. Default: {DEFAULT_OCR_SCALE}')
    parser.add_argument('--timeout', type=int, default=DEFAULT_PAGE_TIMEOUT_SECONDS, help=f'Maximum Tesseract runtime per page in seconds. Default: {DEFAULT_PAGE_TIMEOUT_SECONDS}')
    parser.add_argument('--top-guides', type=int, default=DEFAULT_TOP_GRAPH_GUIDES, help=f'Maximum distinct OCR guide pages to retain per target; 0 retains all. Default: {DEFAULT_TOP_GRAPH_GUIDES}')
    parser.add_argument('--tesseract-cmd', default=None, help='Optional explicit path to the Tesseract executable when it is not available on PATH.')
    args = parser.parse_args()
    selected_tesseract, selected_tessdata = configure_tesseract(args.tesseract_cmd)
    try:
        tesseract_version = str(pytesseract.get_tesseract_version())
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(f'Tesseract OCR could not be launched: {selected_tesseract}') from error
    try:
        available_languages = set(pytesseract.get_languages(config=''))
    except pytesseract.TesseractError as error:
        raise RuntimeError(f'Tesseract launched, but its language data could not be read. Executable: {selected_tesseract}') from error
    if args.lang not in available_languages:
        raise RuntimeError(f"Tesseract language '{args.lang}' is not installed. Available languages: " + ', '.join(sorted(available_languages)))
    input_dir = Path(args.input_dir).expanduser()
    output_path = ensure_output_parent(output_path=Path(args.out))
    results = analyze_page_folder(input_dir=input_dir, language=args.lang, page_segmentation_mode=args.psm, timeout_seconds=args.timeout, ocr_scale=args.ocr_scale, top_graph_guides=args.top_guides, top_afib_candidates=args.top_afib_candidates, top_demographic_candidates=args.top_demographic_candidates)
    results['tesseract_version'] = tesseract_version
    results['tesseract_executable'] = str(selected_tesseract)
    results['tessdata_directory'] = str(selected_tessdata) if selected_tessdata is not None else None
    save_json(data=results, output_path=output_path)
    print()
    print(f"Pages OCRed: {results['page_count']}")
    print(f"Total OCR time: {results['total_ocr_seconds']:.3f}s")
    print(f"Patient name status: {results['patient_name']['status']}")
    if results['patient_name']['selected'] is not None:
        print(f"Patient name: {results['patient_name']['selected']['display_value']}")
    print(f"Date of birth status: {results['date_of_birth']['status']}")
    if results['date_of_birth']['selected'] is not None:
        print(f"Date of birth: {results['date_of_birth']['selected']['display_value']}")
    print(f"AFib burden status: {results['afib_burden']['status']}")
    print(f"AFib candidates retained: {results['afib_burden']['candidate_count']}")
    if results['afib_burden']['selected'] is not None:
        selected = results['afib_burden']['selected']
        print(f"AFib burden: {selected['display_value']} (page {selected['page_number']}, score {selected['score']:.1f})")
    print(f"Average heart-rate guides: {len(results['average_heart_rate']['guides'])}")
    print(f"Average daily-activity guides: {len(results['average_daily_activity']['guides'])}")
    print(f'Results: {output_path}')
if __name__ == '__main__':
    main()
