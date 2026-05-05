from pathlib import Path
import base64
import json
import os
import re
from urllib import error, request

import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from django.conf import settings
from pytesseract import Output
from rapidfuzz import fuzz
from symspellpy import SymSpell, Verbosity
import symspellpy


TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
BASE_LOCAL_MODEL_DIR = Path(settings.OCR_LOCAL_MODEL_DIR)
FINETUNED_MODEL_DIR = Path(settings.OCR_FINETUNED_MODEL_DIR)
CUSTOM_FINETUNED_MODEL_DIR = Path(settings.BASE_DIR) / "local_models" / "trocr-custom_csv-finetuned"
BASE_WORD_MODEL_DIR = Path(settings.BASE_DIR) / "local_models" / "trocr-base-handwritten"
LOCAL_MODEL_ID = settings.OCR_LOCAL_MODEL_ID
SYMSPELL_PACKAGE_DIR = Path(symspellpy.__file__).resolve().parent
SYMSPELL_UNIGRAM_PATH = SYMSPELL_PACKAGE_DIR / "frequency_dictionary_en_82_765.txt"
SYMSPELL_BIGRAM_PATH = SYMSPELL_PACKAGE_DIR / "frequency_bigramdictionary_en_243_342.txt"
CUSTOM_DICTIONARY_PATH = Path(settings.BASE_DIR) / "ocr_app" / "custom_dictionary.txt"
_SYMSPELL_INSTANCE = None
_CUSTOM_TERMS = None
_TROCR_MODEL_CACHE = {}
KNOWN_SECTION_HEADINGS = [
    "Career Objective",
    "Education",
    "Technical Skills",
    "Projects",
    "Project",
    "Languages",
    "Experience",
    "Contact",
    "Summary",
    "Achievements",
]
HEADING_SCORE_THRESHOLD = 72


def _resolve_local_ai_model_dir():
    """Prefer a fine-tuned local model when one has already been trained."""

    if (CUSTOM_FINETUNED_MODEL_DIR / 'config.json').exists():
        return CUSTOM_FINETUNED_MODEL_DIR
    if (FINETUNED_MODEL_DIR / 'config.json').exists():
        return FINETUNED_MODEL_DIR
    return BASE_LOCAL_MODEL_DIR


def _load_color_image(image_path: str):
    """Load the uploaded image in color."""

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the uploaded image.")

    return image


def _order_points(points):
    """Order four contour points for perspective correction."""

    rect = np.zeros((4, 2), dtype="float32")
    points_sum = points.sum(axis=1)
    rect[0] = points[np.argmin(points_sum)]
    rect[2] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(points_diff)]
    rect[3] = points[np.argmax(points_diff)]
    return rect


def _find_document_contour(image):
    """Try to detect the page boundary from a phone photo."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        contour_area = cv2.contourArea(approximation)
        if len(approximation) == 4 and contour_area > image_area * 0.2:
            return approximation.reshape(4, 2)

    return None


def _apply_perspective_transform(image, contour):
    """Flatten the detected document into a top-down scan-like view."""

    rect = _order_points(contour)
    top_left, top_right, bottom_right, bottom_left = rect

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(top_right - bottom_right)
    height_left = np.linalg.norm(top_left - bottom_left)
    max_height = int(max(height_right, height_left))

    if max_width <= 0 or max_height <= 0:
        return image

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _remove_shadow(gray_image):
    """Reduce uneven lighting and page shadows from phone-captured images."""

    dilated = cv2.dilate(gray_image, np.ones((7, 7), np.uint8))
    background = cv2.medianBlur(dilated, 21)
    difference = 255 - cv2.absdiff(gray_image, background)
    normalized = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def _enhance_document_contrast(gray_image):
    """Improve contrast after page cleanup."""

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray_image)
    return cv2.fastNlMeansDenoising(contrasted, None, 12, 7, 21)


def _prepare_document_grayscale(image_path: str):
    """
    Convert a phone photo into a cleaner scan-style grayscale image.

    Steps:
    1. Load the original color image
    2. Detect page contour when possible
    3. Apply perspective correction
    4. Remove shadows and uneven lighting
    5. Enhance local contrast
    """

    color_image = _load_color_image(image_path)
    contour = _find_document_contour(color_image)
    if contour is not None:
        color_image = _apply_perspective_transform(color_image, contour)

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    shadow_free = _remove_shadow(gray)
    enhanced = _enhance_document_contrast(shadow_free)
    return enhanced


def _prepare_basic_grayscale(image_path: str):
    """Load a neutral grayscale image without scan-style cleanup."""

    color_image = _load_color_image(image_path)
    return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


def _prepare_grayscale_candidates(image_path: str):
    """
    Build multiple grayscale candidates for OCR.

    A flat screenshot or already-clean image can perform worse after aggressive
    scan cleanup, so both the normal grayscale and the document-cleaned version
    are evaluated.
    """

    basic_gray = _prepare_basic_grayscale(image_path)
    candidates = [("basic", basic_gray)]

    try:
        scan_gray = _prepare_document_grayscale(image_path)
    except Exception:
        scan_gray = None

    if scan_gray is not None:
        difference = float(np.mean(cv2.absdiff(basic_gray, scan_gray)))
        if difference > 2.0:
            candidates.append(("scan", scan_gray))

    return candidates


def _upscale_image(gray_image):
    """Increase image size to help Tesseract detect characters more clearly."""

    height, width = gray_image.shape[:2]
    scale = 3 if max(height, width) < 1000 else 2
    return cv2.resize(gray_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _crop_text_region(gray_image):
    """Crop the image to the main text block to reduce background noise."""

    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_inv = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_inv, kernel, iterations=1)
    points = cv2.findNonZero(dilated)

    if points is None:
        return gray_image

    x, y, w, h = cv2.boundingRect(points)
    pad_x = max(int(w * 0.03), 12)
    pad_y = max(int(h * 0.08), 12)
    x0 = max(x - pad_x, 0)
    y0 = max(y - pad_y, 0)
    x1 = min(x + w + pad_x, gray_image.shape[1])
    y1 = min(y + h + pad_y, gray_image.shape[0])
    cropped = gray_image[y0:y1, x0:x1]
    return cropped if cropped.size else gray_image


def _crop_primary_foreground_region(gray_image):
    """Tightly crop the main handwritten blob and ignore border shadows."""

    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_inv = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    boxes = []
    height, width = gray_image.shape[:2]
    min_area = max(int((height * width) * 0.0005), 25)

    for index in range(1, component_count):
        x, y, w, h, area = stats[index]
        touches_border = x <= 1 or y <= 1 or (x + w) >= (width - 1) or (y + h) >= (height - 1)
        if area < min_area or touches_border:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return gray_image

    x0 = min(x for x, _, _, _ in boxes)
    y0 = min(y for _, y, _, _ in boxes)
    x1 = max(x + w for x, _, w, _ in boxes)
    y1 = max(y + h for _, y, _, h in boxes)

    pad_x = max(int((x1 - x0) * 0.2), 18)
    pad_y = max(int((y1 - y0) * 0.35), 18)
    x0 = max(x0 - pad_x, 0)
    y0 = max(y0 - pad_y, 0)
    x1 = min(x1 + pad_x, width)
    y1 = min(y1 + pad_y, height)
    cropped = gray_image[y0:y1, x0:x1]
    return cropped if cropped.size else gray_image


def _build_image_variants(gray_image, prefer_word_mode: bool = False):
    """Create several cleaned versions of the same image for OCR comparison."""

    enlarged = _upscale_image(gray_image)
    blurred = cv2.GaussianBlur(enlarged, (5, 5), 0)
    sharpened = cv2.addWeighted(enlarged, 1.4, blurred, -0.4, 0)

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(adaptive)
    denoised = cv2.medianBlur(adaptive, 3)
    morph_kernel = np.ones((2, 2), np.uint8)
    morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, morph_kernel)
    morph_open = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, morph_kernel)

    variants = [
        ("adaptive_psm6", Image.fromarray(adaptive), "--oem 3 --psm 6"),
        ("adaptive_psm7", Image.fromarray(adaptive), "--oem 3 --psm 7"),
        ("adaptive_psm11", Image.fromarray(adaptive), "--oem 3 --psm 11"),
        ("otsu_psm6", Image.fromarray(otsu), "--oem 3 --psm 6"),
        ("otsu_psm7", Image.fromarray(otsu), "--oem 3 --psm 7"),
        ("denoised_psm6", Image.fromarray(denoised), "--oem 3 --psm 6"),
        ("morphed_psm6", Image.fromarray(morphed), "--oem 3 --psm 6"),
        ("opened_psm6", Image.fromarray(morph_open), "--oem 3 --psm 6"),
        ("inverted_psm6", Image.fromarray(inverted), "--oem 3 --psm 6"),
    ]

    if prefer_word_mode:
        variants.extend(
            [
                ("adaptive_psm8", Image.fromarray(adaptive), "--oem 3 --psm 8"),
                ("adaptive_psm10", Image.fromarray(adaptive), "--oem 3 --psm 10"),
                ("otsu_psm8", Image.fromarray(otsu), "--oem 3 --psm 8"),
                ("otsu_psm13", Image.fromarray(otsu), "--oem 3 --psm 13"),
            ]
        )

    return variants


def _clean_text(text: str) -> str:
    """Normalize OCR output by removing empty lines and extra spaces."""

    cleaned_lines = []
    for line in text.splitlines():
        normalized = " ".join(line.strip().split())
        if normalized:
            cleaned_lines.append(normalized)
    return "\n".join(cleaned_lines)


def _get_symspell():
    """Load SymSpell dictionaries once for offline spell correction."""

    global _SYMSPELL_INSTANCE

    if _SYMSPELL_INSTANCE is not None:
        return _SYMSPELL_INSTANCE

    if not SYMSPELL_UNIGRAM_PATH.exists():
        return None

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    loaded = sym_spell.load_dictionary(str(SYMSPELL_UNIGRAM_PATH), 0, 1)
    if not loaded:
        return None

    if SYMSPELL_BIGRAM_PATH.exists():
        sym_spell.load_bigram_dictionary(str(SYMSPELL_BIGRAM_PATH), 0, 2)

    if CUSTOM_DICTIONARY_PATH.exists():
        with CUSTOM_DICTIONARY_PATH.open('r', encoding='utf-8') as dictionary_file:
            for raw_line in dictionary_file:
                term = raw_line.strip().lower()
                if term:
                    sym_spell.create_dictionary_entry(term, 10_000_000)

    _SYMSPELL_INSTANCE = sym_spell
    return _SYMSPELL_INSTANCE


def _get_custom_terms():
    """Load custom OCR-safe terms once."""

    global _CUSTOM_TERMS

    if _CUSTOM_TERMS is not None:
        return _CUSTOM_TERMS

    if not CUSTOM_DICTIONARY_PATH.exists():
        _CUSTOM_TERMS = set()
        return _CUSTOM_TERMS

    try:
        with CUSTOM_DICTIONARY_PATH.open('r', encoding='utf-8') as dictionary_file:
            _CUSTOM_TERMS = {line.strip().lower() for line in dictionary_file if line.strip()}
    except OSError:
        _CUSTOM_TERMS = set()

    return _CUSTOM_TERMS


def _should_correct_token(token: str) -> bool:
    """Avoid damaging numbers or very short fragments during correction."""

    return token.isalpha() and len(token) >= 3


def _correct_token(token: str, sym_spell: SymSpell) -> str:
    """Correct a single token while preserving case when possible."""

    suggestions = sym_spell.lookup(token.lower(), Verbosity.TOP, max_edit_distance=2)
    if not suggestions:
        return token

    best = suggestions[0].term
    if fuzz.ratio(token.lower(), best.lower()) < 55:
        return token

    if token.isupper():
        return best.upper()
    if token.istitle():
        return best.title()
    return best


def _is_custom_dictionary_term(token: str) -> bool:
    """Preserve project-specific words from the custom dictionary."""

    return token.lower() in _get_custom_terms()


def _correct_line_with_nlp(line: str, sym_spell: SymSpell) -> str:
    """Apply line-level and token-level NLP correction to OCR output."""

    if not line.strip():
        return line

    compound = sym_spell.lookup_compound(line, max_edit_distance=2)
    candidate = compound[0].term if compound else line

    pieces = re.findall(r"[A-Za-z]+|\d+|[^\w\s]+|\s+", candidate)
    corrected_parts = []
    for piece in pieces:
        if piece.isspace() or piece.isdigit() or re.fullmatch(r"[^\w\s]+", piece):
            corrected_parts.append(piece)
            continue

        if _is_custom_dictionary_term(piece):
            corrected_parts.append(piece)
            continue

        if _should_correct_token(piece):
            corrected_parts.append(_correct_token(piece, sym_spell))
        else:
            corrected_parts.append(piece)

    corrected = "".join(corrected_parts)
    return _clean_text(corrected) or _clean_text(line)


def _apply_nlp_spell_correction(text: str) -> str:
    """Improve OCR output with offline NLP-based spelling correction."""

    sym_spell = _get_symspell()
    if not sym_spell or not text.strip():
        return text

    corrected_lines = []
    for line in text.splitlines():
        corrected_lines.append(_correct_line_with_nlp(line, sym_spell))

    corrected_text = _clean_text("\n".join(corrected_lines))
    return corrected_text or text


def _normalize_bullet_prefix(line: str) -> str:
    """Turn noisy OCR bullet prefixes into a stable dash bullet."""

    normalized = re.sub(r"^\s*[€¢©oO]\s+", "- ", line)
    normalized = re.sub(r"^\s*e\s+(?=[A-Z0-9])", "- ", normalized)
    normalized = re.sub(r"^\s*[-–—]\s*", "- ", normalized)
    return normalized


def _normalize_email_and_url_patterns(line: str) -> str:
    """Repair common OCR spacing issues in emails and simple URLs."""

    normalized = line
    normalized = re.sub(r"\s*@\s*", "@", normalized)
    normalized = re.sub(r"\bemail\b", "@", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bat\b", "@", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot\b", ".", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?i)\bgit\s+hub\b", "github", normalized)
    normalized = re.sub(r"(?i)\bg\s*mail\b", "gmail", normalized)
    normalized = re.sub(r"(?i)\bya\s*hoo\b", "yahoo", normalized)
    normalized = re.sub(r"(?i)\bout\s*look\b", "outlook", normalized)
    normalized = re.sub(r"(?i)\bhot\s*mail\b", "hotmail", normalized)
    normalized = re.sub(r"(?i)\bgithub\s+com\b", "github.com", normalized)
    normalized = re.sub(r"(?i)\bwww\s*\.\s*", "www.", normalized)
    normalized = re.sub(r"(?i)\bhttps?\s*:\s*/\s*/\s*", "https://", normalized)

    if any(token in normalized.lower() for token in ["@", ".com", ".org", ".net", "github", "www.", "http://", "https://"]):
        normalized = re.sub(r"\s*\.\s*", ".", normalized)
        normalized = re.sub(r"\s*/\s*", "/", normalized)

    return normalized


def _normalize_numeric_patterns(line: str) -> str:
    """Repair simple OCR mistakes around phone numbers and GPA/CGPA style values."""

    normalized = line

    if sum(char.isdigit() for char in normalized) >= 8:
        normalized = re.sub(r"(?<=\d)[oO](?=\d)", "0", normalized)
        normalized = re.sub(r"(?<=\d)[iIl](?=\d)", "1", normalized)
        normalized = re.sub(r"(?<=\d)[sS](?=\d)", "5", normalized)

    normalized = re.sub(r"(?i)\bcqpa\b", "CGPA", normalized)
    normalized = re.sub(r"(?i)\bgqpa\b", "GPA", normalized)
    normalized = re.sub(r"(?i)\bcse student\b", "CSE student", normalized)
    return normalized


def _normalize_heading_line(line: str) -> str:
    """Map noisy heading-like lines to a known section title when there is a close match."""

    stripped = line.strip(" -:.\t")
    if not stripped:
        return line

    best_heading = None
    best_score = 0
    for heading in KNOWN_SECTION_HEADINGS:
        score = fuzz.ratio(stripped.lower(), heading.lower())
        if score > best_score:
            best_heading = heading
            best_score = score

    if best_heading and best_score >= HEADING_SCORE_THRESHOLD:
        return best_heading

    return line


def _normalize_common_handwriting_terms(line: str) -> str:
    """Repair a few common handwriting/OCR confusions without over-correcting the whole line."""

    replacements = {
        r"(?i)\beaucation\b": "Education",
        r"(?i)\blanquages\b": "Languages",
        r"(?i)\blanguages\b": "Languages",
        r"(?i)\btechnicai\b": "Technical",
        r"(?i)\bobjectlve\b": "Objective",
        r"(?i)\bobjectiye\b": "Objective",
        r"(?i)\bdjang0\b": "Django",
        r"(?i)\bpyth0n\b": "Python",
        r"(?i)\bjava5cript\b": "JavaScript",
        r"(?i)\b5qlite\b": "SQLite",
        r"(?i)\bchatt? gram\b": "Chattogram",
        r"(?i)\bbangla desh\b": "Bangladesh",
    }

    normalized = line
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def _apply_pattern_aware_correction(text: str) -> str:
    """
    Apply structure-aware corrections to OCR output.

    This keeps the raw OCR wording mostly intact, but improves common document
    patterns such as section headings, bullets, emails, URLs, and contact lines.
    """

    if not text.strip():
        return text

    corrected_lines = []
    for line in text.splitlines():
        normalized = _clean_text(line)
        if not normalized:
            continue

        normalized = _normalize_bullet_prefix(normalized)
        normalized = _normalize_email_and_url_patterns(normalized)
        normalized = _normalize_numeric_patterns(normalized)
        normalized = _normalize_common_handwriting_terms(normalized)
        normalized = _normalize_heading_line(normalized)
        corrected_lines.append(normalized)

    corrected_text = _clean_text("\n".join(corrected_lines))
    return corrected_text or text


def _score_text_quality(text: str) -> float:
    """Heuristic score for selecting the most plausible OCR output."""

    if not text:
        return -1.0

    characters = len(text)
    alnum_count = sum(char.isalnum() for char in text)
    alpha_count = sum(char.isalpha() for char in text)
    digit_count = sum(char.isdigit() for char in text)
    space_count = sum(char.isspace() for char in text)
    unique_chars = len(set(text))

    alnum_ratio = alnum_count / max(characters, 1)
    space_ratio = space_count / max(characters, 1)
    diversity_ratio = unique_chars / max(characters, 1)
    alpha_bonus = min(alpha_count * 0.25, 8.0)
    digit_bonus = min(digit_count * 0.15, 3.0)
    length_bonus = min(characters * 0.12, 10.0)
    punctuation_penalty = sum(not (char.isalnum() or char.isspace()) for char in text) * 0.15

    return (
        (alnum_ratio * 40.0)
        + (space_ratio * 10.0)
        + (diversity_ratio * 6.0)
        + alpha_bonus
        + digit_bonus
        + length_bonus
        - punctuation_penalty
    )


def _repetition_penalty(text: str) -> float:
    """Penalize outputs that repeat the same token pattern too often."""

    tokens = text.lower().split()
    if not tokens:
        return 0.0

    unique_ratio = len(set(tokens)) / len(tokens)
    repeated_zero_like = sum(token in {'0', '00', '000', 'o', 'oo'} for token in tokens)
    return ((1.0 - unique_ratio) * 20.0) + (repeated_zero_like * 1.5)


def _overall_prediction_score(text: str) -> float:
    """Combined score used to compare OCR/AI candidate outputs."""

    return _score_text_quality(text) - _repetition_penalty(text)


def _word_candidate_bonus(text: str) -> float:
    """Reward plausible single-word handwriting outputs."""

    cleaned = _clean_text(text)
    if not cleaned or "\n" in cleaned:
        return 0.0

    tokens = cleaned.split()
    if len(tokens) > 2:
        return 0.0

    alpha_count = sum(char.isalpha() for char in cleaned)
    digit_count = sum(char.isdigit() for char in cleaned)
    if alpha_count >= 3 and digit_count == 0:
        return 12.0
    if alpha_count >= 3 and digit_count <= 1:
        return 6.0
    return 0.0


def _is_plausible_word_prediction(text: str) -> bool:
    """Detect whether OCR output looks like a clean handwritten word."""

    cleaned = _clean_text(text)
    if not cleaned or "\n" in cleaned:
        return False

    tokens = cleaned.split()
    if len(tokens) > 2:
        return False

    alpha_count = sum(char.isalpha() for char in cleaned)
    digit_count = sum(char.isdigit() for char in cleaned)
    return alpha_count >= 3 and digit_count == 0


def _normalize_alpha_sequence_line(text: str) -> str:
    """Clean short alphabetic OCR lines by removing stray punctuation."""

    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    compact = re.sub(r"[^A-Za-z\s]", "", cleaned)
    compact = _clean_text(compact)
    if not compact:
        return cleaned

    tokens = compact.split()
    if len(tokens) <= 3 and sum(char.isalpha() for char in compact) >= 3:
        return compact

    return cleaned


def _extract_text_with_confidence(image_variant: Image.Image, config: str):
    """Run Tesseract and estimate result quality from word confidences."""

    data = pytesseract.image_to_data(image_variant, config=config, output_type=Output.DICT)
    words = []
    confidences = []

    for text, confidence in zip(data.get('text', []), data.get('conf', [])):
        normalized_text = " ".join(text.split())
        if not normalized_text:
            continue

        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            continue

        if confidence_value >= 0:
            words.append(normalized_text)
            confidences.append(confidence_value)

    candidate_text = _clean_text(" ".join(words))
    average_confidence = sum(confidences) / len(confidences) if confidences else -1.0
    return candidate_text, average_confidence


def _segment_lines(gray_image):
    """Split a text image into line images when it appears to contain multiple lines."""

    enlarged = _upscale_image(gray_image)
    blurred = cv2.GaussianBlur(enlarged, (5, 5), 0)
    _, binary_inv = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    row_projection = np.sum(binary_inv > 0, axis=1)
    threshold = max(int(binary_inv.shape[1] * 0.02), 8)

    line_ranges = []
    start = None
    for index, value in enumerate(row_projection):
        if value > threshold and start is None:
            start = index
        elif value <= threshold and start is not None:
            if index - start > 18:
                line_ranges.append((start, index))
            start = None

    if start is not None and len(row_projection) - start > 18:
        line_ranges.append((start, len(row_projection)))

    if len(line_ranges) <= 1:
        return []

    line_images = []
    for start, end in line_ranges:
        pad = 8
        y0 = max(start - pad, 0)
        y1 = min(end + pad, enlarged.shape[0])
        line_image = enlarged[y0:y1, :]
        line_image = _crop_text_region(line_image)
        if line_image.size:
            line_images.append(line_image)

    return line_images


def _is_short_word_image(gray_image) -> bool:
    """Detect small single-word style inputs that need word-focused OCR settings."""

    height, width = gray_image.shape[:2]
    line_images = _segment_lines(gray_image)

    if line_images:
        return False

    if height <= 0:
        return False

    aspect_ratio = width / height
    return (width * height) < 220_000 and 1.5 <= aspect_ratio <= 8.5


def _pad_line_image(gray_image, padding: int = 24):
    """Add white padding around a line image to help the AI model focus on text."""

    return cv2.copyMakeBorder(
        gray_image,
        padding,
        padding,
        padding,
        padding,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )


def _foreground_strength(gray_image) -> float:
    """Estimate how strongly the crop contains foreground ink."""

    _, binary_inv = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return float(np.mean(binary_inv > 0))


def _prepare_ai_line_images(image_path: str):
    """
    Prepare one or more line images for the TrOCR handwriting model.

    The small TrOCR handwritten checkpoint works best on single-line text images,
    so page-like uploads are cropped and segmented into separate lines first.
    """

    best_lines = []
    best_score = -1.0

    for _candidate_name, gray_image in _prepare_grayscale_candidates(image_path):
        cropped_image = _crop_text_region(gray_image)
        short_word_image = _is_short_word_image(cropped_image)
        line_images = _segment_lines(cropped_image)

        if not line_images:
            if short_word_image:
                cropped_image = _crop_primary_foreground_region(cropped_image)
            line_images = [cropped_image]

        prepared_images = []
        for line_image in line_images:
            padded = _pad_line_image(line_image, 32 if short_word_image else 24)
            rgb_line = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
            prepared_images.append(Image.fromarray(rgb_line))

        if short_word_image:
            candidate_score = _foreground_strength(cropped_image) * 1000.0
        else:
            candidate_score = sum(image.width * image.height for image in prepared_images)
        if prepared_images and candidate_score > best_score:
            best_lines = prepared_images
            best_score = candidate_score

    return best_lines


def _should_skip_ai_for_image(image_path: str) -> bool:
    """
    Skip the local handwriting model for page-like documents.

    The small TrOCR model is best for short handwritten lines. For large pages,
    CVs, or many-line notes, the tuned local OCR pipeline is currently more
    reliable and much faster.
    """

    try:
        for _candidate_name, gray_image in _prepare_grayscale_candidates(image_path):
            cropped_image = _crop_text_region(gray_image)
            primary_crop = _crop_primary_foreground_region(cropped_image)
            if _is_short_word_image(primary_crop):
                return False
            cropped_image = primary_crop
            height, width = cropped_image.shape[:2]
            line_images = _segment_lines(cropped_image)
            if width * height > 500_000:
                return True
            if len(line_images) >= 3:
                return True
    except Exception:
        return False

    return False


def _predict_from_line_segments(gray_image):
    """OCR each detected line separately and combine the best line outputs."""

    line_images = _segment_lines(gray_image)
    if not line_images:
        return "", -1.0

    lines = []
    total_score = 0.0

    for line_image in line_images:
        best_line_text = ""
        best_line_score = -1.0

        for _variant_name, image_variant, config in _build_image_variants(line_image):
            line_config = config.replace("--psm 6", "--psm 7").replace("--psm 11", "--psm 7")
            try:
                candidate_text, candidate_confidence = _extract_text_with_confidence(
                    image_variant,
                    line_config,
                )
            except Exception:
                continue

            candidate_score = candidate_confidence + _overall_prediction_score(candidate_text)
            if candidate_text and candidate_score > best_line_score:
                best_line_text = candidate_text
                best_line_score = candidate_score

        if best_line_text:
            lines.append(best_line_text)
            total_score += best_line_score

    combined_text = _clean_text("\n".join(lines))
    return combined_text, total_score


def _predict_with_local_ocr(uploaded_image) -> str:
    """Extract text from the uploaded image using local Tesseract OCR."""

    if not TESSERACT_PATH.exists():
        return (
            "Tesseract OCR executable was not found.\n"
            "Install Tesseract or update the path in ocr_app/services.py."
        )

    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)

    image_path = uploaded_image.image.path

    best_text = ""
    best_score = -1.0

    try:
        grayscale_candidates = _prepare_grayscale_candidates(image_path)
    except Exception as exc:
        return f"OCR processing failed: {exc}"

    for _candidate_name, gray_image in grayscale_candidates:
        cropped_image = _crop_text_region(gray_image)
        prefer_word_mode = _is_short_word_image(cropped_image)
        if prefer_word_mode:
            cropped_image = _crop_primary_foreground_region(cropped_image)
        image_variants = _build_image_variants(cropped_image, prefer_word_mode=prefer_word_mode)

        for _variant_name, image_variant, config in image_variants:
            try:
                candidate_text, candidate_confidence = _extract_text_with_confidence(
                    image_variant,
                    config,
                )
            except Exception:
                continue

            candidate_score = candidate_confidence + _overall_prediction_score(candidate_text)
            if prefer_word_mode:
                candidate_score += _word_candidate_bonus(candidate_text)
            if candidate_text and candidate_score > best_score:
                best_text = candidate_text
                best_score = candidate_score

        if not prefer_word_mode:
            segmented_text, segmented_score = _predict_from_line_segments(cropped_image)
            if segmented_text and segmented_score > best_score:
                best_text = segmented_text
                best_score = segmented_score

    if best_text:
        return best_text

    return "No readable text was detected in the uploaded image."


def _load_trocr_model(model_dir: Path):
    """Load and cache a local TrOCR model from disk."""

    cache_key = str(model_dir.resolve())
    if cache_key in _TROCR_MODEL_CACHE:
        return _TROCR_MODEL_CACHE[cache_key]

    if not model_dir.exists():
        return None, None

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained(str(model_dir), local_files_only=True, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(str(model_dir), local_files_only=True)
        model.eval()
    except Exception:
        return None, None

    _TROCR_MODEL_CACHE[cache_key] = (processor, model)
    return processor, model


def _predict_short_word_with_base_model(image_path: str) -> str:
    """Use the generic base handwritten model for short single-word images."""

    if not BASE_WORD_MODEL_DIR.exists():
        return ""

    grayscale_candidates = _prepare_grayscale_candidates(image_path)
    if not grayscale_candidates:
        return ""

    basic_gray = grayscale_candidates[0][1]
    cropped_image = _crop_primary_foreground_region(_crop_text_region(basic_gray))
    padded = _pad_line_image(cropped_image, 32)
    rgb_image = Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB))

    processor, model = _load_trocr_model(BASE_WORD_MODEL_DIR)
    if processor is None or model is None:
        return ""

    pixel_values = processor(images=rgb_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=32,
            num_beams=6,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return _clean_text(generated_text)


def _merge_sparse_boxes(boxes):
    """Merge nearby sparse text boxes that belong to the same handwritten row."""

    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
    merged_boxes = []

    for x, y, w, h in boxes:
        if not merged_boxes:
            merged_boxes.append([x, y, w, h])
            continue

        last_x, last_y, last_w, last_h = merged_boxes[-1]
        last_bottom = last_y + last_h
        bottom = y + h
        vertical_overlap = max(0, min(last_bottom, bottom) - max(last_y, y))
        min_height = max(min(last_h, h), 1)
        gap = x - (last_x + last_w)

        if vertical_overlap >= min_height * 0.45 and gap <= max(last_w, w) * 0.45:
            new_x = min(last_x, x)
            new_y = min(last_y, y)
            new_right = max(last_x + last_w, x + w)
            new_bottom = max(last_y + last_h, y + h)
            merged_boxes[-1] = [new_x, new_y, new_right - new_x, new_bottom - new_y]
        else:
            merged_boxes.append([x, y, w, h])

    return [tuple(box) for box in merged_boxes]


def _detect_sparse_text_boxes(gray_image):
    """Find a few isolated handwritten text groups on a mostly empty page."""

    enlarged = cv2.resize(gray_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary_inv = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 15))
    merged = cv2.dilate(binary_inv, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    height, width = enlarged.shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        touches_border = x <= 5 or y <= 5 or (x + w) >= (width - 5) or (y + h) >= (height - 5)
        if area < 20_000 or touches_border:
            continue
        boxes.append((x, y, w, h))

    boxes = _merge_sparse_boxes(boxes)
    return enlarged, boxes


def _predict_sparse_text_clusters_with_base_model(image_path: str) -> str:
    """Read a few isolated handwritten text groups using the base handwritten model."""

    if not BASE_WORD_MODEL_DIR.exists():
        return ""

    grayscale_candidates = _prepare_grayscale_candidates(image_path)
    if not grayscale_candidates:
        return ""

    basic_gray = grayscale_candidates[0][1]
    cropped_image = _crop_text_region(basic_gray)
    enlarged, boxes = _detect_sparse_text_boxes(cropped_image)
    if not boxes or len(boxes) > 5:
        return ""

    processor, model = _load_trocr_model(BASE_WORD_MODEL_DIR)
    if processor is None or model is None:
        return ""

    predictions = []
    image_height, image_width = enlarged.shape[:2]
    for x, y, w, h in boxes:
        pad = 25
        roi = enlarged[max(y - pad, 0):min(y + h + pad, image_height), max(x - pad, 0):min(x + w + pad, image_width)]
        rgb_image = Image.fromarray(cv2.cvtColor(_pad_line_image(roi, 32), cv2.COLOR_GRAY2RGB))
        pixel_values = processor(images=rgb_image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=48,
                num_beams=6,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        cleaned_text = _normalize_alpha_sequence_line(generated_text)
        if cleaned_text:
            predictions.append(cleaned_text)

    return _clean_text("\n".join(predictions))


def _predict_with_local_ai_model(uploaded_image) -> str:
    """Predict handwriting text with a locally stored TrOCR-style model."""

    try:
        grayscale_candidates = _prepare_grayscale_candidates(uploaded_image.image.path)
        short_word_image = any(
            _is_short_word_image(_crop_primary_foreground_region(_crop_text_region(gray_image)))
            for _, gray_image in grayscale_candidates
        )
        prepared_images = _prepare_ai_line_images(uploaded_image.image.path)
    except Exception as exc:
        return f"Local AI OCR failed: {exc}"

    if short_word_image:
        try:
            short_word_prediction = _predict_short_word_with_base_model(uploaded_image.image.path)
            if short_word_prediction:
                return short_word_prediction
        except Exception:
            pass

    preferred_model_dir = BASE_WORD_MODEL_DIR if short_word_image and BASE_WORD_MODEL_DIR.exists() else _resolve_local_ai_model_dir()
    processor, model = _load_trocr_model(preferred_model_dir)
    if processor is None or model is None:
        return (
            "Local AI OCR model is not available yet.\n"
            f"Expected model directory: {preferred_model_dir}\n"
            f"Configured model id: {LOCAL_MODEL_ID}"
        )

    line_predictions = []
    try:
        for line_image in prepared_images:
            pixel_values = processor(images=line_image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=96,
                    num_beams=6 if short_word_image else 4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            cleaned_text = _clean_text(generated_text)
            if cleaned_text:
                line_predictions.append(cleaned_text)
    except Exception as exc:
        return f"Local AI OCR failed: {exc}"

    if not line_predictions:
        return "No readable text was detected in the uploaded image."

    return _clean_text("\n".join(line_predictions))


def _predict_with_hybrid_local_ai(uploaded_image) -> str:
    """
    Use both local OCR and the local AI model, then keep the stronger result.

    This handles an important real-world pattern:
    page-like handwritten notes or CVs often work better with the tuned local OCR
    pipeline, while cleaner single-line handwriting may work better with TrOCR.
    """

    sparse_cluster_text = ""
    try:
        sparse_cluster_text = _predict_sparse_text_clusters_with_base_model(uploaded_image.image.path)
    except Exception:
        sparse_cluster_text = ""

    local_text = _predict_with_local_ocr(uploaded_image)
    short_word_image = False

    try:
        grayscale_candidates = _prepare_grayscale_candidates(uploaded_image.image.path)
        short_word_image = any(
            _is_short_word_image(_crop_primary_foreground_region(_crop_text_region(gray_image)))
            for _, gray_image in grayscale_candidates
        )
    except Exception:
        short_word_image = False

    if _should_skip_ai_for_image(uploaded_image.image.path):
        return local_text

    ai_text = _predict_with_local_ai_model(uploaded_image)
    if _is_plausible_word_prediction(ai_text) and not _is_plausible_word_prediction(local_text):
        return ai_text

    sparse_score = _overall_prediction_score(sparse_cluster_text) if sparse_cluster_text else -1.0
    if sparse_cluster_text and sparse_score > max(_overall_prediction_score(local_text), _overall_prediction_score(ai_text)):
        return sparse_cluster_text

    ai_score = _overall_prediction_score(ai_text)
    local_score = _overall_prediction_score(local_text)
    if short_word_image:
        ai_score += _word_candidate_bonus(ai_text)
        local_score += _word_candidate_bonus(local_text)

    if local_score >= ai_score:
        return local_text
    return ai_text


def _predict_with_api_model(uploaded_image) -> str:
    """
    Send the uploaded image to an external OCR API.

    Expected API response examples:
    - {"predicted_text": "..."}
    - {"text": "..."}
    - {"result": "..."}
    - {"output": "..."}
    """

    if not settings.OCR_API_KEY:
        return "API OCR key is missing. Set OCR_API_KEY in your environment settings."

    # If no generic API URL is configured, fall back to the Gemini SDK flow.
    if not settings.OCR_API_URL:
        return _predict_with_gemini_model(uploaded_image)

    with open(uploaded_image.image.path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {
        'model': settings.OCR_API_MODEL,
        'file_name': Path(uploaded_image.image.name).name,
        'image_base64': image_base64,
    }

    req = request.Request(
        settings.OCR_API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {settings.OCR_API_KEY}',
        },
        method='POST',
    )

    try:
        with request.urlopen(req, timeout=settings.OCR_API_TIMEOUT) as response:
            response_data = json.loads(response.read().decode('utf-8'))
    except error.HTTPError as exc:
        return f"API OCR request failed with HTTP {exc.code}."
    except error.URLError as exc:
        return f"API OCR connection failed: {exc.reason}"
    except Exception as exc:
        return f"API OCR processing failed: {exc}"

    for key in ['predicted_text', 'text', 'result', 'output']:
        value = response_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return "API OCR did not return readable text."


def _predict_with_gemini_model(uploaded_image) -> str:
    """Run handwriting OCR with Gemini, preferring a cleaned document image."""

    try:
        import google.generativeai as genai
    except ImportError:
        return "Gemini SDK is not installed. Install google-generativeai to use API OCR."

    model_name = settings.OCR_API_MODEL or 'gemini-1.5-flash'
    prompt = """
Extract text from this handwritten image.

Use EXACT format with this section header:

EXTRACTED TEXT:
[Write the full handwritten text here line by line]

Rules:
- Read the handwriting carefully and extract only the visible text.
- Preserve the original line order.
- Preserve paragraph breaks if visible.
- Do not summarize.
- Do not correct grammar.
- Do not rewrite in a different style.
- Do not add explanations, comments, labels, or extra headings.
- If a word is unclear, write [unclear].
- If a number or symbol is unclear, write [unclear].
- If a line is partially readable, keep the readable part and mark the unclear part as [unclear].
- Return nothing except the exact format above.

If no readable text is found, return exactly:

    EXTRACTED TEXT:
Not specified
""".strip()

    try:
        genai.configure(api_key=settings.OCR_API_KEY)
        model = genai.GenerativeModel(model_name)
    except Exception as exc:
        return f"Gemini OCR processing failed: {exc}"

    gemini_images = []
    try:
        processed_gray = _prepare_document_grayscale(uploaded_image.image.path)
        gemini_images.append(Image.fromarray(processed_gray).convert('RGB'))
    except Exception:
        pass

    try:
        with Image.open(uploaded_image.image.path) as pil_image:
            gemini_images.append(pil_image.convert('RGB').copy())
    except Exception as exc:
        if not gemini_images:
            return f"Gemini OCR processing failed: {exc}"

    best_text = ""
    best_score = -1.0

    for gemini_image in gemini_images:
        try:
            response = model.generate_content([prompt, gemini_image])
        except Exception:
            continue

        text = getattr(response, 'text', '') or ''
        if 'EXTRACTED TEXT:' in text:
            text = text.split('EXTRACTED TEXT:', 1)[1].strip()

        cleaned_text = _clean_text(text)
        if not cleaned_text:
            continue

        score = _overall_prediction_score(cleaned_text)
        if cleaned_text.lower() == 'not specified':
            score = -0.5

        if score > best_score:
            best_text = cleaned_text
            best_score = score

    if best_text:
        return best_text

    return "Gemini OCR did not return readable text."


def _looks_like_api_error(text: str) -> bool:
    """Detect non-prediction API status/error messages."""

    if not text:
        return True

    lowered = text.lower().strip()
    return (
        lowered.startswith('api ocr is not configured')
        or lowered.startswith('api ocr key is missing')
        or lowered.startswith('gemini sdk is not installed')
        or lowered.startswith('gemini ocr processing failed')
        or lowered.startswith('gemini ocr did not return readable text')
        or lowered.startswith('api ocr request failed')
        or lowered.startswith('api ocr connection failed')
        or lowered.startswith('api ocr processing failed')
        or lowered.startswith('api ocr did not return readable text')
    )


def _predict_with_smart_pipeline(uploaded_image) -> str:
    """Use Gemini/API first and keep local OCR as a fallback safety net."""

    api_text = _predict_with_api_model(uploaded_image)
    local_text = _predict_with_hybrid_local_ai(uploaded_image)

    if _looks_like_api_error(api_text):
        return local_text

    if api_text.strip() and api_text.strip().lower() != 'not specified':
        return api_text

    api_score = _overall_prediction_score(api_text)
    local_score = _overall_prediction_score(local_text)

    if local_score > api_score:
        return local_text

    return api_text


def predict_handwritten_text(uploaded_image, ocr_engine: str) -> str:
    """Route OCR prediction through either local Tesseract or an API model."""

    if ocr_engine == 'smart':
        return _predict_with_smart_pipeline(uploaded_image)

    if ocr_engine == 'ai_local':
        return _predict_with_hybrid_local_ai(uploaded_image)

    if ocr_engine == 'api':
        return _predict_with_api_model(uploaded_image)

    return _predict_with_local_ocr(uploaded_image)


def extract_and_correct_text(uploaded_image, ocr_engine: str):
    """Return both raw OCR text and NLP-corrected final text."""

    raw_text = predict_handwritten_text(uploaded_image, ocr_engine)
    pattern_text = _apply_pattern_aware_correction(raw_text)

    if ocr_engine == 'local':
        corrected_text = _apply_nlp_spell_correction(pattern_text)
        if _overall_prediction_score(corrected_text) < _overall_prediction_score(pattern_text):
            corrected_text = pattern_text
    else:
        corrected_text = pattern_text

    return raw_text, corrected_text
