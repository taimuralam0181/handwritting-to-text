from __future__ import annotations

import csv
import uuid
from pathlib import Path

import cv2
from django.conf import settings
from django.utils.text import slugify


Crop = tuple[str, tuple[float, float, float, float]]


MANUAL_CROPS: dict[str, list[Crop]] = {
    "WhatsApp Image 2026-05-25 at 21.24.08.jpeg": [
        ("Validation", (0.34, 0.02, 0.77, 0.24)),
        ("Early Stopping", (0.33, 0.20, 0.93, 0.50)),
        ("cross", (0.36, 0.44, 0.66, 0.68)),
        ("fitting", (0.35, 0.61, 0.66, 0.86)),
        ("data", (0.36, 0.78, 0.65, 0.99)),
    ],
    "WhatsApp Image 2026-05-25 at 21.24.08 (1).jpeg": [
        ("Shova", (0.34, 0.02, 0.63, 0.24)),
        ("How are you", (0.22, 0.18, 0.81, 0.48)),
        ("How", (0.23, 0.43, 0.48, 0.72)),
        ("Text", (0.55, 0.40, 0.80, 0.73)),
        ("MOM", (0.20, 0.62, 0.48, 0.95)),
        ("Sir", (0.58, 0.58, 0.79, 0.94)),
    ],
    "WhatsApp Image 2026-05-25 at 21.24.08 (2).jpeg": [
        ("My Name is", (0.17, 0.02, 0.82, 0.25)),
        ("Hidden", (0.27, 0.21, 0.65, 0.51)),
        ("Layer", (0.26, 0.43, 0.62, 0.73)),
        ("Output", (0.25, 0.65, 0.67, 0.97)),
    ],
    "WhatsApp Image 2026-05-25 at 21.24.09.jpeg": [
        ("Problem", (0.43, 0.00, 0.76, 0.21)),
        ("overfitting", (0.45, 0.17, 0.93, 0.42)),
        ("Unseen", (0.46, 0.36, 0.75, 0.59)),
        ("data", (0.49, 0.54, 0.69, 0.73)),
        ("poorly", (0.50, 0.67, 0.73, 0.87)),
        ("on new", (0.51, 0.81, 0.79, 0.99)),
    ],
    "WhatsApp Image 2026-05-25 at 21.24.09 (1).jpeg": [
        ("Apply", (0.35, 0.00, 0.64, 0.22)),
        ("Use", (0.39, 0.17, 0.60, 0.42)),
        ("Dropout", (0.34, 0.37, 0.72, 0.65)),
        ("increase", (0.33, 0.57, 0.68, 0.85)),
        ("cross", (0.33, 0.76, 0.61, 0.99)),
    ],
    "WhatsApp Image 2026-05-25 at 21.24.09 (2).jpeg": [
        ("Glucose Level", (0.34, 0.00, 0.90, 0.20)),
        ("Non", (0.35, 0.14, 0.64, 0.34)),
        ("Father", (0.29, 0.26, 0.60, 0.58)),
        ("DAD", (0.64, 0.24, 0.90, 0.58)),
        ("Saitan", (0.28, 0.44, 0.63, 0.77)),
        ("Devil", (0.64, 0.43, 0.90, 0.76)),
        ("god", (0.37, 0.64, 0.61, 0.97)),
    ],
    "WhatsApp Image 2026-05-25 at 21.24.10.jpeg": [
        ("BMI", (0.21, 0.00, 0.48, 0.18)),
        ("Age", (0.21, 0.12, 0.48, 0.35)),
        ("Blood", (0.20, 0.25, 0.55, 0.55)),
        ("Family", (0.21, 0.42, 0.57, 0.72)),
        ("History", (0.21, 0.58, 0.60, 0.89)),
        ("Pressure", (0.20, 0.75, 0.67, 0.99)),
    ],
    "WhatsApp Image 2026-05-25 at 21.24.10 (1).jpeg": [
        ("Digital forensic", (0.17, 0.00, 0.84, 0.22)),
        ("scientific", (0.18, 0.17, 0.66, 0.45)),
        ("process", (0.19, 0.35, 0.60, 0.66)),
        ("digital", (0.19, 0.53, 0.57, 0.82)),
        ("Collecting", (0.18, 0.69, 0.68, 0.99)),
    ],
}


def _append_entries(entries: list[dict[str, str | bytes]]) -> int:
    config = settings.HANDWRITING_DATASET_PROFILES["custom_csv"]
    csv_path = Path(config["csv_path"])
    dataset_root = csv_path.parent
    images_dir = dataset_root / "images"
    dataset_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_exists = csv_path.exists()
    image_name_field = config.get("image_name_field", "image_name")
    text_field = config.get("text_field", "text")

    with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[image_name_field, text_field])
        if not csv_exists:
            writer.writeheader()

        for entry in entries:
            safe_stem = slugify(Path(str(entry["name"])).stem) or "sample"
            image_name = f"{safe_stem}-{uuid.uuid4().hex[:8]}.png"
            (images_dir / image_name).write_bytes(entry["bytes"])
            writer.writerow({image_name_field: image_name, text_field: entry["text"]})

    return len(entries)


def _encode_crop(image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode crop.")
    return encoded.tobytes()


def run() -> str:
    incoming_dir = Path(settings.BASE_DIR) / "datasets" / "incoming_samples"
    entries: list[dict[str, str | bytes]] = []

    for file_name, crops in MANUAL_CROPS.items():
        image_path = incoming_dir / file_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]

        for index, (label, (x1, y1, x2, y2)) in enumerate(crops, start=1):
            left = max(int(width * x1), 0)
            top = max(int(height * y1), 0)
            right = min(int(width * x2), width)
            bottom = min(int(height * y2), height)
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                raise ValueError(f"Empty crop for {file_name} label {label}.")
            entries.append(
                {
                    "name": f"{Path(file_name).stem}-{index}.png",
                    "bytes": _encode_crop(crop),
                    "text": label,
                }
            )

    saved_count = _append_entries(entries)
    return f"Saved {saved_count} entries from {len(MANUAL_CROPS)} files."
