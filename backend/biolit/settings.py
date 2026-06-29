"""Configuration légère du backend de production."""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
LABEL_CONFIG_DIR = Path(
    os.getenv("LABEL_CONFIG_DIR", BACKEND_DIR.parent / "label-studio" / "config")
)

S3_BUCKET = os.getenv("S3_BUCKET", "biolit-temporary")
CROP_PROJECT = os.getenv("LABEL_STUDIO_CROP_PROJECT", "Biolit - Crop manuel")
VALIDATION_PROJECT = os.getenv(
    "LABEL_STUDIO_VALIDATION_PROJECT",
    "Biolit - Validation taxonomique",
)

PIPELINE_BATCH_SIZE = int(os.getenv("PIPELINE_BATCH_SIZE", "20"))
PIPELINE_INTERVAL_SECONDS = int(os.getenv("PIPELINE_INTERVAL_SECONDS", "300"))
PROCESSING_STALE_MINUTES = int(os.getenv("PROCESSING_STALE_MINUTES", "30"))
