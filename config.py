"""Central configuration for EcoSort AI. Every other file imports from here."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- paths
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "ecosort_mobilenetv2.keras"
MODEL_ONNX_PATH = MODEL_DIR / "ecosort_mobilenetv2.onnx"
MODEL_PT_PATH = MODEL_DIR / "ecosort_mobilenetv2.pt"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
HISTORY_PATH = REPORT_DIR / "history.json"
METRICS_PATH = REPORT_DIR / "metrics.json"
DISPOSAL_GUIDE_PATH = BASE_DIR / "disposal_guide.json"


def get_available_model_path() -> Path | None:
    for p in (MODEL_ONNX_PATH, MODEL_PT_PATH, MODEL_PATH):
        if p.exists():
            return p
    return None


# -------------------------------------------------------------- classes
# Folder names exactly as they appear in the Kaggle dataset
# "mostafaabla/garbage-classification" (note: "battery", not "batteries").
CLASS_NAMES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]
NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------------------------------------ data/split
IMG_SIZE = (224, 224)          # (height, width) expected by MobileNetV2
BATCH_SIZE = 32
SEED = 42
SPLIT = (0.70, 0.15, 0.15)     # train / val / test (must sum to 1.0)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# -------------------------------------------------------------- training
EPOCHS_HEAD = 10               # phase 1: feature extraction (base frozen)
EPOCHS_FINE = 10               # phase 2: optional fine-tuning
LR_HEAD = 1e-3
LR_FINE = 1e-5
FINE_TUNE_LAYERS = 40          # how many top layers of MobileNetV2 to unfreeze
DROPOUT = 0.3

# ----------------------------------------------------------------- app
LOW_CONFIDENCE = 0.60          # below this the app warns the user


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
