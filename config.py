"""Central configuration for kelp-observe-classifier.

All data paths are derived from KELP_DATA_DIR (default: ~/kelp-data). Set that
env var to point at your local data directory. The Clay v1.5 checkpoint is
expected to live at $KELP_DATA_DIR/clay/v1.5/clay-v1.5.ckpt (override via
KELP_CLAY_CKPT if you keep it elsewhere).

Nothing here reads from the environment lazily — attribute access is safe.
"""
from pathlib import Path
import os

# Repository root (this file lives at repo/config.py)
REPO_ROOT = Path(__file__).resolve().parent

# --- Data root (all data + intermediate artefacts live here) ------------------
DATA_DIR = Path(os.environ.get("KELP_DATA_DIR", "~/kelp-data")).expanduser()

# --- Sentinel-2 chips (merged across sources) --------------------------------
CHIPS_DIR  = DATA_DIR / "s2_chips_merged"
CHIPS_H5   = CHIPS_DIR / "chips.h5"
CHIPS_META = CHIPS_DIR / "chips_metadata.csv"

# --- Clay v1.5 -----------------------------------------------------------------
CLAY_CKPT = Path(os.environ.get("KELP_CLAY_CKPT", DATA_DIR / "clay" / "v1.5" / "clay-v1.5.ckpt"))
CLAY_META = REPO_ROOT / "configs" / "metadata.yaml"

# --- Cached Clay embeddings ---------------------------------------------------
CLAY_CACHE_DIR = DATA_DIR / "clay_cache"
CLAY_CACHE_NPZ = CLAY_CACHE_DIR / "identity.npz"

# --- Per-record allowed_max_days annotations (built by data_collection) -------
ABSENT_WINDOWS_JSON = DATA_DIR / "absent_windows.json"

# --- Splits + results (regenerated locally, gitignored) -----------------------
SPLITS_DIR  = REPO_ROOT / "training" / "splits"
RESULTS_DIR = REPO_ROOT / "training" / "results"

# --- Logs / tmp (workspace only) ----------------------------------------------
LOGS_DIR = Path(os.environ.get("KELP_LOGS_DIR", DATA_DIR.parent / "logs"))
TMP_DIR  = Path(os.environ.get("KELP_TMP_DIR",  DATA_DIR.parent / "tmp"))
