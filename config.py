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

# --- Centre-pool size (spatial receptive field of the head) -------------------
# Number of Clay patches per side averaged into the head's centre-pool component.
# Full grid is 28×28 after Clay's 128→224 resize; each patch ≈ 46 m native
# (8 px at 224² → 128/28·8 ≈ 8/28·1280 m). The chip centre falls at patch ≈13.9.
# Valid values: 1, 2, 3, 5, 7. 5 matches the original run (~230 m footprint);
# smaller values sharpen the classifier toward the chip centre pixel.
# The cache stores the raw centre 7×7 patches (see training/patch_pool.py
# CACHE_K), so POOL_SIZE is a train-time-only knob — no Clay re-run needed.
# See training/README.md § "Centre-pool size".
POOL_SIZE = int(os.environ.get("KELP_POOL_SIZE", 5))
assert POOL_SIZE in {1, 2, 3, 5, 7}, (
    f"KELP_POOL_SIZE must be one of 1, 2, 3, 5, 7 (got {POOL_SIZE})"
)

# --- Cached Clay embeddings ---------------------------------------------------
# Single cache file for all pool sizes: stores CLS + centre 7×7 patch tokens.
# `train_head.py` slices + means at load time to build the 2048-D input for
# the head.
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
