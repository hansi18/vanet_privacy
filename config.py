"""
config.py  —  EDIT THIS FILE before running
─────────────────────────────────────────────────────────────────
Set DATA_PATH to your VeReMi file or folder.
Accepts a single CSV, a folder of CSVs, or a folder of JSONs.
─────────────────────────────────────────────────────────────────
"""

# ── REQUIRED: path to your VeReMi dataset ────────────────────────────────────
#
# Examples:
#   Single CSV:
#     DATA_PATH = r"C:\Users\hansi\OneDrive\Documents\veremi_combined.csv"
#   Folder of CSVs or JSONs:
#     DATA_PATH = r"C:\Users\hansi\Downloads\veremi_folder"
#   Mac/Linux:
#     DATA_PATH = "/home/yourname/datasets/veremi_combined.csv"

DATA_PATH = r"C:\Users\hansi\OneDrive\Documents\veremi_combined.csv"


# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "results"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Vehicle setup ─────────────────────────────────────────────────────────────
MAX_VEHICLES       = 200
N_VEHICLES         = 100
ATTACKER_FRACTION  = 0.20

# ── FL training ───────────────────────────────────────────────────────────────
N_ROUNDS           = 20
BEACONS_PER_ROUND  = 50
LOCAL_EPOCHS       = 2
LR                 = 0.01

# ── Privacy parameters ────────────────────────────────────────────────────────
EPS_TOTAL          = 1.00
EPS_MIN            = 0.05
EPS_MAX            = 1.50
CLIP_C             = 1.0
DELTA              = 1e-5

# ── Baseline ──────────────────────────────────────────────────────────────────
FIXED_EPS_BASELINE = 0.50

# ── RSU-DT ────────────────────────────────────────────────────────────────────
MIN_TWIN_HISTORY   = 5
