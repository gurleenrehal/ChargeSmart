"""Central configuration for ChargeSmart.

Everything that is *simulated* (battery sizes, prices, grid capacity, ...) lives
here and in ``SimConfig`` so that it is easy to tell real observations (ACN-Data)
apart from assumptions.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = ROOT / "logs"

SAMPLE_CSV = SAMPLE_DIR / "acn_sample_synthetic.csv"   # synthetic, ACN-schema compatible
RAW_CSV = RAW_DIR / "acn_sessions_raw.csv"             # produced by scripts/download_data.py
SESSIONS_CSV = PROCESSED_DIR / "sessions_clean.csv"
HOURLY_CSV = PROCESSED_DIR / "hourly_demand.csv"
DATA_SOURCE_FILE = PROCESSED_DIR / "data_source.json"

GLOBAL_SEED = 42
EVAL_SEEDS = list(range(1000, 1030))    # 30 identical scenarios for every policy

for _d in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
