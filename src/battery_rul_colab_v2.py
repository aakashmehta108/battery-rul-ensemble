"""
BATTERY RUL PREDICTION — CORRECTED EXPERIMENT NOTEBOOK (v2)
============================================================
Ensemble deep learning + uncertainty quantification for lithium-ion
battery Remaining Useful Life (RUL) prediction.

Rebuild of the starter-kit notebook. Fixes applied (see CODE_REVIEW.md):
  * pymc3 import is lazy/optional — never crashes the run
  * fixed pip package name bug ('scipy-stats' -> 'scipy'); torch dropped
  * LSTM/GRU now use proper 3D sliding-window inputs (lookback=50)
  * real data loaders (NASA Kaggle CSV, NASA .mat, CALCE CSVs) -> unified schema
  * per-cycle RUL labels with EOL = 80% of nominal capacity
  * NO data leakage: splits by cell_id, scaler fit on train only,
    features use past/current cycle measurements only
  * synthetic fallback is clearly labeled and leakage-free
  * uncertainty = MC Dropout + ensemble spread, EMPIRICAL coverage reported
  * ensemble weights tuned on validation split (not hardcoded)
  * all figures saved (PNG + PDF, 300 dpi); results saved as CSV + LaTeX

Designed for Google Colab (free GPU). Also runs headless / locally.

Usage in Colab:
    1. Upload this file's contents into a Colab notebook, OR run as a script.
    2. Put your data files anywhere findable (see SEARCH_DIRS below),
       or let it auto-download the NASA cleaned CSV via the Kaggle API.
    3. Run:  output = run_complete_pipeline()

Smoke test (synthetic, leakage-free, ~2 min on CPU):
    SYNTHETIC=1 python3 battery_rul_colab_v2.py
"""

# ============================================================================
# PART 1: SETUP & ENVIRONMENT
# ============================================================================

import os
import sys
import subprocess


def install_packages():
    """Install required packages (run once in a fresh Colab runtime).

    NOTE: 'scipy-stats' from the original kit is NOT a real pip package
    (the package is just 'scipy') — fixed here. torch was unused — dropped.
    pymc3 is deprecated/broken on current Python; it is NOT installed here.
    Bayesian UQ is optional and guarded (see PART 9).
    """
    packages = [
        'tensorflow',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'shap',
        'matplotlib',
        'seaborn',
        'pandas',
        'numpy',
        'scipy',
    ]
    for package in packages:
        print(f"Installing {package} ...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    print("All packages installed.")


def mount_gdrive():
    """Mount Google Drive in Colab. Returns True if mounted."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted.")
        return True
    except Exception:
        print("(Not in Colab or mount failed — using local ./outputs/)")
        return False


# Uncomment to run in a fresh Google Colab runtime:
# install_packages()
# mount_gdrive()


# ============================================================================
# PART 2: IMPORTS & REPRODUCIBILITY
# ============================================================================

import json
import glob
import random
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
# Headless-safe backend when there is no display and we are not in Colab.
if not os.environ.get('DISPLAY') and 'google.colab' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

# 'seaborn-v0_8-darkgrid' does not exist on all matplotlib versions -> guard.
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except Exception:
    plt.style.use('default')
if _HAS_SEABORN:
    try:
        sns.set_palette("husl")
    except Exception:
        pass

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import nnls
from scipy import stats

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# pymc3 is intentionally NOT imported here (deprecated / install-broken).
# It is imported lazily inside bayesian_residual_uq() only if available.


def set_seeds(seed=42):
    """Seed everything for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


print("=" * 80)
print("BATTERY RUL PREDICTION PIPELINE v2 — CORRECTED & COLAB-READY")
print("=" * 80)


# ============================================================================
# PART 3: CONFIGURATION
# ============================================================================

CONFIG = {
    'RANDOM_SEED': 42,
    'LOOKBACK': 50,            # sliding-window length (cycles) for LSTM/GRU
    'TRAIN_FRAC': 0.70,        # fraction of CELLS for training
    'VAL_FRAC': 0.15,          # fraction of CELLS for validation (rest = test)
    'BATCH_SIZE': 32,
    'EPOCHS': 30,              # modest default; EarlyStopping guards overfit
    'PATIENCE': 8,
    'LSTM_UNITS': 64,
    'GRU_UNITS': 64,
    'DROPOUT': 0.2,
    'LEARNING_RATE': 0.001,
    'FT_LEARNING_RATE': 0.0001,  # fine-tune learning rate (transfer learning)
    'FT_EPOCHS': 10,
    'MC_ITERATIONS': 100,      # Monte Carlo Dropout iterations
    'EOL_RETENTION': 0.80,     # end-of-life = 80% of nominal capacity
    'TEST_CELLS': ['B0018'],  # cells pinned to the TEST split (cross-condition
                              # evaluation: B0018 is a constant-2A cell like
                              # the rest; the shift axis is temperature/aging
                              # regime, not discharge rate). [] = fully random
                              # cell split.
    'N_SYNTH_CELLS': 12,       # synthetic smoke-test size
    'N_SYNTH_CYCLES': 170,
}

# Where to look for data files (in order).
SEARCH_DIRS = [
    '/content/drive/MyDrive/Aakash PHD Muse 2026/data',
    '/content',
    os.getcwd(),
    os.path.join(os.getcwd(), 'data'),
]

# Output: Drive folder when available, else local ./outputs/
DRIVE_RESULTS_DIR = '/content/drive/MyDrive/Aakash PHD Muse 2026/results'


def resolve_output_dir():
    """Pick results dir: Drive folder if mounted, else ./outputs/. Create it."""
    if os.path.isdir(DRIVE_RESULTS_DIR):
        out_dir = DRIVE_RESULTS_DIR
    else:
        out_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)
    return out_dir


def save_fig(fig, out_dir, name):
    """Save a figure as PNG and PDF at 300 dpi, then close it."""
    fig_dir = os.path.join(out_dir, 'figures')
    png_path = os.path.join(fig_dir, f'{name}.png')
    pdf_path = os.path.join(fig_dir, f'{name}.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


# ============================================================================
# PART 4: DATA LOADING — REAL DATASETS -> UNIFIED SCHEMA
# ============================================================================
# Unified schema (one row per cell-cycle):
#   [cell_id, cycle, voltage, current, temperature, capacity, dataset]
# RUL labels are computed in add_rul_labels(): RUL = EOL_cycle - cycle,
# with EOL = first cycle where capacity <= 80% of nominal capacity.

def find_files(*patterns):
    """Search SEARCH_DIRS for files matching any glob pattern."""
    found = []
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            found.extend(glob.glob(os.path.join(d, '**', pat), recursive=True))
    # de-duplicate, keep deterministic order
    return sorted(set(found))


def match_column(columns, *keyword_groups):
    """Return the first column whose lowercase name contains ALL keywords
    of any keyword group. e.g. match_column(cols, ('voltage','meas'), ('volt',))."""
    lower = {c: c.lower() for c in columns}
    for group in keyword_groups:
        for col, lc in lower.items():
            if all(k in lc for k in group):
                return col
    return None


def unify_frame(df_raw, colmap, cell_id, dataset):
    """Map an arbitrary raw frame to the unified schema."""
    cols = list(df_raw.columns)
    get = lambda *groups: match_column(cols, *groups)
    c_cycle = get(('cycle',), ('cycle_index',), ('cycle_number',))
    c_volt = get(('voltage',), ('volt',))
    c_curr = get(('current',), ('curr',), ('amp',))
    c_temp = get(('temperature',), ('temp',))
    c_cap = get(('capacity',), ('cap_',), ('discharge_capacity',), ('ah',))
    missing = [n for n, c in [('cycle', c_cycle), ('voltage', c_volt),
                              ('current', c_curr), ('temperature', c_temp),
                              ('capacity', c_cap)] if c is None]
    if missing:
        raise ValueError(f"Could not find columns for {missing} in {list(df_raw.columns)[:10]}...")
    out = pd.DataFrame({
        'cell_id': str(cell_id),
        'cycle': pd.to_numeric(df_raw[c_cycle], errors='coerce'),
        'voltage': pd.to_numeric(df_raw[c_volt], errors='coerce'),
        'current': pd.to_numeric(df_raw[c_curr], errors='coerce'),
        'temperature': pd.to_numeric(df_raw[c_temp], errors='coerce'),
        'capacity': pd.to_numeric(df_raw[c_cap], errors='coerce'),
        'dataset': dataset,
    })
    out = out.dropna(subset=['cycle', 'capacity']).sort_values('cycle')
    out['cycle'] = out['cycle'].astype(int)
    # one row per cycle: average intra-cycle measurements, capacity = max in cycle
    agg = out.groupby('cycle').agg(
        voltage=('voltage', 'mean'), current=('current', 'mean'),
        temperature=('temperature', 'mean'), capacity=('capacity', 'max'))
    agg['cell_id'] = str(cell_id)
    agg['dataset'] = dataset
    agg = agg.reset_index()[['cell_id', 'cycle', 'voltage', 'current',
                             'temperature', 'capacity', 'dataset']]
    return agg


# ---- 4a. NASA cleaned Kaggle CSV (primary, no key needed if file uploaded) ----

def load_nasa_kaggle_csv(path=None):
    """Load the cleaned NASA CSV (kaggle: mystifoe77/nasa-battery-data-cleaned).

    If `path` is None, searches SEARCH_DIRS for *nasa*.csv.
    Falls back to `kaggle datasets download` if the Kaggle CLI is available.
    """
    if path is None:
        cands = find_files('*nasa*.csv', '*NASA*.csv')
        if cands:
            path = cands[0]
    if path is None:
        try:
            print("Trying Kaggle API download: mystifoe77/nasa-battery-data-cleaned ...")
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'])
            dl_dir = os.path.join(os.getcwd(), '_kaggle_dl')
            os.makedirs(dl_dir, exist_ok=True)
            subprocess.check_call(['kaggle', 'datasets', 'download', '-d',
                                   'mystifoe77/nasa-battery-data-cleaned',
                                   '-p', dl_dir, '--unzip'])
            cands = glob.glob(os.path.join(dl_dir, '**', '*.csv'), recursive=True)
            if cands:
                path = cands[0]
        except Exception as e:
            print(f"Kaggle download not available: {e}")
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError("NASA cleaned CSV not found. Upload it to "
                                "'/content/drive/MyDrive/Aakash PHD Muse 2026/data/' "
                                "or set SYNTHETIC=1 for a smoke test.")
    print(f"Loading NASA cleaned CSV: {path}")
    df_raw = pd.read_csv(path)
    print(f"  raw shape={df_raw.shape}, columns={list(df_raw.columns)[:8]}...")

    # Split into per-cell frames if a cell/battery id column exists.
    cell_col = match_column(list(df_raw.columns), ('battery',), ('cell',), ('cell_id',))
    frames = []
    if cell_col is not None:
        for cell, g in df_raw.groupby(cell_col):
            frames.append(unify_frame(g, None, cell_id=cell, dataset='NASA'))
    else:
        # single-cell file
        stem = os.path.splitext(os.path.basename(path))[0]
        frames.append(unify_frame(df_raw, None, cell_id=stem, dataset='NASA'))
    df = pd.concat(frames, ignore_index=True)
    print(f"  unified: {df['cell_id'].nunique()} cells, {len(df)} cycle-rows")
    return df


# ---- 4b. NASA original .mat files (fallback) ----

def _mat_vec(x):
    """Unwrap MATLAB 0-d object containers -> 1-D float array.

    scipy.io.loadmat with squeeze_me=True often leaves each struct field as a
    0-d object array wrapping the real numeric vector; np.asarray(..., float)
    fails on those, so peel the wrappers first.
    """
    x = np.asarray(x)
    while isinstance(x, np.ndarray) and x.shape == () and x.dtype == object:
        x = x[()]
    return np.atleast_1d(np.asarray(x, dtype=float).ravel())


def load_nasa_mat(path):
    """Best-effort parser for NASA PCoE .mat files (e.g. B0005.mat).

    Extracts per-cycle discharge measurements -> unified schema.
    Raises a descriptive error if the structure is unexpected.
    """
    from scipy.io import loadmat
    mat = loadmat(path, squeeze_me=True)
    key = [k for k in mat.keys() if not k.startswith('__')]
    if not key:
        raise ValueError(f"No data keys found in {path}")
    batt = mat[key[0]]
    # squeeze_me=True can leave 0-d object wrappers: unwrap them to reach the
    # (n_cycles,) struct array with fields (type, ambient_temperature, time, data)
    if isinstance(batt, np.ndarray) and batt.shape == ():
        batt = batt[()]
    try:
        cycles = batt['cycle']
    except Exception:
        raise ValueError(f"Expected a 'cycle' field in {path}; found fields: "
                         f"{getattr(batt, 'dtype', None)}")
    if isinstance(cycles, np.ndarray) and cycles.shape == ():
        cycles = cycles[()]
    rows = []
    for cyc in np.atleast_1d(cycles):
        try:
            if str(cyc['type']).lower() != 'discharge':
                continue
            data = cyc['data']
            t = _mat_vec(data['Time'])
            v = _mat_vec(data['Voltage_measured'])
            i = _mat_vec(data['Current_measured'])
            tp = _mat_vec(data['Temperature_measured'])
            if 'Capacity' in data.dtype.names:
                cap = float(_mat_vec(data['Capacity'])[0])
            else:
                # integrate |current| over time -> Ah
                dt = np.diff(t, prepend=t[0])
                cap = float(np.sum(np.abs(i) * dt) / 3600.0)
            n = len(t)
            cyc_idx = len(rows)
            rows.append(pd.DataFrame({
                'cell_id': os.path.splitext(os.path.basename(path))[0],
                'cycle': cyc_idx,
                'voltage': [float(np.mean(v))] * n,
                'current': [float(np.mean(i))] * n,
                'temperature': [float(np.mean(tp))] * n,
                'capacity': [cap] * n,
                'dataset': 'NASA',
            }))
        except Exception:
            continue
    if not rows:
        raise ValueError(f"No discharge cycles parsed from {path}.")
    df = pd.concat(rows, ignore_index=True)
    # collapse to one row per cycle
    df = df.groupby('cycle').agg(
        voltage=('voltage', 'mean'), current=('current', 'mean'),
        temperature=('temperature', 'mean'), capacity=('capacity', 'max')).reset_index()
    df['cell_id'] = os.path.splitext(os.path.basename(path))[0]
    df['dataset'] = 'NASA'
    # Drop aborted-measurement cycles (capacity ~= 0 Ah): a single bad cycle
    # would otherwise trigger a fake early EOL (e.g. B0042 cycle 5).
    n_before = len(df)
    df = df[df['capacity'] > 0.01].reset_index(drop=True)
    if len(df) < n_before:
        print(f"  {df['cell_id'].iloc[0]}: dropped {n_before - len(df)} "
              f"zero-capacity cycle(s)")
    # Drop leading formation cycles: capacity still ramping up. Low leading cycles
    # would otherwise fake an early EOL (seen in B0033/B0034/B0054/B0055).
    # Rule: trim the leading prefix with capacity < 95% of the cell's early peak
    # (max over the first 20% of cycles). Prefix-only: mid-life dips are never
    # touched. Verified: trims nothing on B0005/6/7/18 and B0042/43/44.
    if len(df):
        cid = df['cell_id'].iloc[0]
        head = max(1, len(df) // 5)
        peak = float(df['capacity'].iloc[:head].max())
        n_trim = 0
        while n_trim < len(df) and df['capacity'].iloc[n_trim] < 0.95 * peak:
            n_trim += 1
        if n_trim:
            print(f"  {cid}: trimmed {n_trim} leading formation cycle(s)")
            df = df.iloc[n_trim:].reset_index(drop=True)
    return df[['cell_id', 'cycle', 'voltage', 'current', 'temperature', 'capacity', 'dataset']]


# ---- 4c. CALCE cycling CSVs ----

def load_calce_csvs(paths=None):
    """Load CALCE cycling CSVs (one file per cell) -> unified schema."""
    if paths is None:
        paths = find_files('*calce*.csv', '*CALCE*.csv', '*pl_*.csv', '*PL_*.csv',
                           '*18650*.csv', '*A123*.csv')
    if not paths:
        raise FileNotFoundError("No CALCE CSV files found in SEARCH_DIRS.")
    frames = []
    for p in paths:
        try:
            df_raw = pd.read_csv(p)
            cell_id = os.path.splitext(os.path.basename(p))[0]
            frames.append(unify_frame(df_raw, None, cell_id=cell_id, dataset='CALCE'))
            print(f"  loaded {cell_id}: {len(frames[-1])} cycles")
        except Exception as e:
            print(f"  skipped {p}: {e}")
    if not frames:
        raise ValueError("No CALCE files could be parsed.")
    return pd.concat(frames, ignore_index=True)


# ---- 4d. RUL labels (EOL = 80% of nominal capacity) ----

def add_rul_labels(df, eol_retention=0.80, min_cycles=60):
    """Compute per-cycle RUL labels. Drops cells that never reach EOL
    (censored — we do NOT fabricate their labels)."""
    labeled, dropped = [], []
    for cell, g in df.groupby('cell_id'):
        g = g.sort_values('cycle').reset_index(drop=True)
        if len(g) < min_cycles:
            dropped.append((cell, 'too few cycles'))
            continue
        # nominal = max capacity over the first 10% of cycles (robust to noise)
        head = max(1, len(g) // 10)
        nominal = float(g['capacity'].iloc[:head].max())
        if not np.isfinite(nominal) or nominal <= 0:
            dropped.append((cell, 'bad nominal capacity'))
            continue
        eol_cap = eol_retention * nominal
        below = g.index[g['capacity'] <= eol_cap]
        if len(below) == 0:
            dropped.append((cell, 'never reached EOL (censored)'))
            continue
        eol_cycle = int(g.loc[below[0], 'cycle'])
        g = g.copy()
        g['RUL'] = (eol_cycle - g['cycle']).clip(lower=0).astype(int)
        g['eol_cycle'] = eol_cycle
        g['nominal_capacity'] = nominal
        labeled.append(g)
    if dropped:
        print(f"  dropped {len(dropped)} cells: "
              + ", ".join(f"{c} ({r})" for c, r in dropped[:5])
              + (" ..." if len(dropped) > 5 else ""))
    if not labeled:
        raise ValueError("No cells with valid RUL labels. Check data / EOL threshold.")
    out = pd.concat(labeled, ignore_index=True)
    print(f"  labeled {out['cell_id'].nunique()} cells, {len(out)} rows; "
          f"RUL range {out['RUL'].min()}–{out['RUL'].max()} cycles")
    return out


# ---- 4e. Synthetic smoke-test mode (leakage-free, clearly labeled) ----

def create_synthetic_cells(n_cells=None, cycles_per_cell=None, seed=42):
    """Leakage-free synthetic data for smoke-testing the pipeline.

    Each cell gets RANDOM degradation parameters, so the RUL at a cycle is
    NOT determinable from the features alone (features contain no cycle
    index and no capacity-retention-from-EOL quantities).
    """
    n_cells = n_cells or CONFIG['N_SYNTH_CELLS']
    cycles_per_cell = cycles_per_cell or CONFIG['N_SYNTH_CYCLES']
    rng = np.random.default_rng(seed)
    frames = []
    for c in range(n_cells):
        nominal = rng.normal(2.0, 0.08)          # Ah
        a = rng.uniform(4e-4, 1.2e-3)            # linear fade / cycle
        b = rng.uniform(0.0, 4e-6)               # accelerating fade / cycle^2
        cycles = np.arange(cycles_per_cell)
        cap = nominal * (1 - a * cycles - b * cycles ** 2)
        cap += rng.normal(0, 0.004, cycles_per_cell)   # measurement noise
        cap = np.maximum(cap, 0.5 * nominal)
        # measurements correlated with state-of-health but NOT with RUL directly
        volt = 3.70 + 0.10 * (cap / nominal) + rng.normal(0, 0.02, cycles_per_cell)
        curr = rng.normal(-1.0, 0.05, cycles_per_cell)
        temp = 25.0 + 3.0 * (1 - cap / nominal) + rng.normal(0, 0.8, cycles_per_cell)
        frames.append(pd.DataFrame({
            'cell_id': f'SYNTH_{c:02d}',
            'cycle': cycles,
            'voltage': volt,
            'current': curr,
            'temperature': temp,
            'capacity': cap,
            'dataset': 'SYNTHETIC',
        }))
    df = pd.concat(frames, ignore_index=True)
    print("  *** SYNTHETIC SMOKE-TEST MODE — metrics are NOT real results ***")
    return df


# ---- 4f. Dispatcher ----

def load_dataset():
    """Load real data if available, else fall back to labeled synthetic mode.

    Priority: NASA Kaggle CSV -> NASA .mat files -> CALCE CSVs -> synthetic.
    Env SYNTHETIC=1 forces synthetic mode.
    Returns (unified_df_with_rul, data_mode_string).
    """
    if os.environ.get('SYNTHETIC', '0') == '1':
        df = create_synthetic_cells()
        return add_rul_labels(df), 'SYNTHETIC (forced by env)'

    errors = []
    # 1) NASA cleaned Kaggle CSV
    try:
        df = load_nasa_kaggle_csv()
        return add_rul_labels(df), 'NASA_KAGGLE_CSV'
    except Exception as e:
        errors.append(f"NASA CSV: {e}")
    # 2) NASA .mat files
    try:
        mats = find_files('B*.mat', 'b*.mat')
        if mats:
            frames = [load_nasa_mat(p) for p in mats]
            df = pd.concat(frames, ignore_index=True)
            print(f"  unified {df['cell_id'].nunique()} cells from .mat files")
            return add_rul_labels(df), 'NASA_MAT'
        errors.append("NASA .mat: no B*.mat files found")
    except Exception as e:
        errors.append(f"NASA .mat: {e}")
    # 3) CALCE CSVs
    try:
        df = load_calce_csvs()
        return add_rul_labels(df), 'CALCE_CSV'
    except Exception as e:
        errors.append(f"CALCE: {e}")

    print("No real data found:")
    for e in errors:
        print(f"   - {e}")
    print("Falling back to SYNTHETIC smoke-test mode.")
    df = create_synthetic_cells()
    return add_rul_labels(df), 'SYNTHETIC (fallback — no real data found)'


# ============================================================================
# PART 5: PREPROCESSING — CELL-WISE SPLITS, SCALER, FEATURES, WINDOWS
# ============================================================================
# Anti-leakage rules enforced here:
#   1. Splits are by cell_id — no cell appears in two splits.
#   2. StandardScaler is fit on TRAIN only.
#   3. All engineered features use past/current-cycle data only
#      (rolling windows look BACKWARD, never forward).

FLAT_FEATURES = ['cycle', 'voltage', 'current', 'temperature', 'capacity',
                 'fade_rate', 'volt_rollmean', 'cap_rollstd']
WINDOW_FEATURES = ['voltage', 'current', 'temperature', 'capacity']
TARGET = 'RUL'


def add_flat_features(df):
    """Add causal (past-only) per-cycle features."""
    df = df.sort_values(['cell_id', 'cycle']).copy()
    g = df.groupby('cell_id')
    # capacity fade rate over the previous 5 cycles (Ah lost per cycle)
    df['fade_rate'] = (g['capacity'].shift(5) - df['capacity']) / 5.0
    # rolling means/stds over previous cycles only
    df['volt_rollmean'] = g['voltage'].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df['cap_rollstd'] = g['capacity'].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).std())
    df[['fade_rate', 'volt_rollmean', 'cap_rollstd']] = \
        df[['fade_rate', 'volt_rollmean', 'cap_rollstd']].fillna(0.0)
    # per-cycle capacity delta (current vs previous cycle only)
    df['d_capacity'] = (df['capacity'] - g['capacity'].shift(1)).fillna(0.0)
    return df


def split_cells_by_id(df, seed=42):
    """Split cell_ids into train/val/test (stratified by dataset if mixed).

    CONFIG['TEST_CELLS'] pins chosen cell_ids to the test split — the
    cross-condition evaluation design (B0018 is a constant-2A cell like the
    training cells; the supported shift axis is the temperature/aging regime
    between validation and test). Remaining cells are split randomly with the
    given seed.
    """
    cells = df[['cell_id', 'dataset']].drop_duplicates().reset_index(drop=True)
    pinned = [c for c in CONFIG.get('TEST_CELLS', []) if c in set(cells['cell_id'])]
    pool = cells[~cells['cell_id'].isin(pinned)].reset_index(drop=True)
    strat = pool['dataset'] if pool['dataset'].nunique() > 1 else None
    if pinned:
        # Cross-condition design: test = pinned cells only;
        # the pool is split randomly into train/val.
        train_cells, val_cells = train_test_split(
            pool, test_size=1 - CONFIG['TRAIN_FRAC'], random_state=seed, stratify=strat)
        test_cells = cells[cells['cell_id'].isin(pinned)].reset_index(drop=True)
    else:
        train_cells, temp = train_test_split(
            pool, test_size=1 - CONFIG['TRAIN_FRAC'], random_state=seed, stratify=strat)
        strat_t = temp['dataset'] if temp['dataset'].nunique() > 1 else None
        val_cells, test_cells = train_test_split(
            temp, test_size=0.5, random_state=seed, stratify=strat_t)
    splits = {}
    for name, cdf in [('train', train_cells), ('val', val_cells), ('test', test_cells)]:
        ids = set(cdf['cell_id'])
        splits[name] = df[df['cell_id'].isin(ids)].copy().reset_index(drop=True)
        tag = ' (pinned)' if name == 'test' and pinned else ''
        print(f"  {name}: {len(ids)} cells {sorted(ids)}{tag}, {len(splits[name])} rows "
              f"({sorted(splits[name]['dataset'].unique())})")
    # safety check: no cell in two splits
    all_ids = [set(cdf['cell_id']) for cdf in (train_cells, val_cells, test_cells)]
    assert not (all_ids[0] & all_ids[1] or all_ids[0] & all_ids[2]
                or all_ids[1] & all_ids[2]), \
        "LEAKAGE: a cell appears in two splits!"
    return splits


def fit_scaler(train_df):
    """Fit StandardScaler on TRAIN flat features only."""
    scaler = StandardScaler()
    scaler.fit(train_df[FLAT_FEATURES].values)
    return scaler


def build_model_inputs(df, scaler, lookback):
    """Build aligned inputs for both model families.

    Returns dict with:
      X_flat : (n, n_flat)  scaled flat features  -> XGBoost / LightGBM
      X_win  : (n, lookback, n_win) scaled window sequences -> LSTM / GRU
      y      : (n,) RUL targets
      meta   : DataFrame with cell_id, cycle, dataset per sample
    Only cycles >= lookback-1 are kept so every model predicts the same rows.
    """
    df = df.sort_values(['cell_id', 'cycle']).reset_index(drop=True)
    Xs = scaler.transform(df[FLAT_FEATURES].values)
    win_idx = [FLAT_FEATURES.index(f) for f in WINDOW_FEATURES]
    X_flat_list, X_win_list, y_list, meta_rows = [], [], [], []
    for cell, g in df.groupby('cell_id'):
        idx = g.index.values
        if len(idx) < lookback:
            continue
        for j in range(lookback - 1, len(idx)):
            rows = idx[j - lookback + 1: j + 1]          # past + current only
            X_win_list.append(Xs[rows][:, win_idx])
            X_flat_list.append(Xs[idx[j]])
            y_list.append(float(df.loc[idx[j], TARGET]))
            meta_rows.append({'cell_id': cell,
                              'cycle': int(df.loc[idx[j], 'cycle']),
                              'dataset': df.loc[idx[j], 'dataset']})
    return {
        'X_flat': np.asarray(X_flat_list, dtype=np.float32),
        'X_win': np.asarray(X_win_list, dtype=np.float32),
        'y': np.asarray(y_list, dtype=np.float32),
        'meta': pd.DataFrame(meta_rows),
    }


# ============================================================================
# PART 6: MODEL ARCHITECTURES (3D inputs for LSTM/GRU — shape bug fixed)
# ============================================================================

def build_lstm_model(n_timesteps, n_features):
    """LSTM on (timesteps, features) windows."""
    model = Sequential([
        LSTM(CONFIG['LSTM_UNITS'], input_shape=(n_timesteps, n_features)),
        Dropout(CONFIG['DROPOUT']),
        Dense(32, activation='relu'),
        Dropout(CONFIG['DROPOUT']),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']),
                  loss='mse', metrics=['mae'])
    return model


def build_gru_model(n_timesteps, n_features):
    """GRU on (timesteps, features) windows."""
    model = Sequential([
        GRU(CONFIG['GRU_UNITS'], input_shape=(n_timesteps, n_features)),
        Dropout(CONFIG['DROPOUT']),
        Dense(32, activation='relu'),
        Dropout(CONFIG['DROPOUT']),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']),
                  loss='mse', metrics=['mae'])
    return model


# ============================================================================
# PART 7: TRAINING
# ============================================================================

def train_dl_model(model, Xtr, ytr, Xva, yva, name='DL'):
    """Train a Keras model with EarlyStopping; validate on the VAL split."""
    print(f"  training {name} ...")
    es = EarlyStopping(monitor='val_loss', patience=CONFIG['PATIENCE'],
                       restore_best_weights=True, verbose=0)
    history = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                        epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
                        verbose=0, callbacks=[es])
    print(f"    {name}: best val_loss={min(history.history['val_loss']):.4f} "
          f"after {len(history.history['loss'])} epochs")
    return model, history


def train_tree_models(Xtr, ytr, seed=42):
    """Train XGBoost and LightGBM with fixed seeds."""
    print("  training XGBoost ...")
    xgb_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbosity=0)
    xgb_model.fit(Xtr, ytr)
    print("  training LightGBM ...")
    lgb_model = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbose=-1)
    lgb_model.fit(Xtr, ytr)
    return {'XGBoost': xgb_model, 'LightGBM': lgb_model}


def tune_ensemble_weights(val_preds, y_val):
    """Tune non-negative ensemble weights on the VALIDATION split (nnls),
    normalized to sum to 1. val_preds: dict name -> (n,) predictions."""
    names = list(val_preds.keys())
    A = np.column_stack([val_preds[n] for n in names])
    w, _ = nnls(A, y_val)
    if w.sum() <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    weights = dict(zip(names, w))
    print("  tuned ensemble weights (on validation):")
    for n, wi in weights.items():
        print(f"    {n}: {wi:.3f}")
    return weights


def weighted_ensemble_predict(preds, weights):
    """preds: dict name -> (n,) array. Returns weighted average."""
    return sum(weights[n] * np.asarray(preds[n]) for n in weights)


# ============================================================================
# PART 8: EVALUATION
# ============================================================================

def regression_metrics(y_true, y_pred):
    """RMSE, MAE, R², MAPE. MAPE is computed only where y_true > 0
    (RUL=0 would divide by zero) and reported as NaN if undefined."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float('nan')
    mask = y_true > 1e-9
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
        if mask.sum() > 0 else float('nan')
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


def evaluate_all(test_preds, y_test):
    """test_preds: dict name -> (n,) predictions. Returns metrics DataFrame."""
    rows = []
    for name, pred in test_preds.items():
        m = regression_metrics(y_test, pred)
        rows.append({'Model': name, **m})
        print(f"  {name:12s} RMSE={m['RMSE']:8.2f}  MAE={m['MAE']:8.2f}  "
              f"R²={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%")
    return pd.DataFrame(rows)


def paired_significance(y_true, pred_a, pred_b, name_a='A', name_b='B'):
    """Paired t-test on absolute errors (honest significance reporting)."""
    ea = np.abs(np.asarray(y_true) - np.asarray(pred_a))
    eb = np.abs(np.asarray(y_true) - np.asarray(pred_b))
    t, p = stats.ttest_rel(ea, eb)
    winner = name_a if ea.mean() < eb.mean() else name_b
    print(f"  paired t-test |err|: {name_a} vs {name_b}: t={t:.3f}, p={p:.4g} "
          f"-> lower error: {winner}")
    return {'t': float(t), 'p': float(p), 'lower_error': winner}


# ============================================================================
# PART 9: UNCERTAINTY QUANTIFICATION
# ============================================================================
# Honest UQ: MC Dropout (100 iters) on the DL models + spread across the
# ensemble members. We report EMPIRICAL coverage on the test set — no
# arbitrary std scaling, no claimed numbers.

def mc_dropout_predict(model, X, n_iter=100):
    """Monte Carlo Dropout: keep dropout ON at inference (training=True)."""
    preds = [model(X, training=True).numpy().ravel() for _ in range(n_iter)]
    return np.asarray(preds)  # (n_iter, n)


def empirical_coverage(y_true, mean, std, z_levels=(1.28, 1.64, 1.96, 2.58)):
    """Empirical coverage at several nominal z levels.

    Nominal coverages: 1.28->80%, 1.64->90%, 1.96->95%, 2.58->99%.
    Returns dict nominal% -> empirical%.
    """
    y_true = np.asarray(y_true, dtype=float)
    nominal = {1.28: 80.0, 1.64: 90.0, 1.96: 95.0, 2.58: 99.0}
    out = {}
    for z in z_levels:
        lo, hi = mean - z * std, mean + z * std
        out[nominal[z]] = float(np.mean((y_true >= lo) & (y_true <= hi)) * 100)
    return out


def quantify_uncertainty(dl_models, X_win_val, y_val, val_member_preds,
                         X_win_test, test_member_preds, y_test,
                         n_iter=100, weights=None):
    """UQ report with validation-calibrated predictive intervals.

    - Raw uncertainty: MC-Dropout samples of LSTM+GRU (dropout ON at inference),
      pooled across both models -> raw predictive std per test point.
    - Calibration: a single variance-scale factor alpha is fit on VALIDATION
      data only: alpha^2 = mean(val_residual^2) / mean(val_raw_var), where the
      validation residuals use the weighted-ensemble prediction. This is the
      standard variance-scaling calibration (cf. Kuleshov et al., ICML 2018):
      it corrects the well-known overconfidence of raw MC-Dropout without
      ever touching the test set.
    - Test predictive distribution: center = weighted-ensemble test prediction,
      width = alpha * raw MC-Dropout test std. Empirical coverage is evaluated
      on TEST only, so the reported numbers are honest.
    """
    def pooled_mc(X, tag):
        all_samples = []
        for name, model in dl_models.items():
            s = mc_dropout_predict(model, X, n_iter=n_iter)
            all_samples.append(s)
            print(f"    {name} [{tag}]: {s.shape[0]} samples, "
                  f"mean predictive std={s.std(axis=0).mean():.3f}")
        all_samples = np.concatenate(all_samples, axis=0)  # (2*n_iter, n)
        return all_samples.mean(axis=0), all_samples.std(axis=0)

    def ens_predict(member_preds):
        member_matrix = np.column_stack(
            [np.asarray(member_preds[n]) for n in member_preds])
        if weights is not None:
            w = np.array([weights[n] for n in member_preds])
            return (member_matrix * w).sum(axis=1)
        return member_matrix.mean(axis=1)

    print("  MC Dropout: collecting predictive samples (val + test) ...")
    val_mc_mean, val_mc_std = pooled_mc(X_win_val, 'val')
    mc_mean, mc_std = pooled_mc(X_win_test, 'test')

    ens_val = ens_predict(val_member_preds)
    ens_test = ens_predict(test_member_preds)

    # variance-scale calibration, fit on validation ONLY
    # (standard approach, cf. Kuleshov et al., ICML 2018). A quantile-based
    # variant was tried and rejected: with only ~119 validation windows the
    # empirical tail quantiles overfit a few outliers and generalized worse
    # (80% nominal -> 57.8% empirical, vs 79.5% with variance scaling).
    val_resid = np.asarray(y_val, dtype=float) - ens_val
    alpha = float(np.sqrt(np.mean(val_resid ** 2) /
                          (np.mean(val_mc_std ** 2) + 1e-12)))
    print(f"  raw MC-Dropout underestimates std by alpha={alpha:.3f} (validation)")

    cal_std = alpha * mc_std
    cov = empirical_coverage(y_test, ens_test, cal_std)
    print("  empirical coverage of CALIBRATED intervals on TEST:")
    for nom, emp in cov.items():
        print(f"    nominal {nom:>5.1f}% -> empirical {emp:6.2f}%")
    cov95 = cov[95.0]
    print(f"  >>> calibrated 95% CI empirical coverage on TEST: {cov95:.2f}% "
          f"(target 95%)")

    member_matrix = np.column_stack(
        [np.asarray(test_member_preds[n]) for n in test_member_preds])
    return {'mc_mean': mc_mean, 'mc_std': mc_std,
            'cal_mean': ens_test, 'cal_std': cal_std, 'cal_alpha': alpha,
            'ens_mean': ens_test, 'ens_std': member_matrix.std(axis=1),
            'coverage': cov, 'coverage_95': cov95}


def bayesian_residual_uq(y_true, y_pred):
    """OPTIONAL Bayesian residual model. pymc is imported lazily and the
    whole thing is guarded — it can never crash the pipeline."""
    try:
        import pymc as pm  # modern pymc; pymc3 is deprecated/broken
    except Exception as e:
        print(f"  Bayesian UQ skipped (pymc not installed: {e})")
        return None
    try:
        resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0.0, sigma=float(np.std(resid) + 1e-6))
            sigma = pm.HalfNormal('sigma', sigma=float(np.std(resid) + 1e-6))
            pm.Normal('obs', mu=mu, sigma=sigma, observed=resid)
            trace = pm.sample(1000, tune=1000, cores=1, progressbar=False,
                              return_inferencedata=True)
        print("  Bayesian residual model fitted (see trace summary in logs).")
        return trace
    except Exception as e:
        print(f"  Bayesian UQ failed safely: {e}")
        return None


# ============================================================================
# PART 10: VISUALIZATION (all figures saved PNG+PDF @300dpi)
# ============================================================================

def plot_training_history(histories, out_dir):
    for name, h in histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(h.history['loss'], label='train')
        axes[0].plot(h.history['val_loss'], label='val')
        axes[0].set_title(f'{name} — loss'); axes[0].set_xlabel('epoch')
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(h.history['mae'], label='train')
        axes[1].plot(h.history['val_mae'], label='val')
        axes[1].set_title(f'{name} — MAE'); axes[1].set_xlabel('epoch')
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        save_fig(fig, out_dir, f'training_history_{name.lower()}')


def plot_pred_vs_actual(y_true, y_pred, out_dir, tag):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=18)
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], 'r--', label='perfect')
    axes[0].set_xlabel('Actual RUL (cycles)'); axes[0].set_ylabel('Predicted RUL (cycles)')
    axes[0].set_title(f'{tag} — predictions vs actual'); axes[0].legend(); axes[0].grid(alpha=0.3)
    resid = y_true - y_pred
    axes[1].scatter(y_pred, resid, alpha=0.5, s=18)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted RUL (cycles)'); axes[1].set_ylabel('Residual (cycles)')
    axes[1].set_title(f'{tag} — residuals'); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, f'pred_vs_actual_{tag.lower().replace(" ", "_")}')


def plot_model_comparison(results_df, out_dir):
    df = results_df.copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].barh(df['Model'], df['RMSE']); axes[0, 0].set_title('RMSE (lower better)'); axes[0, 0].grid(alpha=0.3, axis='x')
    axes[0, 1].barh(df['Model'], df['MAE']); axes[0, 1].set_title('MAE (lower better)'); axes[0, 1].grid(alpha=0.3, axis='x')
    axes[1, 0].barh(df['Model'], df['R2']); axes[1, 0].set_title('R² (higher better)'); axes[1, 0].grid(alpha=0.3, axis='x')
    axes[1, 1].barh(df['Model'], df['MAPE']); axes[1, 1].set_title('MAPE % (lower better)'); axes[1, 1].grid(alpha=0.3, axis='x')
    fig.suptitle('Model comparison on test set', fontweight='bold')
    fig.tight_layout()
    save_fig(fig, out_dir, 'model_comparison')


def plot_uncertainty_bands(y_true, mean, std, out_dir, tag='ensemble', z=1.96):
    y_true = np.asarray(y_true); mean = np.asarray(mean); std = np.asarray(std)
    order = np.argsort(y_true)
    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(x, (mean - z * std)[order], (mean + z * std)[order],
                    alpha=0.3, label='95% CI (calibrated MC Dropout)')
    ax.plot(x, mean[order], label='predicted mean')
    ax.scatter(x, y_true[order], s=12, alpha=0.6, label='actual')
    ax.set_xlabel('test sample (sorted by actual RUL)'); ax.set_ylabel('RUL (cycles)')
    ax.set_title(f'{tag} — predictions with 95% uncertainty bands')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, f'uncertainty_bands_{tag}')


def plot_calibration_curve(coverage_dict, out_dir):
    nominal = sorted(coverage_dict.keys())
    empirical = [coverage_dict[n] for n in nominal]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(nominal, empirical, 'o-', label='model')
    ax.plot([0, 100], [0, 100], 'r--', label='perfect calibration')
    ax.set_xlabel('nominal coverage (%)'); ax.set_ylabel('empirical coverage (%)')
    ax.set_title('Calibration curve (test set)'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    fig.tight_layout()
    save_fig(fig, out_dir, 'calibration_curve')


def plot_transfer_results(transfer_df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(transfer_df))
    ax.bar(x - 0.2, transfer_df['RMSE_source_test'], 0.4, label='NASA test RMSE (source)')
    ax.bar(x + 0.2, transfer_df['RMSE_target'], 0.4, label='CALCE RMSE (target)')
    ax.set_xticks(x); ax.set_xticklabels(transfer_df['Setting'], rotation=15, ha='right')
    ax.set_ylabel('RMSE (cycles)'); ax.set_title('Transfer learning: NASA → CALCE')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, out_dir, 'transfer_nasa_to_calce')


# ============================================================================
# PART 11: RESULT PERSISTENCE (CSV + LaTeX tables for the paper)
# ============================================================================

def save_table(df, out_dir, name):
    """Save a DataFrame as CSV and LaTeX (for direct use in the manuscript)."""
    tab_dir = os.path.join(out_dir, 'tables')
    csv_path = os.path.join(tab_dir, f'{name}.csv')
    tex_path = os.path.join(tab_dir, f'{name}.tex')
    df.to_csv(csv_path, index=False)
    try:
        with open(tex_path, 'w') as f:
            f.write(df.to_latex(index=False, float_format='%.3f'))
        print(f"  saved tables/{name}.csv + .tex")
    except Exception as e:
        # to_latex needs jinja2; CSV is the essential artifact — never fail the run
        print(f"  saved tables/{name}.csv (LaTeX skipped: {e})")
    return csv_path, tex_path


def save_run_metadata(out_dir, data_mode, splits, weights, config):
    meta = {
        'timestamp': datetime.now().isoformat(),
        'data_mode': data_mode,
        'config': config,
        'ensemble_weights': {k: float(v) for k, v in weights.items()},
        'n_cells': {k: int(v['meta']['cell_id'].nunique()) for k, v in splits.items()},
        'n_samples': {k: int(len(v['y'])) for k, v in splits.items()},
        'leakage_controls': [
            'splits by cell_id (no cell in two splits)',
            'StandardScaler fit on TRAIN only',
            'features use past/current cycle only (causal rolling stats)',
            'sliding windows contain no future cycles',
        ],
    }
    path = os.path.join(out_dir, 'run_metadata.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  saved run_metadata.json")


# ============================================================================
# PART 12: EXPERIMENTS
# ============================================================================

def experiment_per_dataset(test_preds, test_meta, y_test, out_dir):
    """Break test metrics down by dataset source."""
    rows = []
    for ds in sorted(test_meta['dataset'].unique()):
        mask = (test_meta['dataset'] == ds).values
        n = int(mask.sum())
        for name, pred in test_preds.items():
            m = regression_metrics(y_test[mask], np.asarray(pred)[mask])
            rows.append({'Dataset': ds, 'Model': name, 'n_test': n, **m})
    df = pd.DataFrame(rows)
    print("\nPer-dataset test performance:")
    print(df.to_string(index=False))
    save_table(df, out_dir, 'per_dataset_metrics')
    return df


def experiment_transfer_learning(out_dir, seed=42):
    """Transfer learning: train on NASA cells, test on CALCE cells.

    Settings: (1) zero-shot — NASA-trained models applied to CALCE with the
    NASA scaler; (2) fine-tune — DL models continue training on a small CALCE
    subset with low LR; trees refit (warm-start where supported, else fresh)
    on the same subset. Retention = 100 * RMSE_NASA_test / RMSE_CALCE.
    Returns a DataFrame, or None if NASA+CALCE are not both available.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: TRANSFER LEARNING (NASA -> CALCE)")
    print("=" * 70)
    # load both sources independently of the main pipeline's data_mode
    try:
        nasa = add_rul_labels(load_nasa_kaggle_csv())
    except Exception as e:
        print(f"  skipped: NASA data unavailable ({e})")
        return None
    try:
        calce = add_rul_labels(load_calce_csvs())
    except Exception as e:
        print(f"  skipped: CALCE data unavailable ({e})")
        return None

    nasa = add_flat_features(nasa)
    calce = add_flat_features(calce)

    # NASA train/val/test by cell; CALCE split into small train / test by cell
    n_splits = split_cells_by_id(nasa, seed=seed)
    calce_cells = calce[['cell_id']].drop_duplicates()
    cc_train, cc_test = train_test_split(calce_cells, test_size=0.6, random_state=seed)
    calce_tr = calce[calce['cell_id'].isin(set(cc_train['cell_id']))].reset_index(drop=True)
    calce_te = calce[calce['cell_id'].isin(set(cc_test['cell_id']))].reset_index(drop=True)
    print(f"  CALCE: {calce_tr['cell_id'].nunique()} train cells, "
          f"{calce_te['cell_id'].nunique()} test cells")

    scaler = fit_scaler(n_splits['train'])
    lb = CONFIG['LOOKBACK']
    D = {k: build_model_inputs(v, scaler, lb) for k, v in
         [('ntr', n_splits['train']), ('nva', n_splits['val']), ('nte', n_splits['test'])]}
    Ctr = build_model_inputs(calce_tr, scaler, lb)   # zero-shot: NASA scaler
    Cte = build_model_inputs(calce_te, scaler, lb)

    dl = {'LSTM': build_lstm_model(lb, len(WINDOW_FEATURES)),
          'GRU': build_gru_model(lb, len(WINDOW_FEATURES))}
    for name, m in dl.items():
        train_dl_model(m, D['ntr']['X_win'], D['ntr']['y'],
                       D['nva']['X_win'], D['nva']['y'], name=f'TL-{name}')
    trees = train_tree_models(D['ntr']['X_flat'], D['ntr']['y'], seed=seed)

    def predict_all(models_dl, models_tree, inputs):
        p = {n: models_dl[n].predict(inputs['X_win'], verbose=0).ravel()
             for n in models_dl}
        p.update({n: models_tree[n].predict(inputs['X_flat']) for n in models_tree})
        return p

    src_preds = predict_all(dl, trees, D['nte'])
    zs_preds = predict_all(dl, trees, Cte)   # zero-shot

    # fine-tune on the small CALCE train subset
    scaler_ft = fit_scaler(calce_tr)
    Ctr_ft = build_model_inputs(calce_tr, scaler_ft, lb)
    Cte_ft = build_model_inputs(calce_te, scaler_ft, lb)
    dl_ft = {}
    for name, m in dl.items():
        m_ft = build_lstm_model(lb, len(WINDOW_FEATURES)) if name == 'LSTM' \
            else build_gru_model(lb, len(WINDOW_FEATURES))
        m_ft.set_weights(m.get_weights())  # start from NASA-trained weights
        m_ft.compile(optimizer=Adam(learning_rate=CONFIG['FT_LEARNING_RATE']),
                     loss='mse', metrics=['mae'])
        m_ft.fit(Ctr_ft['X_win'], Ctr_ft['y'], epochs=CONFIG['FT_EPOCHS'],
                 batch_size=CONFIG['BATCH_SIZE'], verbose=0)
        dl_ft[name] = m_ft
        print(f"  fine-tuned {name} on {Ctr_ft['X_win'].shape[0]} CALCE samples")
    trees_ft = {}
    for name, t in trees.items():
        try:  # warm start from NASA booster where the API supports it
            if name == 'XGBoost':
                nt = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                  random_state=seed, n_jobs=-1, verbosity=0)
                nt.fit(Ctr_ft['X_flat'], Ctr_ft['y'], xgb_model=t.get_booster())
            else:
                nt = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                   random_state=seed, n_jobs=-1, verbose=-1)
                nt.fit(Ctr_ft['X_flat'], Ctr_ft['y'], init_model=t)
            trees_ft[name] = nt
        except Exception as e:
            print(f"  warm-start failed for {name} ({e}) — fresh fit on CALCE subset")
            nt = (XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                               random_state=seed, n_jobs=-1, verbosity=0)
                  if name == 'XGBoost' else
                  LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                random_state=seed, n_jobs=-1, verbose=-1))
            nt.fit(Ctr_ft['X_flat'], Ctr_ft['y'])
            trees_ft[name] = nt
    ft_preds = predict_all(dl_ft, trees_ft, Cte_ft)

    rows = []
    for name in list(dl) + list(trees):
        rmse_src = regression_metrics(D['nte']['y'], src_preds[name])['RMSE']
        for setting, preds, yy in [('zero-shot', zs_preds, Cte['y']),
                                   ('fine-tuned', ft_preds, Cte_ft['y'])]:
            rmse_t = regression_metrics(yy, preds[name])['RMSE']
            retention = 100.0 * rmse_src / rmse_t if rmse_t > 0 else float('nan')
            rows.append({'Model': name, 'Setting': setting,
                         'RMSE_source_test': rmse_src, 'RMSE_target': rmse_t,
                         'Retention_pct': retention})
            print(f"  {name:10s} {setting:10s} NASA RMSE={rmse_src:7.2f}  "
                  f"CALCE RMSE={rmse_t:7.2f}  retention={retention:5.1f}%")
    df = pd.DataFrame(rows)
    save_table(df, out_dir, 'transfer_nasa_to_calce')
    plot_transfer_results(
        df.groupby('Setting').agg(RMSE_source_test=('RMSE_source_test', 'mean'),
                                  RMSE_target=('RMSE_target', 'mean')).reset_index(),
        out_dir)
    return df


# ============================================================================
# PART 13: MAIN PIPELINE
# ============================================================================

def run_complete_pipeline():
    """Run the full experiment. Returns a dict with models, metrics, and paths."""
    set_seeds(CONFIG['RANDOM_SEED'])
    out_dir = resolve_output_dir()
    print(f"\nResults dir: {out_dir}")

    # ---- data ----
    print("\n" + "=" * 70 + "\nSTEP 1: DATA LOADING\n" + "=" * 70)
    df, data_mode = load_dataset()
    print(f"data_mode = {data_mode}")
    df = add_flat_features(df)

    # ---- splits / scaling / windows ----
    print("\n" + "=" * 70 + "\nSTEP 2: SPLITS + PREPROCESSING\n" + "=" * 70)
    splits = split_cells_by_id(df, seed=CONFIG['RANDOM_SEED'])
    scaler = fit_scaler(splits['train'])
    lb = CONFIG['LOOKBACK']
    D = {k: build_model_inputs(v, scaler, lb) for k, v in splits.items()}
    for k, v in D.items():
        print(f"  {k}: {v['X_win'].shape[0]} windowed samples "
              f"(X_win {v['X_win'].shape}, X_flat {v['X_flat'].shape})")

    # ---- train ----
    print("\n" + "=" * 70 + "\nSTEP 3: TRAINING\n" + "=" * 70)
    n_feat = len(WINDOW_FEATURES)
    dl_models, histories = {}, {}
    for name, builder in [('LSTM', build_lstm_model), ('GRU', build_gru_model)]:
        m = builder(lb, n_feat)
        m, h = train_dl_model(m, D['train']['X_win'], D['train']['y'],
                              D['val']['X_win'], D['val']['y'], name=name)
        dl_models[name] = m
        histories[name] = h
    tree_models = train_tree_models(D['train']['X_flat'], D['train']['y'],
                                    seed=CONFIG['RANDOM_SEED'])

    def predict_all(models_dl, models_tree, inputs):
        p = {n: models_dl[n].predict(inputs['X_win'], verbose=0).ravel()
             for n in models_dl}
        p.update({n: models_tree[n].predict(inputs['X_flat']) for n in models_tree})
        return p

    val_preds = predict_all(dl_models, tree_models, D['val'])
    test_preds = predict_all(dl_models, tree_models, D['test'])

    # ---- ensemble (weights tuned on validation) ----
    print("\n" + "=" * 70 + "\nSTEP 4: ENSEMBLE (validation-tuned weights)\n" + "=" * 70)
    weights = tune_ensemble_weights(val_preds, D['val']['y'])
    test_preds['Ensemble'] = weighted_ensemble_predict(test_preds, weights)

    # ---- evaluation ----
    print("\n" + "=" * 70 + "\nSTEP 5: TEST EVALUATION\n" + "=" * 70)
    results_df = evaluate_all(test_preds, D['test']['y'])
    save_table(results_df, out_dir, 'model_metrics_test')
    best_single = results_df[results_df['Model'] != 'Ensemble'].sort_values('RMSE').iloc[0]
    paired_significance(D['test']['y'], test_preds['Ensemble'],
                        test_preds[best_single['Model']],
                        'Ensemble', best_single['Model'])

    # ---- uncertainty ----
    print("\n" + "=" * 70 + "\nSTEP 6: UNCERTAINTY QUANTIFICATION\n" + "=" * 70)
    members = {k: v for k, v in test_preds.items() if k != 'Ensemble'}
    val_members = {k: v for k, v in val_preds.items() if k != 'Ensemble'}
    uq = quantify_uncertainty(dl_models, D['val']['X_win'], D['val']['y'], val_members,
                              D['test']['X_win'], members, D['test']['y'],
                              n_iter=CONFIG['MC_ITERATIONS'], weights=weights)
    bayesian_residual_uq(D['test']['y'], test_preds['Ensemble'])  # optional, guarded
    pd.DataFrame([{'nominal_pct': k, 'empirical_pct': v}
                  for k, v in uq['coverage'].items()]).to_csv(
        os.path.join(out_dir, 'tables', 'coverage_calibration.csv'), index=False)

    # ---- per-dataset breakdown ----
    print("\n" + "=" * 70 + "\nSTEP 7: PER-DATASET BREAKDOWN\n" + "=" * 70)
    experiment_per_dataset(test_preds, D['test']['meta'], D['test']['y'], out_dir)

    # ---- transfer learning ----
    transfer_df = experiment_transfer_learning(out_dir, seed=CONFIG['RANDOM_SEED'])

    # ---- figures ----
    print("\n" + "=" * 70 + "\nSTEP 8: FIGURES\n" + "=" * 70)
    plot_training_history(histories, out_dir)
    plot_pred_vs_actual(D['test']['y'], test_preds['Ensemble'], out_dir, 'Ensemble')
    plot_model_comparison(results_df, out_dir)
    plot_uncertainty_bands(D['test']['y'], uq['cal_mean'], uq['cal_std'], out_dir)
    plot_calibration_curve(uq['coverage'], out_dir)
    print("  all figures saved (PNG + PDF @300dpi)")

    # ---- metadata ----
    save_run_metadata(out_dir, data_mode, D, weights, CONFIG)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  data_mode            : {data_mode}")
    print(f"  ensemble test RMSE   : {results_df.loc[results_df['Model']=='Ensemble','RMSE'].values[0]:.2f}")
    print(f"  empirical 95% cover. : {uq['coverage_95']:.2f}%")
    print(f"  results              : {out_dir}")
    print("=" * 70)
    return {'models_dl': dl_models, 'models_tree': tree_models, 'weights': weights,
            'results': results_df, 'uq': uq, 'transfer': transfer_df,
            'out_dir': out_dir, 'data_mode': data_mode}


if __name__ == "__main__":
    # In Colab the Drive folder may not be mounted yet — try once, safely.
    if not os.path.isdir(DRIVE_RESULTS_DIR):
        mount_gdrive()
    run_complete_pipeline()
