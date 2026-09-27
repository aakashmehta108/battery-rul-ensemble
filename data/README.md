# Datasets — download instructions and inventory

Raw data is **not** committed to this repo (size + licensing). Download once,
place under `data/`, and run the pipeline.

## 1. NASA PCoE Battery Data Set (primary — Runs A/B/C)

- Source: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
- Reference: Saha, B. & Goebel, K. (2007). Battery Data Set, NASA Ames Prognostics Data Repository.
- 34 cells (B0005–B0056). After parsing, formation-cycle trimming, and honest
  censoring: **11 usable cells, 1,557 discharge-cycle rows** (Run C).
- Place extracted `.mat` files in `data/nasa/`.

## 2. Oxford Battery Degradation Dataset 1 (independent cross-dataset validation)

- DOI: `10.5287/bodleian:KO2kdmYGg` (Oxford ORA, open access)
- 8 Kokam SLPB533459H4 740 mAh pouch cells, 40 °C chamber, Artemis Urban
  drive-cycle aging; 46–78 characterization snapshots per cell (1-C charge/discharge
  + pseudo-OCV, each with t/v/q/T time series).
- Measured fade e.g. Cell1: 739 → 561 mAh over 8,200 cycles.
- Place in `data/oxford/`.

## 3. Kaggle cleaned NASA (feature enrichment only)

- `mystifoe77/nasa-battery-data-cleaned` — 7,368 rows × 9 cols
  (Capacity, Re, Rct, ambient_temperature 4/22/24/43/44 °C, …).
- **Derivative of the NASA set (same 34 cell IDs) — NOT an independent dataset.**
  Used only for impedance/temperature feature enrichment after leakage-free
  cycle alignment. Never used as a second validation dataset.

## 4. CALCE SP1/SP3 (excluded from RUL training)

- Single characterization/OCV tests at 0/25/45 °C — no aging trajectory.
  Kept for optional OCV–SOC supporting analysis only.

## Inventory

Per-cell usability, EOL cycles, and censoring decisions are documented in the
pipeline (`process_all_cells.py` logic, summarized in `EXPERIMENTS.md` §2).
