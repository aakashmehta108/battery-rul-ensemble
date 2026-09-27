# Run A — 4-cell primary run (2026-09-25)

## Setup
- Data: NASA PCoE — 4 cells (B0005/B0006/B0007/B0018), 636 discharge cycles, EOL = 80% retention.
- Split (cell-level, seed 42): train B0005+B0007 (238 windows), val B0006 (119), **test B0018 pinned (83 windows)** — a constant-2A cell, like the train/val cells; the shift axis is temperature/aging regime, not discharge rate.
- Models: LSTM(64)/GRU(64), lookback 50, early stopping; XGBoost/LightGBM on flat per-cycle features; NNLS ensemble weights on validation.

## Results (B0018, n=83)
| Model | RMSE | MAE | R² |
|---|---|---|---|
| LSTM | 6.82 | 3.56 | 0.093 |
| GRU | 6.59 | 3.66 | 0.152 |
| XGBoost | 5.37 | 3.51 | 0.438 |
| LightGBM | 5.96 | 3.68 | 0.306 |
| **Ensemble** | **4.89** | **2.78** | **0.533** |

- Weights: GRU 0.851, LightGBM 0.149. Paired t-test vs XGBoost: p = **0.0149** (significant).
- UQ: raw MC-Dropout 95% → 27.7% coverage; variance-scaled (α=4.778, val-fit) → 95% → **81.9%**.

## Artifacts
Full tables + figures for Runs A/B were archived to Drive (`results/`, `results/7cell/`) when Run C overwrote the local outputs. The metrics above are verbatim from the archived run logs.
