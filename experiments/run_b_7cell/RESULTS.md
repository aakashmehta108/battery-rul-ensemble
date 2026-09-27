# Run B — 7-cell distribution-shift ablation (2026-09-25)

## Setup
- Added B0042/B0043/B0044 (112 cycles each, EOL 41). Pipeline now drops zero-capacity cycles and pins B0018 to test by design.
- Data: 7 cells, 969 rows. Split: train B0007+B0042+B0043+B0044 (305 windows), val B0005+B0006 (238), **test B0018 pinned (83)** — same test set as Run A.

## Results (B0018, n=83)
| Model | RMSE | MAE | R² |
|---|---|---|---|
| LSTM | 8.58 | 8.09 | −0.437 |
| GRU | 5.23 | 4.77 | 0.466 |
| XGBoost | 10.01 | 4.30 | −0.957 |
| LightGBM | 7.33 | 3.53 | −0.049 |
| **Ensemble** | **7.33** | **3.53** | **−0.049** |

- Weights: LightGBM 1.000. Paired t-test vs GRU: p = 0.0334.
- **Honest verdict: expansion hurt.** Ensemble 7.33 vs Run A's 4.89 on the identical test set — the three new short-lived cells (EOL 41) shifted the training distribution away from B0018's regime (EOL 74).
- Calibration improved: 95% → **92.8%** (larger 2-cell validation set calibrates better).

## Artifacts
Archived to Drive (`results/7cell/`). Reported here as an ablation, not the primary result.
