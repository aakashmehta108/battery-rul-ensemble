# Run C — 11-cell expansion with formation trimming (2026-09-25)

## Setup
- `load_nasa_mat` trims leading formation cycles (prefix < 95% of early-peak capacity): trims 7/1/1/1/1 cycles on B0033/B0034/B0036/B0054/B0055/B0056, nothing on the original cells.
- B0036/B0056 honestly censored (never reach 80% EOL) → **11 usable cells, 1,557 rows**.
- Split (seed 42): train B0007+B0033+B0034+B0043+B0044+B0055 (583 windows), val B0005+B0006+B0042+B0054 (352), **test B0018 pinned (83)** — same test set as A/B.

## Results (B0018, n=83)
| Model | RMSE | MAE | R² |
|---|---|---|---|
| LSTM | 5.13 | 4.67 | 0.485 |
| **GRU** | **3.37** | **2.66** | **0.779** |
| XGBoost | 11.91 | 8.03 | −1.771 |
| LightGBM | 11.65 | 8.22 | −1.651 |
| Ensemble | 11.91 | 8.03 | −1.771 |

- Weights (val-tuned NNLS): XGBoost 1.000, all others 0.
- Paired t-test, ensemble vs GRU: p = **7.0×10⁻⁷ favoring GRU** — complete validation→test rank reversal.
- Calibrated coverage: 95% → 75.9% (miscalibration grows with shift severity).

## Honest verdict
- GRU alone: RMSE **3.37**, R² 0.78 — best cross-condition result, improving monotonically with data (6.59 → 5.23 → 3.37).
- Tree models fit the validation cells well but collapse on the held-out 2A B0018 test cell; NNLS trusted validation → ensemble RMSE **11.91**.
- Lesson: validation-tuned ensemble weights assume validation represents test. Under the temperature/aging-regime shift between validation and test, that assumption fails.

## Artifacts in this folder
- `tables/` — model_metrics_test, per_dataset_metrics, coverage_calibration (csv+tex)
- `figures/` — training histories, pred-vs-actual, model comparison, uncertainty bands, calibration curve (PNG)
- `run_metadata.json` — full config (seed 42), leakage controls, split sizes
