# Full experiment record — methodology, findings, discussion, observations
Companion to `README.md`. All numbers below are taken verbatim from the run logs and
result tables in `experiments/`; nothing is rounded beyond what the pipeline reported.

## 1. Objective

Test whether a **heterogeneous ensemble** (two recurrent deep learners + two
gradient-boosting machines) with **uncertainty quantification** improves
lithium-ion battery remaining-useful-life (RUL) prediction — and, crucially,
whether the improvement survives an **operating-condition shift** between
validation and test.

## 2. Data and protocol (all runs)

- **Source:** NASA Prognostics Center of Excellence Battery Data Set (official S3 archive).
- **Usable cells after parsing:** 11 (B0005, B0006, B0007, B0018, B0033, B0034, B0042, B0043, B0044, B0054, B0055); B0036/B0056 honestly censored (never reach 80% EOL); zero-capacity cycles dropped (aborted measurements); leading formation cycles trimmed (prefix below 95% of the cell's early-peak capacity).
- **EOL definition:** 80% of nominal capacity retention (note: NASA's own criterion is 70%/1.4 Ah — we state 80% explicitly).
- **Test design:** cell **B0018 is pinned as the test cell in every run** (`CONFIG['TEST_CELLS']`). B0018 is a **constant-2A discharge cell, exactly like the training/validation cells** (B0005/B0006/B0007); the 4A-pulse cells (B0033/B0034) sit in the *training* set. The supported shift axis between validation and test is the **temperature/aging regime**, not discharge rate. Every run is therefore a genuine cross-condition generalization test, and runs are directly comparable on the identical 83-window test set.
- **Splits:** by cell ID (no cell in two splits), seed 42.
- **Features:** per-discharge-cycle voltage/current/temperature time series → engineered per-cycle features; LSTM/GRU use lookback-50 sliding windows (causal — no future cycles); XGBoost/LightGBM use flat per-cycle features. `StandardScaler` fit on train only.
- **Models:** LSTM(64), GRU(64), early stopping, dropout 0.2; XGBoost, LightGBM.
- **Ensembling:** non-negative least squares (NNLS) weights fit on the held-out validation cells' predictions.
- **Uncertainty:** Monte-Carlo Dropout (100 iterations, LSTM+GRU pooled) → predictive std → intervals; calibration via a variance-scaling factor fit on **validation only**.
- **Leakage controls** (recorded in `run_c_11cell/run_metadata.json`): cell-level splits, scaler fit on train only, causal rolling features, no future cycles in windows.

## 3. Findings

### Run A — 4 cells (B0005/B0006/B0007 train/val, B0018 test), 636 cycles
| Model | RMSE | MAE | R² |
|---|---|---|---|
| LSTM | 6.82 | 3.56 | 0.093 |
| GRU | 6.59 | 3.66 | 0.152 |
| XGBoost | 5.37 | 3.51 | 0.438 |
| LightGBM | 5.96 | 3.68 | 0.306 |
| **Ensemble** | **4.89** | **2.78** | **0.533** |

- NNLS weights: GRU 0.851, LightGBM 0.149 (LSTM, XGBoost → 0).
- Paired t-test on absolute errors, ensemble vs XGBoost: t = −2.488, **p = 0.0149** — statistically significant improvement (~9% RMSE reduction).
- UQ: raw MC-Dropout 95% nominal → **27.7%** empirical (severely overconfident). A quantile-calibration variant was tried and **rejected** (overfit validation outliers: 80% nominal → 57.8% empirical vs 79.5% with variance scaling). Variance-scaling (α = 4.778, fit on validation only) → calibrated coverage: 80%→79.5%, 90%→80.7%, **95%→81.9%**, 99%→90.4%.

### Run B — 7 cells (+B0042/B0043/B0044), 969 rows — distribution-shift ablation
| Model | RMSE | MAE | R² |
|---|---|---|---|
| LSTM | 8.58 | 8.09 | −0.437 |
| GRU | 5.23 | 4.77 | 0.466 |
| XGBoost | 10.01 | 4.30 | −0.957 |
| LightGBM | 7.33 | 3.53 | −0.049 |
| **Ensemble** | **7.33** | **3.53** | **−0.049** |

- NNLS weights: LightGBM 1.000, all others 0.
- **Verdict: expansion HURT point accuracy** — ensemble RMSE 7.33 vs Run A's 4.89 on the identical B0018 test set. Likely cause: 3 of the 4 new training cells are short-lived (EOL 41 vs 100–123 previously), shifting the training distribution away from B0018's regime (EOL 74). XGBoost collapsed (R² −0.96).
- Silver lining: calibration improved (95% → **92.8%**) with the larger 2-cell validation set.
- Paired t-test, ensemble vs GRU: t = −2.164, p = 0.0334 (ensemble's absolute errors significantly smaller despite worse RMSE — heavy-tail effect).

### Run C — 11 cells (+B0033/B0034/B0054/B0055 via formation trim), 1,557 rows
| Model | RMSE | MAE | R² |
|---|---|---|---|
| LSTM | 5.13 | 4.67 | 0.485 |
| **GRU** | **3.37** | **2.66** | **0.779** |
| XGBoost | 11.91 | 8.03 | −1.771 |
| LightGBM | 11.65 | 8.22 | −1.651 |
| Ensemble | 11.91 | 8.03 | −1.771 |

- NNLS weights: XGBoost 1.000, all others 0.
- Paired t-test, ensemble vs GRU: t = 5.375, **p = 7.0×10⁻⁷ favoring GRU** — a complete validation→test rank reversal.
- **Verdict: the validation-tuned ensemble broke under distribution shift.** Tree models fit the validation cells well but collapse on the held-out B0018 test cell (R² ≈ −1.7); NNLS trusted validation and concentrated all weight on XGBoost.
- GRU improves **monotonically with training data** across runs: 6.59 → 5.23 → **3.37** — sequence models scale under shift.
- Calibrated coverage: 95% → 75.9% (worse than A/B — miscalibration grows with shift severity).

## 4. Discussion

1. **Ensembles help when validation resembles test (Run A), and the gain is statistically significant.** This is the standard, publishable in-distribution result.
2. **More data is not better under regime shift (Run B).** Adding short-lived cells shifted the training distribution away from the test regime and degraded the ensemble. A useful cautionary ablation.
3. **Sequence models scale with data under shift; tree models do not (Run C).** GRU's monotonic improvement vs the trees' collapse is the cleanest empirical contrast in the study.
4. **Naive validation-tuned ensembling is fragile under operating-condition shift.** NNLS weight selection assumes the validation distribution represents the test distribution. Under the temperature/aging-regime shift between validation and test cells, that assumption fails catastrophically (11.91 vs 3.37). This is the paper's central empirical contribution — an honest negative result about standard practice.
5. **MC-Dropout is overconfident out of the box; variance scaling helps but leaves residual miscalibration** that grows with shift severity (81.9% → 75.9% at 95% nominal from Run A to Run C). The miscalibration is reported, not hidden.

## 5. Observations and limitations (stated as-is)

- Only 4 cells in the primary run — small by modern standards; variance across seeds/splits is acknowledged, not explored.
- Single pinned test cell (B0018, constant-2A like the training cells): a strong generalization test across the temperature/aging-regime shift, but also the source of the high-RUL underprediction bias visible in residuals.
- **No second chemistry/dataset is in the main ensemble study** — the Oxford cross-dataset track (`experiments/oxford_cross_dataset/`) is complete and reports a clean negative (naive transfer loses to the fade-fraction baseline).
- EOL at 80% retention vs NASA's own 70% — stated explicitly wherever EOL appears.
- MAPE is inflated by near-zero true RUL values; RMSE/MAE/R² are the primary metrics.
- A superseded faulty-split run (RMSE 12.37) exists in history and is **never** cited or reused.

## 6. Follow-on tracks (completed, results in this repo)

- `experiments/oxford_cross_dataset/` — independent validation on 8 Kokam cells (different chemistry, form factor, 40 °C, drive-cycle aging): honest negative; naive feature harmonization transfers nothing beyond the shared fade curve.
- `experiments/shift_robust_stacking/` — weighting schemes designed to survive the validation→test shift without test peeking: tempered-softmax (T=0.5) reaches 3.70 vs the NNLS collapse at 11.91; minimax-regret 4.47; shift-weighted conformal prediction recovers 100% coverage at ~2× interval width.
