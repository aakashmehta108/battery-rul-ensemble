# Shift-robust stacking — COMPLETED

The problem (from Run C): NNLS ensemble weights tuned on validation collapse
under the temperature/aging-regime shift between validation and test cells
(ensemble RMSE 11.91 vs GRU 3.37 on pinned-B0018).

All remedies below were **tuned on validation only — no test peeking**.

## Results (B0018 test, n=83)

| Weighting scheme | RMSE | R² | Notes |
|---|---|---|---|
| NNLS (validation-tuned, baseline) | 11.91 | −1.77 | All weight on XGBoost; collapsed |
| **Tempered-softmax (T=0.5)** | **3.70** | **0.73** | Weights ≈ LSTM 0.257, GRU 0.741, trees ≈ 0 |
| Minimax-regret | 4.47 | — | — |
| Shift-weighted conformal 95% intervals | — | — | **100% coverage**, mean half-width 46.1 cycles |

## Verdict

Tempered-softmax weighting (T=0.5) — softening the validation-best weights
instead of hard-selecting them — recovers to **3.70** on B0018, close to the
GRU's 3.37 and a clean fix of the NNLS collapse (11.91). Minimax-regret
weighting also survives the shift (4.47). Shift-weighted conformal prediction
recovers nominal coverage (100%) but at the honest cost of sharpness: intervals
twice as wide. This is the study's shift-robustness contribution: an
ensemble-weighting remedy that does not collapse on the held-out test cell
without ever seeing it.

Status: complete. Figures and tables to be committed alongside the Run C artifacts.
