# Oxford cross-dataset validation — COMPLETED (honest negative)

Independent validation on the Oxford Battery Degradation Dataset 1
(DOI `10.5287/bodleian:KO2kdmYGg`): 8 Kokam 740 mAh pouch cells, 40 °C,
Artemis Urban drive-cycle aging — different chemistry, form factor, and
aging protocol from NASA. All 8 cells were extracted (519 snapshots);
every cell reached the 80%-retention EOL.

## Results

| Direction | Best learned model | Baseline: fade-fraction |
|---|---|---|
| NASA → Oxford (test on Oxford) | XGBoost R² **0.807** | R² **0.964** |
| Oxford → NASA (test on B0018) | XGBoost R² **0.947** | R² **0.959** |

## Verdict

The simple fade-fraction baseline beats every learned model in **both**
transfer directions. Naive feature harmonization transfers nothing beyond
the shared fade curve: genuine cross-dataset transfer needs
throughput-normalized and/or physics-informed features, or explicit domain
adaptation. This is reported as a stated limitation of the study, not a
triumph.

Status: complete. No further cross-dataset experiments planned for this paper.
