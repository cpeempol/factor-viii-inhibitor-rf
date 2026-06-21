# Factor VIII Inhibitor Prediction Using aPTT Mixing Study Parameters

This repository contains the source code for a research prototype that predicts factor VIII inhibitor positivity and estimates factor VIII inhibitor titer using routine activated partial thromboplastin time (aPTT) mixing study parameters.

The model was developed as a two-stage hurdle Random Forest model:
1. A classifier for inhibitor positivity, defined as factor VIII inhibitor titer >0.6 BU.
2. A regressor for quantitative inhibitor titer estimation among predicted positive cases.

## Required Input Parameters

| Parameter name | Description |
|---|---|
| `sex` | Patient sex |
| `tage` | Patient age |
| `PatientPTT` | Patient aPTT before mixing |
| `mix1` | aPTT immediately after 1:1 mixing with normal pooled plasma |
| `mix2` | aPTT after 2-hour incubation at 37°C |
| `NPPPTT` | aPTT of normal pooled plasma |

## Output

| Output | Description |
|---|---|
| `inhibitor_probability` | Predicted probability of inhibitor positivity |
| `inhibitor_classification` | Predicted inhibitor-positive or inhibitor-negative status |
| `predicted_FVIIIinh_BU` | Estimated factor VIII inhibitor titer in Bethesda units |

## Notes

This application is intended for research and laboratory decision-support demonstration only. It is not a standalone diagnostic tool and should not replace confirmatory Bethesda assay testing or expert clinical interpretation.

The file `data/example_input.csv` contains synthetic example data provided only to demonstrate the required input format. It does not contain real patient-level data.

## Citation

Citation information will be added after publication.
