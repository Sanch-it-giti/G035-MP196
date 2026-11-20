# G035 — MP196: Auto-Rickshaw Fare (Kochi)

SCSE3040 MLOps Mini Project  
Batch: 72  
Faculty: Dr. Amit Soni

## Members

| Enrolment number | Member |
|---|---|
| S24CSEU2710 | Saksham Sharma |
| S24CSEU2717 | Amay Kothari |
| S24CSEU2729 | Sanchit Khandelwal |
| S24CSEU2724 | Aniket Yadav |

## Objective

Predict ride fare (`fare_rs`) using information available before payment.
Evaluate predictions using Mean Absolute Error (MAE), measured in rupees.

## Data

Original training and later datasets are tracked with DVC in `data/raw/`.
Do not edit these original files.

## Current progress

- Public GitHub repository created.
- DVC initialized and original datasets tracked.
- Stage 1 data investigation saved in `src/check_data_issues.py`.
- Data findings recorded in `reports/findings.yaml`.

## Reproduce the data investigation

From the repository root, with the Python environment activated:

```bash
python src/inspect_data.py > reports/initial_inspection.txt
python src/check_data_issues.py > reports/data_issue_checks.txt
```

The scripts read the original files without changing them. The receipt field
leaks information available after payment. Station Area waiting times use
seconds (divide by 60 to obtain minutes). Negative bag counts represent missing
values, and booking names have spelling variants. The checks found no exact
duplicate rows, prolonged stuck readings, or isolated tenfold target values.
Cleaning and model training are the next stage.
