# G035 team handoff

## What is ready

The data investigation and cleaning code are ready locally. After these changes
are pushed, teammates can obtain the code from GitHub. Model training, MLflow,
API, TestClient tests, Docker, CI, Kubernetes, monitoring and predictions remain
assigned work; they are not completed by this handoff.

## Getting code and data on another computer

1. Accept the GitHub invitation. Clone the repository (or run `git pull` in an
   existing checkout).
2. Create a Python environment and install the data-stage dependencies:

```bash
git clone https://github.com/Sanch-it-giti/G035-MP196.git
cd G035-MP196
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell activation instead: `.venv\Scripts\Activate.ps1`.

3. Extract **only your group's original** `train.csv` and `later.csv` from the
   supplied ZIP into `data/raw/`. Do not overwrite the `.dvc` files. The holdout
   CSV is already in Git. Never open and save the originals through Excel.
4. Run:

```bash
python -m src.fare_data.prepare
python -m unittest discover -s tests -v
```

Preparation checks the original training SHA-256 and writes three generated
CSVs under `data/processed/`. It also creates `reports/cleaning_summary.json`.
The processed files are ignored by Git because they can be regenerated.
Do not run cleaning on processed training data again: the unit correction is
intended for original train only.

**GitHub alone does not supply train/later:** DVC pointers contain checksums,
not the CSV contents. `.dvc/config` currently has no remote, so `dvc pull` cannot
retrieve them. Sharing the original ZIP privately within this group is the
current route. Alternatively, the group can configure accessible DVC storage,
run `dvc push`, and then teammates can use `dvc pull`.

## Sanchit: data handoff

- Original data is unchanged; SHA-256 checks are recorded in the summary.
- `fare_paid_receipt_rs` is removed: a receipt is available after payment.
- Original train's Station Area wait values are divided by 60. Later and
  holdout are already minutes and are not converted.
- `luggage_bags = -1` becomes a missing value, not zero bags.
- Booking whitespace/case and abbreviations are normalized to app/stand/street.
- Exact duplicate original training rows are removed if present; none were found.
- No target values are changed, because no tenfold typo group was found.

## Saksham: training and MLflow

Use `data/processed/train.csv`, and select exactly these features:

```python
from src.fare_data.cleaning import FEATURES, TARGET
import pandas as pd

train = pd.read_csv("data/processed/train.csv")
X = train[FEATURES]
y = train[TARGET]
```

Keep record_id, date and receipt out of X. Retain date for splitting/auditing.
Fit numeric imputation and categorical encoding **inside the sklearn Pipeline**
on the training split only; luggage has 55 missing values. Compare at least
three settings on the same validation split using MAE. Select without looking
at later or holdout targets. Refit the chosen pipeline on cleaned train for v1.
Log every run to `sqlite:///mlflow.db` with `data_sha256` of ORIGINAL train,
parameters and MAE. Register v1 as version 1 and give it alias champion.
Commit the database and save the full fitted preprocessing/model pipeline to
`models/model.joblib`; record tracking fields in reports/findings.yaml.
Append exact pinned versions of your dependencies to requirements.txt. Agree
on versions with Amay so the saved model loads in the API environment.

## Amay: service, tests and Docker

Load the complete saved pipeline and send it a one-row DataFrame with FEATURES.
Service requests already use minutes: do not apply the raw train unit fix.
Use precisely the eight requested fields, reject extras, and put distance <=45
and fuel price >=45 constraints in pydantic Fields. Add /health and /predict,
synthetic-data pytest/TestClient coverage and the required Dockerfile. Existing
unittest tests can also be discovered by pytest. Add pinned API/test libraries.
Coordinate JSON stdout prediction logging with Aniket.

## Aniket: operations, monitoring and final prediction

Build push-triggered GitHub Actions running pytest without original CSVs, and
Kubernetes YAML with deployment mp196-api, service mp196-svc, image
mp196-api:1.0, 3 replicas, container port 8000, service port 80 and both /health
probes (liveness waits longer). Verify an actual green Actions run.
Coordinate with Saksham for PSI, v1 MAE by zone and retraining. Use cleaned
train and later; sort later by date, use its first 70% for retraining and last
30% for comparison. Register v2 as version 2. Promote only if v2 check MAE is
at least 8% lower than v1 check MAE. Update champion/model.joblib consistently.
Predict holdout using champion, and save record_id,prediction exactly.
Complete monitoring fields with measured results, never placeholders.

## Working together

Before starting, pull and create a personal feature branch. Commit your actual
work under your own GitHub identity and open a pull request. Coordinate edits
to shared requirements.txt, params.yaml and reports/findings.yaml. Do not claim
AI-generated work as unaided work; follow the handbook's AI-use rules and ask
faculty if the level of assistance used needs clarification. Every member
must understand the files for the viva and submit their own honest form.
