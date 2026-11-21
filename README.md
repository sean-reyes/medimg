# Heart Tools

A learning project that started as a heart disease risk estimator and grew into a multi-section prototype for exploring how machine learning can work with different kinds of cardiac data: clinical measurements, echocardiograms, ECGs and cardiac MRI.

> **Not a medical tool.** Everything in this project is a research and learning prototype built on small, older, public datasets. It has not been validated on real clinical cases and must not be used to diagnose, screen or make decisions about anyone's health.

## What is built and what is planned

| Section | What it does | Status |
|---|---|---|
| 1. Heart disease risk estimator | Estimates the probability of heart disease from clinical measurements | **Working**, trained and tested |
| 2. Echo ejection fraction | Takes two echo frames (end of filling, end of squeezing) and will estimate ejection fraction | Interface only, no model yet |
| 3. ECG pattern classifier | Estimates five ECG pattern categories from a 12-lead ECG | **Template**: training script and app section written, not yet run on real data |
| 4. Cardiac MRI patterns | Will estimate five cardiomyopathy-related patterns from cine MRI | Interface only, no model yet |
| 5. Aortic stenosis | Will estimate aortic stenosis severity from echo images | Interface only, no model yet |

Sections marked "interface only" accept an upload and show the result layout, but return "Unsure" because no model is connected. They are placeholders for the work described under [Next steps](#next-steps).

## Section 1: heart disease risk estimator

### Data

- **Source:** UCI Heart Disease dataset (add the link to where you downloaded it).
- **Size:** 920 patients, 16 columns, collected at several hospitals (the `dataset` column records which).
- **Label:** the original `num` column (0 = no disease, 1 to 4 = increasing severity) was turned into a yes/no `target` (1 if `num` > 0).

### Cleaning

The raw file is never edited. Cleaning is done in `cleaning_data.py` and written to `data/processed_data/processed_data.csv`.

| Step | Why |
|---|---|
| Dropped `id` | It is only a row number |
| Removed duplicate rows (920 to 918) | Duplicates can land in both train and test sets and inflate the score |
| Dropped `ca` (66% missing) and `thal` (53% missing) | Too little real data to fill reliably |
| Treated zeros in `chol` and `trestbps` as missing | Zero cholesterol or blood pressure is impossible |
| Converted `fbs` and `exang` to 1/0 | Models need numbers |
| Replaced blank `slope` values with an `unknown` category | About a third of rows were blank |
| Created `target`, dropped `num` | `num` is the answer, so it must not be a feature |

Result: 918 rows, 12 features plus the target.

### Method

- 80/20 train/test split (734 / 184 patients), stratified, `random_state=42`.
- Blank values filled inside a scikit-learn `Pipeline`, using the training rows only: median plus scaling for numeric columns, most common value for binary columns, most common value plus one-hot encoding for categories.
- Models compared with 5-fold cross-validation on the training set only. The test set was scored once at the end.

### Results

| Model | Cross-validation accuracy |
|---|---|
| Always guess the majority class | 0.553 (+/- 0.003) |
| **Logistic regression** | **0.816** (+/- 0.004) |
| Random forest | 0.801 (+/- 0.012) |

Logistic regression was chosen: it scored slightly higher, and the gap was close to the random forest's spread, so the simpler model won the tie.

Test set (184 patients, logistic regression), accuracy **0.83**:

| | Precision | Recall | Patients |
|---|---|---|---|
| No disease | 0.86 | 0.74 | 82 |
| Disease | 0.81 | 0.90 | 102 |

| | Predicted healthy | Predicted sick |
|---|---|---|
| **Actually healthy** | 61 | 21 |
| **Actually sick** | 10 | 92 |

The model caught 92 of 102 patients who had heart disease, at the cost of 21 false alarms. Removing `chol`, `slope` or `restecg` one at a time changed the test matrix by only a few patients, which is within the noise for a test set this small. A proper cross-validated comparison of every column is still to do.

### App inputs

The model was trained on total cholesterol only, so the app turns LDL, HDL and triglycerides into total cholesterol (LDL + HDL + triglycerides / 5) before predicting. Fasting blood sugar is entered as a number and converted to the dataset's yes/no (above 120 mg/dl). Blood pressure is the top (systolic) number only. The app warns when the maximum heart rate is below 85% of the age-predicted maximum (220 minus age), because an exercise test that short can be inconclusive.

## Sections 2 to 5: scan models

None of these use the clinical dataset above. Each needs its own labeled data, and each dataset comes from a single institution with limited numbers of patients.

| Section | Dataset | Size | Task | Access |
|---|---|---|---|---|
| 2. Echo ejection fraction | CAMUS | 500 patients | Outline the left ventricle at two moments of the heartbeat, then compute ejection fraction | Download from the Human Heart Project (see the CAMUS challenge page, https://creatis.insa-lyon.fr/Challenge/camus) |
| 3. ECG patterns | PTB-XL | 21,837 ECGs, 18,885 patients | Five categories: normal, myocardial infarction, ST/T change, conduction disturbance, hypertrophy | https://physionet.org/content/ptb-xl/1.0.3/ |
| 3 (smaller option) | ECG Images dataset of Cardiac Patients | 928 images | Four ECG categories from images | https://data.mendeley.com/datasets/gwbz3fsgp8 |
| 4. Cardiac MRI | ACDC | 150 patients | Normal, dilated cardiomyopathy, hypertrophic cardiomyopathy, previous infarction, right ventricular abnormality | Register on the challenge site (non-commercial licence) |
| 5. Aortic stenosis | TMED-2 | 599 labeled studies | None, early or significant aortic stenosis | Apply for access (non-commercial research use) |

The ECG labels are ECG patterns read by cardiologists, not confirmed diseases. A model that predicts them estimates what a cardiologist would say about the tracing.

### Section 3 template

`train_ecg.py` reads PTB-XL, rolls its detailed statements up into the five categories, turns each ECG into 48 summary numbers (mean, spread, minimum and maximum per lead), trains a random forest with one output per category, and reports an AUROC for each category on the dataset's own held-out fold. These simple features give a baseline to improve on, and the usual next step is a neural network on the raw signal.

## Data is not in this repository

Large datasets are listed in `.gitignore` and must be downloaded separately. Check each dataset's terms before reusing it.

```
data/camus/
data/ptbxl/
```

Place each download in its folder, for example CAMUS in `data/camus/`.

## Project structure

```
.
├── data/
│   ├── raw_data/raw_data.csv              # original clinical file, never edited
│   ├── processed_data/processed_data.csv  # cleaned clinical data
│   ├── camus/                             # not in the repo
│   └── ptbxl/                             # not in the repo
├── models/
│   ├── model.joblib                       # clinical risk model
│   └── ecg_model.joblib                   # ECG model, created by train_ecg.py
├── cleaning_data.py                       # raw clinical data to processed data
├── train_model.py                         # section 1: split, compare models, test report
├── train_ecg.py                           # section 3: baseline ECG classifier
├── app.py                                 # Streamlit app with all five sections
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

```
pip install pandas scikit-learn joblib streamlit wfdb
python cleaning_data.py
python train_model.py
python -m streamlit run app.py
```

`train_ecg.py` is only needed for section 3 and requires PTB-XL in `data/ptbxl/`. The app reads the saved model files, so rerun the matching training script after retraining and restart Streamlit. Pin the scikit-learn version in `requirements.txt` to the one you trained with, because a saved model can misbehave when loaded with a different version.

## Limitations

- **Small, older data.** The clinical dataset has 918 patients, and each scan dataset comes from one institution. A model that works here may not carry over to other hospitals, machines or populations.
- **Hospital as a feature.** `dataset` (which hospital) is a model input, so some performance may reflect differences between hospitals, such as how often a measurement was recorded, rather than medical signals. This has not been tested yet.
- **Imputed values.** About 22% of cholesterol values and 10% of fasting blood sugar values were missing and were filled in.
- **Cholesterol is approximated.** The model sees only total cholesterol, so it cannot tell high LDL from high HDL.
- **Small test sets.** With a few dozen to a few hundred test cases, one or two errors move a score by several points.
- **Labels are not diagnoses.** Scan and ECG labels describe patterns or measurements, and real diagnosis combines imaging, ECG, blood tests and clinical history.

## Next steps

1. Load CAMUS, train a model that outlines the left ventricle, compute ejection fraction, and compare it with the dataset's reference values (section 2).
2. Reuse the segmentation work on ACDC and classify the resulting measurements (section 4).
3. Build an image classifier for TMED-2 aortic stenosis (section 5) and a pretrained-network classifier for the ECG images (section 3).
4. Run the cross-validated column comparison for the clinical model, including the effect of removing `dataset`.
5. Report every result with a baseline and a range, not a single number.
