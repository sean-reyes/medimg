# Heart Disease Prediction
 
A beginner machine learning project that predicts whether a patient has heart disease from routine clinical measurements. The goal was to practice the full workflow: cleaning raw data, splitting it honestly, comparing models fairly, and reporting a result on data the model never saw.
 
> **Not a medical tool.** This is a learning project built on a small, older dataset. It must not be used to diagnose or screen real patients.
 
## Problem
 
Given a patient's age, sex, chest pain type, blood pressure, cholesterol, heart rate and similar measurements, predict a yes/no label: does the patient have heart disease?
 
## Data
 
- **Source:** UCI Heart Disease dataset (add the link to where you downloaded it).
- **Size:** 920 patients, 16 columns in the raw file, collected at several hospitals (the `dataset` column records which one).
- **Original label:** `num`, a diagnosis from 0 (no disease) to 4 (most severe).
- **Label used here:** `target`, which is 1 if `num` > 0 and 0 otherwise. This turns a 5-way problem into a yes/no one.
## Cleaning
 
The raw file is never edited. Cleaning is done in code and written to `data/processed_data/processed_data.csv`.
 
| Step | Why |
|---|---|
| Dropped `id` | It is only a row number and adds noise |
| Removed duplicate rows (920 to 918) | Duplicates can land in both train and test sets and inflate the score |
| Dropped `ca` (66% missing) and `thal` (53% missing) | Too little real data to fill reliably |
| Treated zeros in `chol` and `trestbps` as missing | A cholesterol or blood pressure of 0 is impossible, so these are unrecorded values (cholesterol ended up about 22% missing) |
| Converted `fbs` and `exang` from True/False to 1/0 | Models need numbers |
| Replaced blank `slope` values with an `unknown` category | About a third of rows were blank, so it became its own category |
| Created `target` and removed `num` | `num` is the answer, so leaving it in would let the model cheat |
 
Final dataset: **918 rows, 12 features + 1 target.**
 
## Method
 
- **Split:** 80% train / 20% test (734 / 184 patients), random but fixed with `random_state=42`, and stratified so both sets keep the same share of sick patients (about 55%).
- **Remaining blanks** were filled inside a scikit-learn `Pipeline`, using values learned from the training rows only, so no test information leaks in:
  - numeric columns: median, then standardized
  - binary columns (`fbs`, `exang`): most common value
  - categorical columns (`sex`, `dataset`, `cp`, `restecg`, `slope`): most common value, then one-hot encoded
- **Model comparison** used 5-fold cross-validation on the training set only. The test set was held back until the end and scored once.
Three models were compared:
 
1. **Always guess the majority class**, as a floor to beat
2. **Logistic regression**
3. **Random forest** (200 trees)
## Results
 
### Cross-validation (training data, accuracy)
 
| Model | Mean | Spread (+/-) |
|---|---|---|
| Always guess majority | 0.553 | 0.003 |
| **Logistic regression** | **0.816** | 0.004 |
| Random forest | 0.801 | 0.012 |
 
Both real models beat the baseline by about 25 points. Logistic regression scored slightly higher, but the gap (0.015) is close to the random forest's spread, so it is nearly a tie. Logistic regression was chosen because it is the simpler, easier-to-explain model.
 
### Test set (184 held-out patients, logistic regression)
 
| | Precision | Recall | F1 | Patients |
|---|---|---|---|---|
| No disease (0) | 0.86 | 0.74 | 0.80 | 82 |
| Disease (1) | 0.81 | 0.90 | 0.86 | 102 |
 
**Accuracy: 0.83**
 
Confusion matrix:
 
| | Predicted healthy | Predicted sick |
|---|---|---|
| **Actually healthy** | 61 | 21 |
| **Actually sick** | 10 | 92 |
 
### What the results show
 
- Test accuracy (0.83) is close to cross-validation accuracy (0.816), so the model generalizes rather than just memorizing the training data.
- The model catches 90% of patients who actually have heart disease (92 of 102). For a medical problem, missing a sick patient is the costlier mistake, so this is the most important number.
- The trade-off is 21 false alarms: healthy patients flagged as sick (recall of 0.74 for the healthy class).
## Limitations
 
- **Small dataset:** 918 patients, and only 184 in the test set, so the test accuracy has a few points of natural uncertainty.
- **Older, multi-hospital data:** results may not carry over to other populations or modern clinical practice.
- **Hospital as a feature:** `dataset` is included, so some of the model's performance may reflect differences between hospitals (such as how often each records a measurement) rather than medical signals. This has not been tested yet.
- **Imputed values:** about 22% of cholesterol values and 10% of `fbs` values were missing and filled in.
## Possible next steps
 
- Remove `dataset` from the features and check how much the score changes
- Inspect the logistic regression coefficients to see which features push predictions toward "sick"
- Examine the 10 missed patients for shared traits
- Try gradient boosting and basic hyperparameter tuning
## Project structure
 
```
.
├── data/
│   ├── raw_data/raw_data.csv              # original file, never edited
│   └── processed_data/processed_data.csv  # cleaned data used for modeling
├── models/
│   └── model.joblib                       # saved trained pipeline
├── cleaning_data.py                       # raw data to processed data
├── train_model.py                         # split, compare models, final test report
└── README.md
```
 
## How to run
 
```
pip install pandas scikit-learn joblib
python cleaning_data.py
python train_model.py
```
 
`train_model.py` prints the cross-validation scores and then the test-set report.
