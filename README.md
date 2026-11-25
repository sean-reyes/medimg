# Heart Tools

A learning project that started as a heart disease risk estimator and grew into a multi-section prototype exploring how machine learning can work with different kinds of cardiac data: clinical measurements, echocardiograms and ECGs, with cardiac MRI and valve imaging planned.

> **Not a medical tool.** Everything here is a research and learning prototype built on small, older, public datasets. It has not been validated on real clinical cases and must not be used to diagnose, screen or make decisions about anyone's health.

## Status

| Section | What it does | Status |
|---|---|---|
| 1. Heart disease risk estimator | Estimates the probability of heart disease from clinical measurements | **Working**, trained and tested |
| 2. Echo ejection fraction | Takes two echo frames (end of filling, end of squeezing), outlines the left ventricle and estimates ejection fraction | **Working** ONLY on CAMUS data |
| 3. ECG pattern classifier | Estimates how likely each of five ECG patterns is, from a 12-lead ECG | **Model trained and calibrated**; app section written, bad-input testing still to do |
| 4. Cardiac MRI patterns | Will estimate cardiomyopathy-related patterns from cine MRI | Interface only, no model yet |
| 5. Aortic stenosis | Will estimate aortic stenosis severity from echo images | Interface only, no model yet |

Sections 4 and 5 accept an upload but always answer "Unsure", because no model is connected.

## Acknowledgements

This project stands on the work of people who collected, annotated and generously shared their data and methods. Thank you to:

- **The creators of the UCI Heart Disease dataset**: Andras Janosi (Hungarian Institute of Cardiology, Budapest), William Steinbrunn (University Hospital, Zurich), Matthias Pfisterer (University Hospital, Basel) and Robert Detrano (V.A. Medical Center, Long Beach, and the Cleveland Clinic Foundation) [1, 2]. Section 1 exists because of their patient data and the study that introduced it [1].
- **Sarah Leclerc, Olivier Bernard and the co-authors of the CAMUS dataset** and its paper [3], who built a large, openly released, fully annotated echocardiography dataset. Section 2 is built directly on it.
- **The Kaggle contributor "toygarr"**, whose repackaged HDF5 version of CAMUS made the data easy to load.
- **The creators of PTB-XL** [6, 7], the cardiologists who annotated it, and **Nils Strodthoff and colleagues**, whose benchmark study [8] showed what works on it. Section 3 is built on their data and their findings.
- **The authors of U-Net** [4], whose architecture the echo model is based on.
- **The American Society of Echocardiography and the European Association of Cardiovascular Imaging**, whose chamber quantification recommendations [5] define the method of disks that the ejection fraction calculation follows in simplified form.
- **The scikit-learn [9] and PyTorch [10] communities**, whose libraries did most of the heavy lifting.

Any mistakes in this project are my own.

## Section 1: heart disease risk estimator

### Data

- **Source:** the UCI Heart Disease dataset [2], released under CC BY 4.0, which combines four databases: Cleveland, Hungary, Switzerland and the VA Long Beach. This project uses the combined file of 920 patients with 16 columns. (https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data).
- **Label:** the original `num` column records angiographic disease status from 0 (no disease) to 4 (increasing severity). It was turned into a yes/no `target` (1 if `num` > 0). The original study is [1].

### Cleaning

The raw file is never edited. Cleaning is done in `section1_clinical/cleaning_data.py` and written to `data/processed_data/processed_data.csv`.

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
- Blank values filled inside a scikit-learn `Pipeline` using training rows only: median plus scaling for numeric columns, most common value for binary columns, most common value plus one-hot encoding for categories.
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

### Observations

- The model caught 92 of 102 patients with heart disease, at the cost of 21 false alarms.
- Test accuracy (0.83) is close to cross-validation accuracy (0.816), so the model is not simply memorizing the training rows.
- Removing `chol`, `slope` or `restecg` one at a time changed the test matrix by only a few patients, which is within the noise for a test set of 184. A proper cross-validated comparison of every column is still to do.
- Cleaning mattered: about 22% of cholesterol values were impossible zeros or blanks.

### App inputs

The model was trained on total cholesterol only, so the app turns LDL, HDL and triglycerides into total cholesterol (LDL + HDL + triglycerides / 5) before predicting. Fasting blood sugar is entered as a number and converted to the dataset's yes/no (above 120 mg/dl). Blood pressure is the top (systolic) number only. The app warns when the maximum heart rate is below 85% of the age-predicted maximum (220 minus age), because a stress test that short can be inconclusive.

## Section 2: echo ejection fraction

Ejection fraction (EF) is the percentage of the blood in the left ventricle that it pumps out with each beat. It is estimated from the ventricle's size at its fullest (end of filling, ED) and its smallest (end of squeezing, ES). The section outlines the ventricle on an ED and an ES frame, estimates EF, and refuses or flags results that look wrong.

### Data

- **CAMUS** [3]: 500 patients with two- and four-chamber echo, with the ED and ES frames outlined by a cardiologist for the left ventricle, the heart muscle around it and the left atrium.
- **The copy used here** is a repackaged HDF5 version from Kaggle (https://www.kaggle.com/datasets/toygarr/camus-subject-based). It contains 450 of the 500 patients, with ED and ES frames and masks for both views as 384 by 384 grayscale arrays. Only the four-chamber view is used. Check the licence and usage terms on both the CAMUS page and the Kaggle page before reusing or redistributing anything.
- **Mask labels:** 0 background, 1 left ventricle cavity, 2 heart muscle, 3 left atrium. Checked by overlaying masks on the images, and they line up with the anatomy.

### Method

- **Split by patient** (seed 42): 316 train, 67 validation, 67 test. Each patient's two frames stay together, so no one appears in two sets. The split is saved in `models/camus_split.npz`.
- **Preprocessing:** frames are shrunk from 384 to 192 pixels by averaging 2 by 2 blocks (masks keep every second pixel) and standardized per image. An earlier run used 128 pixels.
- **Model:** a small U-Net [4] with 483,204 parameters (three pooling levels, 16 base channels) that labels every pixel as one of the four classes. Trained with cross-entropy and Adam (learning rate 0.001, batch 16) for 10 epochs on a CPU, about 11 minutes, with light brightness and contrast changes and no flips, since flipping changes the heart's orientation. The epoch with the best validation Dice was kept. Built with PyTorch [10].
- **EF calculation:** the ventricle is sliced along its long axis (found by principal component analysis) into 20 slices, and thin discs are added up to give a volume. EF = (volume at ED - volume at ES) / volume at ED. This follows the idea of the method of disks recommended in [5], but it is a single-plane simplification, not the biplane method. Pixel size is unknown in this file, but it cancels out in a ratio.
- **Evaluation:** Dice score for the ventricle outline, and EF error against the EF computed the same way from the cardiologist's masks. The baseline gives every patient the typical EF from the training set.

### Results (67 test patients)

| | 128 pixels | 192 pixels | Baseline |
|---|---|---|---|
| Ventricle outline, test Dice | 0.912 | **0.916** | |
| EF error, mean (points) | 7.6 | **5.8** | 9.1 |
| EF error, median (points) | 6.1 | **4.3** | 7.6 |
| Estimates within 10 points | 72% | **82%** | 64% |
| Correlation with the expert EF | 0.76 | **0.80** | |

(The baseline differs slightly between runs because the training median was recomputed. The 192-pixel baseline is shown.)

### Observations

- **Resolution helped.** Going from 128 to 192 pixels improved every EF number, and the validation Dice was still rising at epoch 10, so more training may help further.
- **Good outlines do not guarantee a good EF.** EF is a ratio, and the squeezed ventricle is small, so a few misplaced pixels there move it a lot. The 128-pixel model had a test Dice of 0.912 but an EF error barely better than the baseline.
- **Cleaning up stray pixels did not help.** Keeping only the largest ventricle region changed nothing at 128 pixels and made the 192-pixel result slightly worse (mean error 6.0, 79% within 10 points), so it is not used.
- **An extreme patient.** One patient had a rough EF near 3%, with ED and ES frames that look almost the same. The outlines looked correct, so it was kept. It may be a genuinely very weak heart or poorly chosen frames.
- **Validation scores are noisy.** With 67 validation patients, epoch-to-epoch swings of a point or two are not meaningful, and one epoch dropped to 0.817 before recovering.
- **Context.** The CAMUS paper reports, for left ventricular EF, a mean correlation of 0.80 and an absolute mean error of 5.6% [3]. The results here are in the same range, but the setups differ (their evaluation protocol on the full dataset, versus 67 patients against EF computed from the experts' outlines here), so this is not a head-to-head comparison.

### What the app does

Upload an ED frame and an ES frame from the same patient and click **Analyse frames**. The app shows both frames with the ventricle outlined in red, then one of:

- **An estimated EF**, with a note that the typical error on test patients was about 4 to 6 points and that 82% of estimates were within 10 points.
- **"Unsure" and no number**, if the outline is fragmented, the squeezed ventricle looks larger than the filled one, or the EF is outside a plausible range (10% to 85%).
- **A rejection**, if the images are far from square or no ventricle is found.

The thresholds for these rules are placeholders chosen by reasoning, not tuned on held-out data. To try it, run `section2_echo/export_test_frames.py` to save a few test patients as PNG files in `test_frames/`, then upload the matching ED and ES images. Frames from the same dataset only show that the app works, not that the model generalizes.

## Section 3: ECG pattern classifier

The section estimates how likely it is that a 12-lead ECG shows each of five patterns: normal, myocardial infarction, ST/T change, conduction disturbance and hypertrophy. These are ECG patterns as read by cardiologists, not confirmed diseases. A record can show more than one pattern, so each category gets its own score.

### Data

- **PTB-XL v1.0.3** [6, 7], released under CC BY 4.0: 21,799 ten-second, 12-lead clinical ECGs from 18,869 patients, annotated by up to two cardiologists, collected between 1989 and 1996. This project uses the 100 Hz version (1,000 samples per lead).
- **Labels:** the detailed ECG statements are rolled up into the five diagnostic categories using the dataset's own statement table. 411 records carry only rhythm or form statements and no diagnostic label, so they are left out, leaving **21,388 ECGs**. Records per category: normal 9,514, myocardial infarction 5,469, ST/T change 5,235, conduction disturbance 4,898, hypertrophy 2,649. Most records have one label, and about a quarter have two or more.
- **Split:** the dataset's recommended folds, which keep each patient inside one fold. Folds 1 to 8 train the model (17,084 ECGs), fold 9 chooses cut-offs and calibration (2,146), and fold 10 is the test set (2,158).

### Method

- **Baseline:** 48 summary numbers per ECG (the mean, spread, lowest and highest value of each lead) and a random forest (300 trees), one yes/no output per category [9].
- **Neural network:** a small 1D residual convolutional network [8, 10] on the raw signals, with 493,893 parameters: one strided convolution, four residual blocks, averaging over time, and five outputs. Trained with binary cross-entropy, AdamW (learning rate 0.001, weight decay 0.01) and a cosine schedule for 15 epochs (18.4 minutes on a CPU), with random amplitude scaling and small time shifts as augmentation. Each lead is standardized with training-set statistics. The epoch with the best validation macro AUROC was kept.
- **Cut-offs:** for each category, the score above which the pattern is "flagged" is chosen on fold 9 to maximize F1, never on the test fold.
- **Calibration:** raw network scores are not true probabilities, so a straight-line correction on the log-odds (Platt scaling) is fitted per category on fold 9. The test fold is used only to check that it worked.
- **Metrics:** AUROC per category (0.5 is guessing, 1.0 is perfect) and the macro average, plus F1 at the chosen cut-off.

### Results (test fold, 2,158 ECGs)

| Category | Random forest AUROC | CNN, 2 epochs | **CNN, 15 epochs** | Test F1 (15 epochs) |
|---|---|---|---|---|
| Normal | 0.824 | 0.938 | **0.949** | 0.870 |
| Myocardial infarction | 0.799 | 0.896 | **0.927** | 0.751 |
| ST/T change | 0.739 | 0.926 | **0.937** | 0.769 |
| Conduction disturbance | 0.822 | 0.900 | **0.923** | 0.754 |
| Hypertrophy | 0.860 | 0.887 | **0.904** | 0.603 |
| **Macro average** | 0.809 | 0.909 | **0.928** | |

Calibration, test fold (lower is better for both measures; AUROC did not change, as expected):

| Category | ECE before | ECE after | Brier before | Brier after |
|---|---|---|---|---|
| Normal | 0.032 | 0.013 | 0.0895 | 0.0881 |
| Myocardial infarction | 0.025 | 0.018 | 0.0926 | 0.0918 |
| ST/T change | 0.017 | 0.022 | 0.0811 | 0.0809 |
| Conduction disturbance | 0.024 | 0.014 | 0.0780 | 0.0776 |
| Hypertrophy | 0.018 | 0.007 | 0.0652 | 0.0649 |

### Observations

- **The neural network beat the feature-based baseline by a wide margin**, from a macro AUROC of 0.809 to 0.928. This matches the PTB-XL benchmark study, which found that convolutional networks clearly outperform feature-based methods [8]. The numbers here are not directly compared with theirs.
- **ST/T change improved the most**, from 0.739 to 0.937 (+0.198), against +0.119 for the macro average. A plausible explanation is that the random forest only sees summary numbers, which discard the shape of the wave, while ST/T change is a finding about wave shape. This has not been tested directly.
- **Hypertrophy is the weakest category** (AUROC 0.904, F1 0.603). It is also the rarest, and F1 falls when a category is uncommon.
- **Validation plateaued.** The validation macro AUROC rose from 0.907 to about 0.930 by epoch 12 and then stayed there while the training loss kept falling, so longer training is unlikely to help much. Validation and test scores agree (0.930 and 0.928), which suggests the cut-offs and choices were not tuned to one fold.
- **The raw scores were already fairly well calibrated** (ECE at most 0.032). Calibration reduced the error in four of five categories and slightly increased it for ST/T change (0.017 to 0.022), a difference small enough to be noise on 2,158 records. Brier scores barely moved, and they are well below those of always predicting a category's overall frequency (roughly 0.11 to 0.25).
- **What a score means.** A displayed 40% means roughly that, among PTB-XL ECGs given that score, about 40% were labeled with that pattern. It is not the chance that a person has a disease, and it depends on PTB-XL's mix of patients.

### What the app does

Upload a CSV of 1,000 rows by 12 lead columns (10 seconds at 100 Hz, millivolts, PTB-XL lead order, no header). The app plots the first lead and shows a calibrated score for each category, marking those above their cut-off as flagged. It rejects a file with the wrong shape, missing or non-numeric values, values too large to be millivolts, or several flat leads. It says "Unsure" when no category reaches its cut-off, and warns when "normal" and an abnormal pattern are flagged together. `sample_ecg.csv`, a test-fold ECG saved by the training script, can be used to try it. Real ECG machines export other formats, so the CSV is a stand-in, and converting photos or PDFs of ECGs is future work.

## Sections 4 and 5: planned

Neither is built. Each needs its own labeled data, and each candidate dataset comes from a single institution. Their references will be added if and when they are used.

| Section | Planned dataset | Task |
|---|---|---|
| 4. Cardiac MRI | ACDC | Normal, dilated cardiomyopathy, hypertrophic cardiomyopathy, previous infarction, right ventricular abnormality |
| 5. Aortic stenosis | TMED-2 | None, early or significant aortic stenosis |

## Limitations

- **Small, older data.** The clinical dataset has 918 patients, the echo model was trained on 316, and the ECG data was collected between 1989 and 1996. Each dataset comes from one source, so results may not carry over to other hospitals, machines or populations.
- **Section 2 has only been tested on CAMUS.** It has not been tried on echo from another source, and the planned bad-input tests (a two-chamber view, a blurry frame, swapped frames, a non-echo photo) have not been done. The model was never trained on non-echo images or other views, so it can outline something confidently on them. The simple rules catch some cases, not all.
- **EF is compared with the experts' outlines, not with clinical EF.** The error measures how faithfully the model copies one cardiologist's outlines, not how close it is to a clinical measurement. Comparing with the reference EF values from the official CAMUS download has not been done.
- **Section 3 has only been evaluated on PTB-XL.** It has not been tested on ECGs from other machines or institutions, and the app's input handling has not been tested with bad files yet. The CSV input assumes the right units, lead order and sampling rate, and the app can only check some of that.
- **ECG labels are cardiologists' statements, not confirmed diagnoses,** and "myocardial infarction" means an ECG pattern consistent with one. Calibrated scores describe PTB-XL-like data, so they may not hold elsewhere.
- **Small test sets and single splits.** Section 1's test set has 184 patients and section 2's has 67, with no confidence intervals. The ECG test set is larger (2,158) but is still one split.
- **Simplified EF method.** Single-plane, from one four-chamber view, with no biplane averaging. Typical errors of 4 to 6 points can move a patient into a different clinical category, so the app shows the outline for a person to check.
- **Hospital as a feature (section 1).** `dataset` (which hospital) is a model input, so some performance may reflect differences between hospitals, such as how often a measurement was recorded, rather than medical signals. This has not been tested yet.
- **Imputed values.** About 22% of cholesterol values and 10% of fasting blood sugar values were missing and were filled in. The app also approximates total cholesterol from LDL, HDL and triglycerides, so the model cannot tell high LDL from high HDL.
- **Labels are not diagnoses.** The clinical label is angiographic disease status, and the scan and ECG labels describe patterns or measurements. Real diagnosis combines imaging, ECG, blood tests and clinical history.
- **Provenance.** Section 2 uses a third-party repackaging of CAMUS. Its licence and any differences from the original release should be checked.

## Data and trained models are not in this repository

Large datasets, generated files and the trained models are not committed. They are listed in `.gitignore`, and the scripts below recreate them. Check each dataset's terms before reusing it. The CAMUS file and the exported test frames should not be redistributed.

```
models/
data/camus/
data/camus_kaggle/
data/ptbxl/
cache/
test_frames/
sample_ecg.csv
__pycache__/
*.zip
```

## Project structure

```
.
├── data/
│   ├── raw_data/raw_data.csv              # original clinical file, never edited
│   ├── processed_data/processed_data.csv  # cleaned clinical data
│   └── ...                                # large datasets, not in the repo
├── models/                                # NOT in the repo: created by the training scripts
│   ├── model.joblib                       # section 1: clinical risk model
│   ├── camus_unet_192.pt                  # section 2: trained echo outliner
│   ├── camus_split.npz                    # the patient split used for echo training and testing
│   ├── ecg_cnn.pt                         # section 3: trained and calibrated ECG network
│   └── ecg_model.joblib                   # section 3: random forest baseline
├── section1_clinical/
│   ├── cleaning_data.py                   # raw clinical data to processed data
│   └── train_model.py                     # split, compare models, test report
├── section2_echo/
│   ├── look_camus.py                      # inspect the CAMUS masks
│   ├── train_camus.py                     # train the U-Net
│   ├── evaluate_camus.py                  # outlines and EF on the test patients
│   ├── export_test_frames.py              # save test patients as PNG files
│   └── echo_model.py                      # model and EF logic used by the app
├── section3_ecg/
│   ├── ptbxl_data.py                      # loads PTB-XL at 100 Hz, with a cache
│   ├── look_ptbxl.py                      # inspect the labels and draw one ECG
│   ├── train_ecg.py                       # random forest baseline
│   ├── train_ecg_cnn.py                   # the neural network
│   ├── calibrate_ecg.py                   # calibrate the network's scores
│   └── ecg_model.py                       # model logic used by the app
├── app.py                                 # Streamlit app with all five sections
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

The trained models are not stored in this repository, so a fresh clone has to recreate them. Run every command from the project root. You only need the steps for the sections you want to use, and the app shows a message for any section whose model is missing.

### 1. Install

```
pip install pandas scikit-learn joblib streamlit torch h5py pillow scipy matplotlib wfdb
```

Pin the scikit-learn version in `requirements.txt` to the one you trained with, because a saved model can misbehave when loaded with a different version.

### 2. Get the data

| For | Download | Then |
|---|---|---|
| Section 1 | The UCI Heart Disease file [2] | Save it as `data/raw_data/raw_data.csv` |
| Section 2 | The CAMUS HDF5 file (see the Section 2 data notes) | Set `DATA_PATH` at the top of `look_camus.py`, `train_camus.py`, `evaluate_camus.py` and `export_test_frames.py` |
| Section 3 | PTB-XL [7]: the zip, or just `records100` plus `ptbxl_database.csv` and `scp_statements.csv` | Set `PTBXL_DIR` in `section3_ecg/ptbxl_data.py` |

### 3. Train the models

| Section | Commands | Creates | Time on the author's CPU |
|---|---|---|---|
| 1 | `python section1_clinical/cleaning_data.py` then `python section1_clinical/train_model.py` | `data/processed_data/processed_data.csv`, `models/model.joblib` | Quick |
| 2 | `python section2_echo/train_camus.py` | `models/camus_unet_192.pt`, `models/camus_split.npz` | About 11 minutes |
| 3 | `python section3_ecg/train_ecg_cnn.py` then `python section3_ecg/calibrate_ecg.py` | `models/ecg_cnn.pt` | The first run reads about 21,400 files into `cache/`; training takes about 18 minutes |

Optional extras: `python section2_echo/evaluate_camus.py` (outline and EF results on the test patients), `python section2_echo/export_test_frames.py` (PNG files to try in the app), and `python section3_ecg/train_ecg.py` (the random forest baseline, which also writes `sample_ecg.csv`, a test ECG to try in the app).

All random seeds are fixed, so the patient splits are the same on every run. Retraining on a different computer or library version can still change the scores slightly, so the numbers in this README are from the author's runs.

### 4. Run the app

```
python -m streamlit run app.py
```

Each section works once its model exists. Section 2 takes a matching end-of-filling and end-of-squeezing image pair from `test_frames/`, and section 3 takes `sample_ecg.csv`. After retraining a model, restart the app, because Streamlit keeps loaded models in memory.

## Next steps

1. Test sections 2 and 3 with bad inputs and with data from other sources, and record what happens.
2. Compare EF with the clinical reference values from the official CAMUS download, if the patients can be matched.
3. Add a view check to section 2, so wrong-view images are rejected instead of outlined.
4. Tune the section 2 reject and "unsure" thresholds on held-out data, and report results with confidence intervals.
5. Test the ECG explanation for ST/T change directly, and look at whether the model fails on particular groups of records.
6. Accept ECG files in real formats (WFDB, or digitized images) in section 3.
7. Build sections 4 and 5, and run the cross-validated column comparison for the clinical model, including the effect of removing `dataset`.

## References

[1] R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, and V. Froelicher, "International application of a new probability algorithm for the diagnosis of coronary artery disease," *Am. J. Cardiol.*, vol. 64, pp. 304-310, 1989.

[2] A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1989. doi: 10.24432/C52P4X. Licensed CC BY 4.0.

[3] S. Leclerc et al., "Deep learning for segmentation using an open large-scale dataset in 2D echocardiography," *IEEE Trans. Med. Imaging*, vol. 38, no. 9, pp. 2198-2210, Sep. 2019, doi: 10.1109/TMI.2019.2900516.

[4] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015, pp. 234-241.

[5] R. M. Lang, L. P. Badano, V. Mor-Avi, et al., "Recommendations for cardiac chamber quantification by echocardiography in adults: an update from the American Society of Echocardiography and the European Association of Cardiovascular Imaging," *J. Am. Soc. Echocardiogr.*, vol. 28, no. 1, pp. 1-39.e14, 2015, doi: 10.1016/j.echo.2014.10.003.

[6] P. Wagner, N. Strodthoff, R.-D. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek, and T. Schaeffter, "PTB-XL, a large publicly available electrocardiography dataset," *Sci. Data*, vol. 7, 2020, doi: 10.1038/s41597-020-0495-6.

[7] P. Wagner, N. Strodthoff, R.-D. Bousseljot, W. Samek, and T. Schaeffter, "PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3)," PhysioNet, 2022, doi: 10.13026/kfzx-aw45. Licensed CC BY 4.0. https://physionet.org/content/ptb-xl/1.0.3/

[8] N. Strodthoff, P. Wagner, T. Schaeffter, and W. Samek, "Deep learning for ECG analysis: Benchmarks and insights from PTB-XL," *IEEE J. Biomed. Health Inform.*, vol. 25, no. 5, pp. 1519-1528, May 2021, doi: 10.1109/JBHI.2020.3022989.

[9] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, 2011.

[10] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019.

Dataset copy used in section 2: "CAMUS echocardiographic image segmentation v2" (subject-based HDF5 version), Kaggle, user toygarr, https://www.kaggle.com/datasets/toygarr/camus-subject-based.
