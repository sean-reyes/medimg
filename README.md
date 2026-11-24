# Heart Tools

A learning project that started as a heart disease risk estimator and grew into a multi-section prototype exploring how machine learning can work with different kinds of cardiac data: clinical measurements, echocardiograms, ECGs and cardiac MRI.

> **Not a medical tool.** Everything here is a research and learning prototype built on small, older, public datasets. It has not been validated on real clinical cases and must not be used to diagnose, screen or make decisions about anyone's health.

## Status

| Section | What it does | Status |
|---|---|---|
| 1. Heart disease risk estimator | Estimates the probability of heart disease from clinical measurements | **Working**, trained and tested |
| 2. Echo ejection fraction | Takes two echo frames (end of filling, end of squeezing), outlines the left ventricle and estimates ejection fraction | **Working** on CAMUS data; not yet tested on images from other sources |
| 3. ECG pattern classifier | Estimates five ECG pattern categories from a 12-lead ECG | Template only: training script and app section written, not yet run on real data |
| 4. Cardiac MRI patterns | Will estimate cardiomyopathy-related patterns from cine MRI | Interface only, no model yet |
| 5. Aortic stenosis | Will estimate aortic stenosis severity from echo images | Interface only, no model yet |

Sections 4 and 5 accept an upload but always answer "Unsure", because no model is connected.

## Acknowledgements

This project stands on the work of people who collected, annotated and generously shared their data and methods. Thank you to:

- **The creators of the UCI Heart Disease dataset**: Andras Janosi (Hungarian Institute of Cardiology, Budapest), William Steinbrunn (University Hospital, Zurich), Matthias Pfisterer (University Hospital, Basel) and Robert Detrano (V.A. Medical Center, Long Beach, and the Cleveland Clinic Foundation) [1, 2]. Section 1 exists because of their patient data and the study that introduced it [1].
- **Sarah Leclerc, Olivier Bernard and the co-authors of the CAMUS dataset** and its accompanying paper [3], who built the largest public, fully annotated echocardiography dataset of its kind and released it openly. Section 2 is built directly on it.
- **The Kaggle contributor "toygarr"**, whose repackaged HDF5 version of CAMUS ("CAMUS echocardiographic image segmentation v2") made the data easy to load.
- **The authors of U-Net** [4], whose architecture the echo model is based on.
- **The American Society of Echocardiography and the European Association of Cardiovascular Imaging**, whose chamber quantification recommendations [5] define the method of disks that the ejection fraction calculation follows in simplified form.
- **The scikit-learn [6] and PyTorch [7] communities**, whose libraries did most of the heavy lifting.

Any mistakes in this project are my own.

## Section 1: heart disease risk estimator

### Data

- **Source:** the UCI Heart Disease dataset [2], released under CC BY 4.0, which combines four databases: Cleveland, Hungary, Switzerland and the VA Long Beach. This project uses the combined file of 920 patients with 16 columns. Add the link to where you downloaded your copy: [add link].
- **Label:** the original `num` column records angiographic disease status from 0 (no disease) to 4 (increasing severity). It was turned into a yes/no `target` (1 if `num` > 0). The original study is [1].

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

## Section 2: Echo Ejection Fraction

Ejection fraction (EF) is the percentage of the blood in the left ventricle that it pumps out with each beat. It is estimated from the ventricle's size at its fullest (end of filling, ED) and its smallest (end of squeezing, ES). The section outlines the ventricle on an ED and an ES frame, estimates EF, and refuses or flags results that look wrong.

### Data

- **CAMUS** [3]: 500 patients with two- and four-chamber echo, with the ED and ES frames outlined by a cardiologist for the left ventricle, the heart muscle around it and the left atrium.
- **The copy used here** is a repackaged HDF5 version from Kaggle (https://www.kaggle.com/datasets/toygarr/camus-subject-based). It contains 450 of the 500 patients, with ED and ES frames and masks for both views as 384 by 384 grayscale arrays. Only the four-chamber view is used. Check the licence and usage terms on both the CAMUS page and the Kaggle page before reusing or redistributing anything.
- **Mask labels:** 0 background, 1 left ventricle cavity, 2 heart muscle, 3 left atrium. Checked by overlaying masks on the images, and they line up with the anatomy.

### Method

- **Split by patient** (seed 42): 316 train, 67 validation, 67 test. Each patient's two frames stay together, so no one appears in two sets. The split is saved in `models/camus_split.npz`.
- **Preprocessing:** frames are shrunk from 384 to 192 pixels by averaging 2 by 2 blocks (masks keep every second pixel) and standardized per image. An earlier run used 128 pixels.
- **Model:** a small U-Net [4] (three pooling levels, 16 base channels) that labels every pixel as one of the four classes. Trained with cross-entropy and Adam (learning rate 0.001, batch 16) for 10 epochs on a CPU, about 11 minutes, with light brightness and contrast changes and no flips, since flipping changes the heart's orientation. The epoch with the best validation Dice was kept. Built with PyTorch [7].
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

(The baseline differs slightly between runs because the model and training set median were recomputed. The 192-pixel baseline is shown.)

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

IMPORTANT: The thresholds for these rules are placeholders chosen by reasoning, not tuned on held-out data.

<u>To try it, run `export_test_frames.py` to save a few test patients as PNG files in `test_frames/`, then upload the matching ED and ES images. Frames from the same dataset only show that the app works, not that the model generalizes.</u>

## Sections 3 to 5: WIP

None of these use the clinical dataset above. Each needs its own labeled data, and each candidate dataset comes from a single institution. They are listed here as plans, and their references will be added if and when they are used.

| Section | Planned dataset | Task |
|---|---|---|
| 3. ECG patterns | PTB-XL, or the Mendeley "ECG Images dataset of Cardiac Patients" | Five ECG categories (or four from images) |
| 4. Cardiac MRI | ACDC | Normal, dilated cardiomyopathy, hypertrophic cardiomyopathy, previous infarction, right ventricular abnormality |
| 5. Aortic stenosis | TMED-2 | None, early or significant aortic stenosis |

`train_ecg.py` is a template for section 3: it rolls PTB-XL's statements up into five categories, trains a random forest on simple per-lead summaries and reports AUROC per category. It has not been run on real data.

## Limitations

- **Small, older data.** The clinical dataset has 918 patients, and the echo model was trained on 316. Each dataset comes from one institution, so results may not carry over to other hospitals, machines or populations.
- **Section 2 has only been tested on CAMUS.** It has not been tried on echo from another source, and the planned bad-input tests (a two-chamber view, a blurry frame, swapped frames, a non-echo photo) have not been done. The model was never trained on non-echo images or other views, so it can outline something confidently on them. The simple rules catch some cases, not all.
- **EF is compared with the experts' outlines, not with clinical EF.** The error measures how faithfully the model copies one cardiologist's outlines, not how close it is to a clinical measurement. Comparing with the reference EF values from the official CAMUS download has not been done.
- **Small test sets.** Section 2's test set has 67 patients and one split, with no confidence intervals. The gap over the baseline is encouraging but not settled.
- **Simplified EF method.** Single-plane, from one four-chamber view, with no biplane averaging, and based on a model that works at 192 pixels.
- **EF is approximate.** Typical errors of 4 to 6 points can move a patient into a different clinical category. The app shows the outline so a person can check it.
- **Hospital as a feature (section 1).** `dataset` (which hospital) is a model input, so some performance may reflect differences between hospitals, such as how often a measurement was recorded, rather than medical signals. This has not been tested yet.
- **Imputed values.** About 22% of cholesterol values and 10% of fasting blood sugar values were missing and were filled in. The app also approximates total cholesterol from LDL, HDL and triglycerides, so the model cannot tell high LDL from high HDL.
- **Labels are not diagnoses.** The clinical label is angiographic disease status, and the scan and ECG labels describe patterns or measurements. Real diagnosis combines imaging, ECG, blood tests and clinical history.
- **Provenance.** Section 2 uses a third-party repackaging of CAMUS. Its licence and any differences from the original release should be checked.

## Data is not in this repository

Large datasets are listed in `.gitignore` and must be downloaded separately. The CAMUS file and the exported test frames should not be redistributed.

```
data/camus/
data/camus_kaggle/
data/ptbxl/
test_frames/
*.zip
```

## Project structure

```
.
├── data/
│   ├── raw_data/raw_data.csv              # original clinical file, never edited
│   ├── processed_data/processed_data.csv  # cleaned clinical data
│   └── ...                                # large datasets, not in the repo
├── models/
│   ├── model.joblib                       # section 1: clinical risk model
│   ├── camus_unet_192.pt                  # section 2: trained echo outliner
│   ├── camus_split.npz                    # the patient split used for training and testing
│   └── ecg_model.joblib                   # section 3, created by train_ecg.py
├── cleaning_data.py                       # raw clinical data to processed data
├── train_model.py                         # section 1: split, compare models, test report
├── look_camus.py                          # section 2: inspect the CAMUS masks
├── train_camus.py                         # section 2: train the U-Net
├── evaluate_camus.py                      # section 2: outlines and EF on the test patients
├── export_test_frames.py                  # section 2: save test patients as PNG files
├── echo_model.py                          # section 2: model and EF logic used by the app
├── train_ecg.py                           # section 3: baseline ECG classifier (template)
├── app.py                                 # Streamlit app with all five sections
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

```
pip install pandas scikit-learn joblib streamlit torch h5py pillow scipy matplotlib wfdb
python cleaning_data.py
python train_model.py
python train_camus.py
python evaluate_camus.py
python export_test_frames.py
python -m streamlit run app.py
```

Set `DATA_PATH` at the top of the CAMUS scripts to the location of your HDF5 file. `train_ecg.py` is only needed for section 3 and requires PTB-XL in `data/ptbxl/`. The app reads the saved model files, so rerun the matching training script after retraining and restart Streamlit. Pin the scikit-learn version in `requirements.txt` to the one you trained with, because a saved model can misbehave when loaded with a different version.

## Next steps

1. Test section 2 with bad inputs and with echo from another source, and record what happens.
2. Compare EF with the clinical reference values from the official CAMUS download, if the patients can be matched.
3. Add a view check, so wrong-view images are rejected instead of outlined.
4. Tune the reject and "unsure" thresholds on held-out data, and report results with confidence intervals.
5. Try longer training and a two-chamber model with a biplane EF.
6. Build sections 3 to 5, and run the cross-validated column comparison for the clinical model, including the effect of removing `dataset`.

## References

[1] R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, and V. Froelicher, "International application of a new probability algorithm for the diagnosis of coronary artery disease," *Am. J. Cardiol.*, vol. 64, pp. 304-310, 1989.

[2] A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1989. doi: 10.24432/C52P4X. Licensed CC BY 4.0.

[3] S. Leclerc et al., "Deep learning for segmentation using an open large-scale dataset in 2D echocardiography," *IEEE Trans. Med. Imaging*, vol. 38, no. 9, pp. 2198-2210, Sep. 2019, doi: 10.1109/TMI.2019.2900516.

[4] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015, pp. 234-241.

[5] R. M. Lang, L. P. Badano, V. Mor-Avi, et al., "Recommendations for cardiac chamber quantification by echocardiography in adults: an update from the American Society of Echocardiography and the European Association of Cardiovascular Imaging," *J. Am. Soc. Echocardiogr.*, vol. 28, no. 1, pp. 1-39.e14, 2015, doi: 10.1016/j.echo.2014.10.003.

[6] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, 2011.

[7] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019.

Dataset copy used in section 2: "CAMUS echocardiographic image segmentation v2" (subject-based HDF5 version), Kaggle, user toygarr, https://www.kaggle.com/datasets/toygarr/camus-subject-based.
