# PRIME-ECG

PRIME-ECG (Prediction of Risk in Mortality Evaluation by Electrocardiography) is a deep-learning model that estimates 28-day mortality risk in critically ill patients from a raw 12-lead ECG, age, and sex. Laboratory results and severity scores are not model inputs.

This repository contains the model definition and the preprocessing, inference, and evaluation code. No patient data or trained weights are included. The trained weights are derived from credentialed-access clinical data and are not distributed here. This code release alone does not reproduce the manuscript results.

## Model

- Output: one logit for 28-day mortality. The scripts report its sigmoid as `probability`; this is an uncalibrated risk score.
- Input: a 10-second, 500 Hz, 12-lead ECG plus age and sex.
- Each lead has its own encoder of four 1D convolutional layers and a bidirectional GRU. Parameters are not shared across leads.
- Lead embeddings are fused by concatenating an attention-weighted representation with the unweighted mean.
- Age and sex pass through a multilayer perceptron and are concatenated with the ECG representation.
- Trainable parameters: 3,510,754.

The single-head model retains the shared encoders and 28-day mortality output head from the original multitask-trained model; conversion details are recorded in `MANIFEST.json`.

## Input contract

| Item | Required value |
|---|---|
| Lead order | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 |
| Signal unit | mV |
| Sampling rate | 500 Hz after preprocessing |
| Duration | 10 seconds after preprocessing |
| ECG array | `[N, 12, 5000]`, float32 |
| Tabular array | `[N, 2]` = `[age / 100, male indicator]` |
| Sex coding | male=1, female=0 |

Lead order and unit cannot be detected from the signal, so check them before running the code. All 12 standard leads are required. Do not substitute posterior or right-sided leads. Inputs must be finite, and age and sex must be known.

## Preprocessing and quality control

`preprocessing.preprocess_ecg` checks input shape and finite values, then applies these steps in order:

1. Resample to 500 Hz with `scipy.signal.resample` when needed.
2. For recordings longer than 10 seconds, select the center window by default. With `window_mode="quality"` (CLI: `--window-mode quality`), score candidate windows in 0.25-second steps, also including the center and final possible windows, using gross-artifact criteria such as near-flat leads, amplitudes beyond +/-5 mV, and abrupt sample-to-sample transitions. Keep the lowest-scoring window, resolving ties toward the center.
3. Apply a 0.5-40 Hz Butterworth band-pass filter (4th order, zero-phase).
4. Apply a centered 5-point moving average.

Signals are not normalized or standardized.

A record raises `ValueError` and receives no score if it contains non-finite values, is shorter than 10 seconds after resampling, exceeds 5 mV in absolute amplitude after preprocessing, or has any all-zero lead after preprocessing. No minimum peak-to-peak amplitude threshold is enforced. No record is modified or imputed to pass quality control.

Already-preprocessed NPZ inputs bypass these signal quality checks and must be validated before inference.

## Installation and smoke test

Python 3.10 or newer is recommended. Run commands from this directory.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The optional smoke test requires a compatible checkpoint at `weights/final_bandpass_cnn_gru.pt`, including a `checkpoint_id` field:

```bash
python smoke_test.py
```

It checks checkpoint loading and a forward pass using synthetic arrays. It needs no patient data, but cannot run without weights. Its scores have no clinical meaning.

## Input NPZ format

```python
import numpy as np

np.savez_compressed(
    "cohort.npz",
    ecg=ecg,                  # float32 [N, 12, 5000], already preprocessed
    tabular=tabular,          # float32 [N, 2], [age/100, male]
    label=label,              # optional binary 0/1 [N]; required for evaluation
    patient_id=patient_id,    # optional fixed-width string or integer [N]
    record_id=record_id,      # optional fixed-width string or integer [N]
)
```

This example assumes the arrays already exist. Omit optional fields when unavailable. Arrays must have the same number of records; do not use object-dtype arrays.

You can provide `age` and `sex` instead of `tabular`. If `ecg` holds raw mV signals, add `--preprocess-raw --sampling-rate <Hz>`. All records in one NPZ must then have the same shape and sampling rate, and any record that fails quality control stops the run.

## Prediction

Prediction requires a compatible weights file, passed with `--weights`. The file must contain `model_state_dict` for the single-head architecture and may include `model_config`; without it, the default architecture is used. Only load checkpoints from a trusted source.

```bash
python predict.py \
  --input cohort.npz \
  --weights path/to/weights.pt \
  --output results/predictions.csv \
  --device cpu
```

For raw recordings with quality-based window selection:

```bash
python predict.py --input raw_cohort.npz --weights path/to/weights.pt \
  --preprocess-raw --sampling-rate 1000 --window-mode quality \
  --output results/predictions.csv --device cpu
```

Each ECG gets one row containing `ecg_index` and `probability`. `label`, `patient_id`, and `record_id` are copied to the output when present.

When `patient_id` is present, an additional `_patient.csv` file contains the mean score per patient and, if supplied, the first label. Confirm that grouped records share the same prediction episode and outcome: this aggregation does not distinguish repeat admissions.

## Evaluation

```bash
mkdir -p results
python evaluate.py --predictions results/predictions.csv \
  --threshold <value> --output results/metrics.json
```

The CSV must contain `label` (binary 0/1) and `probability`, with both outcome classes present for AUROC. The script reports AUROC, average precision (reported as AUPRC), Brier score, sensitivity, specificity, PPV, and NPV at ECG level. With `patient_id`, it also reports patient-mean metrics and rejects groups with inconsistent labels.

The script does not compute confidence intervals or fit calibration models. Replace `<value>` with a threshold chosen on independent validation data before evaluating on test data. Treat ECGs from the same patient as correlated when estimating uncertainty.

Despite its column name, `probability` is not a calibrated mortality probability. Determine calibration and thresholds on independent data for each target population.

## Intended use

This code is for research use only. PRIME-ECG is not a medical device and must not be used for patient-care decisions. Its effect on clinical management and patient outcomes has not been established.
