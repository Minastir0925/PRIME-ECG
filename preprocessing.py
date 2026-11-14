"""PRIME-ECG signal preprocessing, quality control, and age/sex encoding."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
TARGET_RATE = 500
TARGET_SECONDS = 10
TARGET_SAMPLES = TARGET_RATE * TARGET_SECONDS
MAX_ABSOLUTE_MV = 5.0
MIN_LEAD_PEAK_TO_PEAK_MV = 0.05


def encode_sex(value) -> float:
    """Return 1 for male and 0 for female; raise for missing/unknown values."""
    text = str(value).strip().lower()
    if text in {"m", "male", "1", "1.0"}:
        return 1.0
    if text in {"f", "female", "0", "0.0"}:
        return 0.0
    raise ValueError(f"Unknown sex value: {value!r}")


def encode_tabular(age, sex) -> np.ndarray:
    """Encode age and sex as the model input: [age / 100, male]."""
    age_value = float(age)
    if not np.isfinite(age_value) or age_value < 0:
        raise ValueError(f"Invalid age: {age!r}")
    return np.asarray([age_value / 100.0, encode_sex(sex)], dtype=np.float32)


def _artifact_score(window: np.ndarray) -> float:
    finite = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
    nonfinite = int((~np.isfinite(window)).sum())
    amplitude_excess = np.maximum(np.abs(finite) - 5.0, 0.0).sum() / finite.size
    differences = np.abs(np.diff(finite, axis=1))
    abrupt_excess = np.maximum(differences - 1.0, 0.0).sum() / max(differences.size, 1)
    flat_leads = int(np.sum(np.ptp(finite, axis=1) < 1e-4))
    return 1000.0 * nonfinite + 100.0 * flat_leads + 100.0 * amplitude_excess + 10.0 * abrupt_excess


def select_window(
    ecg: np.ndarray,
    window_samples: int = TARGET_SAMPLES,
    sampling_rate: int = TARGET_RATE,
    step_seconds: float = 0.25,
) -> Tuple[np.ndarray, int]:
    """Select the least-artifactual fixed-length window, with ties toward the center."""
    if ecg.shape[1] < window_samples:
        raise ValueError(f"ECG is shorter than {window_samples / sampling_rate:.1f} seconds")
    center = (ecg.shape[1] - window_samples) // 2
    if ecg.shape[1] == window_samples:
        return ecg, 0
    max_start = ecg.shape[1] - window_samples
    step = max(1, int(round(step_seconds * sampling_rate)))
    starts = set(range(0, max_start + 1, step))
    starts.update({center, max_start})
    selected = min(
        starts,
        key=lambda start: (_artifact_score(ecg[:, start : start + window_samples]), abs(start - center), start),
    )
    return ecg[:, selected : selected + window_samples], selected


def preprocess_ecg(ecg: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Convert a 12-lead ECG in mV to the model input, or raise ValueError if it fails QC."""
    ecg = np.asarray(ecg, dtype=np.float32)
    if ecg.ndim != 2:
        raise ValueError(f"Expected 2D ECG, got {ecg.shape}")
    if ecg.shape[0] != 12 and ecg.shape[1] == 12:
        ecg = ecg.T
    if ecg.shape[0] != 12:
        raise ValueError(f"Expected 12 leads ordered as {LEAD_ORDER}, got {ecg.shape}")
    if not np.isfinite(ecg).all():
        raise ValueError("ECG contains NaN or infinity")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    if int(sampling_rate) != TARGET_RATE:
        target_length = int(TARGET_RATE * ecg.shape[1] / float(sampling_rate))
        ecg = signal.resample(ecg, target_length, axis=1).astype(np.float32)
    ecg, _ = select_window(ecg)

    nyquist = 0.5 * TARGET_RATE
    sos = signal.butter(
        N=4,
        Wn=[0.5 / nyquist, 40.0 / nyquist],
        btype="bandpass",
        output="sos",
    )
    ecg = signal.sosfiltfilt(sos, ecg, axis=1).astype(np.float32)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    smoothed = np.stack([np.convolve(lead, kernel, mode="same") for lead in ecg]).astype(np.float32)
    if np.max(np.abs(smoothed)) > MAX_ABSOLUTE_MV:
        raise ValueError(f"Preprocessed ECG exceeds the {MAX_ABSOLUTE_MV} mV quality-control limit")
    if np.min(np.ptp(smoothed, axis=1)) < MIN_LEAD_PEAK_TO_PEAK_MV:
        raise ValueError(f"Preprocessed ECG has a lead below {MIN_LEAD_PEAK_TO_PEAK_MV} mV peak-to-peak")
    return smoothed

