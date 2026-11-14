"""Self-test on synthetic signals; needs no patient data and no trained weights."""

import numpy as np

from model import build_model
from preprocessing import MIN_LEAD_PEAK_TO_PEAK_MV, encode_tabular, preprocess_ecg
from runtime import predict_scores

EXPECTED_PARAMETERS = 3_510_754


def synthetic_ecg(seconds: int = 12, sampling_rate: int = 1000) -> np.ndarray:
    """Return an artificial 12-lead signal in mV. It is not a physiological ECG."""
    rng = np.random.default_rng(0)
    time = np.arange(seconds * sampling_rate) / sampling_rate
    signal = np.stack([np.sin(2 * np.pi * (1.0 + lead / 10) * time) for lead in range(12)])
    return (signal + rng.normal(0, 0.02, signal.shape)).astype(np.float32)  # shape: (12, 12000)


def main() -> None:
    # The network is randomly initialized, so the scores below have no meaning.
    model = build_model().eval()
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert parameters == EXPECTED_PARAMETERS, f"Expected {EXPECTED_PARAMETERS} parameters, got {parameters}"

    raw = synthetic_ecg()
    ecg = np.stack([preprocess_ecg(raw, sampling_rate=1000)] * 2)  # shape: (2, 12, 5000)
    assert ecg.shape == (2, 12, 5000), f"Unexpected ECG shape {ecg.shape}"

    tabular = np.stack([encode_tabular(60, "male"), encode_tabular(75, "female")])  # shape: (2, 2)
    scores = predict_scores(model, ecg, tabular, batch_size=2)
    assert scores.shape == (2,), f"Unexpected score shape {scores.shape}"
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()

    # A lead with almost no signal must be rejected, not scored.
    flat = raw.copy()
    flat[3] = flat[3] * (MIN_LEAD_PEAK_TO_PEAK_MV / 100)
    try:
        preprocess_ecg(flat, sampling_rate=1000)
    except ValueError as error:
        print("Flat lead rejected:", error)
    else:
        raise AssertionError("A flat lead passed quality control")

    print("Smoke test passed")
    print("parameters:", parameters)


if __name__ == "__main__":
    main()
