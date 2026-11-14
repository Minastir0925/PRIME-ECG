"""Generate ECG-level scores from an NPZ file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import encode_tabular, preprocess_ecg
from runtime import load_model, predict_scores


def load_input(path: Path, preprocess_raw: bool, sampling_rate: int):
    data = np.load(path, allow_pickle=False)
    ecg = data["ecg"]
    if preprocess_raw:
        ecg = np.stack([preprocess_ecg(item, sampling_rate) for item in ecg])
    if "tabular" in data:
        tabular = data["tabular"].astype(np.float32)
    elif "age" in data and "sex" in data:
        tabular = np.stack([encode_tabular(age, sex) for age, sex in zip(data["age"], data["sex"])])
    else:
        raise ValueError("NPZ must contain tabular or both age and sex")
    return data, ecg.astype(np.float32), tabular


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preprocess-raw", action="store_true")
    parser.add_argument("--sampling-rate", type=int, default=500)
    args = parser.parse_args()

    source, ecg, tabular = load_input(args.input, args.preprocess_raw, args.sampling_rate)
    model, _ = load_model(args.weights, args.device)
    scores = predict_scores(model, ecg, tabular, args.batch_size, args.device)
    frame = pd.DataFrame({"ecg_index": np.arange(len(scores)), "score": scores})
    for key in ("label", "patient_id", "record_id"):
        if key in source:
            frame[key] = source[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} ECG scores to {args.output}")


if __name__ == "__main__":
    main()
