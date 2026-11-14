"""Evaluate ECG-level discrimination and threshold metrics from a score CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(frame: pd.DataFrame, threshold: float) -> dict:
    y = frame["label"].astype(int).to_numpy()
    score = frame["score"].astype(float).to_numpy()
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    divide = lambda a, b: float(a / b) if b else None
    return {
        "n": int(len(frame)),
        "positives": int(y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "threshold": float(threshold),
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Chosen on independent validation data")
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    if not {"label", "score"}.issubset(frame.columns):
        raise ValueError("Prediction CSV must contain label and score")
    result = {"ecg_level": metrics(frame, args.threshold)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
