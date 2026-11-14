"""Weights loading and batched inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from model import build_model


def load_model(weights: str | Path, device: str = "cpu") -> Tuple[torch.nn.Module, dict]:
    payload = torch.load(weights, map_location=device, weights_only=True)
    if "model_state_dict" not in payload:
        raise ValueError("Weights file does not contain model_state_dict")
    model = build_model(payload.get("model_config", {}))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


@torch.inference_mode()
def predict_scores(
    model: torch.nn.Module,
    ecg: np.ndarray,
    tabular: np.ndarray,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=np.float32)
    tabular = np.asarray(tabular, dtype=np.float32)
    if ecg.ndim != 3 or ecg.shape[1:] != (12, 5000):
        raise ValueError(f"Expected ECG [N, 12, 5000], got {ecg.shape}")
    if tabular.shape != (len(ecg), 2):
        raise ValueError(f"Expected tabular [N, 2], got {tabular.shape}")
    scores = []
    for start in range(0, len(ecg), batch_size):
        x = torch.from_numpy(ecg[start : start + batch_size]).to(device)
        t = torch.from_numpy(tabular[start : start + batch_size]).to(device)
        scores.append(model.mortality_score(x, t).squeeze(1).cpu().numpy())
    return np.concatenate(scores).astype(np.float32)
