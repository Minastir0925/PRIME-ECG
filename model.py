"""Definition of the PRIME-ECG independent-lead CNN-GRU model."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn


TASK_NAME = "mortality_28d"


class LeadCNNGRUEncoder(nn.Module):
    """One independent CNN + bidirectional GRU branch for one ECG lead."""

    def __init__(
        self,
        conv_channels: Iterable[int] = (24, 48, 96, 128),
        gru_hidden: int = 96,
        gru_layers: int = 1,
        embedding_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = [int(value) for value in conv_channels]
        self.cnn = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Dropout1d(dropout * 0.2),
            nn.Conv1d(c1, c2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.Dropout1d(dropout * 0.2),
            nn.Conv1d(c2, c3, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c3),
            nn.GELU(),
            nn.Dropout1d(dropout * 0.2),
            nn.Conv1d(c3, c4, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(c4),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=c4,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(gru_hidden * 4),
            nn.Linear(gru_hidden * 4, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.gru(x)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)
        return self.projection(pooled)


class PrimeECG(nn.Module):
    """Twelve independent lead encoders with attention and age/sex fusion."""

    def __init__(
        self,
        binary_tasks: Optional[Iterable[str]] = None,
        conv_channels: Iterable[int] = (24, 48, 96, 128),
        lead_hidden: int = 96,
        lead_gru_layers: int = 1,
        lead_embedding_dim: int = 128,
        tabular_dim: int = 2,
        tabular_hidden: int = 32,
        dropout: float = 0.25,
        classifier_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.binary_tasks = list(binary_tasks or [TASK_NAME])
        self.lead_encoders = nn.ModuleList(
            LeadCNNGRUEncoder(
                conv_channels=conv_channels,
                gru_hidden=lead_hidden,
                gru_layers=lead_gru_layers,
                embedding_dim=lead_embedding_dim,
                dropout=dropout,
            )
            for _ in range(12)
        )
        self.lead_attention = nn.Sequential(
            nn.Linear(lead_embedding_dim, lead_embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(lead_embedding_dim // 2, 1),
        )
        ecg_feature_dim = lead_embedding_dim * 2
        self.feature_norm = nn.LayerNorm(ecg_feature_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tabular_hidden, tabular_hidden),
            nn.GELU(),
        )
        feature_dim = ecg_feature_dim + tabular_hidden
        self.fusion_norm = nn.LayerNorm(feature_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim, classifier_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, 1),
            )

        self.binary_heads = nn.ModuleDict({task: make_head() for task in self.binary_tasks})

    def encode_leads(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 12:
            raise ValueError(f"Expected ECG [batch, 12, samples], got {tuple(x.shape)}")
        encoded = [
            encoder(x[:, index : index + 1, :])
            for index, encoder in enumerate(self.lead_encoders)
        ]
        return torch.stack(encoded, dim=1)

    def fuse_lead_features(self, lead_features: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.lead_attention(lead_features), dim=1)
        attended = (lead_features * attention).sum(dim=1)
        averaged = lead_features.mean(dim=1)
        return self.feature_norm(torch.cat([attended, averaged], dim=1))

    def forward_features(self, x: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        ecg_features = self.embedding_dropout(self.fuse_lead_features(self.encode_leads(x)))
        tabular_features = self.tabular_encoder(tabular)
        fused = torch.cat([ecg_features, tabular_features], dim=1)
        return self.fusion_dropout(self.fusion_norm(fused))

    def forward(self, x: torch.Tensor, tabular: torch.Tensor) -> Dict[str, object]:
        features = self.forward_features(x, tabular)
        binary_logits = {task: head(features) for task, head in self.binary_heads.items()}
        return {
            "features": features,
            "binary": binary_logits,
            "logit": binary_logits["mortality_28d"],
        }

    def mortality_score(self, x: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Sigmoid of the 28-day mortality logit: a ranking score, not a calibrated probability."""
        return torch.sigmoid(self.forward(x, tabular)["logit"])


def build_model(config: Optional[dict] = None) -> PrimeECG:
    """Build the PRIME-ECG architecture; the defaults give the 3,510,754-parameter model."""
    config = config or {}
    return PrimeECG(
        binary_tasks=config.get("binary_tasks", [TASK_NAME]),
        conv_channels=config.get("conv_channels", [24, 48, 96, 128]),
        lead_hidden=int(config.get("lead_hidden", 96)),
        lead_gru_layers=int(config.get("lead_gru_layers", 1)),
        lead_embedding_dim=int(config.get("lead_embedding_dim", 128)),
        tabular_dim=int(config.get("tabular_dim", 2)),
        tabular_hidden=int(config.get("tabular_hidden", 32)),
        dropout=float(config.get("dropout", 0.25)),
        classifier_hidden=int(config.get("classifier_hidden", 256)),
    )

