from __future__ import annotations

import torch
import torch.nn as nn


class WindowMLP(nn.Module):
    """对每个肌电窗口提取特征，再对一个样本的有效窗口做平均。"""

    def __init__(self, channels: int, samples: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        input_dim = channels * samples
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, windows: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, samples = windows.shape
        flat = windows.reshape(batch_size * sequence_length, channels * samples)
        features = self.encoder(flat).reshape(batch_size, sequence_length, -1)
        mask = attention_mask.to(features.dtype).unsqueeze(-1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)
