from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """把一个 NCT EMG token 映射到固定维度的表示。"""

    def __init__(self, channels: int, samples: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = channels * samples
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.network(tokens)


class WindowMLP(nn.Module):
    """旧的监督 MLP 接口，保留用于兼容 DB4-only 对照实验。"""

    def __init__(self, channels: int, samples: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = MLPEncoder(channels, samples, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, windows: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, samples = windows.shape
        flat = windows.reshape(batch_size * sequence_length, channels * samples)
        features = self.encoder(flat).reshape(batch_size, sequence_length, -1)
        mask = attention_mask.to(features.dtype).unsqueeze(-1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


class MaskedTokenAutoencoder(nn.Module):
    """用可见 token 的上下文重建被 mask 的 token。"""

    def __init__(self, channels: int, samples: int, hidden_dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.encoder = MLPEncoder(channels, samples, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * samples),
        )

    def reconstruct(
        self,
        windows: torch.Tensor,
        attention_mask: torch.Tensor,
        time_positions: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, channels, samples = windows.shape
        if channels != self.channels or samples != self.samples:
            raise ValueError(
                f"输入 token 维度为 {channels}x{samples}，"
                f"但模型配置为 {self.channels}x{self.samples}。"
            )
        flat = windows.reshape(batch_size, sequence_length, channels * samples)
        masked_input = flat.masked_fill(token_mask.unsqueeze(-1), 0.0)
        encoded = self.encoder(masked_input.reshape(-1, channels * samples))
        encoded = encoded.reshape(batch_size, sequence_length, -1)

        visible_mask = attention_mask & ~token_mask
        visible = visible_mask.to(encoded.dtype).unsqueeze(-1)
        context = (encoded * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1.0)
        context = context.unsqueeze(1).expand(-1, sequence_length, -1)
        decoder_input = torch.cat([context, time_positions.unsqueeze(-1)], dim=-1)
        prediction = self.decoder(decoder_input.reshape(-1, decoder_input.shape[-1]))
        prediction = prediction.reshape(batch_size, sequence_length, channels * samples)
        return prediction, flat
