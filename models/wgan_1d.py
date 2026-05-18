from __future__ import annotations

from models.dcgan_1d import ConditionalGenerator1D
import torch
import torch.nn as nn


def _label_channel(labels: torch.Tensor, embedding: nn.Embedding, length: int) -> torch.Tensor:
    return embedding(labels).unsqueeze(-1).expand(-1, -1, length)


class ConditionalCritic1D(nn.Module):
    def __init__(self, num_classes: int = 5, signal_length: int = 2048, base_channels: int = 64):
        super().__init__()
        if signal_length % 32 != 0:
            raise ValueError("signal_length must be divisible by 32")
        self.label_embed = nn.Embedding(num_classes, 1)
        self.features = nn.Sequential(
            nn.Conv1d(2, base_channels, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base_channels, base_channels * 2, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base_channels * 2, base_channels * 4, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base_channels * 4, base_channels * 8, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base_channels * 8, base_channels * 8, 4, 2, 1), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Linear(base_channels * 8 * (signal_length // 32), 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = torch.cat([x, _label_channel(labels, self.label_embed, x.shape[-1])], dim=1)
        return self.head(self.features(h).flatten(1)).squeeze(1)
