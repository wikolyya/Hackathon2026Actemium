from __future__ import annotations
import torch
from torch import nn
from ._torch_common import TorchSequenceRegressorBase

class _GRUNet(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Linear(hidden_size//2, 1))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

class GRURegressor(TorchSequenceRegressorBase):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
    def _build_model(self, n_features: int):
        return _GRUNet(n_features, self.hidden_size, self.num_layers, self.dropout)
