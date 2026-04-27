from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class TorchSequenceRegressorBase:
    def __init__(self, lr=1e-3, batch_size=64, epochs=15, device=None, verbose=False):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.model = None

    def _build_model(self, n_features: int):
        raise NotImplementedError

    def fit(self, X, y):
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
        self.model = self._build_model(X_t.shape[-1]).to(self.device)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in range(self.epochs):
            losses = []
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            if self.verbose:
                print(f"epoch={epoch+1} loss={float(np.mean(losses)):.6f}")
        return self

    @torch.no_grad()
    def predict(self, X):
        X_t = torch.as_tensor(X, dtype=torch.float32)
        self.model.eval()
        preds = []
        for i in range(0, len(X_t), self.batch_size):
            xb = X_t[i:i+self.batch_size].to(self.device)
            preds.append(self.model(xb).cpu().numpy())
        return np.concatenate(preds, axis=0).reshape(-1)
