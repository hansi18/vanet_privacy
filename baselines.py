"""
baselines.py
Implements the three baseline models for comparison:
  1. FL + DP         — fixed global ε, no anomaly detection
  2. DT + FL         — digital twin for resource allocation only, no DP
  3. Adaptive DP-FL  — per-round adaptive ε based on gradient norm, no trust
"""

import numpy as np
import torch
import torch.nn as nn
from models import AttackDetector


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: FL + DP (Static Differential Privacy)
# ─────────────────────────────────────────────────────────────────────────────

class FLDPBaseline:
    """
    Standard FL with fixed global ε for all vehicles.
    No per-vehicle budget tracking, no anomaly detection.
    Represents the simplest privacy-preserving FL.
    """

    def __init__(self, fixed_eps=0.50, clip_C=1.0):
        self.fixed_eps = fixed_eps
        self.clip_C = clip_C
        self.global_params = AttackDetector().get_flat_params().copy()
        self.model = AttackDetector()

    def run_round(self, vehicle_data_list, class_weights=None):
        updates = []
        for vd in vehicle_data_list:
            X, y = vd["X"], vd["y"]
            if len(np.unique(y)) < 2 or len(y) < 4:
                continue

            model = AttackDetector()
            model.set_flat_params(self.global_params)
            model.train()

            if class_weights is not None:
                cw = torch.tensor(class_weights, dtype=torch.float32)
                criterion = nn.CrossEntropyLoss(weight=cw)
            else:
                criterion = nn.CrossEntropyLoss()

            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)

            for _ in range(2):
                opt.zero_grad()
                loss = criterion(model(Xt), yt)
                loss.backward()
                opt.step()

            grad = model.get_flat_grads()

            # Fixed ε for all vehicles
            norm = np.linalg.norm(grad)
            if norm > self.clip_C:
                grad = grad * (self.clip_C / norm)

            delta = 1e-5
            sigma = self.clip_C * np.sqrt(2 * np.log(1.25 / delta)) / self.fixed_eps
            dp_grad = grad + np.random.normal(0, sigma, grad.shape)
            updates.append({"gradient": dp_grad, "n_samples": len(y)})

        if not updates:
            return self.global_params

        total_n = sum(u["n_samples"] for u in updates) + 1e-9
        agg = sum(u["n_samples"] * u["gradient"] for u in updates) / total_n
        self.global_params = self.global_params - 0.01 * agg
        self.model.set_flat_params(self.global_params)
        return self.global_params.copy()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
        return out.argmax(dim=1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: DT + FL (Digital Twin for resource allocation, no DP)
# ─────────────────────────────────────────────────────────────────────────────

class DTFLBaseline:
    """
    Digital Twin used only for resource allocation (bandwidth, participation).
    No differential privacy. No gradient anomaly screening.
    Represents DT used purely for optimisation, not security.
    """

    def __init__(self):
        self.global_params = AttackDetector().get_flat_params().copy()
        self.model = AttackDetector()
        # Lightweight twin: only tracks participation count for weighting
        self.participation_counts = {}

    def run_round(self, vehicle_data_list, class_weights=None):
        updates = []
        for vd in vehicle_data_list:
            X, y = vd["X"], vd["y"]
            vid = vd.get("vid", "unknown")
            if len(np.unique(y)) < 2 or len(y) < 4:
                continue

            model = AttackDetector()
            model.set_flat_params(self.global_params)
            model.train()

            if class_weights is not None:
                cw = torch.tensor(class_weights, dtype=torch.float32)
                criterion = nn.CrossEntropyLoss(weight=cw)
            else:
                criterion = nn.CrossEntropyLoss()

            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)

            for _ in range(2):
                opt.zero_grad()
                loss = criterion(model(Xt), yt)
                loss.backward()
                opt.step()

            grad = model.get_flat_grads()  # No DP noise
            self.participation_counts[vid] = self.participation_counts.get(vid, 0) + 1

            # DT weights by participation (resource allocation)
            weight = 1.0 / (1.0 + self.participation_counts[vid] * 0.1)
            updates.append({"gradient": grad, "n_samples": len(y), "weight": weight})

        if not updates:
            return self.global_params

        total_w = sum(u["weight"] * u["n_samples"] for u in updates) + 1e-9
        agg = sum(u["weight"] * u["n_samples"] * u["gradient"] for u in updates) / total_w
        self.global_params = self.global_params - 0.01 * agg
        self.model.set_flat_params(self.global_params)
        return self.global_params.copy()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
        return out.argmax(dim=1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: Adaptive DP-FL Non-IID
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveDPFLBaseline:
    """
    Adapts ε per round based on gradient norm magnitude.
    High norm variance → lower ε (more noise).
    No trust scores, no cross-round memory, no cosine checks.
    Based on Fu et al. (2022) style adaptive clipping.
    """

    def __init__(self, eps_min=0.05, eps_max=1.50, clip_C=1.0):
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.clip_C = clip_C
        self.global_params = AttackDetector().get_flat_params().copy()
        self.model = AttackDetector()
        self.round_norm_history = []

    def _adapt_eps(self, grad_norm):
        """Adapt ε based on current round's gradient norm vs history."""
        if len(self.round_norm_history) < 2:
            return (self.eps_min + self.eps_max) / 2.0
        mu = np.mean(self.round_norm_history)
        sigma = np.std(self.round_norm_history) + 1e-9
        # Normalise norm to [0,1] and map to ε range
        alpha = 1.0 - np.tanh(abs(grad_norm - mu) / sigma)
        eps = self.eps_min + alpha * (self.eps_max - self.eps_min)
        return float(np.clip(eps, self.eps_min, self.eps_max))

    def run_round(self, vehicle_data_list, class_weights=None):
        updates = []
        round_norms = []

        for vd in vehicle_data_list:
            X, y = vd["X"], vd["y"]
            if len(np.unique(y)) < 2 or len(y) < 4:
                continue

            model = AttackDetector()
            model.set_flat_params(self.global_params)
            model.train()

            if class_weights is not None:
                cw = torch.tensor(class_weights, dtype=torch.float32)
                criterion = nn.CrossEntropyLoss(weight=cw)
            else:
                criterion = nn.CrossEntropyLoss()

            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)

            for _ in range(2):
                opt.zero_grad()
                loss = criterion(model(Xt), yt)
                loss.backward()
                opt.step()

            grad = model.get_flat_grads()
            norm = np.linalg.norm(grad)
            round_norms.append(norm)

            if norm > self.clip_C:
                grad = grad * (self.clip_C / norm)

            eps_v = self._adapt_eps(norm)
            delta = 1e-5
            sigma = self.clip_C * np.sqrt(2 * np.log(1.25 / delta)) / eps_v
            dp_grad = grad + np.random.normal(0, sigma, grad.shape)
            updates.append({"gradient": dp_grad, "n_samples": len(y)})

        if round_norms:
            self.round_norm_history.append(np.mean(round_norms))

        if not updates:
            return self.global_params

        total_n = sum(u["n_samples"] for u in updates) + 1e-9
        agg = sum(u["n_samples"] * u["gradient"] for u in updates) / total_n
        self.global_params = self.global_params - 0.01 * agg
        self.model.set_flat_params(self.global_params)
        return self.global_params.copy()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
        return out.argmax(dim=1).numpy()
