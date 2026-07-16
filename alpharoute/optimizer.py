"""Augmented Lagrangian penalty optimizer."""
import numpy as np
from typing import Dict, Tuple


class AugmentedLagrangianOptimizer:

    def __init__(self, grid_shape: Tuple[int, int, int],
                 rho: float = 1.0, rho_growth: float = 1.05,
                 rho_max: float = 50.0):
        nL, H, W = grid_shape
        self.grid_shape = grid_shape
        self.lambda_cap = np.zeros((nL, H, W), dtype=np.float32)
        self.lambda_cut: Dict[str, float] = {}
        self.rho = rho
        self.rho_growth = rho_growth
        self.rho_max = rho_max
        self.overflow_history: list = []
        self._plateau_window = 3
        self._plateau_thresh = 0.02
        self._last_overflow: np.ndarray = None

    def compute_loss(self, cap_orig: np.ndarray, cap_cur: np.ndarray):
        g = np.maximum(-cap_cur, 0).astype(np.float32)
        lag = float((self.lambda_cap * g).sum())
        quad = 0.5 * self.rho * float((g ** 2).sum())
        total = lag + quad

        of_l2 = float((g ** 2).sum())
        self.overflow_history.append(of_l2)
        self._last_overflow = g

        return total, g, {
            'lagrangian': lag,
            'quadratic': quad,
            'overflow_l1': float(g.sum()),
            'overflow_l2': of_l2,
            'congested_cells': int((cap_cur < 0).sum()),
            'max_overflow': float(g.max()),
        }

    def update_multipliers(self, cap_cur: np.ndarray):
        g = np.maximum(-cap_cur, 0).astype(np.float32)
        self.lambda_cap = np.maximum(0, self.lambda_cap + self.rho * g)


        if len(self.overflow_history) >= 2:
            prev, cur = self.overflow_history[-2], self.overflow_history[-1]
            rel_improv = (prev - cur) / (prev + 1e-9)
            if rel_improv < 0.05:
                self.rho = min(self.rho_max, self.rho * self.rho_growth)

    def update_cut_multipliers(self, cut_nets: set, overflow_at_boundary: Dict[str, float]):
        for net in cut_nets:
            g = overflow_at_boundary.get(net, 0.0)
            cur = self.lambda_cut.get(net, 0.0)
            self.lambda_cut[net] = max(0.0, cur + self.rho * g)

    def get_penalty_map(self, cap_cur: np.ndarray) -> np.ndarray:
        g = np.maximum(-cap_cur, 0).astype(np.float32)
        return self.lambda_cap + self.rho * g

    def detect_plateau(self) -> bool:
        if len(self.overflow_history) < self._plateau_window + 1:
            return False
        recent = self.overflow_history[-self._plateau_window:]
        if abs(recent[0]) < 1e-9:
            return True
        return abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-9) < self._plateau_thresh

    def get_overflow_delta(self) -> float:
        if len(self.overflow_history) < 2:
            return 0.0
        return self.overflow_history[-1] - self.overflow_history[-2]
