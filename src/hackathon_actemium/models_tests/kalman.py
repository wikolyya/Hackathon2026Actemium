from __future__ import annotations
import numpy as np

class KalmanLevelFilter:
    """Simple 2-state Kalman filter: level + slope."""
    def __init__(self, dt: float = 1.0, q_level: float = 1e-4, q_slope: float = 1e-4, r_measure: float = 1e-2):
        self.dt = dt
        self.q_level = q_level
        self.q_slope = q_slope
        self.r_measure = r_measure
        self.x_ = None
        self.P_ = None

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        self.x_ = np.array([[y[0]], [0.0]])
        self.P_ = np.eye(2)
        return self

    def filter(self, y):
        y = np.asarray(y, dtype=float)
        if self.x_ is None:
            self.fit(y)
        A = np.array([[1.0, self.dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.array([[self.q_level, 0.0], [0.0, self.q_slope]])
        R = np.array([[self.r_measure]])
        I = np.eye(2)
        xs = []
        x, P = self.x_.copy(), self.P_.copy()
        for obs in y:
            x = A @ x
            P = A @ P @ A.T + Q
            z = np.array([[obs]])
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (I - K @ H) @ P
            xs.append(x[0, 0])
        self.x_, self.P_ = x, P
        return np.array(xs)

    def predict_next(self, steps: int = 1):
        if self.x_ is None:
            raise ValueError('Call fit/filter first.')
        A = np.array([[1.0, self.dt], [0.0, 1.0]])
        x = self.x_.copy()
        preds = []
        for _ in range(steps):
            x = A @ x
            preds.append(x[0, 0])
        return np.array(preds)
