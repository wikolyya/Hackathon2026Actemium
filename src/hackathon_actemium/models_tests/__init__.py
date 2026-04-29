from .baseline import PersistenceBaseline
from .kalman import KalmanLevelFilter
from .linear_local import RegimeLocalLinearRegressor
from .xgb_model import XGBTimeSeriesRegressor

__all__ = [
    "PersistenceBaseline",
    "KalmanLevelFilter",
    "RegimeLocalLinearRegressor",
    "XGBTimeSeriesRegressor",
]

try:
    from .gru_model import GRURegressor
    from .lstm_model import LSTMRegressor
    from .tcn_model import TCNRegressor
    from .temporal_transformer import TemporalTransformerRegressor

    __all__ += [
        "GRURegressor",
        "LSTMRegressor",
        "TCNRegressor",
        "TemporalTransformerRegressor",
    ]
except Exception:
    # Torch is optional for the quick tabular test runner.
    pass
