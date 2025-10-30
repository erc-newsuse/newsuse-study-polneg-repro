import numpy as np
from dlordinal.metrics import amae

__all__ = ("amae_score",)


def amae_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
) -> float:
    """Calculate the Average Mean Absolute Error (AMAE) score."""
    if num_classes is None:
        num_classes = np.unique(np.concatenate([y_true, y_pred])).size
    error = amae(y_true, y_pred)
    return (1 - error / (num_classes - 1)).item()
