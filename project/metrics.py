from collections.abc import Callable
from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
from dlordinal.metrics import amae
from scipy.stats import hmean

__all__ = ("amae_score", "mae_precision_score", "mae_recall_score", "o1_score")


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


@singledispatch
def mae_precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args: Any,
    suport: np.ndarray | None = None,
    **kwargs: Any,
) -> float | np.ndarray:
    cm = _confusion_matrix(y_true, y_pred, support=suport)
    return mae_precision_score(cm, *args, **kwargs)


@mae_precision_score.register
def _(cm: pd.DataFrame, value: int | None = None, **kwargs: Any) -> float | np.ndarray:
    return _prediction_margin_score(cm, value, axis=1, **kwargs)


@singledispatch
def mae_recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args: Any,
    suport: np.ndarray | None = None,
    **kwargs: Any,
) -> float | np.ndarray:
    cm = _confusion_matrix(y_true, y_pred, support=suport)
    return mae_recall_score(cm, *args, **kwargs)


@mae_recall_score.register
def _(cm: pd.DataFrame, value: int | None = None, **kwargs: Any) -> float | np.ndarray:
    return _prediction_margin_score(cm, value, axis=0, **kwargs)


@singledispatch
def o1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args: Any,
    suport: np.ndarray | None = None,
    **kwargs: Any,
) -> float | np.ndarray:
    cm = _confusion_matrix(y_true, y_pred, support=suport)
    return o1_score(cm, *args, **kwargs)


@o1_score.register
def _(
    cm: pd.DataFrame,
    value: int | None = None,
    *,
    average: Callable[[np.ndarray], float] | None = hmean,
    **kwargs: Any,
) -> float | np.ndarray:
    """Calculate O1 (MAE F1) score from confusion matrix."""
    if value is not None:
        prec = mae_precision_score(cm, value=value, **kwargs)
        rec = mae_recall_score(cm, value=value, **kwargs)
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)
    scores = np.array([o1_score(cm, value=v) for v in cm.index])
    if average is not None:
        return average(scores)
    return scores


# Internals --------------------------------------------------------------------------


def _confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    support: np.ndarray | int | None = None,
) -> pd.DataFrame:
    y_true = y_true
    y_pred = y_pred
    u_true = np.unique(y_true)
    u_pred = np.unique(y_pred)
    if support is None:
        support = np.unique(np.concatenate([u_true, u_pred]))
    cm = (
        pd.crosstab(index=y_true, columns=y_pred)
        .reindex(support, fill_value=0, axis=0)
        .reindex(support, fill_value=0, axis=1)
    )
    return cm


def _mad_error(
    cm: pd.DataFrame,
    value: int,
    axis: int,
    *,
    nan_as_zero: bool = True,
    normalize: bool = False,
) -> float:
    dist = cm.xs(value, axis=axis).to_numpy()
    if (dist_sum := dist.sum()) == 0:
        return 0.0 if nan_as_zero else np.nan
    probs = dist / dist_sum
    support = (cm.index if axis == 0 else cm.columns).to_numpy()
    error = (probs * np.abs(value - support)).sum()
    if normalize:
        max_err = (max(support) - min(support)).item()
        error /= max_err
    return error.item()


def _prediction_margin_score(
    cm: pd.DataFrame,
    value: int | None,
    axis: int,
    *,
    average: Callable[[np.ndarray], float] | None = hmean,
    **kwargs: Any,
) -> float | np.ndarray:
    """Calculate MAE precision from confusion matrix."""
    if value is not None:
        error = _mad_error(cm, value, axis=axis, normalize=True, **kwargs)
        return 1 - error
    scores = np.array([_prediction_margin_score(cm, value=v, axis=axis) for v in cm.index])
    if average is not None:
        return average(scores)
    return scores
