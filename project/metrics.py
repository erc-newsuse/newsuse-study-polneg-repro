from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import f1_score

__all__ = ("h1_score", "mad_score")


def h1_score(y_true: Sequence[int], y_pred: Sequence[int], **kwargs: Any) -> float:
    """Calculate H1 score."""
    scores = f1_score(y_true, y_pred, average=None, **kwargs)
    return hmean(scores)


def mad_score(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    normalize: float = 1.0,
) -> float:
    """Calculate Mean Absolute Deviation score."""
    return (np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean() / normalize).item()


def nmad_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    support: Sequence[Any] | None = None,
) -> float:
    """Calculate Normalized Mean Absolute Deviation score."""
    y_true = pd.Series(y_true)  # type: ignore
    y_pred = pd.Series(y_pred)  # type: ignore
    pt = y_true.value_counts(normalize=True)
    pp = y_pred.value_counts(normalize=True)
    if support is None:
        support = pt.index.union(pp.index).sort_values()
    pt = pt.reindex(support, fill_value=0).to_numpy()
    pp = pp.reindex(support, fill_value=0).to_numpy()
    xs = support.to_numpy()
    baseline = (np.outer(pt, pp) * np.abs(xs - xs[:, None]) / 2).sum()
    observed = np.abs(y_pred - y_true).mean()  # type: ignore
    score = 1 - observed / baseline if baseline > 0 else 1.0
    return score.item()
