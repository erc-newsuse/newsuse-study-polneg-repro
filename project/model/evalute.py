from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from scipy.stats import hmean

MultiOutputT = Mapping[str, torch.Tensor]


class NewsuseNegativityEvaluator:
    """Evaluator for :class:`~project.model.newsuse.NewsuseNegativityModel` models.

    Attributes
    ----------
    metric
        Metric function to evaluate predictions.
    **kwargs
        Additional keyword arguments to pass to the metric function.
    """

    def __init__(
        self,
        metric: Callable[[Sequence[Any], Sequence[Any], ...], float],
        *,
        reduction: Callable[[Sequence[float]], float] = hmean,
        **kwargs: Any,
    ) -> None:
        self.metric = metric
        self.reduction = reduction
        self.kwargs = kwargs

    def __call__(self, scores: MultiOutputT, labels: MultiOutputT) -> float:
        """Evaluate the model predictions.

        Parameters
        ----------
        scores
            Model output scores.
        labels
            Ground truth labels.

        Returns
        -------
        float
            The aggregated evaluation metric.
        """
        metrics = self._compute_metrics(scores, labels)
        metrics = {"overall": self.reduction(list(metrics.values())), **metrics}
        return metrics

    def _compute_metrics(
        self, scores: MultiOutputT, labels: MultiOutputT
    ) -> Mapping[str, float]:
        """Compute metrics for multiple outputs."""
        assert list(scores) == (
            list(labels),
            "'scores' and 'labels' must have the same keys",
        )
        return {k: self._compute_metric(scores[k], labels[k]) for k in scores}

    def _compute_metric(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute the metric for the given scores and labels."""
        y_pred = scores.cpu().numpy().argmax(axis=-1)
        y_true = labels.cpu().numpy()
        return self.metric(y_true, y_pred, **self.kwargs)
