from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import hmean
from sklearn.metrics import f1_score
from transformers import EvalPrediction

from .model import NewsuseValenceClassifierConfig
from .ordinal import ordinal_probs

MultiOutputT = Mapping[str, torch.Tensor]


class NewsuseValenceEvaluator:
    """Evaluator for :class:`~project.model.newsuse.NewsuseValenceModel` models.

    Attributes
    ----------
    config
        Model configuration.
    metric
        Metric function to evaluate predictions.
    """

    def __init__(
        self,
        config: NewsuseValenceClassifierConfig,
        *,
        reduction: Callable[[Sequence[float]], float] = hmean,
    ) -> None:
        self.config = config
        self.reduction = reduction

    def __call__(self, eval_pred: EvalPrediction) -> float:
        """Evaluate the model predictions.

        Parameters
        ----------
        eval_pred
            Evaluation prediction object containing model outputs and labels.

        Returns
        -------
        float
            The aggregated evaluation metric.
        """
        metrics = self._compute_metrics(eval_pred)
        metrics = {"overall": self.reduction(list(metrics.values())), **metrics}
        return metrics

    def _compute_metrics(self, eval_pred: EvalPrediction) -> Mapping[str, float]:
        """Compute metrics for multiple outputs."""
        logits, labels = eval_pred
        logits = torch.swapaxes(torch.tensor(logits), 0, 1)
        metrics = {}
        for target, logit, labs in zip(self.config.targets, logits, labels, strict=True):
            true = np.asarray([self.config.label2id[target][label] for label in labs])
            pred = ordinal_probs(logit).argmax(axis=-1)
            values = f1_score(true, pred, average=None)
            target_scores = {}
            for i, v in zip(np.unique(true), values, strict=True):
                target_scores[f"{target}_f1_{i}"] = v.item()
            target_scores = {
                target: self.reduction(list(target_scores.values())).item(),
                **target_scores,
            }
            metrics.update(target_scores)
        return metrics

    def _compute_metric(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute the metric for the given scores and labels."""
        pred = scores.argmax(axis=-1)
        return self.metric(labels, pred, **self.kwargs)
