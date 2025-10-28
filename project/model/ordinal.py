from functools import singledispatch

import numpy as np
import torch
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

__all__ = (
    "OrdinalLogit",
    "ordinal_loss",
    "extend_ordinal_labels",
    "ordinal_probs",
)


class OrdinalLogit(nn.Module):
    """Ordinal logit layer for classification tasks."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        bias_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            errmsg = "'num_classes' must be greater than 1."
            raise ValueError(errmsg)
        bias = -(torch.arange(num_classes - 1) + 0.5 - (num_classes - 1) / 2) * bias_scale
        self.bias = nn.Parameter(torch.as_tensor(bias).float())
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.bias_scale = bias_scale

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        indim = self.linear.in_features
        outdim = self.num_classes
        scale = self.bias_scale
        return f"{cn}(in_features={indim}, num_classes={outdim}, bias_scale={scale})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ordinal logit layer."""
        return self.linear(x) + self.bias

    @property
    def num_classes(self) -> int:
        """Number of ordinal classes."""
        return len(self.bias) + 1


def ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute ordinal loss for the logits and labels."""
    loss = sum(
        F.cross_entropy(torch.column_stack([-logits[:, i], logits[:, i]]), labels[:, i])
        for i in range(num_classes - 1)
    )
    return loss


def extend_ordinal_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer ordinal labels to extended binary vector format."""
    labels = labels - labels.min(0).values
    vlen = num_classes - 1
    ext = torch.arange(vlen * labels.size(0), device=labels.device).reshape(-1, vlen)
    elabs = (ext % vlen < labels[:, None]).long()
    return elabs


@singledispatch
def ordinal_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert ordinal logits to class probabilities."""
    probs = F.sigmoid(logits)
    surv = torch.zeros(
        (*probs.shape[:-1], probs.shape[-1] + 2),
        dtype=probs.dtype,
        device=logits.device,
    )
    surv[..., 0] = 1.0
    surv[..., 1:-1] = probs
    return surv[..., :-1] - surv[..., 1:]


@ordinal_probs.register
def _(logits: np.ndarray) -> np.ndarray:
    probs = expit(logits)
    cmf = np.zeros(
        (*probs.shape[:-1], probs.shape[-1] + 2),
        dtype=probs.dtype,
    )
    cmf[..., 0] = 1.0
    cmf[..., 1:-1] = probs
    return cmf[..., :-1] - cmf[..., 1:]
