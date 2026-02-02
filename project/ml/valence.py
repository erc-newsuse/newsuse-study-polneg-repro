"""Newsuse valence classification model with ordinal classification heads.

This is a classifier for detecting valence of news articles, posts, headlines etc.
It distinguishes between event valence (whether the reported event is likely to be
perceived as negative, neutral, or positive), and sentiment
(whether the language and framing used in the text expresses negative, neutral, or positive
attitudes or emotions).

It is implemented as an extension of XLM-RoBERTa-Large with two ordinal classification
heads (negative < neutral < positive).

It is implemented using Hugging Face Transformers and PyTorch, and tested with:
- transformers[torch]==4.45.2

It comes with a custom pipeline for multi-target text classification tasks registered
under the label "text-multi-classification". The pipeline uses the following defalts:
- truncation: True
- padding: "max_length"
- max_length: model.config.model_max_length
- tokenizer: model's tokenizer
"""
from collections.abc import Mapping
from copy import deepcopy
from functools import singledispatch, singledispatchmethod, wraps
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import torch
import transformers
from scipy.special import expit
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.pipelines import PIPELINE_REGISTRY, Pipeline, TextClassificationPipeline
from transformers.utils import ModelOutput

ID2LABEL = MappingProxyType(
    {
        "event": {0: -1, 1: 0, 2: 1},
        "sentiment": {0: -1, 1: 0, 2: 1},
    }
)

__all__ = (
    "NewsuseValenceClassifier",
    "NewsuseValenceClassifierConfig",
)

# TEXT MULTI-CLASSIFICATION PIPELINE -------------------------------------------------

PipeOutputT = dict[str, torch.Tensor]
PipeParamsT = tuple[dict[str, Any], dict[str, Any], dict[str, Any]]


@wraps(transformers.pipeline)
def pipeline(task: str, model: Any, *args: Any, **kwargs: Any) -> transformers.Pipeline:
    kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu", **kwargs}
    pipe = transformers.pipeline(task, model, *args, **kwargs)
    if task in ("text-classification", "text-multi-classification"):
        kwargs = {
            "truncation": True,
            "padding": "max_length",
            "max_length": pipe.tokenizer.model_max_length,
            "tokenizer": pipe.tokenizer,
            **kwargs,
        }
    return transformers.pipeline(task, pipe.model, *args, **kwargs)


class TextMultiClassificationPipeline(Pipeline):
    """Text classification pipeline for tasks with multiple targets."""

    task: ClassVar[str] = "text-multi-classification"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.framework != "pt":
            cn = self.__class__.__name__
            errmsg = f"'{cn}' only supports PyTorch models."
            raise ValueError(errmsg)

    def _sanitize_parameters(self, *args: Any, **kwargs: Any) -> PipeParamsT:
        return TextClassificationPipeline._sanitize_parameters(self, *args, **kwargs)

    def _forward(self, *args: Any, **kwargs: Any) -> ModelOutput:
        """Forward pass through the model."""
        return TextClassificationPipeline._forward(self, *args, **kwargs)

    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        """Pre-process the inputs before passing them to the model."""
        return TextClassificationPipeline.preprocess(self, *args, **kwargs)

    def postprocess(
        self,
        model_outputs: ModelOutput,
        *,
        top_k: int | None = 1,
        **kwargs: Any,  # noqa
    ) -> PipeOutputT:
        """Post-process the model outputs to return multi-target classification results."""
        logits = torch.swapaxes(model_outputs["logits"], 0, 1)
        scores = {
            target: self.compute_scores(self.model, lp)
            for target, lp in zip(self.model.config.targets, logits, strict=True)
        }
        output = {
            target: [
                {"label": self.model.config.id2label[target][int(i)], "score": float(s)}
                for i, s in enumerate(score)
            ]
            for target, score in scores.items()
        }
        if top_k is not None:
            for k in output:
                output[k].sort(key=lambda x: x["score"], reverse=True)
            if top_k == 1:
                output = {k: v[0] for k, v in output.items()}
            else:
                output = {k: v[:top_k] for k, v in output.items()}
        return output

    @singledispatchmethod
    def compute_scores(
        self,
        model: PreTrainedModel,  # noqa
        logits: torch.Tensor,
    ) -> PipeOutputT:
        """Model-specific post-processing for multi-target classification."""
        return F.softmax(logits, dim=-1).squeeze()


PIPELINE_REGISTRY.register_pipeline(
    task=TextMultiClassificationPipeline.task,
    pipeline_class=TextMultiClassificationPipeline,
)


# ORDINAL LOGIT ----------------------------------------------------------------------


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
    cprobs = surv[..., :-1] - surv[..., 1:]
    return cprobs


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


@singledispatch
def ordinal_inverse(probs: torch.Tensor) -> torch.Tensor:
    """Convert ordinal class probabilities to logits."""
    return -torch.log(1 / (1 - probs.cumsum(-1)[..., :-1]) - 1)


@ordinal_inverse.register
def _(probs: np.ndarray) -> np.ndarray:
    return -np.log(1 / (1 - probs.cumsum(-1)[..., :-1]) - 1)


# VALENCE CLASSIFIER -----------------------------------------------------------------


class NewsuseValenceClassifierConfig(PretrainedConfig):
    """Configuration for `NewsuseValenceClassifier`.

    Attributes
    ----------
    base_name_or_path
        Name or path of the base model.
    ordinal_logit_bias_scale
        Scale of the bias term in the ordinal logit layers.
    num_shared_layers
        Number of shared layers in the classification model.
    num_head_layers
        Number of layers in each classification head.
    head_poolers
        Use tanh poolers at the end of feed forward networks within each head.
    dropout
        Dropout rate for the classification head.
    layer_norm_eps
        Epsilon value for layer normalization.
    id2label
        Mapping from label IDs to label names.
    """

    model_type = "newsuse-valence-classifier"

    def __init__(
        self,
        base: PretrainedConfig | Mapping[str, Any] | str | None = None,
        *,
        ordinal_logit_bias_scale: float = 2.0,
        num_shared_layers: int = 0,
        num_head_layers: int = 1,
        head_poolers: bool = False,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        id2label: Mapping[str, Mapping[int, int]] | None = None,
        **kwargs: Any,
    ) -> None:
        if ordinal_logit_bias_scale <= 0:
            errmsg = "'ordinal_logit_bias_scale' must be positive."
            raise ValueError(errmsg)
        super().__init__()
        # Set base model config
        if base is None:
            self.base = PretrainedConfig(**kwargs)
        elif isinstance(base, PretrainedConfig):
            self.base = base
            self.base.update(kwargs)
        elif isinstance(base, Mapping):
            base = deepcopy(base)
            base.update(kwargs)
            _name_or_path = base.pop("_name_or_path")
            self.base = AutoConfig.from_pretrained(_name_or_path, **base)
        elif isinstance(base, str):
            self.base = AutoConfig.from_pretrained(base, **kwargs)
        else:
            errmsg = f"unsupported base config type: '{type(base)}'"
            raise TypeError(errmsg)
        # Set model config parameters
        self.ordinal_logit_bias_scale = ordinal_logit_bias_scale
        self.num_shared_layers = num_shared_layers
        self.num_head_layers = num_head_layers
        self.head_poolers = head_poolers
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        # Keep the original order of targets
        self.id2label = id2label or dict(ID2LABEL)
        self.id2label = {
            target: {int(k): int(v) for k, v in self.id2label[target].items()}
            for target in ID2LABEL
        }
        self.label2id = {
            feature: {int(label): int(i) for i, label in mapping.items()}
            for feature, mapping in self.id2label.items()
        }

    @property
    def base_name_or_path(self) -> str | None:
        """Name or path of the base model."""
        return getattr(self.base, "_name_or_path", None)

    @property
    def targets(self) -> list[str]:
        """List of target tasks."""
        return list(self.id2label)


class Pooler(nn.Sequential):
    """Pooling layer applied in-between feed forward and classification branches."""

    def __init__(self, config: NewsuseValenceClassifierConfig) -> None:
        super().__init__(
            nn.Linear(config.base.hidden_size, config.base.hidden_size), nn.Tanh()
        )


class FeedForward(nn.Sequential):
    """Feed forward layer used in shared and head-specific networks."""

    def __init__(self, config: NewsuseValenceClassifierConfig) -> None:
        super().__init__(
            nn.Linear(config.base.hidden_size, config.base.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.base.hidden_size, config.layer_norm_eps),
            nn.Dropout(config.dropout),
        )


class NewsuseValenceClassifierHead(nn.Module):
    """Classification head for a single target in `NewsuseValenceClassifier`."""

    def __init__(
        self,
        config: NewsuseValenceClassifierConfig,
        target: str,
    ) -> None:
        super().__init__()
        self.target = target
        self._use_gradient_checkpointing = False
        self.ff = nn.ModuleList(
            [FeedForward(config) for _ in range(config.num_head_layers)]
        )
        self.pooler = Pooler(config) if config.head_poolers else nn.Identity()
        self.ordinal = OrdinalLogit(
            config.base.hidden_size,
            num_classes=len(config.id2label[target]),
            bias_scale=config.ordinal_logit_bias_scale,
        )

    @property
    def num_classes(self) -> int:
        """Number of classes for the target."""
        return self.ordinal.num_classes

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the classification head.

        Parameters
        ----------
        hidden
            Hidden states obtained from the shared feed forward network.
        """
        forward = (
            self._forward_with_checkpointing
            if self._use_gradient_checkpointing
            else self._forward
        )
        return forward(hidden)

    def _forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.ff:
            hidden = layer(hidden)
        pooled = self.pooler(hidden)
        logits = self.ordinal(pooled)
        return logits

    def _forward_with_checkpointing(self, hidden: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""
        return checkpoint(self._forward, hidden, use_reentrant=False)


class NewsuseValenceClassifier(PreTrainedModel):
    """Newsuse valence classifier with ordinal classification heads."""

    config_class: ClassVar[type[PretrainedConfig]] = NewsuseValenceClassifierConfig
    supports_gradient_checkpointing: ClassVar[bool] = True

    def __init__(self, config: NewsuseValenceClassifierConfig) -> None:
        super().__init__(config)
        self.base = AutoModel.from_pretrained(
            (d := self.config.base.to_dict()).pop("_name_or_path"), **d
        )
        self.ff = nn.ModuleList(
            [FeedForward(config) for _ in range(config.num_shared_layers)]
        )
        self.heads = nn.ModuleDict(
            {
                target: NewsuseValenceClassifierHead(config, target)
                for target in config.targets
            }
        )
        self._use_gradient_checkpointing = False
        # Initialize weights and biases
        self.init_weights()

    def feed_forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Feed forward through the shared layers."""
        feed_forward = (
            self._feed_forward_with_checkpointing
            if self._use_gradient_checkpointing
            else self._feed_forward
        )
        return feed_forward(pooled_output)

    @wraps(feed_forward)
    def _feed_forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        hidden = pooled_output
        for layer in self.ff:
            hidden = layer(hidden)
        return hidden

    @wraps(feed_forward)
    def _feed_forward_with_checkpointing(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Feed forward with gradient checkpointing."""
        return checkpoint(self._feed_forward, pooled_output, use_reentrant=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        """Forward pass through the model.

        Parameters
        ----------
        input_ids : torch.Tensor, optional
            Token IDs of shape (batch_size, seq_len).
        attention_mask : torch.Tensor, optional
            Attention mask of shape (batch_size, seq_len).
        token_type_ids : torch.Tensor, optional
            Token type IDs.
        position_ids : torch.Tensor, optional
            Position IDs.
        head_mask : torch.Tensor, optional
            Head mask.
        inputs_embeds : torch.Tensor, optional
            Input embeddings.
        labels : dict[str, torch.Tensor], optional
            Dictionary with ground truth labels for training.
        output_attentions : bool, optional
            Whether to return attention weights.
        output_hidden_states : bool, optional
            Whether to return hidden states.
        return_dict : bool, optional
            Whether to return a dict or tuple.

        Returns
        -------
        SequenceClassifierOutput or tuple
            SequenceClassifierOutput containing logits and optionally loss.
        """
        if not (labels := kwargs.get("labels")):
            labels = {t: kwargs.pop(t) for t in self.config.targets if t in kwargs}
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if isinstance(outputs, tuple):
            errmsg = "'forward' method requires 'return_dict=True'"
            raise ValueError(errmsg)
        pooled_output = self._get_pooled_output(outputs)
        hidden = self.feed_forward(pooled_output)
        # logits = {target: head(hidden) for target, head in self.heads.items()}
        logits = torch.stack([head(hidden) for head in self.heads.values()], dim=1)

        loss = None
        # Calculate multi-target loss if labels are provided
        if labels:
            loss = self.compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _get_pooled_output(self, outputs: Mapping) -> torch.Tensor:
        """Get pooled output from the model outputs."""
        if self.config.base_name_or_path.startswith("distilbert"):
            return outputs.last_hidden_state[:, 0]
        return outputs.pooler_output

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: Mapping[str, int],
    ) -> torch.Tensor:
        """Compute multi-target ordinal loss.

        Parameters
        ----------
        logits
            Dictionary with logits for each target.
        labels
            Ground truth labels for each target.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from target names to their respective losses.
        """
        loss = 0.0
        logits = torch.swapaxes(logits, 0, 1)
        for target, target_logits in zip(self.config.targets, logits, strict=True):
            num_classes = self.heads[target].num_classes
            label = torch.as_tensor(labels[target])
            if label.ndim == 0:
                label = label[None, ...]
            extended_labels = extend_ordinal_labels(label, num_classes)
            loss += ordinal_loss(target_logits, extended_labels, num_classes)
        return loss / len(self.config.targets)

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any) -> None:
        self.base.gradient_checkpointing_enable(*args, **kwargs)
        self._use_gradient_checkpointing = True
        for head in self.heads.values():
            head._use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.base.gradient_checkpointing_disable()
        self._use_gradient_checkpointing = False
        for head in self.heads.values():
            head._use_gradient_checkpointing = False


# Register the model configuration and model class
AutoConfig.register("newsuse-valence-classifier", NewsuseValenceClassifierConfig)
AutoModel.register(NewsuseValenceClassifierConfig, NewsuseValenceClassifier)


@TextMultiClassificationPipeline.compute_scores.register
def _(
    self,  # noqa
    model: NewsuseValenceClassifier,  # noqa
    logits: torch.Tensor,
) -> PipeOutputT:
    """Model-specific logits post-processing for multi-target classification."""
    return ordinal_probs(logits).squeeze()
