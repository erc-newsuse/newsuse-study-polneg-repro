from collections.abc import Mapping
from copy import deepcopy
from functools import wraps
from types import MappingProxyType
from typing import Any, ClassVar

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from project.pipelines.pipelines import (
    PipeOutputT,
    TextMultiClassificationPipeline,
)

from .ordinal import OrdinalLogit, extend_ordinal_labels, ordinal_loss, ordinal_probs

ID2LABEL = MappingProxyType(
    {
        "event": {0: -1, 1: 0, 2: 1},
        "sentiment": {0: -1, 1: 0, 2: 1},
    }
)

__all__ = (
    "NewsuseNegativityClassifier",
    "NewsuseNegativityClassifierConfig",
)


class NewsuseNegativityClassifierConfig(PretrainedConfig):
    """Configuration for `NewsuseNegativityClassifier`.

    Attributes
    ----------
    base_name_or_path
        Name or path of the base model.
    ordinal_logit_bias_scale
        Scale of the bias term in the ordinal logit layers.
    shared_layers
        Number of shared layers in the classification model.
    head_layers
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

    model_type = "newsuse-negativity-classifier"

    def __init__(
        self,
        base: PretrainedConfig | Mapping[str, Any] | str | None = None,
        *,
        ordinal_logit_bias_scale: float = 2.0,
        shared_layers: int = 0,
        head_layers: int = 1,
        head_poolers: bool = False,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        id2label: Mapping[str, Mapping[int, str]] | None = None,
        **kwargs: Any,
    ) -> None:
        if ordinal_logit_bias_scale <= 0:
            errmsg = "'ordinal_logit_bias_scale' must be positive."
            raise ValueError(errmsg)
        super().__init__(**kwargs)
        # Set base model config
        if base is None:
            self.base = PretrainedConfig()
        elif isinstance(base, PretrainedConfig):
            self.base = base
        elif isinstance(base, Mapping):
            base = deepcopy(base)
            _name_or_path = base.pop("_name_or_path")
            self.base = AutoConfig.from_pretrained(_name_or_path, **base)
        elif isinstance(base, str):
            self.base = AutoConfig.from_pretrained(base)
        else:
            errmsg = f"unsupported base config type: '{type(base)}'"
            raise TypeError(errmsg)
        # Set model config parameters
        self.ordinal_logit_bias_scale = ordinal_logit_bias_scale
        self.shared_layers = shared_layers
        self.head_layers = head_layers
        self.head_poolers = head_poolers
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        # Keep the original order of targets
        self.id2label = id2label or dict(ID2LABEL)
        self.id2label = {k: self.id2label[k] for k in ID2LABEL}
        self.label2id = {
            feature: {label: i for i, label in mapping.items()}
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

    def __init__(self, config: NewsuseNegativityClassifierConfig) -> None:
        super().__init__(
            nn.Linear(config.base.hidden_size, config.base.hidden_size), nn.Tanh()
        )


class FeedForward(nn.Sequential):
    """Feed forward layer used in shared and head-specific networks."""

    def __init__(self, config: NewsuseNegativityClassifierConfig) -> None:
        super().__init__(
            nn.Linear(config.base.hidden_size, config.base.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.base.hidden_size, config.layer_norm_eps),
            nn.Dropout(config.dropout),
        )


class NewsuseNegativityClassifierHead(nn.Module):
    """Classification head for a single target in `NewsuseNegativityClassifier`."""

    def __init__(
        self,
        config: NewsuseNegativityClassifierConfig,
        target: str,
    ) -> None:
        super().__init__()
        self.target = target
        self._use_gradient_checkpointing = False
        self.ff = nn.ModuleList([FeedForward(config) for _ in range(config.head_layers)])
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


class NewsuseNegativityClassifier(PreTrainedModel):
    """Newsuse negativity classifier with ordinal classification heads."""

    config_class: ClassVar[type[PretrainedConfig]] = NewsuseNegativityClassifierConfig
    supports_gradient_checkpointing: ClassVar[bool] = True

    def __init__(self, config: NewsuseNegativityClassifierConfig) -> None:
        super().__init__(config)
        self.base = AutoModel.from_pretrained(
            (d := self.config.base.to_dict()).pop("_name_or_path"), **d
        )
        self.ff = nn.ModuleList([FeedForward(config) for _ in range(config.shared_layers)])
        self.heads = nn.ModuleDict(
            {
                target: NewsuseNegativityClassifierHead(config, target)
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
        logits = {target: head(hidden) for target, head in self.heads.items()}

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
        logits: Mapping[str, torch.Tensor],
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
        loss = {}
        for target in logits:
            num_classes = self.heads[target].num_classes
            label = torch.as_tensor(labels[target])
            if label.ndim == 0:
                label = label[None, ...]
            extended_labels = extend_ordinal_labels(label, num_classes)
            loss[target] = ordinal_loss(logits[target], extended_labels, num_classes)
        return sum(loss.values()) / len(loss)

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
AutoConfig.register("newsuse-negativity-classifier", NewsuseNegativityClassifierConfig)
AutoModel.register(NewsuseNegativityClassifierConfig, NewsuseNegativityClassifier)


@TextMultiClassificationPipeline.compute_scores.register
def _(
    self,  # noqa
    model: NewsuseNegativityClassifier,  # noqa
    logits: torch.Tensor,
) -> PipeOutputT:
    """Model-specific logits post-processing for multi-target classification."""
    return ordinal_probs(logits).squeeze()
