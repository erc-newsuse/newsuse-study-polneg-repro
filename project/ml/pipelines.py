from functools import singledispatchmethod, wraps
from typing import Any, ClassVar

import torch
import transformers
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.pipelines import PIPELINE_REGISTRY, Pipeline, TextClassificationPipeline
from transformers.utils import ModelOutput

__all__ = ("pipeline",)

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
        model: PreTrainedModel,
        logits: torch.Tensor,
    ) -> PipeOutputT:
        """Model-specific post-processing for multi-target classification."""
        return F.softmax(logits, dim=-1).squeeze()


PIPELINE_REGISTRY.register_pipeline(
    task=TextMultiClassificationPipeline.task,
    pipeline_class=TextMultiClassificationPipeline,
)
