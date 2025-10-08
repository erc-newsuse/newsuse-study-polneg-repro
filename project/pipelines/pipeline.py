from functools import wraps
from typing import Any

import torch
import transformers

__all__ = ("pipeline",)


@wraps(transformers.pipeline)
def pipeline(task: str, model: Any, *args: Any, **kwargs: Any) -> transformers.Pipeline:
    kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu", **kwargs}
    pipe = transformers.pipeline(task, model, *args, **kwargs)
    if task == "text-classification":
        kwargs = {
            "truncation": True,
            "padding": "max_length",
            "max_length": pipe.tokenizer.model_max_length,
            "tokenizer": pipe.tokenizer,
            **kwargs,
        }
    return transformers.pipeline(task, pipe.model, *args, **kwargs)
