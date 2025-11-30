from typing import Any

import arviz as az
import bambi as bmb
import numpy as np
import pymc as pm
from pymc.backends.base import MultiTrace

__all__ = ("advi_trace_to_inference",)


def advi_trace_to_inference(
    trace: MultiTrace,
    model: bmb.Model | None = None,
    **kwargs: Any,
) -> az.InferenceData:
    P = trace["p"]
    P = np.concatenate([np.zeros((*P.shape[:-1], 1)), P], axis=-1)
    P = np.concatenate([P, np.ones((*P.shape[:-1], 1))], axis=-1)
    P = np.diff(P, axis=-1)
    backend = trace.__dict__["_straces"][0]
    backend.var_shapes["p"] = P.shape[1:]
    backend.samples["p"] = P
    return pm.to_inference_data(trace, model=model.backend.model, **kwargs)
