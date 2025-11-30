from collections.abc import Mapping
from typing import Any

import arviz as az
import bambi as bmb
import numpy as np
import pymc as pm
from pymc.backends.base import MultiTrace

__all__ = ("advi_trace_to_inference", "ConvergenceCallback", "make_priors")


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


class ConvergenceCallback:
    def __init__(
        self,
        k: int = 10,
        every: int = 100,
        tolerance: float = 5e-4,
        logging: bool = True,
    ) -> None:
        self.history = np.full((k,), np.nan)
        self.every = every
        self.tolerance = tolerance
        self.logging = logging

    @property
    def k(self) -> int:
        return self.history.shape[0]

    @property
    def elbo_mean(self) -> float:
        return np.nanmean(self.history).item()

    @property
    def elbo_std(self) -> float:
        return np.nanstd(self.history).item()

    def __call__(self, _, hist: np.ndarray, i: int) -> None:
        if (i + 1) % self.every != 0:
            return
        self.history = np.roll(self.history, -1)
        self.history[-1] = hist[-self.every :].mean()
        if i < self.every * 2:
            return
        elbo_mean, elbo_std = self.elbo_mean, self.elbo_std
        elbo_cv = elbo_std / elbo_mean
        if self.logging:
            msg = f"Iteration {i + 1}: ELBO={elbo_mean:.4f}, CV={elbo_cv:.4f}"
            print(msg)
        if not np.isnan(elbo_cv) and elbo_cv < self.tolerance:
            msg = (
                f"Converged at iteration {i + 1} with ELBO = {elbo_mean:.3f}; "
                f" ELBO-CV = {elbo_cv:.6f} < {self.tolerance:.6} (tolerance)."
            )
            raise StopIteration(msg)


def make_priors(priors: Mapping[str, Mapping[str, Any]]) -> Mapping[str, bmb.Prior]:
    def make_prior(opts: Mapping[str, Any]) -> bmb.Prior:
        opts = opts.copy()
        name = opts.pop("name")
        opts = {k: make_prior(v) if isinstance(v, Mapping) else v for k, v in opts.items()}
        return bmb.Prior(name, **opts)

    return {k: make_prior(v) for k, v in priors.items()}
