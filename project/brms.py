"""Routines for handling Python-level access to `brms` package for R."""
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import arviz as az
import pandas as pd
from brmspy import prior
from brmspy.helpers import conversion
from brmspy.helpers.priors import _build_priors
from brmspy.runtime._state import get_brms, get_cmdstanr, get_rstan
from brmspy.types import FitResult, PriorSpec
from rpy2 import robjects as ro
from rpy2.rinterface_lib import openrlib

__all__ = ("make_prior", "build_priors", "df_to_r", "brm")


def df_to_r(data: pd.DataFrame) -> ro.DataFrame:
    """Convert a pandas DataFrame to an R DataFrame."""
    data = data.copy().convert_dtypes(dtype_backend="numpy_nullable")
    cat = {}
    for col in data.select_dtypes(["category"]):
        if pd.api.types.is_string_dtype(data[col].cat.categories):
            rtype = ro.StrVector
        elif pd.api.types.is_integer_dtype(data[col].cat.categories):
            rtype = ro.IntVector
        else:
            rtype = ro.FloatVector
        cat[col] = {
            "levels": rtype(data[col].cat.categories.tolist()),
            "ordered": data[col].cat.ordered,
        }
        data[col] = data[col].astype(data[col].cat.categories.dtype)

    rdf = conversion.py_to_r(data)
    for col, info in cat.items():
        idx = data.columns.tolist().index(col)
        rcol = ro.r["[["](rdf, col)
        rfac = ro.FactorVector(rcol, levels=info["levels"], ordered=info["ordered"])
        rdf[idx] = rfac  # type: ignore
    return rdf


def make_prior(spec: str | Mapping[str, Any]) -> PriorSpec:
    """Create a prior specification for use in `brms` models."""
    if isinstance(spec, str):
        return ro.r(f"brms::prior({spec})")
    spec = dict(spec)
    if (field := "class") in spec:
        class_ = spec.pop(field)
        spec = {**spec, f"{field}_": class_}
    return prior(**spec)


def build_priors(specs: Iterable[str | Mapping[str, Any]]) -> ro.DataFrame:
    """Build a sequence of prior specifications for use in `brms` models."""
    if all(isinstance(spec, str) for spec in specs):
        priors = [make_prior(spec) for spec in specs]
        return ro.r["c"](*priors)
    return _build_priors([make_prior(spec) for spec in specs])


def brm(
    formula: str,
    data: pd.DataFrame,
    *,
    prior: Iterable[Mapping[str, Any]] | None = None,
    backend: Literal["rstan", "cmdstanr"] = "cmdstanr",
    **kwargs: Any,
) -> FitResult:
    """Fit a Bayesian regression model using `brms::brm`."""
    # Handle initilization of brms and STAN backends
    brms = get_brms()
    if backend == "rstan":
        _backend = get_rstan()
    elif backend == "cmdstanr":
        _backend = get_cmdstanr()
    else:
        errmsg = f"unsupported backend '{backend}'"
        raise ValueError(errmsg)
    if _backend is None:
        errmsg = f"'{backend}' backend is not available"
        raise RuntimeError(errmsg)

    # Handle model fit arguments
    formula_r = ro.Formula(re.sub(r"\s+", " ", formula, flags=re.MULTILINE))
    prior_r = build_priors(prior) if prior is not None else ro.NULL
    data_r = df_to_r(data)
    brm_kwargs = {
        "formula": formula_r,
        "data": data_r,
        "prior": prior_r,
        "backend": backend,
        **kwargs,
    }

    # Call brms::brm
    with openrlib.rlock:
        fit = brms.brm(**brm_kwargs)
    idata = az.InferenceData()
    return FitResult(fit, idata)
