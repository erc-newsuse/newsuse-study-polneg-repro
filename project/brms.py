"""Routines for handling Python-level access to `brms` package for R."""
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
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

__all__ = ("make_prior", "build_priors", "df_to_r", "brm", "brm_large")


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


def brm_large(
    formula: str,
    data: pd.DataFrame,
    *,
    prior: Iterable[Mapping[str, Any]] | None = None,
    backend: Literal["cmdstanr"] = "cmdstanr",
    tmpdir: str | Path | None = None,
    **kwargs: Any,
) -> FitResult:
    """Fit a Bayesian regression model for large datasets.

    This function works around R's 2GB string limit when serializing
    large datasets to JSON by using file-based data transfer:

    1. Create an empty brmsfit with formula, priors, and Stan code
    2. Write Stan data to a JSON file using cmdstanr::write_stan_json
    3. Compile and fit the model with cmdstanr using data_file
    4. Reconstruct the brmsfit object from the cmdstanr output

    Parameters
    ----------
    formula
        Model formula as a string.
    data
        DataFrame with model data.
    prior
        Prior specifications (optional).
    backend
        Only "cmdstanr" is supported for this function.
    tmpdir
        Directory for temporary files. If None, uses system temp directory.
    **kwargs
        Additional arguments passed to cmdstanr sampling method.
        Common options: algorithm, iter, chains, cores, seed, control, refresh.

    Returns
    -------
    FitResult
        A FitResult containing the fitted brmsfit object.

    Notes
    -----
    This approach bypasses the R string limit by writing data directly
    to disk instead of serializing it in memory.
    """
    # ruff: noqa: C901,E501
    if backend != "cmdstanr":
        errmsg = "brm_large only supports 'cmdstanr' backend"
        raise ValueError(errmsg)

    brms = get_brms()
    cmdstanr = get_cmdstanr()
    if cmdstanr is None:
        errmsg = "'cmdstanr' backend is not available"
        raise RuntimeError(errmsg)

    # Prepare formula, priors, and data
    formula_r = ro.Formula(re.sub(r"\s+", " ", formula, flags=re.MULTILINE))
    prior_r = build_priors(prior) if prior is not None else ro.NULL
    data_r = df_to_r(data)

    # Extract algorithm and sampling kwargs
    algorithm = kwargs.pop("algorithm", "sampling")
    seed = kwargs.pop("seed", ro.NULL)

    # Remove brms-specific kwargs that don't apply to cmdstanr
    kwargs.pop("backend", None)

    # Map brms kwargs to cmdstanr kwargs
    # For variational: iter -> iter, output_samples -> draws
    # For sampling: iter -> iter, chains -> chains, cores -> parallel_chains
    iter_val = kwargs.pop("iter", None)
    chains_val = kwargs.pop("chains", None)
    cores_val = kwargs.pop("cores", None)
    control_val = kwargs.pop("control", None)

    # Extract control parameters (adapt_delta, max_treedepth) if provided
    # These are used for MCMC sampling
    adapt_delta = None
    max_treedepth = None
    if control_val is not None and hasattr(control_val, "names") and control_val.names:
        control_names = list(control_val.names)
        if "adapt_delta" in control_names:
            adapt_delta = control_val.rx2("adapt_delta")[0]
        if "max_treedepth" in control_names:
            max_treedepth = control_val.rx2("max_treedepth")[0]

    # Set up temp directory (use unique temp dir by default to avoid conflicts)
    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="brm_large_"))
    else:
        tmpdir = Path(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)

    stan_file = str(tmpdir / "model.stan")
    data_file = str(tmpdir / "data.json")

    with openrlib.rlock:
        # 1. Create empty brmsfit (no sampling)
        print("Creating empty brmsfit object...")
        empty_fit = brms.brm(
            formula=formula_r,
            data=data_r,
            prior=prior_r,
            backend=backend,
            empty=True,
        )

        # 2. Get Stan code and data
        stancode = ro.r("brms::stancode")(empty_fit)
        standata = ro.r("brms::standata")(empty_fit)

        # 3. Write Stan code to file
        ro.r("writeLines")(stancode, stan_file)

        # 4. Write data to JSON file (bypasses R string limit)
        print(f"Writing Stan data to {data_file}...")
        ro.r("cmdstanr::write_stan_json")(standata, data_file)

        # 5. Compile and fit using R code directly (R6 objects don't work well with rpy2)
        print("Compiling Stan model...")
        ro.r(f'mod <- cmdstanr::cmdstan_model("{stan_file}")')

        # Build cmdstanr-specific kwargs based on algorithm
        r_kwargs = []
        if seed != ro.NULL:
            r_kwargs.append(f"seed = {seed}")

        print(f"Fitting model with algorithm='{algorithm}'...")
        if algorithm == "sampling":
            if iter_val:
                r_kwargs.append(f"iter_sampling = {iter_val // 2}")
                r_kwargs.append(f"iter_warmup = {iter_val // 2}")
            if chains_val:
                r_kwargs.append(f"chains = {chains_val}")
            if cores_val:
                r_kwargs.append(f"parallel_chains = {cores_val}")
            if adapt_delta is not None:
                r_kwargs.append(f"adapt_delta = {adapt_delta}")
            if max_treedepth is not None:
                r_kwargs.append(f"max_treedepth = {max_treedepth}")
            r_kwargs_str = ", ".join(r_kwargs)
            fit_cmd = f'cmdstan_fit <- mod$sample(data = "{data_file}", {r_kwargs_str})'
        elif algorithm in ("meanfield", "fullrank"):
            if iter_val:
                r_kwargs.append(f"iter = {iter_val}")
            # Note: threads requires model compiled with stan_threads=TRUE
            r_kwargs_str = ", ".join(r_kwargs)
            fit_cmd = f'cmdstan_fit <- mod$variational(data = "{data_file}", algorithm = "{algorithm}", {r_kwargs_str})'
        elif algorithm == "pathfinder":
            r_kwargs_str = ", ".join(r_kwargs)
            fit_cmd = f'cmdstan_fit <- mod$pathfinder(data = "{data_file}", {r_kwargs_str})'
        elif algorithm == "laplace":
            r_kwargs_str = ", ".join(r_kwargs)
            fit_cmd = f'cmdstan_fit <- mod$laplace(data = "{data_file}", {r_kwargs_str})'
        else:
            errmsg = f"unsupported algorithm '{algorithm}'"
            raise ValueError(errmsg)

        ro.r(fit_cmd)

        # Check if fitting succeeded
        output_files = ro.r("cmdstan_fit$output_files()")
        if len(output_files) == 0 or output_files[0] == "":  # type: ignore
            errmsg = "model fitting failed - no output files generated"
            raise RuntimeError(errmsg)

        # 6. Convert cmdstanr output to stanfit and reconstruct brmsfit
        print("Reconstructing brmsfit object...")
        ro.r(
            "stanfit <- brms::read_csv_as_stanfit(cmdstan_fit$output_files(), model = mod)"
        )

        # Insert stanfit into brmsfit and rename parameters
        ro.globalenv["empty_fit"] = empty_fit
        ro.r("empty_fit$fit <- stanfit")
        fit = ro.r("brms::rename_pars(empty_fit)")

    idata = az.InferenceData()
    return FitResult(fit, idata)
