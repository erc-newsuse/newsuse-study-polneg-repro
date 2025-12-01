import typing
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import arviz as az
import brmspy
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import xarray as xr
from brmspy.helpers import singleton
from brmspy.helpers.conversion import py_to_r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

__all__ = (
    "make_data_coords",
    "brms_posterior",
    "brms_observed_data",
    "brms_posterior_epred",
    "brms_posterior_predictive",
    "brms_log_likelihood",
)

DataCoordsT = Mapping[str, np.ndarray | tuple[str, np.ndarray]]
_OBS_DIM = "__obs__"


def pydf(df: ro.RObject) -> object:
    df = ro.r["as.data.frame"](df)
    with localconverter(ro.default_converter + pandas2ri.converter):
        return pandas2ri.rpy2py(df)


def make_data_coords(df: pd.DataFrame, index_col: Hashable | None = None) -> DataCoordsT:
    if any(df.index.names):
        df = df.reset_index()
    if index_col is None:
        index_col = df.columns[0]
    assert df[index_col].is_unique, f"index column '{index_col}' must be unique!"
    cols = [index_col, *(col for col in df.columns if col != index_col)]
    coords = {}
    for col in cols:
        coords[col] = (_OBS_DIM, pd.Index(df[col].to_numpy(), name=col))
    return coords


def as_index(ds: xr.Dataset | pd.DataFrame, cols: Sequence[str]) -> pd.MultiIndex:
    if isinstance(ds, xr.Dataset):
        ds = ds.coords.to_dataset().to_dataframe()
    if any(ds.index.names):
        ds = ds.reset_index()
    return pd.MultiIndex.from_frame(ds[cols])


def brms_observed_data(
    brms: brmspy.FitResult,
    response_name: str = "y",
    coords: DataCoordsT | None = None,
    dtype: type | None = None,
) -> brmspy.FitResult:
    _base = singleton._get_base()
    # Extract data from the fit object: fit$data
    if _base:
        r_data = _base.getElement(brms.r, "data")
    else:
        errmsg = "Base uninitialized (Should not happen if _get_brms was done)!"
        raise Exception(errmsg)

    with localconverter(ro.default_converter + pandas2ri.converter):
        df_data = pandas2ri.rpy2py(r_data)

    y = df_data[response_name].to_numpy()
    if dtype is not None:
        y = y.astype(dtype)

    if coords is None:
        coords = make_data_coords(pd.DataFrame({}, index=np.arange(len(y))))

    observed_data = xr.Dataset(
        {response_name: (_OBS_DIM, y)},
        coords=coords,
    )
    brms.idata.add_groups(observed_data=observed_data)
    return brms


def brms_posterior(brms: brmspy.FitResult, **kwargs: Any) -> brmspy.FitResult:
    # Safely get the as_draws_df function
    as_draws_df = typing.cast(typing.Callable, ro.r("posterior::as_draws_df"))
    draws_r = as_draws_df(brms.r, **kwargs)

    with localconverter(ro.default_converter + pandas2ri.converter):
        df = pandas2ri.rpy2py(draws_r)

    # Handle Chain/Draw Indexing
    chain_col = ".chain" if ".chain" in df.columns else "chain"
    draw_col = ".draw" if ".draw" in df.columns else "draw"

    # Create a clean 0..N index for draws within each chain
    df["draw_idx"] = df.groupby(chain_col)[draw_col].transform(lambda x: np.arange(len(x)))

    posterior = {}
    for col in df.columns:
        if col not in [chain_col, draw_col, ".iteration", "draw_idx"]:
            # Pivot ensures we respect the chain/draw structure explicitly
            mat = df.pivot(index="draw_idx", columns=chain_col, values=col)
            samples = mat.to_numpy().T
            posterior[col] = samples

    posterior = az.from_dict(posterior=posterior).posterior
    brms.idata.add_groups(posterior=posterior)
    return brms


def brms_posterior_epred(
    brms: brmspy.FitResult,
    response_name: str = "y",
    newdata: pd.DataFrame | None = None,
    *,
    re_formula: str | ro.Formula | None = None,
    support: np.ndarray | None = None,
    **kwargs: Any,
) -> brmspy.FitResult:
    if isinstance(re_formula, str):
        re_formula = ro.Formula(re_formula)
    if re_formula is not None:
        kwargs["re_formula"] = re_formula
    if newdata is not None:
        kwargs["newdata"] = py_to_r(newdata)
    posterior_epred = ro.r("brms::posterior_epred")
    epred = posterior_epred(brms.r, **kwargs)
    epred = np.asarray(epred)
    # Handle observed data subsetting
    observed = brms.idata.observed_data
    if newdata is not None:
        coords_cols = list(observed.coords)
        observed_index = as_index(observed, coords_cols)
        newdata_index = as_index(newdata, coords_cols)
        mask = observed_index.isin(newdata_index)
        observed = observed.sel(**{_OBS_DIM: mask})
    # Build dims and coordinates
    if brms.idata.posterior.sizes["chain"] == 1:
        epred = np.expand_dims(epred, axis=0)
    coords = {
        "chain": ("chain", brms.idata.posterior.coords["chain"].values),
        "draw": ("draw", brms.idata.posterior.coords["draw"][: epred.shape[1]].values),
        **{k: (_OBS_DIM, v.values) for k, v in observed.coords.items()},
    }
    posterior_dims = ["chain", "draw"]
    dims = [*posterior_dims, _OBS_DIM]
    # Handle categorical support
    if epred.ndim > len(dims):
        if support is None:
            support = np.unique(observed[response_name])
        support = np.asarray(support)
        dims.append(response_name)
        coords[response_name] = (response_name, np.asarray(support))
    # Build dataset
    ds = xr.Dataset(
        {"epred": (dims, epred)},
        coords=coords,
    )
    brms.idata.add_groups(posterior_epred=ds.sortby(_OBS_DIM))
    return brms


def brms_posterior_predictive(
    brms: brmspy.FitResult,
    response_name: str = "y",
    newdata: pd.DataFrame | None = None,
    *,
    re_formula: str | ro.Formula | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    **kwargs: Any,
) -> brmspy.FitResult:
    if isinstance(re_formula, str):
        re_formula = ro.Formula(re_formula)
    if re_formula is not None:
        kwargs["re_formula"] = re_formula
    if newdata is not None:
        kwargs["newdata"] = py_to_r(newdata)
    posterior_predictive = ro.r("brms::posterior_predict")
    ppred = posterior_predictive(brms.r, **kwargs)
    ppred = np.asarray(ppred)
    if transform is not None:
        ppred = transform(ppred)
    # Handle observed data subsetting
    observed = brms.idata.observed_data
    if newdata is not None:
        coords_cols = list(observed.coords)
        observed_index = as_index(observed, coords_cols)
        newdata_index = as_index(newdata, coords_cols)
        mask = observed_index.isin(newdata_index)
        observed = observed.sel(**{_OBS_DIM: mask})
    # Build dims and coordinates
    if brms.idata.posterior.sizes["chain"] == 1:
        ppred = np.expand_dims(ppred, axis=0)
    coords = {
        "chain": ("chain", brms.idata.posterior.coords["chain"].values),
        "draw": ("draw", brms.idata.posterior.coords["draw"][: ppred.shape[1]].values),
        **{k: (_OBS_DIM, v.values) for k, v in observed.coords.items()},
    }
    dims = ["chain", "draw", _OBS_DIM]
    # Build dataset
    ds = xr.Dataset(
        {response_name: (dims, ppred)},
        coords=coords,
    )
    brms.idata.add_groups(posterior_predictive=ds.sortby(_OBS_DIM))
    return brms


def brms_log_likelihood(
    brms: brmspy.FitResult,
    response_name: str,
    newdata: pd.DataFrame | None = None,
    *,
    re_formula: str | ro.Formula | None = None,
    pointwise: bool = True,
    **kwargs: Any,
) -> brmspy.FitResult:
    if not pointwise:
        errmsg = "'pointwise=False' is not supported"
        raise NotImplementedError(errmsg)
    if isinstance(re_formula, str):
        re_formula = ro.Formula(re_formula)
    if re_formula is not None:
        kwargs["re_formula"] = re_formula
    if newdata is not None:
        kwargs["newdata"] = py_to_r(newdata)
    log_lik_func = ro.r("brms::log_lik")
    log_lik = log_lik_func(brms.r, **kwargs)
    log_lik = np.asarray(log_lik)
    # Handle observed data subsetting
    observed = brms.idata.observed_data
    if newdata is not None:
        coords_cols = list(observed.coords)
        observed_index = as_index(observed, coords_cols)
        newdata_index = as_index(newdata, coords_cols)
        mask = observed_index.isin(newdata_index)
        observed = observed.sel(**{_OBS_DIM: mask})
    # Build dims and coordinates
    if brms.idata.posterior.sizes["chain"] == 1:
        log_lik = np.expand_dims(log_lik, axis=0)
    coords = {
        "chain": ("chain", brms.idata.posterior.coords["chain"].values),
        "draw": ("draw", brms.idata.posterior.coords["draw"][: log_lik.shape[1]].values),
        **{k: (_OBS_DIM, v.values) for k, v in observed.coords.items()},
    }
    dims = ["chain", "draw", _OBS_DIM]
    # Build dataset
    ds = xr.Dataset({response_name: (dims, log_lik)}, coords=coords)
    brms.idata.add_groups(log_likelihood=ds.sortby(_OBS_DIM))
    return brms
