import typing
from collections.abc import Callable, Hashable, Mapping, Sequence
from itertools import product
from typing import Any, ClassVar

import arviz as az
import brmspy
import numpy as np
import pandas as pd
import xarray as xr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from project.rutils import df_to_r

__all__ = (
    "make_data_coords",
    "brms_posterior",
    "brms_observed_data",
    "brms_posterior_epred",
    "brms_posterior_predictive",
    "brms_log_likelihood",
    "set_xindex",
    "waic_metrics",
    "StatsAccessor",
)

DataCoordsT = Mapping[str, np.ndarray | tuple[str, np.ndarray]]
FormulaT = ro.Formula | str
OBS_DIM = "__obs__"
GROUP_DIM = "__group__"


def waic_metrics(waic: az.ELPDData, null_elpd: float) -> dict[str, Any]:
    """Compute out-of-sample predictive performance metrics from WAIC results."""
    mean_elpd = waic.elpd_waic / waic.n_data_points
    mean_elpd_se = waic.se / np.sqrt(waic.n_data_points)
    z = (mean_elpd - null_elpd) / mean_elpd_se
    pseudo_r2 = 1 - (mean_elpd / null_elpd)
    return pd.Series(
        {
            "mean_elpd": mean_elpd,
            "mean_elpd_se": mean_elpd_se,
            "pseudo_r2": pseudo_r2,
            "z": z,
        }
    )


def make_data_coords(df: pd.DataFrame, index_col: Hashable | None = None) -> DataCoordsT:
    """Make coordinates for observed data.

    Coordiantes are returned as a mapping from column names to tuples of
    (dimension name, values).
    """
    if any(df.index.names):
        df = df.reset_index()
    if index_col is None:
        index_col = df.columns[0]
    assert df[index_col].is_unique, f"index column '{index_col}' must be unique!"
    cols = [index_col, *(col for col in df.columns if col != index_col)]
    coords = {}
    for col in cols:
        coords[col] = (OBS_DIM, pd.Index(df[col].to_numpy(), name=col))
    return coords


def set_xindex(idata: az.InferenceData, xindex: Sequence[Hashable]) -> az.InferenceData:
    """Set xindex for all groups in InferenceData.

    The xindex is used by ArviZ for plotting and indexing. The function is a convenience
    wrapper around xarray's `set_xindex` method.

    Parameters
    ----------
    idata
        An ArviZ InferenceData object.
    xindex
        A sequence of coordinate names to set as xindex.
    """
    for group in idata.groups():
        ds = getattr(idata, group)
        xs = [col for col in xindex if col in ds.coords]
        if not xs:
            continue
        ds = ds.set_xindex(xs)
        setattr(idata, group, ds)
    return idata


def brms_observed_data(
    model: brmspy.FitResult,
    response_name: Hashable = "y",
    data: pd.DataFrame | None = None,
    coords: Sequence[Hashable] | str | None = None,
    dtype: type | None = None,
    **kwargs: Any,
) -> brmspy.FitResult:
    """Prepare observed data for the model.

    Parameters
    ----------
    model
        A brmspy.FitResult object containing the fitted model.
    response_name
        The name of the response variable in the data.
    data
        A pandas DataFrame containing the observed data.
        If `None`, uses the data from the model.
    coords
        A sequence of coordinate names to use for the observed data.
        If `None`, uses all columns except the response variable.
    dtype
        The data type to which the response variable should be cast.
    **kwargs
        Additional keyword arguments to pass to the `make_data_coords` function.
    """
    if data is None:
        r_data = ro.r["[["](model.r, "data")
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = pandas2ri.rpy2py(r_data)

    response_cols = [
        c for c in data.columns if c == response_name or c.startswith(f"{response_name}_")
    ]

    data = pd.DataFrame(data)
    if not coords:
        coords = [c for c in data if c not in response_cols]

    data_y = {}
    for col in response_cols:
        y = data.pop(col).to_numpy()
        if dtype is not None:
            y = y.astype(dtype)
        data_y[col] = (OBS_DIM, y)

    data_coords = make_data_coords(data, **kwargs)
    observed_data = xr.Dataset(data_y, coords=data_coords).sortby(OBS_DIM)
    model.idata.add_groups(observed_data=observed_data)
    return model


def brms_posterior(model: brmspy.FitResult, **kwargs: Any) -> brmspy.FitResult:
    """Compute posterior distributions.

    Parameters
    ----------
    model
        A brmspy.FitResult object containing the fitted model.
    **kwargs
        Additional keyword arguments to pass to the `posterior::as_draws_df` function.
    """
    # Safely get the as_draws_df function
    as_draws_df = typing.cast(typing.Callable, ro.r("posterior::as_draws_df"))
    draws_r = as_draws_df(model.r, **kwargs)

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
    model.idata.add_groups(posterior=posterior)
    return model


def brms_posterior_epred(
    model: brmspy.FitResult,
    data: pd.DataFrame | None = None,
    *,
    re_formula: FormulaT | None = None,
    support: np.ndarray | None = None,
    **kwargs: Any,
) -> brmspy.FitResult:
    """Compute posterior expected values (epred) for new data.

    Parameters
    ----------
    model
        A brmspy.FitResult object containing the fitted model and posterior samples.
    data
        A pandas DataFrame containing the new data for which to compute epred.
        If `None`, uses the observed data from the model.
    re_formula
        A random effects formula to specify which random effects
        to include in the epred computation. If `None`, includes all random effects.
    support
        An array of possible response values (for categorical outcomes).
        If `None`, uses the unique values from the observed data.
    **kwargs
        Additional keyword arguments to pass to the `brms::posterior_epred` function.
    """
    # Handle observed data
    observed = model.idata.observed_data
    response_name = list(observed.data_vars)[0]
    if support is None:
        support = np.unique(observed[response_name])
    if data is None:
        data = pd.DataFrame(
            model.idata.observed_data.to_pandas()
            .reset_index(drop=True)
            .drop(columns=response_name)
        )
    else:
        observed = (
            data.reset_index(names=GROUP_DIM)
            .set_index(GROUP_DIM)
            .pipe(xr.Dataset.from_dataframe)
        )
        observed = observed.assign_coords(observed.data_vars)

    # Build dims and coordinates
    ndraws = model.idata.posterior.sizes["draw"]
    nchains = model.idata.posterior.sizes["chain"]
    coords = {
        "chain": ("chain", model.idata.posterior.coords["chain"].values),
        "draw": ("draw", model.idata.posterior.coords["draw"][:ndraws].values),
        **{k: (GROUP_DIM, v.values) for k, v in observed.coords.items()},
    }
    posterior_dims = ["chain", "draw"]
    dims = [*posterior_dims, GROUP_DIM]

    # Handle formula
    if re_formula is None:
        re_formula = ro.NULL  # type: ignore
    elif isinstance(re_formula, str):
        re_formula = ro.Formula(re_formula)

    # Prepare shared kwargs
    kwargs.update(newdata=df_to_r(data), draws=ndraws, re_formula=re_formula)

    # Compute
    epreds = {}
    posterior_epred = ro.r("brms::posterior_epred")
    epred = np.asarray(posterior_epred(model.r, **kwargs))

    # Reshape from (nchains*ndraws, nobs, ...) to (nchains, ndraws, nobs, ...)
    nobs = epred.shape[1]
    extra_dims = epred.shape[2:] if epred.ndim > 2 else ()
    epred = epred.reshape(nchains, ndraws, nobs, *extra_dims)

    # Handle categorical support
    if epred.ndim > len(dims):
        support = np.asarray(support)
        dims.append(response_name)
        coords[response_name] = (response_name, np.asarray(support))
    epreds[response_name] = (dims, epred)

    # Build dataset
    ds = xr.Dataset(epreds, coords=coords).sortby(GROUP_DIM)
    model.idata.add_groups(posterior_epred=ds)
    return model


def brms_posterior_predictive(
    model: brmspy.FitResult,
    data: pd.DataFrame | None = None,
    *,
    re_formula: str | ro.Formula | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    group_name: str = "posterior_predictive",
    **kwargs: Any,
) -> brmspy.FitResult:
    if isinstance(re_formula, str):
        re_formula = ro.Formula(re_formula)
    if re_formula is not None:
        kwargs["re_formula"] = re_formula

    observed = model.idata.observed_data
    response_name = list(observed.data_vars)[0]

    if data is None:
        data = pd.DataFrame(
            model.idata.observed_data.to_pandas()
            .reset_index(drop=True)
            .drop(columns=response_name)
        )
    else:
        observed = (
            data.reset_index(names=GROUP_DIM)
            .set_index(GROUP_DIM)
            .pipe(xr.Dataset.from_dataframe)
        )
        observed = observed.assign_coords(observed.data_vars)

    # Compute
    kwargs["newdata"] = df_to_r(data)
    posterior_predictive = ro.r("brms::posterior_predict")
    ppred = posterior_predictive(model.r, **kwargs)
    ppred = np.asarray(ppred)
    if transform is not None:
        ppred = transform(ppred)

    # Reshape from (nchains*ndraws, nobs) to (nchains, ndraws, nobs)
    nchains = 1  # brms returns combined chains in posterior_predict
    ndraws, nobs = ppred.shape
    ppred = ppred.reshape(nchains, ndraws, nobs)

    # Build dims and coordinates
    coords = {
        "chain": ("chain", np.arange(nchains)),
        "draw": ("draw", np.arange(ndraws)),
        **{k: (OBS_DIM, v.values) for k, v in observed.coords.items()},
    }
    dims = ["chain", "draw", OBS_DIM]

    # Build dataset
    ds = xr.Dataset({response_name: (dims, ppred)}, coords=coords).sortby(OBS_DIM)
    model.idata.add_groups(**{group_name: ds})
    return model


def brms_log_likelihood(
    model: brmspy.FitResult,
    *,
    re_formula: str | ro.Formula | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
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

    # Handle observed data
    observed = model.idata.observed_data
    response_name = list(observed.data_vars)[0]
    newdata = pd.DataFrame(model.idata.observed_data.to_pandas().reset_index(drop=True))
    if transform is not None:
        newdata[response_name] = transform(newdata[response_name].to_numpy())

    # Compute
    kwargs["newdata"] = df_to_r(newdata)
    log_lik_func = ro.r("brms::log_lik")
    log_lik = log_lik_func(model.r, **kwargs)
    log_lik = np.asarray(log_lik)

    # Reshape from (nchains*ndraws, nobs) to (nchains, ndraws, nobs)
    nchains = 1
    ndraws, nobs = log_lik.shape
    log_lik = log_lik.reshape(nchains, ndraws, nobs)

    # Build dims and coordinates
    coords = {
        "chain": ("chain", np.arange(nchains)),
        "draw": ("draw", np.arange(ndraws)),
        **{k: (OBS_DIM, v.values) for k, v in observed.coords.items()},
    }
    dims = ["chain", "draw", OBS_DIM]

    # Build dataset
    ds = xr.Dataset({response_name: (dims, log_lik)}, coords=coords).sortby(OBS_DIM)
    model.idata.add_groups(log_likelihood=ds)
    return model


# XArray Accessor -------------------------------------------------------------------


class StatsAccessor:
    """XArray accessor for statistical summaries."""  # noqa: E501  #

    alpha: ClassVar[float] = 0.05
    quantiles: ClassVar[list[float]]

    def __init_subclass__(cls) -> None:
        cls.quantiles = [cls.alpha / 2, 0.5, 1 - cls.alpha / 2]

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def quantile(self, q=None, dim="sample", **kwargs) -> pd.DataFrame:
        if q is None:
            q = self.quantiles
        q0, q1 = q[0], q[-1]
        return (
            self._ds.quantile(q, dim, **kwargs)
            .to_pandas()
            .rename(index={q0: "lb", 0.5: "median", q1: "ub"})
            .T
        )

    def diff(self, *, marginalize: str | None = None, **kwargs) -> pd.DataFrame:
        sel1 = {}
        sel2 = {}
        for name, values in kwargs.items():
            v1, v2 = values
            sel1[name] = v1
            sel2[name] = v2
        ds1 = self._ds.sel(**sel1).drop_vars(list(sel1))
        ds2 = self._ds.sel(**sel2).drop_vars(list(sel2))
        if marginalize:
            ds1 = ds1.mean(marginalize)
            ds2 = ds2.mean(marginalize)
        return ds1 - ds2

    def marginalize(self, *dims: str) -> xr.DataArray | xr.Dataset:
        coords = {dim: np.unique(self._ds.coords[dim].values) for dim in dims}
        n = 0
        carry = None
        for vals in product(*coords.values()):
            sel = dict(zip(coords.keys(), vals, strict=True))
            try:
                ds = self._ds.sel(**sel).drop_vars(list(sel))
                carry = ds if carry is None else carry + ds
                n += 1
            except KeyError:
                continue
        return carry / n  # type: ignore

    def average_by(
        self, *dims: str, obs_dim: str = OBS_DIM, group_dim: str = GROUP_DIM
    ) -> xr.DataArray | xr.Dataset:
        groups = []
        coords = {dim: np.unique(self._ds.coords[dim].values) for dim in dims}
        for vals in product(*coords.values()):
            sel = dict(zip(coords.keys(), vals, strict=True))
            try:
                ds = self._ds.sel(**sel).mean(obs_dim)
                groups.append(ds)
            except KeyError:
                continue
        return xr.concat(groups, dim=group_dim)
