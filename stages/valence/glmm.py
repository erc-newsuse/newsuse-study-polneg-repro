# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import xarray as xr
from newsuse.data import DataFrame
from omegaconf import OmegaConf

from project import config, paths
from project.bayes import store_model_metadata

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)

# %% ---------------------------------------------------------------------------------

target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target name (event): ").strip() or "event"
opts = config.glmm.valence.targets[target]

dirpath = paths.glmm / "valence"
dirpath.mkdir(parents=True, exist_ok=True)

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.final)
    .assign(
        country=lambda df: pd.Categorical(
            df["country"], categories=[*config.categorical.country]
        ),
        outlet=lambda df: pd.Categorical(df["outlet"]),
        **{
            target: lambda df: pd.Categorical(
                df[target],
                categories=[*config.categorical[target]],
                ordered=True,
            )
        },
    )[["key", target, *opts.predictors.fixed, *opts.predictors.groups]]
    .convert_dtypes(dtype_backend="numpy_nullable")
)

# %% ---------------------------------------------------------------------------------

if sample := opts.get("sample"):
    print(f"Subsampling data for model fitting to {sample.n} observations...")
    model_data = data.sample(**sample, ignore_index=True)
else:
    model_data = data


# %% ---------------------------------------------------------------------------------

print(f"Building GLMM for '{target}' using 'bambi'...")
model = bmb.Model(
    formula=opts.formula.format(target=target).strip().replace("\n", " "),
    data=model_data,
    **opts.model,
)

# %% ---------------------------------------------------------------------------------

print(f"Fitting GLMM for '{target}' using '{opts.fit.inference_method}'...")
idata = model.fit(**OmegaConf.to_object(opts.fit))

# %% ---------------------------------------------------------------------------------

print("Prepare observed data group in inference data...")
if (group := "observed_data") in idata.groups():
    del idata["observed_data"]

observed = xr.Dataset(
    {target: ("__obs__", model.data[target].to_numpy())},
    coords={n: ("__obs__", c.to_numpy()) for n, c in model.data.items() if n != target},
)
idata.add_groups(**{group: observed})

# %% ---------------------------------------------------------------------------------

print("Prepare posterior predictive group in inference data...")
if (group := "posterior_predictive") in idata.groups():
    del idata["posterior_predictive"]

kwargs = OmegaConf.to_object(opts.ppd)
ppd = (
    model.predict(
        idata.isel(draw=slice(0, kwargs.pop("draws"))),
        data=model.data,
        inplace=False,
        **kwargs,
    )
    .posterior_predictive.drop_vars("__obs__")
    .assign_coords(
        {n: ("__obs__", c.to_numpy()) for n, c in model.data.items() if n != target}
    )
)
ppd[target].values -= 1

idata.add_groups(**{group: ppd})

# %% ---------------------------------------------------------------------------------

print("Prepare posterior expectations group in inference data...")
if (group := "posterior_epred") in idata.groups():
    del idata["posterior_epred"]

grid = model.data[opts.predictors.fixed].drop_duplicates(ignore_index=True)
grid = (
    # Make dummy values for group effects
    # to allow independent sampling of group-level effects
    # for proper marginalization
    grid.loc[grid.index.repeat(opts.epred.samples_per_simple_effect)]
    .groupby(level=0)
    .apply(
        lambda df: df.assign(
            **{
                n: str(df.name) + "_" + np.arange(len(df)).astype(str)
                for n in opts.predictors.groups
            }
        )
    )
    .reset_index(drop=True)
)

epred = (
    model.predict(idata, data=grid, inplace=False, **opts.epred.predict)
    .posterior["p"]
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in grid.items()})
    .rename({f"{target}_dim": target})
    .groupby([*opts.predictors.fixed])
    .mean()
    .stack(__obs__=tuple(opts.predictors.fixed))
    .transpose("chain", "draw", "__obs__", target)
    .reset_index("__obs__")
)

idata.add_groups(**{group: epred.to_dataset()})

# %% ---------------------------------------------------------------------------------

print("Prepare log-likelihood group in inference data...")
if (group := "log_likelihood") in idata.groups():
    del idata["log_likelihood"]

epred = (
    model.predict(
        idata.isel(draw=slice(0, opts.ppd.draws)),
        data=model.data,
        inplace=False,
        **opts.epred.predict,
    )
    .posterior["p"]
    .rename({f"{target}_dim": target})
)

# Use observed category codes from model data
obs_codes = model.data[target].cat.codes.to_numpy()

# For cumulative family: p contains P(Y <= k), need P(Y = k)
# P(Y = 0) = P(Y <= 0)
# P(Y = k) = P(Y <= k) - P(Y <= k-1) for k > 0
cumprobs = epred.values
cat_probs = np.diff(cumprobs, axis=-1, prepend=0)
cat_probs = np.concatenate([cumprobs[..., :1], cat_probs[..., 1:]], axis=-1)
cat_probs = xr.DataArray(
    cat_probs,
    dims=epred.dims,
    coords=epred.coords,
)

# Log-likelihood: log(p_observed)
obs_probs = cat_probs.isel({target: xr.DataArray(obs_codes, dims="__obs__")})
loglik = (
    np.log(np.clip(obs_probs, 1e-10, 1.0))
    .drop_vars(target)  # drop the scalar coordinate to avoid name conflict
    .assign_coords(
        {n: ("__obs__", c.to_numpy()) for n, c in model.data.items() if n != target}
    )
    .reset_index("__obs__")
)

idata.add_groups(**{group: loglik.to_dataset(name=target)})

# %% ---------------------------------------------------------------------------------

print("Storing model metadata in inference data...")
store_model_metadata(
    idata,
    model,
    formula=opts.formula.format(target=target),
    family=opts.model.get("family", "categorical"),
    target=target,
)

# %% ---------------------------------------------------------------------------------

print("Saving model inference data as NetCDF file...")
idata.to_netcdf(dirpath / f"{target}.nc")

# %% ---------------------------------------------------------------------------------
