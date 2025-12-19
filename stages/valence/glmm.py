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
                df[target], categories=[*config.categorical[target]]
            )
        },
    )[["key", target, *opts.predictors.fixed, *opts.predictors.groups]]
    .convert_dtypes(dtype_backend="numpy_nullable")
)

# %% ---------------------------------------------------------------------------------

if sample := opts.get("sample"):
    model_data = data.sample(**sample, ignore_index=True)
else:
    model_data = data


# %% ---------------------------------------------------------------------------------

print(f"Building GLMM for '{target}' using 'bambi'...")
kwargs = {
    **opts.model,
    "formula": opts.model.formula.format(target=target).strip().replace("\n", " "),
}

model = bmb.Model(data=model_data, **kwargs)

# %% ---------------------------------------------------------------------------------

print(f"Fitting GLMM for '{target}' using '{opts.inference_method}'...")
kwargs = {
    "inference_method": (method := opts.inference_method),
    **OmegaConf.to_object(opts.fit[method]),
}
idata = model.fit(**kwargs)

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

if len(model.data) < len(data):
    print("Prepare held out dataset subsample...")
    valid = data.pipe(lambda df: df[~df["key"].isin(model.data["key"])]).sample(
        **opts.validation.sample, ignore_index=True
    )
else:
    valid = data

# %% ---------------------------------------------------------------------------------

print("Prepare posterior predictive group in inference data...")
if (group := "posterior_predictive") in idata.groups():
    del idata["posterior_predictive"]

kwargs = OmegaConf.to_object(opts.ppd)
ppd = (
    model.predict(
        idata.isel(draw=slice(0, kwargs.pop("draws"))), data=valid, inplace=False, **kwargs
    )
    .posterior_predictive.drop_vars("__obs__")
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in valid.items() if n != target})
)
ppd[target].values -= 1

idata.add_groups(**{group: ppd})

# %% ---------------------------------------------------------------------------------

print("Prepare posterior expectations group in inference data...")
if (group := "posterior_epred") in idata.groups():
    del idata["posterior_epred"]

grid = model.data[opts.predictors.fixed].drop_duplicates(ignore_index=True)
epred = (
    model.predict(idata, data=grid, inplace=False, **opts.epred)
    .posterior["p"]
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in grid.items()})
    .rename({f"{target}_dim": target})
)

idata.add_groups(**{group: epred.to_dataset()})

# %% ---------------------------------------------------------------------------------

print("Prepare log-likelihood group in inference data...")
if (group := "log_likelihood") in idata.groups():
    del idata["log_likelihood"]

epred = (
    model.predict(
        idata.isel(draw=slice(0, opts.ppd.draws)), data=valid, inplace=False, **opts.epred
    )
    .posterior["p"]
    .rename({f"{target}_dim": target})
)
p = epred.values.reshape(-1, epred.sizes[target])
t = ppd[target].values.flatten() + 1
i = np.arange(len(p))

loglik = np.log((1 - epred).prod(target))
loglik -= np.log(1 - p[i, t]).reshape(loglik.shape)
loglik += np.log(p[i, t]).reshape(loglik.shape)

loglik = loglik.assign_coords(
    {n: ("__obs__", c.to_numpy()) for n, c in valid.items() if n != target}
)

idata.add_groups(**{group: loglik.to_dataset(name=target)})

# %% ---------------------------------------------------------------------------------

print("Saving model inference data as NetCDF file...")
idata.to_netcdf(dirpath / f"{target}.nc")

# %% ---------------------------------------------------------------------------------
