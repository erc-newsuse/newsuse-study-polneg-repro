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
by = os.environ.get("BY")
if by is None:
    by = input("Enter grouping variable: ").strip() or ""
by = by.removeprefix("-")

opts = config.glmm.valence.targets[target]
support = np.asarray([*config.categorical[target]])

dirpath = paths.glmm / "valence"
dirpath.mkdir(parents=True, exist_ok=True)

predictors_fixed = [*opts.predictors.fixed, *([by] if by else [])]
predictors_groups = [*opts.predictors.groups]

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
    )[["key", target, *predictors_fixed, *predictors_groups]]
    .dropna(ignore_index=True)
    .convert_dtypes(dtype_backend="numpy_nullable")
)

# if (col := "month") in data:
#     data[col] = pd.Categorical(data[col])

# %% ---------------------------------------------------------------------------------

if sample := opts.get("sample"):
    print(f"Subsampling data for model fitting to {sample.n} observations...")
    model_data = data.sample(**sample, ignore_index=True)
else:
    model_data = data


# %% ---------------------------------------------------------------------------------

print(f"Building GLMM for '{target}' using 'bambi'...")
if by:
    formula = opts.formula.extended.format(target=target, by=by)
else:
    formula = opts.formula.base.format(target=target)
formula = formula.replace("\n", " ").strip()
print("Model formula:", formula)

model = bmb.Model(
    formula=formula,
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
ppd[target].values += min(support)  # adjust for 0-indexing

idata.add_groups(**{group: ppd})

# %% ---------------------------------------------------------------------------------

print("Prepare posterior expectations group in inference data...")
if (group := "posterior_epred") in idata.groups():
    del idata["posterior_epred"]

grid = model.data[predictors_fixed].drop_duplicates(ignore_index=True)
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
                for n in predictors_groups
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
    .groupby(predictors_fixed)
    .mean()
    .stack(__obs__=tuple(predictors_fixed))
    .transpose("chain", "draw", "__obs__", target)
    .reset_index("__obs__")
)

idata.add_groups(**{group: epred.to_dataset()})

# %% ---------------------------------------------------------------------------------

print("Prepare log-likelihood group in inference data...")
if (group := "log_likelihood") in idata.groups():
    del idata["log_likelihood"]

# Get category probabilities from posterior predictions
# NOTE: For cumulative family with kind="response_params", bambi returns
# category probabilities P(Y=k) directly, NOT cumulative probabilities P(Y<=k)
cat_probs = (
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
    formula=formula,
    family=opts.model.get("family", "categorical"),
    target=target,
)

# %% ---------------------------------------------------------------------------------

print("Saving model inference data as NetCDF file...")
name = f"{target}.nc" if not by else f"{target}-{by}.nc"
idata.to_netcdf(dirpath / name)

# %% ---------------------------------------------------------------------------------
