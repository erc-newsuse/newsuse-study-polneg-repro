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

TARGET = os.environ.get("TARGET")
if TARGET is None:
    TARGET = input("Enter target name (event): ").strip() or "event"

opts = config.glmm.valence.targets[TARGET]
support = np.asarray([*config.categorical[opts.response]])


dirpath = paths.glmm / "valence"
dirpath.mkdir(parents=True, exist_ok=True)

predictors_fixed = [*opts.common]
predictors_groups = [*opts.group]

rng = np.random.default_rng(opts.seed)

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.final)
    .assign(
        country=lambda df: pd.Categorical(
            df["country"], categories=[*config.categorical.country]
        ),
        outlet=lambda df: pd.Categorical(df["outlet"]),
    )[["key", opts.response, *predictors_fixed, *predictors_groups]]
    .dropna(ignore_index=True)
    .convert_dtypes(dtype_backend="numpy_nullable")
)

for col in ["event", "sentiment"]:
    if col in data.columns:
        data[col] = pd.Categorical(
            data[col],
            categories=[*config.categorical[col]],
            ordered=True,
        )

# %% ---------------------------------------------------------------------------------

if sample := opts.get("sample"):
    print(f"Subsampling data for model fitting to {sample} observations...")
    model_data = data.sample(n=sample, random_state=rng, ignore_index=True)
else:
    model_data = data

# %% ---------------------------------------------------------------------------------

print(f"Building GLMM for '{opts.response}' using 'bambi'...")
formula = opts.formula.format(response=opts.response)
formula = formula.replace("\n", " ").strip()
print("Model formula:", formula)

model = bmb.Model(
    formula=formula,
    data=model_data,
    family=opts.family,
    noncentered=opts.noncentered,
)

# %% ---------------------------------------------------------------------------------

print(f"Fitting GLMM for '{opts.response}' using '{opts.fit.inference_method}'...")
idata = model.fit(**OmegaConf.to_object(opts.fit), random_seed=rng)

# %% ---------------------------------------------------------------------------------

print("Prepare observed data group in inference data...")
if (group := "observed_data") in idata.groups():
    del idata["observed_data"]

observed = xr.Dataset(
    {opts.response: ("__obs__", model.data[opts.response].to_numpy())},
    coords={
        n: ("__obs__", c.to_numpy()) for n, c in model.data.items() if n != opts.response
    },
)
idata.add_groups(**{group: observed})

# %% ---------------------------------------------------------------------------------

print("Prepare posterior predictive group in inference data...")
if (group := "posterior_predictive") in idata.groups():
    del idata[group]

kwargs = OmegaConf.to_object(opts.ppd)
ppd = (
    model.predict(
        idata.isel(draw=slice(0, kwargs.pop("draws"))),
        data=model.data,
        inplace=False,
        random_seed=rng,
        **kwargs,
    )
    .posterior_predictive.drop_vars("__obs__")
    .assign_coords(
        {n: ("__obs__", c.to_numpy()) for n, c in model.data.items() if n != opts.response}
    )
)
ppd[opts.response].values += min(support)  # adjust for 0-indexing

idata.add_groups(**{group: ppd})

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
        random_seed=rng,
        **opts.epred.predict,
    )
    .posterior["p"]
    .rename({f"{opts.response}_dim": opts.response})
)

# Use observed category codes from model data
obs_codes = model.data[opts.response].cat.codes.to_numpy()

# Log-likelihood: log(p_observed)
obs_probs = cat_probs.isel({opts.response: xr.DataArray(obs_codes, dims="__obs__")})
loglik = (
    np.log(np.clip(obs_probs, 1e-10, 1.0))
    .drop_vars(opts.response)  # drop the scalar coordinate to avoid name conflict
    .assign_coords(
        {n: ("__obs__", c.to_numpy()) for n, c in model.data.items() if n != opts.response}
    )
    .reset_index("__obs__")
)

idata.add_groups(**{group: loglik.to_dataset(name=opts.response)})

# %% ---------------------------------------------------------------------------------

print("Storing model metadata in inference data...")
store_model_metadata(
    idata,
    model,
    formula=formula,
    family=opts.get("family", "cumulative"),
    response=opts.response,
)

# %% ---------------------------------------------------------------------------------

print("Saving model inference data as NetCDF file...")
idata.to_netcdf(dirpath / f"{TARGET}.nc")

# %% ---------------------------------------------------------------------------------
