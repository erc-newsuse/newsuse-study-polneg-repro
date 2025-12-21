# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import xarray as xr
from newsuse.data import DataFrame
from omegaconf import OmegaConf
from scipy.special import gammaln

from project import config, paths
from project.bayes import store_model_metadata

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)

# %% ---------------------------------------------------------------------------------

target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target name (reactions): ").strip() or "reactions"
opts = config.glmm.engagement.targets[target]

dirpath = paths.glmm / "engagement"
dirpath.mkdir(parents=True, exist_ok=True)

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.final)
    .assign(
        country=lambda df: pd.Categorical(
            df["country"], categories=[*config.categorical.country]
        ),
        outlet=lambda df: pd.Categorical(df["outlet"]),
        event=lambda df: pd.Categorical(
            df["event"],
            categories=[*config.categorical.event],
            ordered=True,
        ),
        sentiment=lambda df: pd.Categorical(
            df["sentiment"],
            categories=[*config.categorical.sentiment],
            ordered=True,
        ),
    )[
        [
            "key",
            target,
            *opts.predictors.fixed,
            *opts.predictors.groups,
        ]
    ]
    .convert_dtypes(dtype_backend="numpy_nullable")
)

# %% ---------------------------------------------------------------------------------

if sample := opts.get("sample"):
    print(f"Subsampling data for model fitting to {sample.n} observations...")
    model_data = data.sample(**sample, ignore_index=True)
else:
    model_data = data

# %% ---------------------------------------------------------------------------------

# Build formula from list of formulas (mean, dispersion, zero-inflation)
formula_strings = [f.format(target=target).strip().replace("\n", " ") for f in opts.formula]
formula = bmb.Formula(*formula_strings)

print(f"Building GLMM for '{target}' using 'bambi'...")
print(f"Formula: {formula}")
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

ppd_kwargs = OmegaConf.to_object(opts.ppd)
ppd_draws = ppd_kwargs.pop("draws")
ppd = (
    model.predict(
        idata.isel(draw=slice(0, ppd_draws)),
        data=model.data,
        inplace=False,
        **ppd_kwargs,
    )
    .posterior_predictive.drop_vars("__obs__")
    .assign_coords(
        {n: ("__obs__", c.to_numpy()) for n, c in model.data.items() if n != target}
    )
)

idata.add_groups(**{group: ppd})

# %% ---------------------------------------------------------------------------------

print("Prepare posterior expectations group in inference data...")
if (group := "posterior_epred") in idata.groups():
    del idata["posterior_epred"]

# Create grid for simple effects (fixed effects only)
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

epred_kwargs = OmegaConf.to_object(opts.epred.predict)
epred = (
    model.predict(idata, data=grid, inplace=False, **epred_kwargs)
    .posterior["mu"]  # For hurdle models, 'mu' is the expected value
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in grid.items()})
    .groupby([*opts.predictors.fixed])
    .mean()
    .stack(__obs__=tuple(opts.predictors.fixed))
    .transpose("chain", "draw", "__obs__")
    .reset_index("__obs__")
    .dropna("__obs__")
)

idata.add_groups(**{group: epred.to_dataset()})

# %% ---------------------------------------------------------------------------------

print("Prepare log-likelihood group in inference data...")
if (group := "log_likelihood") in idata.groups():
    del idata["log_likelihood"]

# Get response parameters for log-likelihood computation
epred_for_ll = model.predict(
    idata.isel(draw=slice(0, ppd_draws)),
    data=model.data,
    inplace=False,
    kind="response_params",
    include_group_specific=True,
    sample_new_groups=True,
)

# Get model parameters from posterior
posterior = epred_for_ll.posterior
mu = posterior["mu"].values  # Expected count (chain, draw, obs)
alpha = posterior["alpha"].values  # Dispersion parameter

# Get observed values
y = model.data[target].to_numpy()

# Negative binomial PMF: NegBin(y; mu, alpha) where variance = mu + alpha * mu^2
# Using scipy parameterization: n = 1/alpha, p = n/(n + mu)
nb_n = 1.0 / alpha  # Number of successes
nb_p = nb_n / (nb_n + mu)  # Success probability


def negbin_logpmf(y, n, p):
    """Log PMF of negative binomial distribution."""
    return gammaln(y + n) - gammaln(y + 1) - gammaln(n) + n * np.log(p) + y * np.log(1 - p)


# Compute log-likelihood for all observations
loglik_vals = negbin_logpmf(y, nb_n, nb_p)

loglik = xr.DataArray(
    loglik_vals,
    dims=["chain", "draw", "__obs__"],
    coords={
        "chain": idata.posterior.chain,
        "draw": idata.posterior.draw[:ppd_draws],
    },
).assign_coords(
    {col: ("__obs__", c.to_numpy()) for col, c in model.data.items() if col != target}
)

idata.add_groups(**{group: loglik.to_dataset(name=target)})

# %% ---------------------------------------------------------------------------------

print("Storing model metadata in inference data...")
# Join formula strings for storage
formula_str = " ; ".join(formula_strings)
store_model_metadata(
    idata,
    model,
    formula=formula_str,
    family=opts.model.get("family", "hurdle_negativebinomial"),
    target=target,
)

# %% ---------------------------------------------------------------------------------

print("Saving model inference data as NetCDF file...")
idata.to_netcdf(dirpath / f"{target}.nc")

# %% ---------------------------------------------------------------------------------
