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
by = os.environ.get("BY")
if by is None:
    by = input("Enter grouping variable: ").strip() or ""
by = by.removeprefix("-")

opts = config.glmm.engagement.targets[target]

dirpath = paths.glmm / "engagement"
dirpath.mkdir(parents=True, exist_ok=True)

predictors_fixed = [*opts.predictors.fixed, *([by] if by else [])]
predictors_groups = [*opts.predictors.groups]

rng = np.random.default_rng(opts.seed + 303)

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
        valence=lambda df: pd.Categorical(
            df["valence"],
            categories=[*config.categorical.valence],
            ordered=True,
        ),
        quality=lambda df: pd.Categorical(
            df["quality"],
            categories=[*config.categorical.quality],
            ordered=True,
        ),
        ideology=lambda df: pd.Categorical(
            df["ideology"],
            categories=[*config.categorical.ideology],
        ),
    )[["key", target, *predictors_fixed, *predictors_groups]]
    .dropna(ignore_index=True)
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
formula_list = opts.formula.extended if by else opts.formula.base
# Only the first formula (conditional mean) uses target/by interpolation
formula_strings = [
    formula_list[0].format(target=target, by=by).strip().replace("\n", " "),
    *[f.strip().replace("\n", " ") for f in formula_list[1:]],
]
formula = bmb.Formula(*formula_strings)
print("Model formula:", formula)
model = bmb.Model(
    formula=formula,
    data=model_data,
    **opts.model,
)

# %% ---------------------------------------------------------------------------------

print(f"Fitting GLMM for '{target}' using '{opts.fit.inference_method}'...")
idata = model.fit(**OmegaConf.to_object(opts.fit), random_seed=rng)

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
        random_seed=rng,
        **ppd_kwargs,
    )
    .posterior_predictive.drop_vars("__obs__")
    .assign_coords(
        {n: ("__obs__", c.to_numpy()) for n, c in model.data.items() if n != target}
    )
)

idata.add_groups(**{group: ppd})

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
    random_seed=rng,
)

# Get model parameters from posterior
posterior = epred_for_ll.posterior
mu = posterior["mu"].values  # Expected count (chain, draw, obs)
alpha = posterior["alpha"].values  # Dispersion parameter

# Get observed values
y = model.data[target].to_numpy()

# Negative binomial PMF using PyMC/Bambi parameterization:
# Mean = mu, Variance = mu + mu^2/alpha
# PMF: binom(x + alpha - 1, x) * (alpha/(mu+alpha))^alpha * (mu/(mu+alpha))^x
# where n = alpha (not 1/alpha), p = alpha/(mu+alpha)


def negbin_logpmf(y, mu, alpha):
    """Log PMF of negative binomial distribution (PyMC parameterization)."""
    p = alpha / (mu + alpha)  # Success probability
    q = mu / (mu + alpha)  # Failure probability (1 - p)
    return (
        gammaln(y + alpha)
        - gammaln(y + 1)
        - gammaln(alpha)
        + alpha * np.log(p)
        + y * np.log(q)
    )


# Compute log-likelihood for all observations
loglik_vals = negbin_logpmf(y, mu, alpha)

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
    response=target,
)

# %% ---------------------------------------------------------------------------------

print("Saving model inference data as NetCDF file...")
name = f"{target}.nc" if not by else f"{target}-{by}.nc"
idata.to_netcdf(dirpath / name)

# %% ---------------------------------------------------------------------------------
