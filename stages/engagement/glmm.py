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

# Load and prepare data
predictors_fixed = opts.predictors.fixed
predictors_groups = opts.predictors.groups

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
            *predictors_fixed,
            *predictors_groups,
        ]
    ]
    .convert_dtypes(dtype_backend="numpy_nullable")
)

# %% ---------------------------------------------------------------------------------

if sample := opts.get("sample"):
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

print(f"Fitting GLMM for '{target}' using '{config.glmm.defaults.inference_method}'...")
fit_kwargs = OmegaConf.to_object(opts.fit)
idata = model.fit(
    inference_method=config.glmm.defaults.inference_method,
    **fit_kwargs,
)

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
    # Model fitted on subsample - use held-out data
    print("Prepare held-out dataset subsample...")
    held_out = data.pipe(lambda df: df[~df["key"].isin(model.data["key"])])
    valid = held_out.sample(**opts.validation.sample, ignore_index=True)
elif len(data) > (n := opts.validation.sample.n) and n >= 1:
    # Model fitted on full data but data is large - sample for efficiency
    print("Prepare validation dataset subsample...")
    valid = data.sample(**opts.validation.sample, ignore_index=True)
else:
    # Model fitted on full data and data is small enough
    valid = data

# %% ---------------------------------------------------------------------------------

print("Prepare posterior predictive group in inference data...")
if (group := "posterior_predictive") in idata.groups():
    del idata["posterior_predictive"]

ppd_kwargs = OmegaConf.to_object(opts.ppd)
ppd_draws = ppd_kwargs.pop("draws")
ppd = (
    model.predict(
        idata.isel(draw=slice(0, ppd_draws)),
        data=valid,
        inplace=False,
        **ppd_kwargs,
    )
    .posterior_predictive.drop_vars("__obs__")
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in valid.items() if n != target})
)

idata.add_groups(**{group: ppd})

# %% ---------------------------------------------------------------------------------

print("Prepare posterior expectations group in inference data...")
if (group := "posterior_epred") in idata.groups():
    del idata["posterior_epred"]

# Create grid for simple effects (fixed effects only)
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

epred_kwargs = OmegaConf.to_object(opts.epred.predict)
epred = (
    model.predict(idata, data=grid, inplace=False, **epred_kwargs)
    .posterior["mu"]  # For hurdle models, 'mu' is the expected value
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in grid.items()})
    .groupby([*predictors_fixed])
    .mean()
    .stack(__obs__=tuple(predictors_fixed))
    .transpose("chain", "draw", "__obs__")
    .reset_index("__obs__")
)

idata.add_groups(**{group: epred.to_dataset()})

# %% ---------------------------------------------------------------------------------

print("Prepare log-likelihood group in inference data...")
if (group := "log_likelihood") in idata.groups():
    del idata["log_likelihood"]

# For hurdle negative binomial, we need to compute log-likelihood manually
# The model has three components: mu (mean), alpha (dispersion), psi (zero prob)
epred_for_ll = model.predict(
    idata.isel(draw=slice(0, ppd_draws)),
    data=valid,
    inplace=False,
    kind="response_params",
    include_group_specific=True,
    sample_new_groups=True,
)

# Get model parameters from posterior
posterior = epred_for_ll.posterior
mu = posterior["mu"].values  # Expected count (chain, draw, obs)
alpha = posterior["alpha"].values  # Dispersion parameter
psi = posterior["psi"].values  # Zero-inflation probability

# Get observed values
y = valid[target].to_numpy()

# Compute log-likelihood for hurdle negative binomial
# P(Y=0) = psi
# P(Y=y|Y>0) = (1-psi) * NegBin(y; mu, alpha) / (1 - NegBin(0; mu, alpha))

# Negative binomial PMF: NegBin(y; mu, alpha) where variance = mu + alpha * mu^2
# Using scipy parameterization: n = 1/alpha, p = 1/(1 + alpha*mu)
nb_n = 1.0 / alpha  # Number of successes
nb_p = nb_n / (nb_n + mu)  # Success probability


def negbin_logpmf(y, n, p):
    """Log PMF of negative binomial distribution."""
    return gammaln(y + n) - gammaln(y + 1) - gammaln(n) + n * np.log(p) + y * np.log(1 - p)


# Compute log-likelihood
loglik_vals = np.zeros_like(mu)

# For y = 0: log(psi)
zero_mask = y == 0
loglik_vals[..., zero_mask] = np.log(np.clip(psi[..., zero_mask], 1e-10, 1.0))

# For y > 0: log(1-psi) + log(NegBin(y)) - log(1 - NegBin(0))
nonzero_mask = y > 0
if nonzero_mask.any():
    log_one_minus_psi = np.log(np.clip(1 - psi[..., nonzero_mask], 1e-10, 1.0))
    log_negbin_y = negbin_logpmf(
        y[nonzero_mask],
        nb_n[..., nonzero_mask],
        nb_p[..., nonzero_mask],
    )
    log_negbin_0 = negbin_logpmf(0, nb_n[..., nonzero_mask], nb_p[..., nonzero_mask])
    log_one_minus_negbin_0 = np.log(np.clip(1 - np.exp(log_negbin_0), 1e-10, 1.0))
    loglik_vals[..., nonzero_mask] = (
        log_one_minus_psi + log_negbin_y - log_one_minus_negbin_0
    )

loglik = xr.DataArray(
    loglik_vals,
    dims=["chain", "draw", "__obs__"],
    coords={
        "chain": idata.posterior.chain,
        "draw": idata.posterior.draw[:ppd_draws],
    },
).assign_coords(
    {col: ("__obs__", c.to_numpy()) for col, c in valid.items() if col != target}
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
