# %% ---------------------------------------------------------------------------------

import arviz as az
import numpy as np
import pandas as pd
from newsuse.data import DataFrame
from omegaconf import OmegaConf

from project import config, paths
from project.brms import brm, ro
from project.inference import (
    brms_log_likelihood,
    brms_observed_data,
    brms_posterior,
    brms_posterior_epred,
    brms_posterior_predictive,
)

az.rcParams.update(config.arviz)

target = "event"
opts = config.glmm.valence.targets[target]

dirpath = paths.glmm / "valence"
dirpath.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(opts.seed)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final).assign(
    country=lambda df: pd.Categorical(
        df["country"], categories=[*config.categorical.country]
    ),
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
)[["key", target, *opts.predictors.fixed, *opts.predictors.groups]]

# %% ---------------------------------------------------------------------------------

if (n := opts.model.get("subsample")) and n > 0:
    print(f"Subsampling to {n} data points for faster model fitting...")
    model_data = data.sample(n=n, random_state=rng)
else:
    model_data = data

# %% ---------------------------------------------------------------------------------

print(f"Fitting GLMM for '{target}' target with {model_data.shape[0]} observations")

kwargs = dict(OmegaConf.to_object(opts.solver))
kwargs["control"] = ro.ListVector(kwargs.get("control", {}))

model = brm(
    formula=opts.model.formula.format(target=target),
    data=model_data,
    prior=opts.model.get("prior"),
    family=ro.StrVector([opts.model.family, opts.model.link]),
    seed=int(rng.integers(0, 2**16 - 1)),
    **kwargs,
)

# %% ---------------------------------------------------------------------------------

print("Saving fitted 'brms' model as RDS file...")
ro.r["saveRDS"](model.r, str(dirpath / f"{target}-model.rds"))

# %% ---------------------------------------------------------------------------------

print("Start building inference data...")
quantized = data[[*opts.predictors.fixed]].drop_duplicates(ignore_index=True)

# %% ---------------------------------------------------------------------------------

print("Preparing observed data...")
model = brms_observed_data(model, target, data, dtype=int)

# %% ---------------------------------------------------------------------------------

print("Preparing posterior samples...")
model = brms_posterior(model)

# %% ---------------------------------------------------------------------------------

print("Preparing posterior expectations...")
model = brms_posterior_epred(model, quantized, **opts.inference.epd)

# %% ---------------------------------------------------------------------------------

if (subsample := opts.inference.get("subsample")) is not None and subsample > 0:
    print("Subsampling data for posterior predictive computations...")
    data = data.sample(n=subsample, random_state=rng)

# %% ---------------------------------------------------------------------------------

print("Preparing posterior predictive samples...")
model = brms_posterior_predictive(
    model, data, transform=lambda x: (x - 2).astype(int), **opts.inference.ppd
)

# %% ---------------------------------------------------------------------------------

print("Preparing log-likelihood...")
model = brms_log_likelihood(model, data, **opts.inference.loglik)

# %% ---------------------------------------------------------------------------------

print("Downsampling and separating random effects...")
ranef = [k for k in model.idata.posterior if k.startswith("r_")]
if ranef:
    isel_kwargs = {"draw": slice(-opts.inference.ranef.ndraws, None)}
    ranef_idata = az.InferenceData(
        posterior=model.idata.posterior[ranef].isel(**isel_kwargs)
    )
    model.idata.posterior = model.idata.posterior.drop_vars(ranef)
else:
    ranef_idata = None

# %% ---------------------------------------------------------------------------------

print("Saving inference data to NetCDF...")
model.idata.to_netcdf(dirpath / f"{target}.nc")
if ranef_idata is not None:
    print("Saving random effects inference data to NetCDF...")
    ranef_idata.to_netcdf(dirpath / f"{target}-ranef.nc")

# %% ---------------------------------------------------------------------------------
