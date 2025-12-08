# %% ---------------------------------------------------------------------------------


import arviz as az
import brmspy
import numpy as np
from newsuse.data import DataFrame
from rpy2.robjects import r as R

from project import config, paths
from project.inference import (
    brms_log_likelihood,
    brms_observed_data,
    brms_posterior,
    brms_posterior_epred,
    brms_posterior_predictive,
)

target = "event"
output_dir = paths.glmm / "valence"
output_dir.mkdir(parents=True, exist_ok=True)

opts = config.glmm.valence.targets[target]

rng = np.random.default_rng(opts.seed)

az.rcParams.update(config.arviz)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(
    paths.final,
    columns=[opts.index_col, target, *opts.predictors.fixed, *opts.predictors.groups],
)
quantized = data[[*opts.predictors.fixed]].drop_duplicates(ignore_index=True)

if subsample := opts.inference.get("subsample"):
    print(f"Subsampling data to {subsample} rows for faster processing...")
    data = data.sample(n=subsample, random_state=rng)

# %% ---------------------------------------------------------------------------------

model = brmspy.FitResult(
    idata=az.InferenceData(), r=R(f"readRDS('{output_dir / f'{target}.rds'}')")
)

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

print("Preparing posterior predictive samples...")
model = brms_posterior_predictive(
    model, transform=lambda x: (x - 2).astype(int), **opts.inference.ppd
)

# %% ---------------------------------------------------------------------------------

print("Preparing log-likelihood...")
model = brms_log_likelihood(model, **opts.inference.loglik)

# %% ---------------------------------------------------------------------------------

print("Downsampling and separating random effects...")
ranef = [k for k in model.idata.posterior if k.startswith("r_")]
ranef_idata = az.InferenceData(
    posterior=(
        model.idata.posterior[ranef].isel(draw=slice(-opts.inference.ranef.ndraws, None))
    )
)
model.idata.posterior = model.idata.posterior.drop_vars(ranef)

# %% ---------------------------------------------------------------------------------

print("Saving InferenceData to NetCDF...")
model.idata.to_netcdf(output_dir / f"{target}.nc")

# %% ---------------------------------------------------------------------------------

print("Saving random effects InferenceData to NetCDF...")
ranef_idata.to_netcdf(output_dir / f"{target}-ranef.nc")

# %% ---------------------------------------------------------------------------------
