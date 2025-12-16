# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import brmspy
import numpy as np
from newsuse.data import DataFrame
from rpy2 import robjects as ro

from project import config, paths
from project.inference import (
    brms_log_likelihood,
    brms_observed_data,
    brms_posterior,
    brms_posterior_epred,
    brms_posterior_predictive,
)

target = os.environ.get("TARGET")
if not target:
    target = input("Enter target name (event): ").strip() or "event"

output_dir = paths.glmm / "valence"
output_dir.mkdir(parents=True, exist_ok=True)

opts = config.glmm.valence.targets[target]

rng = np.random.default_rng(opts.seed)

az.rcParams.update(config.arviz)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(
    paths.final,
    columns=[
        opts.index_col,
        f"{target}_latent",
        *opts.predictors.fixed,
        *opts.predictors.groups,
    ],
).assign(
    **{target: lambda df: df[f"{target}_latent"]},
)
quantized = data[[*opts.predictors.fixed]].drop_duplicates(ignore_index=True)

if (n := opts.inference.get("subsample")) and n > 0:
    print(f"Subsampling data to {n} rows for faster processing...")
    data = data.sample(n=n, random_state=rng)

# %% ---------------------------------------------------------------------------------

model = brmspy.FitResult(
    r=ro.r["readRDS"](str(output_dir / f"{target}.rds")),
    idata=az.InferenceData(),
    # idata=az.from_netcdf(str(output_dir / f"{target}.nc")),
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
model = brms_posterior_predictive(model, **opts.inference.ppd)

# %% ---------------------------------------------------------------------------------

print("Preparing population posterior predictive samples...")
model = brms_posterior_predictive(model, quantized, **opts.inference["pop"])

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
