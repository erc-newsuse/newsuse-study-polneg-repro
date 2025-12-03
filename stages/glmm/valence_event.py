# %% ---------------------------------------------------------------------------------

from types import SimpleNamespace

import arviz as az
import brmspy
import numpy as np
from newsuse.data import DataFrame
from rpy2.robjects.packages import importr

from project import config, paths
from project.inference import (
    brms_log_likelihood,
    brms_observed_data,
    brms_posterior,
    brms_posterior_epred,
    brms_posterior_predictive,
    make_data_coords,
)

target = "event"
output_dir = paths.glmm / "valence"
output_dir.mkdir(parents=True, exist_ok=True)

opts = config.glmm.valence[target]
coords_cols = list(opts.coords)

rng = np.random.default_rng(opts.seed)

R = SimpleNamespace(
    base=importr("base"),
    brms=importr("brms"),
    posterior=importr("posterior"),
    emmeans=importr("emmeans"),
)

az.rcParams.update(config.arviz)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final)
sample = data.sample(10000, random_state=42)

# %% ---------------------------------------------------------------------------------

data_coords = make_data_coords(data[coords_cols])

# %% ---------------------------------------------------------------------------------

model = brmspy.FitResult(
    idata=az.InferenceData(), r=R.base.readRDS(str(output_dir / f"{target}.rds"))
)

# %% ---------------------------------------------------------------------------------

print("Preparing observed data...")
model = brms_observed_data(model, target, data_coords, dtype=int)

# %% ---------------------------------------------------------------------------------

print("Preparing posterior samples...")
model = brms_posterior(model)

# %% ---------------------------------------------------------------------------------

print("Preparing posterior expectations...")
model = brms_posterior_epred(model, target, **opts.epd)

# %% ---------------------------------------------------------------------------------

print("Preparing posterior predictive samples...")
model = brms_posterior_predictive(
    model, target, transform=lambda x: (x - 2).astype(int), **opts.ppd
)

# %% ---------------------------------------------------------------------------------

print("Preparing log-likelihood...")
model = brms_log_likelihood(model, target, **opts.ppd)

# %% ---------------------------------------------------------------------------------

print("Downsampling and separating random effects...")
ranef = [k for k in model.idata.posterior if k.startswith("r_")]
model.idata.add_groups(
    posterior_ranef=model.idata.posterior[ranef].isel(draw=slice(0, opts.ranef.ndraws))
)
model.idata.posterior = model.idata.posterior.drop_vars(ranef)

# %% ---------------------------------------------------------------------------------

print("Saving InferenceData to NetCDF...")
model.idata.to_netcdf(output_dir / f"{target}.nc")

# %% ---------------------------------------------------------------------------------
