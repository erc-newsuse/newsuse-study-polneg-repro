# %% ---------------------------------------------------------------------------------
"""Analysis of engagement differences by political content using arviz/bambi machinery.

This script tests whether the properly marginalized expected engagement counts
differ between political and non-political posts overall and by country and valence.
Results are illustrated by point+interval estimates for expected counts
and corresponding ratios (political / non-political).
"""

import os

import arviz as az
import matplotlib as mpl
import seaborn.objects as so
import xarray as xr

from project import config, paths
from project.bayes import index_idata

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target (reactions): ").strip() or "reactions"

opts = config.glmm.engagement.targets[target]

figpath = paths.figures / "glmm" / "engagement"
figpath.mkdir(parents=True, exist_ok=True)

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))
event_map = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}
sentiment_map = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}

# %% Load inference data -------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "engagement" / f"{target}.nc")
idata = index_idata(idata, ["key", *sum(opts.predictors.values(), start=[])])

# Extract posterior expected values (mu)
epred = az.extract(idata, group="posterior_epred")
# Get expected values as pandas Series
mu = epred.mu.to_dataframe()["mu"]

# %% Event posterior -----------------------------------------------------------------

(mu.groupby(["event"]).quantile([0.5, 0.025, 0.975]))

# %% -------------------------------------------------------------------------------
