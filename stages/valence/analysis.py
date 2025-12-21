# %% ---------------------------------------------------------------------------------
"""Analysis of political valence differences using arviz/bambi machinery.

This script tests whether the properly marginalized expected class probabilities
differ between political and non-political posts overall and by country.
Results are illustrated by point+interval estimates for class probabilities
and corresponding odds ratios.
"""

import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
import xarray as xr
from scipy.special import logit

from project import config, paths
from project.bayes import index_idata, rebuild_model

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target (event): ").strip() or "event"
support = [*config.categorical[target]]
opts = config.glmm.valence.targets[target]

figpath = paths.figures / "valence" / target
figpath.mkdir(parents=True, exist_ok=True)

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))

# %% Load inference data -------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}.nc")
idata = index_idata(idata, ["key", *sum(opts.predictors.values(), start=[])])

# Rebuild model for bambi.interpret functions
model = rebuild_model(idata)

# Extract posterior expected probabilities
epred = az.extract(idata, group="posterior_epred")
# Marginalize out weekend effects to focus on main effects
if "weekend" in epred.coords:
    # Average over weekend levels while preserving other coordinates
    weekend_vals = np.unique(epred.coords["weekend"].values)
    epred = sum(epred.sel(weekend=w).drop_vars("weekend") for w in weekend_vals) / len(
        weekend_vals
    )

# Get probabilities as xarray DataArray
probs = epred.p.to_dataframe()["p"]

# %% -----------------------------------------------------------------------------


def hdi(s: pd.Series, prob: float = config.arviz["stats.ci_prob"]) -> pd.Series:
    """Compute HDI for a pandas Series."""
    hdi_bounds = az.hdi(s.values, hdi_prob=prob)
    return pd.Series({"median": s.median(), "lower": hdi_bounds[0], "upper": hdi_bounds[1]})


# %% Compute posterior expected class probabilities ----------------------------------

posterior = pd.concat(
    [
        probs.groupby(["country", "political", target])
        .apply(hdi)
        .unstack(-1)
        .reset_index(),
        probs.groupby(["political", target]).apply(hdi).unstack(-1).reset_index(),
    ]
).fillna({"country": "overall"})

# %% Plot posterior expectations -----------------------------------------------------

country_order = ["overall", *config.categorical.country]
fig, axes = plt.subplots(ncols=len(country_order), figsize=(21, 3), sharey=True)

for ax, country in zip(axes, country_order, strict=True):
    gdf = posterior.query("country == @country")
    range_kw = config.plotting.objects.range
    (
        so.Plot(gdf, x=target, y="median", color="political")
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(so.Range(**range_kw), so.Dodge(), ymin="lower", ymax="upper")
        .scale(color=[*config.plotting.color.political])
        .on(ax)
        .plot()
    )
    title = country.title() if country == "overall" else countries_map[country]
    ax.set_title(title, fontsize="xx-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(0, 1)
    ax.set_xticks(support)

fig.legends.clear()
axes[0].set_ylabel("Posterior expectation", fontsize="xx-large")
fig.tight_layout()
fig.savefig(figpath / "posterior-expectations.pdf")


# %% Compute political/non-political odds ratios -------------------------------------

odds_ratios = (
    probs.pipe(logit)
    .groupby(["country", "chain", "draw", target])
    .diff()
    .dropna()
    .droplevel(0)
    .pipe(np.exp)
)

posterior_or = pd.concat(
    [
        odds_ratios.groupby(["country", target]).apply(hdi).unstack(-1).reset_index(),
        odds_ratios.groupby(target).apply(hdi).unstack(-1).reset_index(),
    ]
).fillna({"country": "overall"})

# %% Plot odds ratios ----------------------------------------------------------------

fig, axes = plt.subplots(ncols=len(country_order), figsize=(21, 3), sharey=True)

for ax, country in zip(axes, country_order, strict=True):
    gdf = posterior_or.query("country == @country")
    range_kw = config.plotting.objects.range
    (
        so.Plot(gdf, x=target, y="median", color=target)
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(so.Range(**range_kw), so.Dodge(), ymin="lower", ymax="upper")
        .scale(color=[*config.plotting.color.valence])
        .on(ax)
        .plot()
    )
    title = country.title() if country == "overall" else countries_map[country]
    ax.set_title(title, fontsize="xx-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(10**-1, 10**1)
    ax.set_xticks(support)

fig.legends.clear()
axes[0].set_ylabel("Posterior odds ratio", fontsize="xx-large")
fig.tight_layout()
fig.savefig(figpath / "posterior-odds-ratio.pdf")

# %% ---------------------------------------------------------------------------------
