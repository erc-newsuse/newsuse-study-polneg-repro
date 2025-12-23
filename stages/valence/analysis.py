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
from project.bayes import eti, index_idata

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target (event): ").strip() or "event"

opts = config.glmm.valence.targets[target]
support = np.asarray([*config.categorical[target]])

figpath = paths.figures / "valence" / target
figpath.mkdir(parents=True, exist_ok=True)

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))

if target == "valence":
    target_map = {x: x for x in config.categorical.valence}
else:
    target_map = {
        -1: "negative",
        0: "neutral",
        1: "positive",
    }

# %% Load inference data -------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}.nc")
idata = index_idata(idata, ["key", *sum(opts.predictors.values(), start=[])])

# Extract posterior expected probabilities
epred = az.extract(idata, group="posterior_epred")
# Get probabilities as xarray DataArray
probs = epred.p.to_dataframe()["p"]

# %% Compute posterior expected class probabilities ----------------------------------

posterior = pd.concat(
    [
        probs.groupby(["country", "political", target])
        .apply(eti)
        .unstack(-1)
        .reset_index(),
        probs.groupby(["political", target]).apply(eti).unstack(-1).reset_index(),
    ]
).fillna({"country": "overall"})

# %% Plot posterior expectations -----------------------------------------------------

country_order = ["overall", *config.categorical.country]
fig, axes = plt.subplots(ncols=len(country_order), figsize=(21, 3.5), sharey=True)

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
    ax.set_xticks(
        support - support.min(), labels=[str(target_map[t]).title() for t in support]
    )

fig.legends.clear()
ax = axes[0]
ax.set_ylabel("Posterior class proportions", fontsize="xx-large")
# Add custom legend for political
handles = [
    mpl.lines.Line2D(
        [], [], color=color, markersize=10, marker="o", linestyle="", label=label
    )
    for color, label in zip(
        config.plotting.color.political,
        political_map.values(),
        strict=True,
    )
]
ax.legend(handles=handles, fontsize="large", title_fontsize="x-large", frameon=False)
fig.suptitle(f"{target.capitalize()} valence", fontsize="xx-large", x=0.00, ha="left")
fig.tight_layout()
fig.savefig(figpath / "posterior-expectations.pdf")


# %% Compute political/non-political odds ratios -------------------------------------

odds_ratios = (
    probs.pipe(logit)
    .groupby(["country", "chain", "draw", target])
    .diff()
    .dropna()
    .droplevel("political")
    .pipe(np.exp)
)

posterior_or = (
    pd.concat(
        [
            odds_ratios.groupby(["country", target]).apply(eti).unstack(-1).reset_index(),
            odds_ratios.groupby(target).apply(eti).unstack(-1).reset_index(),
        ]
    )
    .fillna({"country": "overall"})
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
    )
)

# %% Plot odds ratios ----------------------------------------------------------------

fig, axes = plt.subplots(ncols=len(country_order), figsize=(21, 3), sharey=True)

for ax, country in zip(axes, country_order, strict=True):
    gdf = posterior_or.query("country == @country")
    range_kw = config.plotting.objects.range
    (
        so.Plot(gdf, x=target, y="median", color=target)
        .add(so.Range(**range_kw), so.Dodge(), ymin="lower", ymax="upper")
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(
            so.Dot(marker="*", edgecolor="black", color="red"),
            so.Dodge(),
            so.Shift(x=0.15, y=0.1),
            pointsize="sig",
        )
        .scale(
            color=[*config.plotting.color[target]],
            pointsize={True: 15, False: 0},
        )
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
    ax.set_xticks(
        support - support.min(), labels=[str(target_map[t]).title() for t in support]
    )

fig.legends.clear()
fig.suptitle("Political / Non-Political", fontsize="xx-large", x=0.00, ha="left")
ax = axes[0]
ax.set_ylabel("Posterior odds ratio", fontsize="xx-large")

fig.tight_layout()
fig.savefig(figpath / "posterior-odds-ratio.pdf")

# %% Country effect size contrasts ---------------------------------------------------

country_effects = (
    probs.pipe(logit)
    .groupby(["political", "chain", "draw", target])
    .apply(lambda s: s - s.mean())
    .droplevel([0, 1, 2, 3])
    .pipe(np.exp)
)

posterior_ce = (
    country_effects.groupby(["country", "political", target])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        **{target: lambda df: df[target].astype(int)},
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
    )
)

# %% Plot country effect size contrasts ----------------------------------------------

fig, axes = plt.subplots(ncols=len(support), figsize=(15, 4), sharey=True)
for ax, valence in zip(axes, support, strict=True):
    gdf = (
        posterior_ce.query(f"{target} == @valence")
        .set_index("country")
        .loc[[*config.categorical.country]]
        .reset_index()
    )
    range_kw = config.plotting.objects.range
    (
        so.Plot(gdf, x="country", y="median", color="political")
        .add(so.Range(**range_kw), so.Dodge(), ymin="lower", ymax="upper")
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(
            so.Dot(marker="*", edgecolor="black", color="red"),
            so.Dodge(),
            so.Shift(x=0.15, y=0.1),
            pointsize="sig",
        )
        .scale(
            color=[*config.plotting.color.political],
            pointsize={True: 15, False: 0},
        )
        .on(ax)
        .plot()
    )
    ax.set_title(str(target_map[valence]).title(), fontsize="xx-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_yscale("log", base=2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(2**-1.25, 2**1.25)
    ax.set_xticks(gdf["country"])
    ax.set_xticklabels([countries_map[c] for c in gdf["country"]], rotation=30, ha="right")

fig.legends.clear()
axes[0].set_ylabel("Country / Grand Mean", fontsize="xx-large")
fig.tight_layout()
fig.savefig(figpath / "country-effect-sizes.pdf")

# %% ---------------------------------------------------------------------------------
