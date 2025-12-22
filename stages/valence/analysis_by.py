# %% ---------------------------------------------------------------------------------
"""Analysis of political valence differences by quality/ideology.

This script analyzes whether quality or ideology moderate the effect of political
content on valence. It uses models that include quality or ideology as interaction
terms with political content to determine scientifically significant effects.
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
from project.bayes import hdi, index_idata

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target (event): ").strip() or "event"

by = os.environ.get("BY")
if by is None:
    by = input("Enter grouping variable (quality): ").strip() or "quality"

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

# Setup grouping variable configuration
by_categories = config.categorical[by]
by_colors = config.plotting.color[by]

# %% Load inference data -------------------------------------------------------------

model_path = paths.glmm / "valence" / f"{target}-{by}.nc"
idata = az.from_netcdf(model_path)
idata = index_idata(idata, ["key", *sum(opts.predictors.values(), start=[]), by])

# Extract posterior expected probabilities
epred = az.extract(idata, group="posterior_epred")
# Get probabilities as xarray DataArray
probs = epred.p.to_dataframe()["p"]

# %% Compute posterior expected class probabilities ----------------------------------

# Overall (marginalized over country) by political and grouping variable
posterior = probs.groupby(["political", by, target]).apply(hdi).unstack(-1).reset_index()

# %% Plot posterior expectations -----------------------------------------------------

fig, axes = plt.subplots(ncols=len(by_categories), figsize=(15, 4), sharey=True)

for ax, by_val in zip(axes, by_categories, strict=True):
    gdf = posterior.query(f"{by} == @by_val")
    range_kw = config.plotting.objects.range
    (
        so.Plot(gdf, x=target, y="median", color="political")
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(so.Range(**range_kw), so.Dodge(), ymin="lower", ymax="upper")
        .scale(color=[*config.plotting.color.political])
        .on(ax)
        .plot()
    )
    ax.set_title(by_val.title(), fontsize="xx-large")
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
fig.suptitle(
    f"{target.capitalize()} valence by {by.capitalize()}",
    fontsize="xx-large",
    x=0.00,
    ha="left",
)
fig.tight_layout()
fig.savefig(figpath / f"{target}-posterior-expectations-by-{by}.pdf")


# %% Compute political/non-political odds ratios by grouping variable ----------------

odds_ratios = (
    probs.pipe(logit)
    .groupby([by, "country", "chain", "draw", target])
    .diff()
    .dropna()
    .droplevel("political")
    .pipe(np.exp)
)

posterior_or = (
    odds_ratios.groupby([by, target])
    .apply(hdi)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
    )
)

# %% Plot odds ratios ----------------------------------------------------------------

fig, axes = plt.subplots(ncols=len(by_categories), figsize=(15, 3), sharey=True)

for ax, by_val in zip(axes, by_categories, strict=True):
    gdf = posterior_or.query(f"{by} == @by_val")
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
    ax.set_title(by_val.title(), fontsize="xx-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(10**-1, 10**1)
    ax.set_xticks(
        support - support.min(), labels=[str(target_map[t]).title() for t in support]
    )

fig.legends.clear()
fig.suptitle(
    f"Political / Non-Political by {by.capitalize()}",
    fontsize="xx-large",
    x=0.00,
    ha="left",
)
ax = axes[0]
ax.set_ylabel("Posterior odds ratio", fontsize="xx-large")

fig.tight_layout()
fig.savefig(figpath / f"{target}-posterior-odds-ratio-by-{by}.pdf")


# %% Compute differences in political effect across grouping variable ----------------
# Test whether the political effect differs by quality/ideology level

# First compute political effect (log-odds) for each level of grouping variable
political_effects = odds_ratios.pipe(np.log)

# Compute pairwise contrasts between levels of the grouping variable
# Reference level is the first category
ref_level = by_categories[1]
contrast_effects = []

for level in by_categories:
    if level == ref_level:
        continue
    # Difference in political effect between level and reference
    contrast = political_effects.xs(level, level=by) - political_effects.xs(
        ref_level, level=by
    )
    contrast_df = (
        contrast.groupby(target)
        .apply(hdi)
        .unstack(-1)
        .reset_index()
        .assign(
            contrast=f"{level} vs {ref_level}",
            sig=lambda df: df[["lower", "upper"]].pipe(np.sign).prod(axis=1).eq(1),
        )
    )
    contrast_effects.append(contrast_df)

posterior_contrasts = pd.concat(contrast_effects, ignore_index=True)

# %% Plot effect contrasts -----------------------------------------------------------

contrasts = posterior_contrasts["contrast"].unique()
fig, axes = plt.subplots(ncols=len(contrasts), figsize=(5 * len(contrasts), 4), sharey=True)
if len(contrasts) == 1:
    axes = [axes]

for ax, contrast in zip(axes, contrasts, strict=True):
    gdf = posterior_contrasts.query("contrast == @contrast")
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
    ax.set_title(contrast.title(), fontsize="x-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(
        support - support.min(), labels=[str(target_map[t]).title() for t in support]
    )

fig.legends.clear()
fig.suptitle(
    f"Difference in Political Effect by {by.capitalize()}",
    fontsize="xx-large",
    x=0.00,
    ha="left",
)
ax = axes[0]
ax.set_ylabel("Posterior contrast (log-odds)", fontsize="xx-large")

fig.tight_layout()
fig.savefig(figpath / f"{target}-political-effect-contrasts-by-{by}.pdf")

# %% ---------------------------------------------------------------------------------
