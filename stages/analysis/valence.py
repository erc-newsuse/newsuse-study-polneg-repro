# %% ---------------------------------------------------------------------------------
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
from project.bayes import contr_ref, eti

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
TARGET = os.environ.get("TARGET")
if TARGET is None:
    TARGET = input("Enter target (event): ").strip() or "event"

opts = config.glmm.valence.targets[TARGET]
support = np.asarray([*config.categorical[opts.response]])

analysis_name = "valence"

figpath = paths.figures / analysis_name
tabpath = paths.tables / analysis_name
figpath.mkdir(parents=True, exist_ok=True)
tabpath.mkdir(parents=True, exist_ok=True)

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))

predictors_fixed = [*opts.common]
predictors_groups = [*opts.group]

# %% Load inference data -------------------------------------------------------------

ppd = xr.open_dataset(paths.glmm / "ppd.nc")[TARGET]

# %% ---------------------------------------------------------------------------------

probs = (
    ppd.to_dataframe()
    .reset_index()
    .rename(columns={f"sample_{TARGET}": "sample"})
    .groupby(predictors_fixed + ["sample"])[TARGET]
    .value_counts(normalize=True)
    .sort_index()
)

# %% Compute posterior expected class probabilities ----------------------------------

probs_country = (
    probs.groupby(["country", "political", TARGET]).apply(eti).unstack(-1).reset_index()
)

probs_overall = (
    probs.groupby(["political", TARGET, "sample"])
    .mean()
    .groupby(["political", TARGET])
    .apply(eti)
    .unstack(-1)
    .reset_index()
)

posterior = pd.concat([probs_country, probs_overall], ignore_index=True).fillna(
    {"country": "overall"}
)

# %% Political effects ---------------------------------------------------------------

political_diffs = (
    probs.pipe(logit)
    .groupby(["country", TARGET, "sample"])
    .diff()
    .dropna()
    .droplevel("political")
)

political_country = (
    political_diffs.pipe(np.exp)
    .groupby(["country", TARGET])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

political_overall = (
    political_diffs.groupby([TARGET, "sample"])
    .mean()
    .pipe(np.exp)
    .groupby(TARGET)
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

political_posterior = pd.concat(
    [political_country, political_overall], ignore_index=True
).fillna({"country": "overall"})

# %% Valence effects vs neutral ------------------------------------------------------

neutral_diffs = (
    probs.pipe(logit)
    .groupby(["country", "political", "sample"])
    .apply(contr_ref, ref=0, level=TARGET)
)

neutral_country = (
    neutral_diffs.pipe(np.exp)
    .groupby(["country", "political", "contrast"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

neutral_overall = (
    neutral_diffs.groupby(["political", "contrast", "sample"])
    .mean()
    .pipe(np.exp)
    .groupby(["political", "contrast"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

neutral_posterior = pd.concat([neutral_country, neutral_overall], ignore_index=True).fillna(
    {"country": "overall"}
)

# %% Plot posterior expectations -----------------------------------------------------

country_order = ["overall", *config.categorical.country]
fig, axes = plt.subplots(ncols=len(country_order), figsize=(21, 3.5), sharey=True)

for ax, country in zip(axes, country_order, strict=True):
    gdf = posterior.query("country == @country")
    range_kw = config.plotting.objects.range
    (
        so.Plot(gdf, x=TARGET, y="median", color="political")
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(so.Range(**range_kw), so.Dodge(), ymin="lower", ymax="upper")
        .scale(color=[*config.plotting.color.political])
        .on(ax)
        .plot()
    )
    title = country.title() if country == "overall" else countries_map[country]
    ax.set_title(title, fontsize=24)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(0, 1)
    ax.set_xticks(support)
    # Mark regions with colors
    for boundary in [*config.categorical[TARGET]][:-1]:
        ax.axvline(
            x=boundary + 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )
    # Mark political significant differences
    sigs = (
        gdf.groupby(["country", TARGET])["median"]
        .mean()
        .reset_index(name="y")
        .merge(political_posterior[["country", TARGET, "sig", "up"]])
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = np.where(support == int(row[TARGET]))[0][0] - 1
        ax.plot(
            x,
            row["y"],
            marker="^" if row["up"] else "v",
            color="black",
            markersize=12,
            zorder=10,
        )
    # Mark difference vs neutral
    sigs = (
        gdf[["country", "political", TARGET, "median"]]
        .rename(columns={"median": "y"})
        .merge(
            neutral_posterior[["country", "political", "contrast", "sig"]].rename(
                columns={"contrast": TARGET}
            )
        )
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = np.where(support == int(row[TARGET]))[0][0] - 1
        ax.plot(
            x + 0.4 * (-1 + row["political"] * 2),
            row["y"],
            marker="*",
            color="red",
            markersize=12,
            zorder=10,
        )

fig.legends.clear()
ax = axes[0]
ax.set_ylabel("Posterior class proportions", fontsize=18)
fig.suptitle(
    f"{('overall' if TARGET == 'valence' else TARGET).capitalize()} valence",
    fontsize=28,
    x=0.00,
    ha="left",
)
fig.tight_layout()
fig.savefig(figpath / f"{TARGET}-posterior.pdf")

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(21, 0.5))

# Add custom legend for political (colors) and contrasts
handles_political = [
    mpl.lines.Line2D(
        [], [], color=color, marker="o", linestyle="", markersize=10, label=label
    )
    for label, color in zip(
        config.categorical.political,
        config.plotting.color.political,
        strict=True,
    )
]
handles_contr_neutral = [
    mpl.lines.Line2D(
        [],
        [],
        color="red",
        marker="*",
        linestyle="",
        markersize=10,
        label="different than neutral",
    )
]
handles_contr_political = [
    mpl.lines.Line2D(
        [],
        [],
        color="black",
        marker=marker,
        label=label,
        markersize=10,
        linestyle="",
    )
    for label, marker in zip(
        ["political higher", "political lower"],
        ["^", "v"],
        strict=True,
    )
]
handles = handles_political + handles_contr_neutral + handles_contr_political
ax.axis("off")
fig.legend(
    handles=handles,
    ncols=len(handles),
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
    fontsize=14,
    frameon=False,
    title=None,
)
fig.tight_layout()
fig.savefig(figpath / "valence-legend.pdf")

# %% ---------------------------------------------------------------------------------
