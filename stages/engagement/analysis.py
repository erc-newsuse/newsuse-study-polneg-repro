# %% ---------------------------------------------------------------------------------
import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn.objects as so
import xarray as xr
from newsuse.data import DataFrame  # noqa

from project import config, paths
from project.bayes import contr_ref, eti, index_idata, rebuild_model

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

rng = np.random.default_rng(303)

# %% -------------------------------------------------------------------------------

# Configuration
target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target (reactions): ").strip() or "reactions"

opts = config.glmm.engagement.targets[target]

figpath = paths.figures / "engagement"
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

predictors_fixed = [*opts.predictors.fixed]
predictors_groups = [*opts.predictors.groups]

# %% Load inference data -------------------------------------------------------------

data = DataFrame.from_(paths.final)
idata = az.from_netcdf(paths.glmm / "engagement" / f"{target}.nc")
model = rebuild_model(idata)
# ppd = idata.posterior_predictive.to_dataframe()[target]

# Observed data
observed = idata.observed_data.to_dataframe().reset_index()

# %% ---------------------------------------------------------------------------------

print("Prepare posterior expectations group in inference data...")
if (group := "posterior_epred") in idata.groups():
    del idata["posterior_epred"]

# Create grid for simple effects (fixed effects only)
grid = (
    # model.data.groupby([*predictors_fixed, "month"], observed=False)
    model.data.groupby([*predictors_fixed], observed=False)
    .sample(n=opts.epred.samples_per_simple_effect, replace=True)
    .reset_index(drop=True)
)

epred_kwargs = {
    **opts.epred.predict,
    "include_group_specific": False,
}
epred = (
    model.predict(idata, data=grid, inplace=False, **epred_kwargs)
    .posterior[["mu", "alpha"]]
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in grid.items()})
    .groupby(predictors_fixed)
    .mean()
    .stack(__obs__=tuple(predictors_fixed))
    .transpose("chain", "draw", "__obs__")
    .reset_index("__obs__")
    .dropna("__obs__")
)

sigma2 = (
    idata.posterior["1|outlet_sigma"] ** 2
    + idata.posterior["1|country:year:month_sigma"] ** 2
)
epred = np.exp(np.log(epred) + sigma2 / 2)

idata.add_groups(**{group: epred})

# %% ---------------------------------------------------------------------------------

idata = index_idata(idata, ["key", *sum(opts.predictors.values(), start=[])])
# Extract posterior expected values (mu)
epred = az.extract(idata, group="posterior_epred")
# Get expected values as pandas Series
mu = epred.mu.to_dataframe()["mu"]

# %% Derive estimated engagement rates -----------------------------------------------

rates = mu.pipe(np.log).groupby(["event", "sentiment", "political", "chain", "draw"]).mean()

# %% ---------------------------------------------------------------------------------

valence_means = (
    pd.concat(
        {
            valence: (
                rates.groupby([valence, "political", "chain", "draw"])
                .mean()
                .pipe(np.exp)
                .groupby([valence, "political"])
                .apply(eti)
                .unstack(-1)
            )
            for valence in ["event", "sentiment"]
        },
        axis=0,
        names=["valence"],
    )
    .reset_index()
    .rename(columns={"event": "value"})
)

valence_effects = (
    pd.concat(
        {
            valence: (
                rates.groupby([valence, "political", "chain", "draw"])
                .mean()
                .groupby(["political", "chain", "draw"])
                .apply(contr_ref, ref=0, level=0)
                .pipe(np.exp)
                .groupby(["political", "contrast"])
                .apply(eti)
                .unstack(-1)
            )
            for valence in ["event", "sentiment"]
        },
        axis=0,
        names=["valence"],
    )
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].pipe(np.log).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

valence_political_effects = (
    pd.concat(
        {
            valence: (
                rates.groupby([valence, "political", "chain", "draw"])
                .mean()
                .groupby([valence, "chain", "draw"])
                .diff()
                .dropna()
                .droplevel("political")
                .pipe(np.exp)
                .groupby([valence])
                .apply(eti)
                .unstack(-1)
                .reset_index(names=["value"])
            )
            for valence in ["event", "sentiment"]
        },
        axis=0,
        names=["valence"],
    )
    .reset_index("valence")
    .reset_index(drop=True)
    .assign(
        sig=lambda df: df[["lower", "upper"]].pipe(np.log).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=2, figsize=(7, 3), sharex=True, sharey=True)

for ax, (valence, gdf) in zip(axes, valence_means.groupby("valence"), strict=True):
    (
        so.Plot(gdf, x="value", y="median", color="political")
        .add(
            so.Line(linestyle="--"),
            so.Dodge(),
        )
        .add(
            so.Range(**config.plotting.objects.range),
            so.Dodge(),
            ymin="lower",
            ymax="upper",
        )
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .scale(color=[*config.plotting.color.political])
        .on(ax)
        .plot()
    )
    sigdata = valence_effects.query("valence == @valence").merge(
        gdf.loc[gdf["value"] != 0, ["median", "value", "political"]].rename(
            columns={"median": "y", "value": "contrast"}
        ),
    )
    for _, row in sigdata.iterrows():
        if not row["sig"]:
            continue
        x = row["contrast"] + 0.5 * (-1 + 2 * row["political"])
        xy = (x, row["y"])
        starkw = {
            "marker": "*",
            "edgecolor": "black",
            "color": "red",
            "s": 200,
            "zorder": 100,
        }
        ax.scatter([x], [row["y"]], **starkw)
    poldata = valence_political_effects.query("valence == @valence").merge(
        valence_means.groupby(["valence", "value"])["median"].mean().reset_index(name="y")
    )
    for _, row in poldata.iterrows():
        if not row["sig"]:
            continue
        kw = {
            **starkw,
            "marker": "^" if row["up"] else "v",
            "color": "black",
            "s": 100,
        }
        ax.scatter([row["value"]], [row["y"]], **kw)
    ax.set_yscale("log")
    ax.set_title(f"{valence.capitalize()} valence", fontsize=20)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks(
        config.categorical[valence],
        labels=[
            (event_map if valence == "event" else sentiment_map)[t].title()
            for t in config.categorical[valence]
        ],
        fontsize="x-large",
    )

axes[0].set_ylabel(target.capitalize(), fontsize=20)
fig.legends.clear()

# Custom legend handles for political
handles_political = [
    mpl.lines.Line2D(
        [],
        [],
        color=c,
        marker="o",
        linestyle="",
        markersize=8,
        label=label,
    )
    for c, label in zip(
        config.plotting.color.political, config.categorical.political, strict=True
    )
]
fig.tight_layout()
fig.savefig(figpath / f"{target}-rates.pdf")

# %% ---------------------------------------------------------------------------------

joint_means = (
    rates.pipe(np.exp)
    .groupby(["event", "sentiment", "political"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
)

joint_effects = (
    rates.groupby(["event", "political", "chain", "draw"])
    .apply(contr_ref, ref=0, level="sentiment")
    .pipe(np.exp)
    .groupby(["event", "political", "contrast"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].pipe(np.log).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

joint_political_effects = (
    rates.groupby(["event", "sentiment", "political", "chain", "draw"])
    .mean()
    .groupby(["event", "sentiment", "chain", "draw"])
    .diff()
    .dropna()
    .droplevel("political")
    .pipe(np.exp)
    .groupby(["event", "sentiment"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].pipe(np.log).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=3, figsize=(12, 4), sharex=True, sharey=True)

for ax, (event, gdf) in zip(axes.flat, joint_means.groupby("event"), strict=True):
    (
        so.Plot(gdf, x="sentiment", y="median", color="political")
        .add(
            so.Line(linestyle="--"),
            so.Dodge(),
        )
        .add(
            so.Range(**config.plotting.objects.range),
            so.Dodge(),
            ymin="lower",
            ymax="upper",
        )
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .scale(color=[*config.plotting.color.political])
        .on(ax)
        .plot()
    )
    sigdata = joint_effects.query("event == @event").merge(
        gdf.loc[gdf["sentiment"] != 0, ["median", "sentiment", "political"]].rename(
            columns={"median": "y", "sentiment": "contrast"}
        ),
    )
    for _, row in sigdata.iterrows():
        if not row["sig"]:
            continue
        x = row["contrast"] + 0.5 * (-1 + 2 * row["political"])
        xy = (x, row["y"])
        starkw = {
            "marker": "*",
            "edgecolor": "black",
            "color": "red",
            "s": 100,
            "zorder": 100,
        }
        ax.scatter([x], [row["y"]], **starkw)
    poldata = joint_political_effects.query("event == @event").merge(
        joint_means.groupby(["event", "sentiment"])["median"].mean().reset_index(name="y")
    )
    for _, row in poldata.iterrows():
        if not row["sig"]:
            continue
        kw = {
            **starkw,
            "marker": "^" if row["up"] else "v",
            "color": "black",
            "s": 50,
        }
        ax.scatter([row["sentiment"]], [row["y"]], **kw)
    ax.set_yscale("log")
    ax.set_title(f"{event_map[event].title()}", fontsize=18)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks(
        config.categorical["sentiment"],
        labels=[sentiment_map[t].title() for t in config.categorical["sentiment"]],
        fontsize="x-large",
    )

fig.legends.clear()
fig.suptitle("Event valence", fontsize=24, y=0.95)
fig.supxlabel("Sentiment valence", fontsize=20, y=0.05)
fig.supylabel(target.capitalize(), fontsize=24, x=0.02)
fig.tight_layout()
fig.savefig(figpath / f"{target}-joint-rates.pdf")

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 0.5))
ax.axis("off")

# Custom legend handles for significance markers
handles_sig = [
    mpl.lines.Line2D(
        [],
        [],
        color=c,
        marker=m,
        markeredgecolor="black",
        linestyle="",
        markersize=8,
        label=label,
    )
    for c, m, label in [
        ("red", "*", "different than neutral"),
        ("black", "^", "political higher"),
        ("black", "v", "political lower"),
    ]
]
# Make legend at the bottom of the figure
fig.legend(
    handles=(handles := [*handles_political, *handles_sig]),
    loc="center",
    ncol=len(handles),
    frameon=False,
    bbox_to_anchor=(0.5, 0.5),
    handletextpad=0.05,
)
fig.tight_layout()
fig.savefig(figpath / "engagement-legend.pdf")

# %% ---------------------------------------------------------------------------------
