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
from project.bayes import eti, index_idata, rebuild_model

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

predictors_fixed = [*opts.predictors.fixed]
predictors_groups = [*opts.predictors.groups]

# %% Contrast functions --------------------------------------------------------------


def contr_effect(s: pd.Series, level: int | str = 0) -> pd.Series:
    index = s.index.get_level_values(level)
    x = s.to_numpy()
    contr = x - (x.sum() - x) / (x.size - 1)
    contr = pd.Series(contr, index=pd.Series(index, name="contrast"))
    return contr


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

valence_rates = (
    pd.concat(
        {
            valence: (
                rates.groupby([valence, "chain", "draw"])
                .mean()
                .pipe(np.exp)
                .groupby([valence])
                .apply(eti)
                .unstack(-1)
            )
            for valence in ["event", "sentiment"]
        },
        axis=0,
        names=["valence"],
    )
    .reset_index()
    .rename(columns={"event": "rate"})
)

valence_effects = (
    pd.concat(
        {
            valence: (
                rates.groupby([valence, "chain", "draw"])
                .mean()
                .groupby(["chain", "draw"])
                .apply(contr_effect)
                .pipe(np.exp)
                .groupby(["contrast"])
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

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=2, figsize=(7, 3), sharex=True, sharey=True)

for ax, (valence, gdf) in zip(axes, valence_rates.groupby("valence"), strict=True):
    (
        so.Plot(gdf, x="rate", y="median", color="rate")
        .add(
            so.Range(**config.plotting.objects.range),
            ymin="lower",
            ymax="upper",
        )
        .add(so.Dot(**config.plotting.objects.dot))
        .add(
            so.Dot(edgecolor="black", color="red"),
            so.Shift(x=0.2),
            marker="up",
            pointsize="sig",
            data=valence_effects.query("valence == @valence"),
        )
        .scale(
            color=[*config.plotting.color[valence]],
            marker={True: "^", False: "v"},
            pointsize={True: 10, False: 0},
        )
        .on(ax)
        .plot()
    )
    ax.set_yscale("log")
    ax.set_title(f"{valence.capitalize()} valence", fontsize="xx-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks(
        config.categorical[valence],
        labels=[
            (event_map if valence == "event" else sentiment_map)[t].title()
            for t in config.categorical[valence]
        ],
    )

axes[0].set_ylabel(target.capitalize(), fontsize="xx-large")
fig.legends.clear()
fig.tight_layout()
fig.savefig(figpath / f"{target}-valence-rates.pdf")


# %% Overall valence rates -----------------------------------------------------------

valence_overall_rates = (
    rates.groupby(["event", "sentiment", "chain", "draw"])
    .mean()
    .pipe(np.exp)
    .groupby(["event", "sentiment"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
)

sentiment_by_event_effects = (
    rates.groupby(["event", "sentiment", "chain", "draw"])
    .mean()
    .groupby(["event", "chain", "draw"])
    .apply(contr_effect, level="sentiment")
    .pipe(np.exp)
    .groupby(["event", "contrast"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].pipe(np.log).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

event_by_sentiment_effects = (
    rates.groupby(["event", "sentiment", "chain", "draw"])
    .mean()
    .groupby(["sentiment", "chain", "draw"])
    .apply(contr_effect, level="event")
    .pipe(np.exp)
    .groupby(["sentiment", "contrast"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].pipe(np.log).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

ax = axes[0]
(
    so.Plot(valence_overall_rates, x="event", y="median", color="sentiment")
    .add(
        so.Range(**config.plotting.objects.range),
        so.Dodge(),
        ymin="lower",
        ymax="upper",
    )
    .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
    .add(
        so.Dot(edgecolor="black", color="red", artist_kws={"zorder": 100}),
        so.Dodge(),
        marker="up",
        pointsize="sig",
        y="median",
        data=sentiment_by_event_effects.assign(
            median=lambda df: valence_overall_rates["median"] * (0.9 + df["up"] * 0.2)
        ),
    )
    .scale(
        color=[*config.plotting.color.sentiment],
        marker={True: "^", False: "v"},
        pointsize={True: 10, False: 0},
    )
    .on(ax)
    .plot()
)
ax.set_yscale("log")
ax.set_xticks(
    (pal := config.categorical.event),
    labels=[event_map[t].title() for t in pal],
)
ax.set_xlabel("Event", fontsize="xx-large")
ax.set_ylabel(target.capitalize(), fontsize="xx-large")
# Custom legend for sentiment
handles = [
    mpl.lines.Line2D(
        [],
        [],
        color=c,
        marker="o",
        linestyle="",
        markersize=8,
        label=sentiment_map[i].title(),
    )
    for c, i in zip(
        config.plotting.color.sentiment, config.categorical.sentiment, strict=True
    )
]
ax.legend(
    title="Sentiment",
    handles=handles,
    frameon=True,
)

ax = axes[1]
(
    so.Plot(event_by_sentiment_effects, x="sentiment", y="median", color="contrast")
    .add(
        so.Range(**config.plotting.objects.range),
        so.Dodge(),
        so.Shift(x=0.12),
        ymin="lower",
        ymax="upper",
    )
    .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
    .add(
        so.Dot(edgecolor="black", color="red", artist_kws={"zorder": 100}),
        # so.Shift(x=-.1),
        so.Dodge(),
        marker="up",
        pointsize="sig",
        y="median",
        data=event_by_sentiment_effects.assign(
            median=lambda df: df["median"] * (0.8 + df["up"] * 0.4)
        ),
    )
    .scale(
        color=[*config.plotting.color.event],
        marker={True: "^", False: "v"},
        pointsize={True: 10, False: 0},
    )
    .on(ax)
    .plot()
)
ax.axhline(1, ls="--", color=config.plotting.color.palette.gray, zorder=-99)
ax.set_yscale("log")
ax.set_ylim(10**-1, 10**1)
ax.set_xticks(
    (pal := config.categorical.sentiment), labels=[sentiment_map[t].title() for t in pal]
)
ax.set_xlabel("Sentiment", fontsize="xx-large")
ax.set_ylabel("Category / average over others", fontsize="xx-large")
# Custom legend for event
handles = [
    mpl.lines.Line2D(
        [],
        [],
        color=c,
        marker="o",
        linestyle="",
        markersize=8,
        label=event_map[i].title(),
    )
    for c, i in zip(config.plotting.color.event, config.categorical.event, strict=True)
]
ax.legend(
    title="Event",
    handles=handles,
    frameon=True,
)

fig.legends.clear()
fig.tight_layout()
fig.savefig(figpath / f"{target}-event-by-sentiment.pdf")

# %% ---------------------------------------------------------------------------------
