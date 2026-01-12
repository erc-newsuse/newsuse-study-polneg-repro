# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
import xarray as xr

from project import config, paths
from project.bayes import contr_ref, eti, rebuild_model

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

figpath = paths.figures / "engagement"
tabpath = paths.tables / "engagement"
figpath.mkdir(parents=True, exist_ok=True)
tabpath.mkdir(parents=True, exist_ok=True)

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))

TARGET = (
    os.environ.get("TARGET") or input("Enter target (reactions): ").strip() or "reactions"
)

# Configuration
opts = config.glmm.engagement.targets[TARGET]
predictors_fixed = [*opts.common]
predictors_groups = [*opts.group]

rng = np.random.default_rng(opts.seed + 6165)

# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "engagement" / f"{TARGET}.nc")
idata.attrs["formula"] = idata.attrs["formula"].format(response=opts.response)
model = rebuild_model(idata)

# %% ---------------------------------------------------------------------------------

grid = (
    # Generate a grid of `n` values per political-country combination
    # with randomly sampled random effects
    # We use the observed group levels to keep correlations
    # between event and sentiment random effects
    model.data.drop(columns=["key", "__obs__", opts.response])
    .drop_duplicates(ignore_index=True)
    .groupby(["political", "country", "event", "sentiment"], observed=True)
    .apply(
        lambda df: df.sample(n=100, random_state=rng, replace=True), include_groups=False
    )
    .droplevel(-1)
    .reset_index()
)

n_obs = len(grid)

kwargs = {**opts.ppd}
engagement = (
    model.predict(
        idata.isel(draw=slice(kwargs.pop("draws"))),
        data=grid,
        inplace=False,
        random_seed=rng,
        **kwargs,
    )
    .posterior_predictive.drop_vars("__obs__")
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in grid.items()})
    .to_dataframe()
    .reset_index()
    .assign(sample=lambda df: df.pop("draw") + df.pop("chain") * opts.ppd.draws)
)

# %% ---------------------------------------------------------------------------------

rates = pd.concat(
    [
        engagement.groupby(["country", "political", "event", "sentiment", "sample"])[
            opts.response
        ]
        .mean()
        .reset_index(),
    ],
    ignore_index=True,
)

rates = (
    pd.concat(
        [
            rates,
            rates.groupby(["political", "event", "sentiment", "sample"])[opts.response]
            .apply(lambda x: np.mean(np.log(x)))
            .pipe(np.exp)
            .reset_index(),
        ],
        ignore_index=True,
    )
    .fillna({"country": "overall"})
    .assign(
        country=lambda df: pd.Categorical(
            df["country"], categories=[*config.categorical.country, "overall"]
        )
    )
    .pipe(lambda df: df.set_index([c for c in df.columns if c != opts.response]))[
        opts.response
    ]
)


# %% ---------------------------------------------------------------------------------

posterior = (
    rates.groupby(["country", "political", "event", "sentiment"], observed=False)
    .apply(eti)
    .unstack(-1)
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

political_diffs = (
    rates.pipe(np.log)
    .groupby(["country", "event", "sentiment", "sample"], observed=False)
    .diff()
    .dropna()
    .droplevel("political")
    .pipe(np.exp)
)

political_posterior = (
    political_diffs.groupby(["country", "event", "sentiment"], observed=False)
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

# %% ---------------------------------------------------------------------------------

sentiment_effects = (
    rates.pipe(np.log)
    .groupby(["country", "political", "event", "sample"], observed=False)
    .apply(contr_ref, ref=0, level="sentiment")
    .pipe(np.exp)
    .reset_index("contrast")
    .rename(columns={"contrast": "sentiment"})
    .set_index("sentiment", append=True)[opts.response]
)

sentiment_posterior = (
    sentiment_effects.groupby(
        ["country", "political", "event", "sentiment"], observed=False
    )
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    figsize=(9, 3),
    ncols=3,
    sharex=True,
    sharey=True,
)

# Political effects
ax = axes[0]
df = political_posterior.query("country == 'overall'")
(
    so.Plot(df, x="event", y="median", color="sentiment")
    .add(so.Line(**config.plotting.objects.line), so.Dodge())
    .add(
        so.Range(**config.plotting.objects.range),
        so.Dodge(),
        ymin="lower",
        ymax="upper",
    )
    .add(
        so.Dot(**config.plotting.objects.dot),
        so.Dodge(),
    )
    .scale(
        color=[*config.plotting.color.sentiment],
    )
    .on(ax)
    .plot()
)
ax.axhline(1, color="gray", linestyle="--", linewidth=1)
ax.set_yscale("log")
ax.set_ylim(10**-1, 10**1)
ax.xaxis.set_ticks(config.categorical.event)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.set_title("Political / non-political", fontsize="x-large")
# Add custom legend for 'sentiment
handles = [
    mpl.lines.Line2D([], [], color=color, marker="o", linestyle="", label=sentiment)
    for sentiment, color in zip(
        config.categorical.sentiment, config.plotting.color.sentiment, strict=True
    )
]
legend = ax.legend(
    handles=handles,
    ncols=len(handles),
    title="Sentiment",
    frameon=False,
    title_fontsize="large",
    loc="lower center",
)

# Negative sentiment effects
for ax, sentiment in zip(axes[1:], [-1, 1], strict=True):
    df = sentiment_posterior.query(f"country == 'overall' & sentiment == {sentiment}")
    (
        so.Plot(df, x="event", y="median", color="political")
        .add(so.Line(**config.plotting.objects.line), so.Dodge())
        .add(
            so.Range(**config.plotting.objects.range),
            so.Dodge(),
            ymin="lower",
            ymax="upper",
        )
        .add(
            so.Dot(**config.plotting.objects.dot),
            so.Dodge(),
        )
        .scale(
            color=[*config.plotting.color.political],
        )
        .on(ax)
        .plot()
    )
    ax.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(10**-1, 10**1)
    ax.xaxis.set_ticks(config.categorical.event)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    focal = "Positive" if sentiment == 1 else "Negative"
    ax.set_title(
        f"{focal} / neutral",
        fontsize="x-large",
    )

ax = axes[1]
handles = [
    mpl.lines.Line2D([], [], color=color, marker="o", linestyle="", label=political)
    for political, color in zip(
        config.categorical.political, config.plotting.color.political, strict=True
    )
]
legend = ax.legend(
    handles=handles,
    frameon=False,
    title_fontsize="large",
    loc="lower center",
)

fig.legends.clear()


fig.suptitle("Sentiment", fontsize="xx-large", x=0.69, y=0.9)
fig.supxlabel("Event valence", fontsize="xx-large", y=0.05)
fig.supylabel("Odds ratio", fontsize="xx-large", y=0.51)
fig.tight_layout()
fig.savefig(figpath / f"{TARGET}.pdf")

# %% RELATIVE VOLUME ANALYSIS _-------------------------------------------------------

ppd = xr.load_dataset(paths.glmm / "ppd.nc")

# %% ---------------------------------------------------------------------------------

volumes = (
    ppd[["event", "sentiment_structural", opts.response]]
    .rename_vars({"sentiment_structural": "sentiment"})
    .to_dataframe()
    .reset_index()
    .assign(
        sample=lambda df: (
            df.pop("sample_event").astype(str)
            + "_"
            + df.pop("sample_structural").astype(str)
            + "_"
            + df.pop(f"sample_{opts.response}").astype(str)
        ),
        valence=lambda df: df[["event", "sentiment"]].sum(axis=1),
    )
    .set_index(["country", "political", "event", "sentiment", "sample"])[opts.response]
    .groupby(["country", "political", "event", "sentiment", "sample"])
    .sum()
    .reset_index()
    .pipe(
        lambda df: (
            pd.concat(
                [
                    df,
                    df.groupby(["political", "event", "sentiment", "sample"])[opts.response]
                    .mean()
                    .reset_index(),
                ],
                ignore_index=True,
            )
            .reset_index()
            .fillna({"country": "overall"})
            .set_index([c for c in df.columns if c != opts.response])
        )
    )[opts.response]
    .groupby(["country", "political", "sample"])
    .transform(lambda x: x / x.sum())
)

assert (
    volumes.groupby(["country", "political", "sample"]).sum().round(6).eq(1).all()
), "Volumes do not sum to 1."

# %% ---------------------------------------------------------------------------------

volume_posterior = (
    volumes.groupby(["country", "political", "event", "sentiment"])
    .apply(eti)
    .unstack(-1)
    .sort_index()
    .loc[[*config.categorical.country, "overall"]]
    .reset_index()
)

volume_political = (
    volumes.groupby(["country", "event", "sentiment", "sample"])
    .diff()
    .dropna()
    .droplevel("political")
    .groupby(["country", "event", "sentiment"])
    .apply(eti)
    .unstack(-1)
    .sort_index()
    .loc[[*config.categorical.country, "overall"]]
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 0,
    )
)

volume_sentiment = (
    volumes.groupby(["country", "political", "event", "sample"])
    .apply(contr_ref, ref=0, level="sentiment")
    .reset_index("contrast")
    .rename(columns={"contrast": "sentiment"})
    .set_index("sentiment", append=True)[opts.response]
    .groupby(["country", "political", "event", "sentiment"])
    .apply(eti)
    .unstack(-1)
    .sort_index()
    .loc[[*config.categorical.country, "overall"]]
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 0,
    )
)

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    figsize=(9, 3),
    ncols=3,
    sharex=True,
    sharey=True,
)

for ax, (event, df) in zip(
    axes.flat,
    volume_posterior.query("country == 'overall'").groupby("event"),
    strict=True,
):
    (
        so.Plot(df, x="sentiment", y="median", color="political")
        .add(so.Line(**config.plotting.objects.line), so.Dodge())
        .add(
            so.Range(**config.plotting.objects.range),
            so.Dodge(),
            ymin="lower",
            ymax="upper",
        )
        .add(
            so.Dot(**config.plotting.objects.dot),
            so.Dodge(),
        )
        .scale(
            color=[*config.plotting.color.political],
        )
        .on(ax)
        .plot()
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(config.categorical.event[event].capitalize(), fontsize="x-large")
    ax.set_xticks(
        config.categorical.sentiment,
        labels=[*map(str.capitalize, config.categorical.event.values())],
        fontsize="large",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax = axes[0]
ax.set_ylabel("Relative engagement volume", fontsize="x-large")
handles = [
    mpl.lines.Line2D([], [], color=color, marker="o", linestyle="", label=political)
    for political, color in zip(
        config.categorical.political, config.plotting.color.political, strict=True
    )
]
legend = ax.legend(
    handles=handles,
    frameon=False,
    fontsize="large",
)

fig.legends.clear()
fig.suptitle("Sentiment", fontsize="xx-large", x=0.525, y=0.925)
fig.supxlabel("Event valence", fontsize="xx-large", x=0.53, y=0.05)
fig.tight_layout()


# # %% ---------------------------------------------------------------------------------
