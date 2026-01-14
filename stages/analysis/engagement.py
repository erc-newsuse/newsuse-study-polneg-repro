# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
import xarray as xr
from newsuse.data import DataFrame

from project import config, paths
from project.bayes import eti, rebuild_model

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

idata = az.from_netcdf(paths.glmm / "engagement" / f"{opts.response}.nc")
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

political_posterior = (
    rates.pipe(np.log)
    .groupby(["country", "event", "sentiment", "sample"], observed=True)
    .diff()
    .dropna()
    .droplevel("political")
    .pipe(np.exp)
    .groupby(["country", "event", "sentiment"], observed=True)
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

# %% ---------------------------------------------------------------------------------

sentiment_posterior = (
    rates.pipe(np.log)
    .pipe(lambda df: df.sub(df.xs(0, level="sentiment")).drop(index=0, level="sentiment"))
    .pipe(np.exp)
    .groupby(["country", "political", "event", "sentiment"], observed=True)
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
ax.set_title("political / non-political", fontsize="x-large")

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
    focal = "positive" if sentiment == 1 else "negative"
    ax.set_title(
        f"{focal} / neutral",
        fontsize="x-large",
    )

# Mark regions between categories with vertical lines
for ax in axes.flat:
    ax.set_xticks(
        config.categorical.event,
        labels=[*map(str.capitalize, config.categorical.event.values())],
        fontsize="large",
    )
    for boundary in [*config.categorical.event][:-1]:
        ax.axvline(
            x=boundary + 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )

ax = axes[1]
handles = [
    mpl.lines.Line2D(
        [],
        [],
        color=color,
        marker="o",
        linestyle="",
        label=political,
    )
    for political, color in zip(
        config.categorical.political, config.plotting.color.political, strict=True
    )
]
legend = ax.legend(
    handles=handles,
    frameon=True,
    title_fontsize="large",
    loc="best",
)

fig.legends.clear()
# Add custom legend for 'sentiment
handles = [
    mpl.lines.Line2D(
        [],
        [],
        color=color,
        marker="o",
        linestyle="",
        label=sentiment.capitalize(),
    )
    for sentiment, color in zip(
        config.categorical.sentiment.values(), config.plotting.color.sentiment, strict=True
    )
]
legend = fig.legend(
    handles=handles,
    ncols=len(handles),
    title="Sentiment",
    frameon=False,
    title_fontsize="large",
    loc="upper left",
    bbox_to_anchor=(0.095, 1.02),
)

ax = axes.flatten()[0]
ax.set_ylabel("Rates ratio", fontsize="x-large")

fig.suptitle("Sentiment", fontsize="xx-large", x=0.70, y=0.9)
fig.supxlabel("Event valence", fontsize="xx-large", y=0.05)
fig.supylabel(rf"\textbf{{{opts.response.capitalize()}}}", fontsize="xx-large", y=0.51)
fig.tight_layout()
fig.savefig(figpath / f"{opts.response}.pdf")


# %% RELATIVE VOLUME ANALYSIS _-------------------------------------------------------

ppd = xr.load_dataset(paths.glmm / "ppd.nc")

# %% ---------------------------------------------------------------------------------
# Compute posterior predictive engagement rates.

rates = (
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
    .reset_index()
    .assign(valence=lambda df: df[["event", "sentiment"]].sum(axis=1))
    .set_index(["country", "political", "event", "sentiment", "valence", "sample"])[
        opts.response
    ]
)

# %% ---------------------------------------------------------------------------------

volumes = rates.groupby(["country", "political", "sample"]).transform(lambda x: x / x.sum())

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
    volumes.droplevel("valence")
    .pipe(lambda df: df.sub(df.xs(0, level="sentiment")))
    .drop(index=0, level="sentiment")
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
    ncols=len(config.categorical.event),
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
    # Mark political significant differences
    sigs = (
        df.groupby(["country", "event", "sentiment"])["median"]
        .mean()
        .reset_index(name="y")
        .merge(volume_political)
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        ax.plot(
            *row[["sentiment", "y"]],
            marker="^" if row["up"] else "v",
            color="black",
            markersize=8,
            zorder=10,
        )
    # Mark difference vs neutral
    sigs = (
        df[["country", "political", "event", "sentiment", "median"]]
        .rename(columns={"median": "y"})
        .merge(volume_sentiment)
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = row["sentiment"]
        ax.plot(
            x + 0.4 * (-1 + row["political"] * 2),
            row["y"],
            marker="*",
            color="red",
            markersize=12,
            zorder=10,
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

# Mark boundaries between categories with vertical lines
for ax in axes.flat:
    for boundary in [*config.categorical.sentiment][:-1]:
        ax.axvline(
            x=boundary + 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )

ax = axes[0]
ax.set_ylabel("Relative volume", fontsize="x-large")
handles = [
    mpl.lines.Line2D([], [], color=color, marker="o", linestyle="", label=political)
    for political, color in zip(
        config.categorical.political, config.plotting.color.political, strict=True
    )
]
# legend = ax.legend(
#     handles=handles,
#     frameon=True,
#     fontsize="large",
# )

fig.legends.clear()
fig.suptitle("Event valence", fontsize="xx-large", x=0.53, y=0.925)
fig.supxlabel("Sentiment", fontsize="xx-large", x=0.53, y=0.05)
fig.supylabel(rf"\textbf{{{opts.response.capitalize()}}}", fontsize="xx-large", y=0.51)
fig.tight_layout()
fig.savefig(figpath / f"{opts.response}-volume.pdf")

# %% EVENT VOLUME --------------------------------------------------------------------

volumes_event = (
    rates.groupby(["country", "political", "event", "sample"])
    .sum()
    .groupby(["country", "political", "sample"])
    .transform(lambda x: x / x.sum())
)

assert (
    volumes_event.groupby(["country", "political", "sample"]).sum().round(6).eq(1).all()
), "Volumes do not sum to 1."

# %% ---------------------------------------------------------------------------------

volumes_event_posterior = (
    volumes_event.groupby(["country", "political", "event"])
    .apply(eti)
    .unstack(-1)
    .sort_index()
    .loc[[*config.categorical.country, "overall"]]
    .reset_index()
)

volumes_event_political = (
    volumes_event.groupby(["country", "event", "sample"])
    .diff()
    .dropna()
    .droplevel("political")
    .groupby(["country", "event"])
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

volumes_event_effects = (
    volumes_event.pipe(lambda df: df.sub(df.xs(0, level="event")))
    .drop(index=0, level="event")
    .groupby(["country", "political", "event"])
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

for by_countries in [True, False]:
    fig, axes = plt.subplots(
        ncols=(ncols := len(config.categorical.country) + 1 if by_countries else 1),
        figsize=(ncols * 4, 4) if by_countries else (5, 4),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (country, df) in zip(
        axes.flat,
        volumes_event_posterior.assign(
            country=lambda df: pd.Categorical(
                df["country"], categories=["overall", *config.categorical.country]
            )
        )
        .pipe(lambda df: df if by_countries else df.query("country == 'overall'"))  # noqa
        .groupby("country", observed=True),
        strict=True,
    ):
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
        # Mark political significant differences
        sigs = (
            df.groupby(["country", "event"], observed=True)["median"]
            .mean()
            .reset_index(name="y")
            .merge(volumes_event_political)
            .dropna()
            .query("sig")
        )
        for _, row in sigs.iterrows():
            ax.plot(
                *row[["event", "y"]],
                marker="^" if row["up"] else "v",
                color="black",
                markersize=12,
                zorder=10,
            )
        # Mark difference vs neutral
        sigs = (
            df[["country", "political", "event", "median"]]
            .rename(columns={"median": "y"})
            .merge(volumes_event_effects)
            .dropna()
            .query("sig")
        )
        for _, row in sigs.iterrows():
            x = row["event"]
            ax.plot(
                x + 0.4 * (-1 + row["political"] * 2),
                row["y"],
                marker="*",
                color="red",
                markersize=12,
                zorder=10,
            )
        ax.set_title(
            config.categorical.country.get(country, "Overall"), fontsize="xx-large"
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks(config.categorical.event)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        ax.set_xticks(
            config.categorical.event,
            labels=[*map(str.capitalize, config.categorical.event.values())],
            fontsize="large",
        )
        # Mark boundaries between categories with vertical lines
        for boundary in [*config.categorical.event][:-1]:
            ax.axvline(
                x=boundary + 0.5,
                color="gray",
                linestyle="--",
                linewidth=1,
                zorder=-1,
            )
    ax = axes[0]
    ax.set_ylabel(
        "Relative volume",
        fontsize="x-large",
    )

    fig.legends.clear()

    fig.supxlabel(
        "Event valence",
        fontsize="xx-large",
        x=0.53 if by_countries else 0.63,
        y=0.02 if by_countries else 0.05,
    )
    fig.supylabel(
        rf"\textbf{{{opts.response.capitalize()}}}",
        fontsize="xx-large",
        x=0.007 if by_countries else 0.04,
        y=0.55,
    )
    fig.tight_layout()
    name = f"{opts.response}-volume-event"
    if by_countries:
        name += "-country"
    fig.savefig(figpath / f"{name}.pdf")


# %% ---------------------------------------------------------------------------------
# Compute posterior predictive valence class frequencies.

sentiment = (
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
    .set_index("sample")
    .reset_index()
)

freqs = (
    pd.concat(
        [
            (
                sentiment.groupby(["political", "country", "sample"])["valence"]
                .value_counts(normalize=True)
                .sort_index()
                .reset_index()
            ),
            (
                sentiment.groupby(["political", "sample"])["valence"]
                .value_counts(normalize=True)
                .sort_index()
                .reset_index()
            ),
        ],
        ignore_index=True,
    )
    .fillna({"country": "overall"})
    .set_index(["political", "country", "sample", "valence"])["proportion"]
)

assert (
    freqs.groupby(["country", "political", "sample"]).sum().round(6).eq(1).all()
), "Frequencies do not sum to 1."

# %% ---------------------------------------------------------------------------------

freqs_posterior = (
    freqs.groupby(["country", "political", "valence"])
    .apply(eti)
    .unstack(-1)
    .sort_index()
    .loc[[*config.categorical.country, "overall"]]
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

volumes_valence = (
    rates.groupby(["country", "political", "valence", "sample"])
    .sum()
    .groupby(["country", "political", "sample"])
    .transform(lambda x: x / x.sum())
)

assert (
    volumes_valence.groupby(["country", "political", "sample"]).sum().round(6).eq(1).all()
), "Volumes do not sum to 1."


# %% ---------------------------------------------------------------------------------

volume_valence_posterior = (
    volumes_valence.groupby(["country", "political", "valence"])
    .apply(eti)
    .unstack(-1)
    .sort_index()
    .loc[[*config.categorical.country, "overall"]]
    .reset_index()
)

volume_valence_political = (
    volumes_valence.groupby(["country", "valence", "sample"])
    .diff()
    .dropna()
    .droplevel("political")
    .groupby(["country", "valence"])
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

volume_valence_effects = (
    volumes_valence.pipe(lambda df: df.sub(df.xs(0, level="valence")))
    .drop(index=0, level="valence")
    .groupby(["country", "political", "valence"])
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

baseline_posterior = (
    freqs.groupby(["country", "political", "valence"])
    .apply(eti)
    .unstack(-1)
    .sort_index()
    .loc[[*config.categorical.country, "overall"]]
    .reset_index()
)

volume_valence_baseline = (
    (volumes_valence - freqs)
    .groupby(["country", "political", "valence"])
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

for by_countries in [True, False]:
    fig, axes = plt.subplots(
        ncols=(ncols := len(config.categorical.country) + 1) if by_countries else 1,
        figsize=(ncols * 4, 4) if by_countries else (5, 4),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (country, df) in zip(
        axes.flat,
        volume_valence_posterior.assign(
            country=lambda df: pd.Categorical(
                df["country"], categories=["overall", *config.categorical.country]
            )
        )
        .pipe(lambda df: df if by_countries else df.query("country == 'overall'"))  # noqa
        .groupby("country", observed=True),
        strict=True,
    ):
        baseline = baseline_posterior.query(f"country == '{country}'")
        (
            so.Plot(df, x="valence", y="median", color="political")
            .add(
                so.Band(alpha=0.3),
                so.Dodge(),
                data=baseline,
                x="valence",
                ymin="lower",
                ymax="upper",
            )
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
        # Mark political significant differences
        # sigs = (
        #     df.groupby(["country", "valence"], observed=True)["median"]
        #     .mean()
        #     .reset_index(name="y")
        #     .merge(volume_valence_political)
        #     .dropna()
        #     .query("sig")
        # )
        # for _, row in sigs.iterrows():
        #     ax.plot(
        #         *row[["valence", "y"]],
        #         marker="^" if row["up"] else "v",
        #         color="black",
        #         markersize=12,
        #         zorder=10,
        #     )
        # Mark difference vs neutral
        # sigs = (
        #     df[["country", "political", "valence", "median"]]
        #     .rename(columns={"median": "y"})
        #     .merge(volume_valence_effects)
        #     .dropna()
        #     .query("sig")
        # )
        # for _, row in sigs.iterrows():
        #     x = row["valence"]
        #     ax.plot(
        #         x + 0.4 * (-1 + row["political"] * 2),
        #         row["y"],
        #         marker="*",
        #         color="red",
        #         markersize=12,
        #         zorder=10,
        #     )
        # Mark difference vs valence frequency baseline
        sigs = (
            df[["country", "political", "valence", "median"]]
            .rename(columns={"median": "y"})
            .merge(volume_valence_baseline)
            .dropna()
            .query("sig")
        )
        for _, row in sigs.iterrows():
            x = row["valence"]
            ax.plot(
                x + 0.4 * (-1 + row["political"] * 2),
                row["y"],
                marker="^" if row["up"] else "v",
                color="red",
                markersize=8,
                zorder=10,
            )
        ax.set_title(
            config.categorical.country.get(country, "Overall"), fontsize="xx-large"
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks(config.categorical.valence)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    # Mark boundaries between categories with vertical lines
    for ax in axes.flat:
        for boundary in [*config.categorical.valence][:-1]:
            ax.axvline(
                x=boundary + 0.5,
                color="gray",
                linestyle="--",
                linewidth=1,
                zorder=-1,
            )

    ax = axes[0]
    ax.set_ylabel("Relative volume", fontsize="x-large")
    # Custom legend for baseline effects
    handles = [
        mpl.patches.Patch(
            facecolor=(color := config.plotting.color.semantics.other),
            edgecolor=color,
            alpha=0.5,
            label="valence frequency",
        )
    ] + [
        mpl.lines.Line2D(
            [],
            [],
            color="red",
            marker=marker,
            linestyle="",
            label=f"volume {label}",
        )
        for label, marker in zip(["higher", "lower"], ["^", "v"], strict=True)
    ]
    legend = ax.legend(
        handles=handles,
        frameon=True,
        loc="upper left",
        columnspacing=0.2,
    )

    fig.legends.clear()
    fig.supxlabel(
        "Joint valence", fontsize="xx-large", x=0.53 if by_countries else 0.6, y=0.05
    )
    fig.supylabel(
        rf"\textbf{{{opts.response.capitalize()}}}",
        fontsize="xx-large",
        x=0.007 if by_countries else 0.04,
        y=0.55,
    )
    fig.tight_layout()
    name = f"{opts.response}-volume-valence"
    if by_countries:
        name += "-country"
    fig.savefig(figpath / f"{name}.pdf")

# %% ---------------------------------------------------------------------------------

tables = {
    "effects": {
        "posterior": posterior,
        "political": political_posterior,
        "sentiment": sentiment_posterior,
    },
    "volume": {
        "posterior": volume_posterior,
        "political": volume_political,
        "sentiment": volume_sentiment,
        "posterior-valence": volume_valence_posterior,
        "political-valence": volume_valence_political,
        "sentiment-valence": volume_valence_effects,
        "baseline-valence": volume_valence_baseline,
    },
}

for analysis, tabs in tables.items():
    for name, tab in tabs.items():
        DataFrame(tab).to_(tabpath / f"{TARGET}-{analysis}-{name}.tsv", index=False)

# %% ---------------------------------------------------------------------------------
