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
from rich.progress import track
from scipy.special import logit

from project import config, paths
from project.bayes import eti

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
valences = ["event", "sentiment", "valence"]
predictors_fixed = [*opts.common]
predictors_groups = [*opts.group]

# %% RELATIVE VOLUME ANALYSIS _-------------------------------------------------------

ppd = xr.load_dataset(paths.glmm / "ppd.nc")

# %% ---------------------------------------------------------------------------------
# Compute posterior predictive engagement rates.

rates = []
for valence in track(valences, description="Computing engagement rates..."):
    # ruff: noqa: B023
    response = f"{opts.response}_{valence}"
    s = (
        ppd[["event", "sentiment_structural", response]]
        .rename_vars({"sentiment_structural": "sentiment"})
        .to_dataframe()
        .reset_index()
        .rename(columns={f"sample_{response}": "sample_engagement"})
        .assign(
            sample=lambda df: (
                df["sample_event"].astype(str)
                + "_"
                + df["sample_structural"].astype(str)
                + "_"
                + df["sample_engagement"].astype(str)
            ),
            valence=lambda df: df[["event", "sentiment"]].sum(axis=1),
        )
        .set_index(
            ["__obs__", "sample", *predictors_fixed, "event", "sentiment", "valence"]
        )[response]
    )
    rates.append(s)

# %% ---------------------------------------------------------------------------------

rates = pd.concat(rates, axis=1, ignore_index=False)

# %% ---------------------------------------------------------------------------------

volumes = rates.copy()
for col in volumes:
    _, valence = col.split("_")
    div = rates.groupby(["sample", *predictors_fixed])[col].transform("sum")
    volumes[col] = rates[col] / div

for valence in valences:
    assert (
        volumes.groupby(["sample", *predictors_fixed, valence])[
            f"{opts.response}_{valence}"
        ]
        .sum()
        .groupby(["sample", *predictors_fixed])
        .sum()
        .round(6)
        .eq(1)
        .all()
    ), f"{valence.capitalize()} volumes do not sum to 1."

# %% ---------------------------------------------------------------------------------


def make_posteriors(
    volumes: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make posterior summaries for valence frequencies and volumes.

    Parameters
    ----------
    volumes
        Volume series with MultiIndex including 'country', 'political', 'sample', and
        valence level.

    Returns
    -------
    volume_posterior
        Posterior summaries for volume frequencies.
    volume_political
        Posterior summaries for political differences in volumes.
    volume_sentiment
        Posterior summaries for sentiment effects in volumes.
    valence_posterior
        Posterior summaries for valence frequencies.
    baseline_diffs
        Posterior summaries for volume minus frequency baseline differences.
    """
    _, valence = volumes.name.split("_")
    freqs_overall = (
        volumes.reset_index()
        .groupby(["political", "sample"])[valence]
        .value_counts(normalize=True)
    )
    freqs = (
        pd.concat(
            [
                volumes.reset_index()
                .groupby(["country", "political", "sample"])[valence]
                .value_counts(normalize=True)
                .reset_index(),
                freqs_overall.reset_index(),
            ]
        )
        .fillna({"country": "overall"})
        .set_index(["country", "political", "sample", valence])["proportion"]
    )
    valence_posterior = (
        freqs.groupby(["country", "political", valence])
        .apply(eti)
        .unstack(-1)
        .sort_index()
        .loc[[*config.categorical.country, "overall"]]
        .reset_index()
    )
    volumes = volumes.groupby(["sample", "country", "political", valence]).sum()
    overall = volumes.groupby(["sample", "political", valence]).mean()
    volumes = (
        pd.concat(
            [
                volumes.reset_index(),
                overall.reset_index(),
            ]
        )
        .fillna({"country": "overall"})
        .set_index(volumes.index.names)[volumes.name]
    )
    volume_posterior = (
        volumes.groupby(["country", "political", valence])
        .apply(eti)
        .unstack(-1)
        .sort_index()
        .loc[[*config.categorical.country, "overall"]]
        .reset_index()
    )
    volume_political = (
        volumes.groupby(["country", valence, "sample"])
        .diff()
        .dropna()
        .droplevel("political")
        .groupby(["country", valence])
        .apply(eti)
        .unstack(-1)
        .sort_index()
        .loc[[*config.categorical.country, "overall"]]
        .assign(
            sig=lambda df: df[["lower", "upper"]].pipe(np.sign).prod(axis=1).eq(1),
            up=lambda df: df["median"] > 0,
        )
        .reset_index()
    )
    volume_valence = (
        volumes.pipe(lambda df: df.sub(df.xs(0, level=valence)))
        .drop(index=0, level=valence)
        .dropna()
        .groupby(["country", "political", valence])
        .apply(eti)
        .unstack(-1)
        .sort_index()
        .loc[[*config.categorical.country, "overall"]]
        .assign(
            sig=lambda df: df[["lower", "upper"]].pipe(np.sign).prod(axis=1).eq(1),
            up=lambda df: df["median"] > 0,
        )
        .reset_index()
    )
    baseline_diffs = (
        volumes.pipe(logit)
        .sub(logit(freqs))
        .pipe(np.exp)
        .groupby(["country", "political", valence])
        .apply(eti)
        .unstack(-1)
        .sort_index()
        .loc[[*config.categorical.country, "overall"]]
        .assign(
            sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
            up=lambda df: df["median"] > 1,
        )
        .reset_index()
    )
    return (
        volume_posterior,
        volume_political,
        volume_valence,
        valence_posterior,
        baseline_diffs,
    )


# %% ---------------------------------------------------------------------------------

valences = ["event", "sentiment"]
fig, axes = plt.subplots(
    ncols=(ncols := len(valences)),
    figsize=((size := 3) * ncols, size),
    sharex=False,
    sharey=True,
)

for ax, valence in zip(axes.flat, valences, strict=True):
    (
        volume_posterior,
        volume_political,
        volume_valence,
        valence_posterior,
        baseline_diffs,
    ) = (
        df.query("country == 'overall'")
        for df in make_posteriors(volumes[f"{opts.response}_{valence}"])
    )
    (
        so.Plot(volume_posterior, x=valence, y="median", color="political")
        .add(
            so.Band(alpha=0.3),
            so.Dodge(),
            data=valence_posterior,
            x=valence,
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
    sigs = (
        valence_posterior.groupby(["country", valence])["median"]
        .mean()
        .reset_index(name="y")
        .merge(volume_political)
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        ax.plot(
            *row[[valence, "y"]],
            marker="^" if row["up"] else "v",
            color="black",
            markersize=8,
            zorder=10,
        )
    # Mark differences from neutral category
    sigs = (
        volume_posterior[["country", "political", valence, "median"]]
        .rename(columns={"median": "y"})
        .merge(volume_valence)
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = row[valence]
        ax.plot(
            x + 0.4 * (-1 + row["political"] * 2),
            row["y"],
            marker="*",
            color="red",
            markersize=8,
            zorder=10,
        )
    # Mark baseline differences
    on = ["country", "political", valence]
    sigs = (
        volume_posterior[[*on, "lower", "upper"]]
        .merge(baseline_diffs[[*on, "sig", "up"]])
        .assign(
            y=lambda df: (
                np.where(df["up"], df["upper"], df["lower"])
                + np.where(df["up"], 0.06, -0.06)
            )
        )
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = row[valence]
        ax.plot(
            x + 0.2 * (-1 + row["political"] * 2),
            row["y"],
            marker="^" if row["up"] else "v",
            color="red",
            markersize=8,
            zorder=10,
        )
    ax.set_title(valence.capitalize(), fontsize="x-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks(
        config.categorical[valence],
        labels=[*map(str.capitalize, config.categorical[valence].values())],
    )
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

# Mark boundaries between categories with vertical lines
for ax in axes.flat:
    for boundary in [*config.categorical[valence]][:-1]:
        ax.axvline(
            x=boundary + 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )

ax = axes[0]
ax.set_ylabel("Relative volume", fontsize="large")

fig.legends.clear()
fig.supylabel(
    rf"\textbf{{{opts.response.capitalize()}}}",
    fontsize="x-large",
    va="center",
    x=0.04,
    y=0.50,
)
fig.tight_layout()
fig.savefig(figpath / f"{TARGET}-volume-event-sentiment.pdf")

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    ncols=(ncols := len(config.categorical.country) + 1),
    figsize=((size := 3) * ncols, size),
    sharex=True,
    sharey=True,
)

(
    volume_posterior,
    volume_political,
    volume_valence,
    valence_posterior,
    baseline_diffs,
) = make_posteriors(volumes[f"{opts.response}_valence"])

for ax, (country, df) in zip(
    axes.flat,
    volume_posterior.assign(
        country=lambda df: pd.Categorical(
            df["country"],
            categories=["overall", *config.categorical.country],
        )
    ).groupby("country", observed=True),
    strict=True,
):
    (
        so.Plot(df, x="valence", y="median", color="political")
        .add(
            so.Band(alpha=0.3),
            so.Dodge(),
            data=valence_posterior.query("country == @country"),
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
    sigs = (
        valence_posterior.query("country == @country")
        .groupby("valence", observed=True)["median"]
        .mean()
        .reset_index(name="y")
        .merge(volume_political.query("country == @country"))
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        ax.plot(
            *row[["valence", "y"]],
            marker="^" if row["up"] else "v",
            color="black",
            markersize=8,
            zorder=10,
        )
    # Mark differences from neutral category
    sigs = (
        volume_posterior.query("country == @country")[
            ["country", "political", "valence", "median"]
        ]
        .rename(columns={"median": "y"})
        .merge(volume_valence)
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = row["valence"]
        ax.plot(
            x + 0.4 * (-1 + row["political"] * 2),
            row["y"],
            marker="*",
            color="red",
            markersize=8,
            zorder=10,
        )
    # Mark baseline differences
    on = ["country", "political", "valence"]
    sigs = (
        volume_posterior.query("country == @country")[[*on, "lower", "upper"]]
        .merge(baseline_diffs[[*on, "sig", "up"]])
        .assign(
            y=lambda df: (
                np.where(df["up"], df["upper"], df["lower"])
                + np.where(df["up"], 0.05, -0.05)
            )
        )
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = row["valence"]
        ax.plot(
            x + 0.2 * (-1 + row["political"] * 2),
            row["y"],
            marker="^" if row["up"] else "v",
            color="red",
            markersize=8,
            zorder=10,
        )
    ax.set_title(
        countries_map.get(country, country).capitalize(),
        fontsize="large",
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks(config.categorical["valence"])
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax = axes.flatten()[0]
ax.set_ylabel("Relative volume", fontsize="x-large")

# Mark boundaries between categories with vertical lines
for ax in axes.flat:
    for boundary in [*config.categorical["valence"]][:-1]:
        ax.axvline(
            x=boundary + 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )

fig.legends.clear()
fig.supylabel(
    rf"\textbf{{{opts.response.capitalize()}}}",
    fontsize="xx-large",
    va="center",
    x=0.01,
    y=0.5,
)
fig.tight_layout()
fig.savefig(figpath / f"{TARGET}-volume-valence-country.pdf")

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
handles_valence = [
    mpl.lines.Line2D(
        [],
        [],
        color="red",
        marker=marker,
        label=label,
        markersize=10,
        linestyle="",
    )
    for label, marker in zip(
        ["different than neutral"],
        ["*"],
        strict=True,
    )
]
handles_baseline = [
    mpl.patches.Patch(
        color=config.plotting.color.semantics.other,
        alpha=0.5,
        label="Valence frequency baseline",
    )
] + [
    mpl.lines.Line2D(
        [],
        [],
        color="red",
        marker=marker,
        linestyle="",
        markersize=10,
        label=label,
    )
    for label, marker in zip(
        ["higher than baseline", "lower than baseline"],
        ["^", "v"],
        strict=True,
    )
]
handles = handles_political + handles_contr_political + handles_valence + handles_baseline
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
fig.savefig(figpath / "engagement-legend.pdf")

# %% ---------------------------------------------------------------------------------

for valence in ["event", "sentiment", "valence"]:
    (
        volume_posterior,
        volume_political,
        volume_valence,
        valence_posterior,
        baseline_diffs,
    ) = make_posteriors(volumes[f"{opts.response}_{valence}"])
    tables = {
        "volume-posterior": volume_posterior,
        "volume-political": volume_political,
        "volume-valence": volume_valence,
        "valence-posterior": valence_posterior,
        "baseline-diffs": baseline_diffs,
    }
    for name, table in tables.items():
        DataFrame(table).to_(
            tabpath / f"{TARGET}-{valence}-{name}.tsv",
            index=False,
        )

# %% ---------------------------------------------------------------------------------
