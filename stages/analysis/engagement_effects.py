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

rates_data = []
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
    rates_data.append(s)

# %% ---------------------------------------------------------------------------------

rates = pd.concat(rates_data, axis=1, ignore_index=False)

# %% ---------------------------------------------------------------------------------


def make_posteriors(
    rates: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make posterior summaries for engagement effects.

    Parameters
    ----------
    volumes
        Volume series with MultiIndex including 'country', 'political', 'sample', and
        valence level.

    Returns
    -------
    posterior_political
        DataFrame with posterior summaries for political effects.
    posterior_valence
        DataFrame with posterior summaries for valence effects.
    """
    _, valence = rates.name.split("_")
    rates = rates.groupby(["sample", "country", "political", valence]).mean().pipe(np.log)
    posterior_political = (
        rates.groupby(["country", valence, "sample"])
        .diff()
        .dropna()
        .droplevel("political")
        .unstack("country")
        .assign(overall=lambda df: df.mean(axis=1))
        .stack("country")
        .pipe(np.exp)
        .groupby(["country", valence])
        .apply(eti)
        .unstack(-1)
        .sort_index()
        .loc[[*config.categorical.country, "overall"]]
        .assign(
            sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
            up=lambda df: df["median"] > 1,
        )
    )
    posterior_valence = (
        rates.pipe(lambda df: df.sub(df.xs(0, level=valence)))
        .drop(index=0, level=valence)
        .dropna()
        .unstack("country")
        .assign(overall=lambda df: df.mean(axis=1))
        .stack("country")
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
    )
    return posterior_political, posterior_valence


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
        posterior_political,
        posterior_valence,
    ) = (
        df.query("country == 'overall'").reset_index()
        for df in make_posteriors(rates[f"{opts.response}_{valence}"])
    )
    posterior = (
        pd.concat(
            [
                posterior_political,
                posterior_valence,
            ]
        )
        .assign(political=lambda df: df["political"].map(political_map))
        .fillna("diff")
    )
    (
        so.Plot(posterior_political, x=valence, y="median")
        .add(so.Line(**config.plotting.objects.line, color="k"))
        .add(
            so.Range(**config.plotting.objects.range),
            ymin="lower",
            ymax="upper",
        )
        .add(
            so.Dot(**{**config.plotting.objects.dot, "edgecolor": "w", "color": "k"}),
        )
        .on(ax)
        .plot()
    )
    (
        so.Plot(posterior_valence, x=valence, y="median", color="political")
        .add(
            so.Range(**config.plotting.objects.range),
            so.Dodge(),
            ymin="lower",
            ymax="upper",
        )
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .scale(
            color=[*config.plotting.color.political],
        )
        .on(ax)
        .plot()
    )
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(
        config.categorical[valence],
        labels=[*map(str.capitalize, config.categorical[valence].values())],
    )
    ax.set_yscale("log", base=2)
    ax.set_title(f"{valence.capitalize()}", fontsize="x-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
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
ax.set_ylabel("Ratio of rates", fontsize="large")

fig.legends.clear()
fig.supylabel(
    rf"\textbf{{{opts.response.capitalize()}}}",
    fontsize="x-large",
    va="center",
    x=0.04,
    y=0.51,
)
fig.tight_layout()
fig.savefig(figpath / f"{TARGET}-rr-event-sentiment.pdf")

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 0.5))

handles_political = [
    mpl.lines.Line2D(
        [],
        [],
        color=color,
        marker="o",
        linestyle="",
        markersize=14,
        label=f"relative to neutral ({label})",
    )
    for label, color in zip(
        config.categorical.political,
        config.plotting.color.political,
        strict=True,
    )
]
handles_diff = [
    mpl.lines.Line2D(
        [],
        [],
        color="black",
        markeredgecolor="white",
        marker="o",
        label="political / non-political",
        markersize=14,
        linestyle="",
    )
]

handles = handles_political + handles_diff
ax.axis("off")
fig.legend(
    handles=handles,
    ncols=len(handles),
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
    fontsize=16,
    frameon=False,
)
fig.tight_layout()
fig.savefig(figpath / "engagement-rr-legend.pdf")

# %% ---------------------------------------------------------------------------------

for valence in ["event", "sentiment"]:
    (
        posterior_political,
        posterior_valence,
    ) = make_posteriors(rates[f"{opts.response}_{valence}"])
    tables = {
        "political": posterior_political.reset_index(),
        "valence": posterior_valence.reset_index(),
    }
    for name, table in tables.items():
        DataFrame(table).to_(
            tabpath / f"{TARGET}-{valence}-rr-{name}.tsv",
            index=False,
        )

# %% Pooled effects ------------------------------------------------------------------

R = pd.concat(
    [
        rates[[f"{opts.response}_{valence}"]]
        .groupby(["sample", "country", "political", valence])
        .mean()
        .pipe(np.log)
        .rename_axis(index={valence: "valence"})
        for valence in valences
    ],
    axis=1,
)

# %% ---------------------------------------------------------------------------------

print(opts.response.upper())
pooled_valence = (
    R.pipe(lambda df: df.sub(df.xs(0, level="valence")))
    .drop(index=0, level="valence")
    .dropna()
    .groupby(["sample", "valence"])
    .mean()
    .assign(**{opts.response: lambda df: df.mean(axis=1)})[opts.response]
    .pipe(np.exp)
    .groupby(["valence"])
    .apply(eti)
    .unstack(-1)
    .sort_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)
pooled_valence.head()

# %% ---------------------------------------------------------------------------------

pooled_valence_by = (
    R.pipe(lambda df: df.sub(df.xs(0, level="valence")))
    .drop(index=0, level="valence")
    .dropna()
    .groupby(["sample", "valence"])
    .mean()
    .pipe(np.exp)
    .groupby(["valence"])
    .apply(lambda df: df.apply(eti, axis=0))
    .unstack(-1)
)
pooled_valence_by.head()

# %% ---------------------------------------------------------------------------------

pooled_political = (
    R.groupby(["sample", "country", "valence"])
    .diff()
    .dropna()
    .droplevel("political")
    .assign(**{opts.response: lambda df: df.mean(axis=1)})[opts.response]
    .groupby(["sample"])
    .mean()
    .pipe(np.exp)
    .pipe(eti)
    .to_frame()
    .T.assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)
pooled_political.head()

# %% ---------------------------------------------------------------------------------
