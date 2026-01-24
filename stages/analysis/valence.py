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
from scipy.special import expit, logit

from project import config, paths
from project.bayes import eti

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
TARGET = os.environ.get("TARGET") or input("Enter target (event): ").strip() or "event"

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

ppd = xr.open_dataset(paths.glmm / "ppd.nc")[opts.response]

# %% ---------------------------------------------------------------------------------

probs = (
    pd.concat(
        [
            (
                df := (
                    ppd.to_dataframe()
                    .reset_index()
                    .rename(columns={f"sample_{opts.response}": "sample"})
                )
            )
            .groupby(predictors_fixed + ["sample"])[opts.response]
            .value_counts(normalize=True)
            .sort_index()
            .reset_index(),
            df.groupby([c for c in predictors_fixed if c != "country"] + ["sample"])[
                opts.response
            ]
            .value_counts(normalize=True)
            .sort_index()
            .reset_index(),
        ],
        ignore_index=True,
    )
    .fillna({"country": "overall"})
    .set_index(predictors_fixed + [opts.response, "sample"])["proportion"]
)

assert (
    probs.groupby(predictors_fixed + ["sample"]).sum().round(6).eq(1).all()
), "Proportions do not sum to 1."

# %% Compute posterior expected class probabilities ----------------------------------

posterior = (
    probs.groupby([*predictors_fixed, opts.response])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        country=lambda df: pd.Categorical(
            df["country"],
            categories=["overall", *config.categorical.country],
        ),
    )
)

posterior_overall = (
    probs.pipe(logit)
    .groupby([opts.response, "sample"])
    .mean()
    .pipe(expit)
    .groupby(opts.response)
    .apply(eti)
    .unstack(-1)
    .reset_index()
)

posterior_political = (
    probs.pipe(logit)
    .groupby(["country", opts.response, "sample"])
    .diff()
    .dropna()
    .droplevel("political")
    .pipe(np.exp)
    .groupby(["country", opts.response])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        country=lambda df: pd.Categorical(
            df["country"],
            categories=["overall", *config.categorical.country],
        ),
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

posterior_valence = (
    probs.pipe(logit)
    .pipe(lambda df: df.sub(0, level=opts.response))
    .drop(index=0, level=opts.response)
    .dropna()
    .pipe(np.exp)
    .groupby(["country", "political", opts.response])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        country=lambda df: pd.Categorical(
            df["country"],
            categories=["overall", *config.categorical.country],
        ),
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

# %% Plot posterior expectations -----------------------------------------------------

country_order = ["overall", *config.categorical.country]
fig, axes = plt.subplots(ncols=len(country_order), figsize=(21, 3.5), sharey=True)

for ax, country in zip(axes, country_order, strict=True):
    gdf = posterior.query("country == @country")
    range_kw = config.plotting.objects.range
    (
        so.Plot(gdf, x=opts.response, y="median", color="political")
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
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_xticks(
        support,
        [*map(str.capitalize, config.categorical[opts.response].values())],
        fontsize="x-large",
    )
    # Mark regions with colors
    for boundary in [*config.categorical[opts.response]][:-1]:
        ax.axvline(
            x=boundary + 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )
    # Mark political significant differences
    sigs = (
        gdf.groupby(["country", opts.response], observed=True)["median"]
        .mean()
        .reset_index(name="y")
        .merge(posterior_political[["country", opts.response, "sig", "up"]])
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = np.where(support == int(row[opts.response]))[0][0] - 1
        ax.plot(
            x,
            row["y"],
            marker="^" if row["up"] else "v",
            color="black",
            markersize=10,
            zorder=10,
        )
    # Mark difference vs neutral
    sigs = (
        gdf[["country", "political", opts.response, "median"]]
        .rename(columns={"median": "y"})
        .merge(posterior_valence[["country", "political", opts.response, "sig"]])
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = np.where(support == int(row[opts.response]))[0][0] - 1
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
ax.set_ylabel("Prevalence", fontsize=18)
fig.suptitle(
    {
        "event": "Event valence",
        "sentiment": "Sentiment",
        "valence": "Joint valence",
    }[opts.response],
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

# %% Save tables ---------------------------------------------------------------------

tables = {
    "posterior": posterior,
    "political": posterior_political,
    "valence": posterior_valence,
}

for name, table in tables.items():
    DataFrame(table).to_(tabpath / f"{TARGET}-{name}.tsv", index=False)

# %% ---------------------------------------------------------------------------------
