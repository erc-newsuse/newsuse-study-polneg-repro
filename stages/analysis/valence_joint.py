# %% ---------------------------------------------------------------------------------

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
import xarray as xr
from newsuse.data import DataFrame
from scipy.special import logit

from project import config, paths
from project.bayes import eti

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
opts = config.glmm.valence.targets["event"]
support = np.asarray([*config.categorical["valence"]])

figpath = paths.figures / "valence"
tabpath = paths.tables / "valence"
figpath.mkdir(parents=True, exist_ok=True)
tabpath.mkdir(parents=True, exist_ok=True)

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))

predictors_fixed = [*opts.common]
predictors_groups = [*opts.group]

# %% ---------------------------------------------------------------------------------

ppd = xr.load_dataset(paths.glmm / "ppd.nc")

# %% ---------------------------------------------------------------------------------

sentiment = (
    ppd[["event", "sentiment_structural"]]
    .rename_vars({"sentiment_structural": "sentiment"})
    .to_dataframe()
    .reset_index()
    .assign(
        sample=lambda df: (
            df.pop("sample_event").astype(str)
            + "_"
            + df.pop("sample_structural").astype(str)
        ),
        valence=lambda df: df[["event", "sentiment"]].sum(axis=1),
    )
    .set_index("sample")
    .reset_index()
)

# %% JOINT VALENCE ANALYSIS ----------------------------------------------------------

probs = (
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

posterior = (
    probs.groupby(["country", "political", "valence"])
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
    probs.groupby(["country", "valence"])
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

posterior_political = (
    probs.pipe(logit)
    .groupby(["country", "valence", "sample"])
    .diff()
    .dropna()
    .droplevel("political")
    .pipe(np.exp)
    .groupby(["country", "valence"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

posterior_valence = (
    probs.pipe(logit)
    .pipe(lambda df: df.sub(df.xs(0, level="valence")))
    .drop(index=0, level="valence")
    .dropna()
    .pipe(np.exp)
    .groupby(["country", "political", "valence"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
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
        so.Plot(gdf, x="valence", y="median", color="political")
        .add(so.Dot(**{**config.plotting.objects.dot, "pointsize": 8}), so.Dodge())
        .add(so.Range(**range_kw), so.Dodge(), ymin="lower", ymax="upper")
        .scale(color=[*config.plotting.color.political])
        .on(ax)
        .plot()
    )
    title = country.title() if country == "overall" else countries_map[country]
    ax.set_title(title, fontsize=24)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    # Mark regions with colors
    for boundary in [*config.categorical.valence][:-1]:
        ax.axvline(
            x=boundary + 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )
    # Format x-axis as integers
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(support))
    # Mark political significant differences
    sigs = (
        gdf.groupby(["country", "valence"], observed=True)["median"]
        .mean()
        .reset_index(name="y")
        .merge(posterior_political[["country", "valence", "sig", "up"]])
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = np.where(support == int(row["valence"]))[0][0] - 2
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
        gdf[["country", "political", "valence", "median"]]
        .rename(columns={"median": "y"})
        .merge(posterior_valence[["country", "political", "valence", "sig"]])
        .dropna()
        .query("sig")
    )
    for _, row in sigs.iterrows():
        x = np.where(support == int(row["valence"]))[0][0] - 2
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
    "Joint valence",
    fontsize=28,
    x=0.00,
    ha="left",
)
fig.tight_layout()
fig.savefig(figpath / "valence-posterior.pdf")

# %% ---------------------------------------------------------------------------------

tables = {
    "valence": {
        "posterior": posterior,
        "political": posterior_political,
        "valence": posterior_valence,
    }
}

for analysis, tabs in tables.items():
    for name, tab in tabs.items():
        DataFrame(tab).to_(tabpath / f"{analysis}-{name}.tsv", index=False)

# %% ---------------------------------------------------------------------------------
