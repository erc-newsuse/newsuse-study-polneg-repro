# %% ---------------------------------------------------------------------------------

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

figpath = paths.figures / "engagement"
figpath.mkdir(parents=True, exist_ok=True)

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    ncols=3,
    nrows=len(config.engagement),
    figsize=(10, 8),
    sharex=False,
    sharey=True,
)
support = np.asarray(config.categorical.sentiment)

for axrow, target in zip(axes, config.engagement, strict=True):
    # Configure analysis
    print(f"Analyze {target}...")
    opts = config.glmm.engagement.targets[target]
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

    predictors_fixed = [*opts.common]
    predictors_groups = [*opts.group]
    # Get models and inference data
    data = DataFrame.from_(paths.final)
    idata = az.from_netcdf(paths.glmm / "engagement" / f"{target}.nc")
    model = rebuild_model(idata)
    # Observed data
    observed = idata.observed_data.to_dataframe().reset_index()
    print("Prepare posterior expectations group in inference data...")
    if (group := "posterior_epred") in idata.groups():
        del idata["posterior_epred"]
    # Create grid for simple effects (fixed effects only)
    # And introduce group variation by scaling thanks to the properties
    # of the Gaussianity of group effects and log-normal distribution
    grid = (
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
    idata = index_idata(idata, ["key", *opts.common, *opts.group])
    # Extract posterior expected values (mu)
    epred = az.extract(idata, group="posterior_epred")
    # Get expected values as pandas Series
    mu = epred.mu.to_dataframe()["mu"]
    # Compute posteriors
    logrates = (
        mu.pipe(np.log).groupby(["political", "event", "sentiment", "chain", "draw"]).mean()
    )
    posterior = (
        logrates.pipe(np.exp)
        .groupby(["event", "sentiment", "political"])
        .apply(eti)
        .unstack(-1)
        .reset_index()
    )
    political_effects = (
        logrates.groupby(["event", "sentiment", "chain", "draw"])
        .diff()
        .dropna()
        .droplevel("political")
        .pipe(np.exp)
        .groupby(["event", "sentiment"])
        .apply(eti)
        .unstack(-1)
        .reset_index()
        .assign(
            sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
            up=lambda df: df["median"] > 1,
        )
    )
    valence_effects = (
        logrates.groupby(["political", "event", "chain", "draw"])
        .apply(contr_ref, ref=0, level="sentiment")
        .pipe(np.exp)
        .groupby(["political", "event", "contrast"])
        .apply(eti)
        .unstack(-1)
        .reset_index()
        .assign(
            sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
            up=lambda df: df["median"] > 1,
        )
    )
    # Plot
    for ax, (event, df) in zip(axrow, posterior.groupby("event"), strict=True):
        (
            so.Plot(df, x="sentiment", y="median", color="political")
            .add(
                so.Line(**config.plotting.objects.line),
                so.Dodge(),
            )
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
            .scale(color=[*config.plotting.color.political])
            .on(ax)
            .plot()
        )
        # Mark political significant differences
        sigs = (
            df.groupby(["sentiment", "event"])["median"]
            .mean()
            .reset_index(name="y")
            .merge(political_effects[["sentiment", "event", "sig", "up"]])
            .dropna()
            .query("sig")
        )
        for _, row in sigs.iterrows():
            x = np.where(support == int(row["sentiment"]))[0][0] - 1
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
            df[["sentiment", "event", "political", "median"]]
            .rename(columns={"median": "y"})
            .merge(
                valence_effects[["political", "event", "contrast", "sig"]].rename(
                    columns={"contrast": "sentiment"}
                )
            )
            .dropna()
            .query("sig")
        )
        for _, row in sigs.iterrows():
            x = np.where(support == int(row["sentiment"]))[0][0] - 1
            ax.plot(
                x + 0.4 * (-1 + row["political"] * 2),
                row["y"],
                marker="*",
                color="red",
                markersize=12,
                zorder=10,
            )
        # Set titles and labels
        ax.set_yscale("log")
        if ax in axes[0]:
            ax.set_title(f"{event_map[event].capitalize()} event", fontsize="x-large")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        if ax in axes[-1]:
            ax.set_xticks(
                support,
                [event_map[x].capitalize() for x in support],
                fontsize="x-large",
            )
        else:
            ax.set_xticks(support, [])
        ax.tick_params(axis="y", which="both", labelsize="large")
    axrow[0].set_ylabel(target.capitalize(), fontsize="xx-large")

for ax in axes.flat:
    ax.set_xlim(axes[0, 0].get_xlim())

fig.legends.clear()
fig.supxlabel("Sentiment valence", fontsize="xx-large", x=0.535)
fig.tight_layout()
fig.savefig(figpath / "engagement.pdf")

# %% ---------------------------------------------------------------------------------
