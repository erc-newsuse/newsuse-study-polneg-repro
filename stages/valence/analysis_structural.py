# %% ---------------------------------------------------------------------------------

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn.objects as so
import xarray as xr
from scipy.special import expit, logit

from project import config, paths
from project.bayes import contr_ref, eti, index_idata, rebuild_model

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

TARGET = "structural"

opts = config.glmm.valence.targets[TARGET]
support = np.asarray([*config.categorical[opts.response]])

figpath = paths.figures / "valence"
figpath.mkdir(parents=True, exist_ok=True)

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))

if TARGET == "valence":
    target_map = {x: x for x in config.categorical.valence}
else:
    target_map = {
        -1: "negative",
        0: "neutral",
        1: "positive",
    }

predictors_fixed = [*opts.common]
predictors_groups = [*opts.group]

# %% Load inference data -------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{TARGET}.nc")
model = rebuild_model(idata)

# %% ---------------------------------------------------------------------------------

print("Prepare posterior expectations group in inference data...")
if (group := "posterior_epred") in idata.groups():
    del idata["posterior_epred"]

grid = model.data[predictors_fixed].drop_duplicates(ignore_index=True)
grid = (
    # Make dummy values for group effects
    # to allow independent sampling of group-level effects
    # for proper marginalization
    grid.loc[grid.index.repeat(opts.epred.samples_per_simple_effect)]
    .groupby(level=0)
    .apply(
        lambda df: df.assign(
            **{
                n: str(df.name) + "_" + np.arange(len(df)).astype(str)
                for n in predictors_groups
            }
        )
    )
    .reset_index(drop=True)
)

epred = (
    model.predict(idata, data=grid, inplace=False, **opts.epred.predict)
    .posterior["p"]
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in grid.items()})
    .rename({f"{opts.response}_dim": opts.response})
    .groupby(predictors_fixed)
    .mean()
    .stack(__obs__=tuple(predictors_fixed))
    .transpose("chain", "draw", "__obs__", opts.response)
    .reset_index("__obs__")
)

idata.add_groups(**{group: epred.to_dataset()})

# %% ---------------------------------------------------------------------------------

idata = index_idata(idata, ["key", *opts.common, *opts.group])
# Extract posterior expected probabilities
epred = az.extract(idata, group="posterior_epred")
# Get probabilities as xarray DataArray
probs = epred.p.to_dataframe()["p"]

# %% ---------------------------------------------------------------------------------

posterior = (
    probs.pipe(logit)
    .groupby(["sentiment", "event", "political", "chain", "draw"])
    .mean()
    .pipe(expit)
    .groupby(["sentiment", "event", "political"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
)

political_effects = (
    probs.pipe(logit)
    .groupby(["sentiment", "event", "country", "chain", "draw"])
    .diff()
    .dropna()
    .droplevel("political")
    .groupby(["sentiment", "event", "chain", "draw"])
    .mean()
    .pipe(np.exp)
    .groupby(["sentiment", "event"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

valence_effects = (
    probs.pipe(logit)
    .groupby(["event", "country", "political", "chain", "draw"])
    .apply(contr_ref, ref="0", level="sentiment")
    .dropna()
    .groupby(["event", "contrast", "political", "chain", "draw"])
    .mean()
    .pipe(np.exp)
    .groupby(["event", "contrast", "political"])
    .apply(eti)
    .unstack(-1)
    .reset_index()
    .assign(
        sig=lambda df: df[["lower", "upper"]].sub(1).pipe(np.sign).prod(axis=1).eq(1),
        up=lambda df: df["median"] > 1,
    )
)

# %% ----------------------------------------------------------------------------------

fig, axes = plt.subplots(
    ncols=3,
    figsize=(10, 4),
    sharex=True,
    sharey=True,
)

for ax, (event, df) in zip(axes, posterior.groupby("event"), strict=True):
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
        x = np.where(support == int(row["sentiment"]))[0][0]
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
        x = np.where(support == int(row["sentiment"]))[0][0]
        ax.plot(
            x + 0.4 * (-1 + row["political"] * 2),
            row["y"],
            marker="*",
            color="red",
            markersize=12,
            zorder=10,
        )
    # Set titles and labels
    ax.set_title(target_map[event].capitalize(), fontsize="x-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks(
        support + 1,
        [target_map[x].capitalize() for x in support],
        fontsize="x-large",
    )

fig.legends.clear()
fig.suptitle("Event valence", fontsize="xx-large", x=0.535, y=0.95)
fig.supxlabel("Sentiment valence", fontsize="xx-large", x=0.535, y=0.05)
fig.supylabel("Posterior class probability", fontsize="xx-large")
fig.tight_layout()
fig.savefig(figpath / "sentiment-structural.pdf")

# %% ---------------------------------------------------------------------------------
