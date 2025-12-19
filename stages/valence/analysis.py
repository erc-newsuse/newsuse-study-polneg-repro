# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import brmspy  # noqa
import matplotlib as mpl  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn as sns  # noqa
import seaborn.objects as so  # noqa
import xarray as xr  # noqa
from scipy.special import logit

import project.model  # noqa
from project import config, paths
from project.bayes import StatsAccessor, set_xindex

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

alpha = 1 - az.rcParams["stats.ci_prob"]
conf = (1 - alpha) * 100
q0, q1 = alpha / 2, 1 - alpha / 2

target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target (event): ").strip() or "event"
support = [*config.categorical[target]]
opts = config.glmm.valence.targets[target]

figpath = paths.figures / "valence" / target
figpath.mkdir(parents=True, exist_ok=True)

countries = config.categorical.country
political = dict(enumerate(config.categorical.political))

rng = np.random.default_rng(opts.seed + 303)

sample_cols = ["chain", "draw"]

# %% Load the valence transformer ----------------------------------------------------


@xr.register_dataarray_accessor("stats")
class _StatsAccessor(StatsAccessor):
    alpha = config.inference.alpha


@xr.register_dataset_accessor("stats")
class _DatasetAccessor(StatsAccessor):
    alpha = config.inference.alpha


# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}.nc")
idata = set_xindex(idata, ["key", *sum(opts.predictors.values(), start=[])])
epred = az.extract(idata, group="posterior_epred")
ppd = az.extract(idata, group="posterior_predictive")
# Average out week effects to focus on main effects
if "weekend" in epred.coords:
    epred = epred.stats.marginalize("weekend")

# %% ---------------------------------------------------------------------------------

eprobs = epred.epred.to_dataframe().iloc[:, 0]

# %% ---------------------------------------------------------------------------------

overall = (
    eprobs.groupby(["political", *sample_cols, target])
    .mean()
    .groupby(["political", target])
    .quantile([q0, 0.5, q1])
    .unstack(-1)
    .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    .reset_index()
)
overall.insert(0, "country", "overall")

countries = (
    eprobs.groupby(["country", "political", target])
    .quantile([q0, 0.5, q1])
    .unstack(-1)
    .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    .loc[[*config.categorical.country]]
    .reset_index()
)

df = pd.concat([overall, countries], axis=0, ignore_index=True)

# %% Plot overall and by country -----------------------------------------------

fig, axes = plt.subplots(ncols=df["country"].nunique(), figsize=(21, 3))

for ax, country in zip(axes, ["overall", *config.categorical.country], strict=True):
    gdf = df.query("country == @country")
    (
        so.Plot(gdf, x=target, y="median", color="political")
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(so.Range(**config.plotting.objects.range), so.Dodge(), ymin="lb", ymax="ub")
        .scale(color=[*config.plotting.color.political])
        .on(ax)
        .plot()
    )
    title = country.title() if country == "overall" else config.categorical.country[country]
    ax.set_title(title, fontsize="xx-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(0, 1)
    ax.set_xticks(support)

fig.legends.clear()
axes[0].set_ylabel("Posterior expectation", fontsize="xx-large")
fig.tight_layout()
fig.savefig(figpath / "posterior-expectations.pdf")

# %% ---------------------------------------------------------------------------------

overall = (
    eprobs.pipe(logit)
    .groupby(["country", *sample_cols, target])
    .diff()
    .dropna()
    .droplevel(0)
    .groupby([*sample_cols, target])
    .mean()
    .groupby([target])
    .quantile([q0, 0.5, q1])
    .unstack(-1)
    .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    .pipe(np.exp)
    .reset_index()
)
overall.insert(0, "country", "overall")

countries = (
    eprobs.pipe(logit)
    .groupby(["country", *sample_cols, target])
    .diff()
    .dropna()
    .droplevel(0)
    .groupby(["country", target])
    .quantile([q0, 0.5, q1])
    .unstack(-1)
    .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    .loc[[*config.categorical.country]]
    .pipe(np.exp)
    .reset_index()
)

df = pd.concat([overall, countries], axis=0, ignore_index=True)

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=df["country"].nunique(), figsize=(21, 3))

for ax, country in zip(axes, ["overall", *config.categorical.country], strict=True):
    gdf = df.query("country == @country")
    (
        so.Plot(gdf, x=target, y="median", color=target)
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(so.Range(**config.plotting.objects.range), so.Dodge(), ymin="lb", ymax="ub")
        .scale(color=[*config.plotting.color.valence])
        .on(ax)
        .plot()
    )
    title = country.title() if country == "overall" else config.categorical.country[country]
    ax.set_title(title, fontsize="xx-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(10**-1, 10**1)
    ax.set_xticks(support)

fig.legends.clear()
axes[0].set_ylabel("Posterior odds ratio", fontsize="xx-large")
fig.tight_layout()
fig.savefig(figpath / "posterior-odds-ratio.pdf")

# %% ---------------------------------------------------------------------------------
