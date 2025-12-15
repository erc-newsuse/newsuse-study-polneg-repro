# %% ---------------------------------------------------------------------------------

import arviz as az
import brmspy  # noqa
import matplotlib as mpl  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn as sns  # noqa
import seaborn.objects as so  # noqa
import xarray as xr  # noqa

from project import config, paths
from project.inference import StatsAccessor, set_xindex
from project.plotting import annotate_ci, make_legend

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

alpha = 1 - az.rcParams["stats.ci_prob"]
conf = (1 - alpha) * 100
q0, q1 = alpha / 2, 1 - alpha / 2

target = "sentiment"
support = [*config.categorical[target]]
opts = config.glmm.valence.targets[target]

figpath = paths.figures / "glmm" / "valence" / "latent"
figpath.mkdir(parents=True, exist_ok=True)

countries = config.categorical.country
political = dict(enumerate(config.categorical.political))

# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}-latent.nc")
idata = set_xindex(idata, [opts.index_col, *sum(opts.predictors.values(), start=[])])
epred = az.extract(idata, group="posterior_epred")
# Average out week effects to focus on main effects
if "weekend" in epred.coords:
    epred = (epred.sel(weekend=0) + epred.sel(weekend=1)) / 2

# %% ---------------------------------------------------------------------------------


@xr.register_dataarray_accessor("stats")
class _StatsAccessor(StatsAccessor):
    alpha = config.inference.alpha


# %%

(
    pd.concat(
        {
            label: epred.full.sel(political=pol).mean("country").stats.quantile()
            for pol, label in political.items()
        },
        names=["political"],
    )
    .unstack("quantile")
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

est_political = (
    pd.concat(
        {
            label: epred.full.sel(political=pol).mean("country").stats.quantile()
            for pol, label in political.items()
        },
        names=["political"],
    )
    .unstack("quantile")
    .reset_index()
)

est_countries = (
    pd.concat(
        {
            country: epred.full.sel(country=country).mean("political").stats.quantile()
            for country in countries
        },
        names=["country"],
    )
    .unstack("quantile")
    .loc[[*countries]]
    .reset_index()
)

est_political_country = (
    pd.concat(
        {
            (country, label): epred.full.sel(
                country=country, political=pol
            ).stats.quantile()
            for pol, label in political.items()
            for country in countries
        },
        names=["country", "political"],
    )
    .unstack("quantile")
    .loc[[*countries]]
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 5))
df = est_political
(
    so.Plot(df, x="political", y="median", color="political")
    .add(so.Range(**config.plotting.objects.range), so.Dodge(), ymin="lb", ymax="ub")
    .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
    .scale(
        color=[*config.plotting.color.political],
    )
    .label(x=str.capitalize, y=str.capitalize)
    .on(ax)
    .plot()
)
ax.set_title(target.capitalize(), x=0.0, ha="left", fontsize="x-large")
ax.set_xlabel(None)
ax.set_ylabel("Posterior expected latent valence")

ax.set_xticks(support)
fig.legends.clear()
# make_legend(fig, (0.95, 0.95))

diffs = (
    epred.full.stats.diff(political=[1, 0], marginalize="country")
    .stats.quantile()
    .to_frame()
    .T.reset_index(drop=True)
)
diffs["anchor"] = df[["lb", "ub"]].mean(axis=1)

for value, row in diffs.iterrows():
    annotate_ci(
        ax,
        [value + 0.5, row["anchor"]],
        row[["lb", "ub"]],
        prefix=r"$\Delta$ ",
        marker_offset=0.45,
        fontsize=8,
    )

fig.tight_layout()
fig.savefig(figpath / f"{target}-latent-political.pdf")

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

df = est_political_country
(
    so.Plot(
        df,
        x="country",
        y="median",
        color="political",
    )
    .add(so.Range(**config.plotting.objects.range), so.Dodge(), ymin="lb", ymax="ub")
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

anchors = df.groupby("country")["ub"].max() + 0.05
for country in countries:
    diffs = epred.full.sel(country=country).stats.diff(political=[1, 0]).stats.quantile()
    annotate_ci(
        ax,
        [country, anchors[country]],
        diffs[["lb", "ub"]],
        prefix=r"$\Delta$ ",
        marker_offset=0.5,
        digits=2,
        fontsize=6,
    )

legend = make_legend(fig, (0.06, 0.95), loc="upper left")
fig.legends = [legend]

ax.set_xticklabels([])
ax.set_xticks([*countries], [*countries.values()])

fig.tight_layout()
fig.savefig(figpath / f"{target}-latent-political-country.pdf")

# %% TABLES --------------------------------------------------------------------------

az.summary(idata)

# %% ---------------------------------------------------------------------------------
