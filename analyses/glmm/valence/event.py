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
from project.inference import set_xindex
from project.plotting import make_legend

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

alpha = 1 - az.rcParams["stats.ci_prob"]
q0, q1 = alpha / 2, 1 - alpha / 2

target = "event"
support = [*config.categorical[target]]
opts = config.glmm.valence[target]

figpath = paths.figures / "glmm" / "valence" / "validation"
figpath.mkdir(parents=True, exist_ok=True)

countries = config.categorical.countries
political = dict(enumerate(config.categorical.political))

# %% ---------------------------------------------------------------------------------


@xr.register_dataset_accessor("stats")
class StatsAccessor:
    q = [q0, 0.5, q1]
    dim = ["__group__", "sample"]
    groupby = ["outlet"]
    sample = "sample"
    varname = "epred"
    label = "expectation"

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def average(self, marginal: bool = False) -> xr.Dataset:
        ds = self._ds.groupby(self.groupby).mean()
        if marginal:
            ds = ds.mean(self.groupby)
        return ds

    def quantile(self, *coords: str) -> pd.DataFrame:
        dim = [*coords, self.sample]
        return (
            self._ds.quantile(q=self.q, dim=dim)[self.varname]
            .to_pandas()
            .rename(index={q0: "lo", 0.5: self.label, q1: "hi"})
            .T
        )

    def mean_diffs(self, **kwargs) -> pd.DataFrame:
        sel1 = {}
        sel2 = {}
        for name, values in kwargs.items():
            v1, v2 = values
            sel1[name] = v1
            sel2[name] = v2
        ds1 = self._ds.sel(**sel1).stats.average(marginal=True).drop_vars(list(sel1))
        ds2 = self._ds.sel(**sel2).stats.average(marginal=True).drop_vars(list(sel2))
        return (ds1 - ds2).stats.quantile()


# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}.nc")
idata = set_xindex(idata, [opts.index_col, *opts.predictors])
epred = az.extract(idata, group="posterior_epred")

# %% ---------------------------------------------------------------------------------

est_political = pd.concat(
    {
        label: epred.sel(political=pol).stats.average().stats.quantile("outlet")
        for pol, label in political.items()
    },
    names=["political"],
).reset_index()

est_countries = pd.concat(
    {
        country: epred.sel(country=country).stats.average().stats.quantile("outlet")
        for country in countries
    },
    names=["country"],
).reset_index()

est_political_country = pd.concat(
    {
        (country, label): epred.sel(country=country, political=pol)
        .stats.average()
        .stats.quantile("outlet")
        for pol, label in political.items()
        for country in countries
    },
    names=["country", "political"],
).reset_index()

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 5))

(
    so.Plot(est_political, x="event", y="expectation", color="political")
    .add(so.Range(**config.plotting.objects.range), so.Dodge(), ymin="lo", ymax="hi")
    .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
    .scale(
        color=[*config.plotting.color.political],
    )
    .limit(y=(0, 1))
    .label(x=str.capitalize, y=str.capitalize)
    .on(ax)
    .plot()
)

ax.set_xticks(support)

legend = make_legend(fig, (0.95, 0.95))
fig.tight_layout()

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(nrows=len(support), figsize=(7, 7))

for ax, event in zip(axes.flat, support, strict=True):
    (
        so.Plot(
            est_political_country.query("event == @event"),
            x="country",
            y="expectation",
            color="political",
        )
        .add(so.Range(**config.plotting.objects.range), so.Dodge(), ymin="lo", ymax="hi")
        .add(
            so.Dot(**config.plotting.objects.dot),
            so.Dodge(),
        )
        .scale(
            color=[*config.plotting.color.political],
        )
        .limit(y=(0, 1))
        .label(x=str.capitalize, y=str.capitalize, title=f"Event: {event}")
        .on(ax)
        .plot()
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)

legend = make_legend(fig, (0.95, 0.3))
fig.legends = [legend]

for ax in axes.flatten()[:-1]:
    ax.set_xticklabels([])
ax = axes.flatten()[-1]
ax.set_xticklabels(countries.values())

fig.tight_layout()

# %% ---------------------------------------------------------------------------------


# %% ---------------------------------------------------------------------------------
