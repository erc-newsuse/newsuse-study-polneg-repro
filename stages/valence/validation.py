# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import matplotlib as mpl  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn as sns  # noqa
import xarray as xr
from newsuse.data import DataFrame

from project import config, paths
from project.bayes import set_xindex
from project.plotting import ArvizLabeller

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)

target = os.environ.get("TARGET")
if not target:
    target = input("Enter target (event): ").strip() or "event"
opts = config.glmm.valence.targets[target]
support = config.categorical[target]

figpath = paths.figures / "glmm" / "valence" / "validation" / target
figpath.mkdir(parents=True, exist_ok=True)

labels = {
    "countries": config.categorical.country,
    "political": dict(enumerate(config.categorical.political)),
    target: config.categorical[target],
}

data = DataFrame.from_(paths.final)

# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}.nc")
idata = set_xindex(idata, ["key", *opts.predictors.fixed, *opts.predictors.groups])

# %% ---------------------------------------------------------------------------------

az.summary(idata)

# %% ---------------------------------------------------------------------------------

axes = az.plot_trace(
    idata,
    var_names=["b_", "sd_"],
    filter_vars="like",
    combined=True,
    figsize=(8, 24),
    labeller=ArvizLabeller(),
)
fig = axes.flatten()[0].figure
fig.tight_layout()

axes[0, 0].set_title("Posterior density", fontsize="x-large")
axes[0, 1].set_title("Trace plot", fontsize="x-large")

fig.savefig(figpath / f"{target}-trace.pdf")

# %% ---------------------------------------------------------------------------------

axes = az.plot_ess(
    idata,
    var_names=["b_", "sd_"],
    filter_vars="like",
    labeller=ArvizLabeller(),
    figsize=(10, 14),
)
ylabel = axes[0, 0].get_ylabel()
for ax in axes.flat:
    ax.set_xlabel(None)
    ax.set_ylabel(None)
fig = axes.flatten()[0].figure
fig.supxlabel("Quantile", fontsize="xx-large")
fig.supylabel(ylabel, fontsize="xx-large")
fig.tight_layout()

fig.savefig(figpath / f"{target}-ess.pdf")

# %% ---------------------------------------------------------------------------------

axes = az.plot_autocorr(
    idata,
    var_names=["b_", "sd_"],
    filter_vars="like",
    labeller=ArvizLabeller(),
    figsize=(12, 12),
)
fig = axes.flatten()[0].figure
fig.tight_layout()

fig.savefig(figpath / f"{target}-autocorr.pdf")

# %% ---------------------------------------------------------------------------------


def plot_ppc(
    idata: az.InferenceData,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot posterior predictive check with mean observed value."""
    kwargs = {
        "kind": "cumulative",
        "legend": False,
        "mean": False,
        **kwargs,
    }
    az.plot_ppc(idata, ax=ax, **kwargs)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    return ax


# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=3, figsize=(7, 3))
plot_ppc(idata, ax=axes[0])
plot_ppc(idata.sel(political=1), ax=axes[1])
plot_ppc(idata.sel(political=0), ax=axes[2])

axes[0].set_title("Overall")
axes[1].set_title("Political")
axes[2].set_title("Other")

fig.tight_layout()
fig.supxlabel(target.capitalize(), y=-0.03)
fig.supylabel(r"$\mathbb{P}(X \leq x)$", x=-0.02)
fig.savefig(figpath / f"{target}-ppc.pdf")

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(7, 4))

for ax, country in zip(axes.flat, labels["countries"], strict=True):
    plot_ppc(idata.sel(country=country), ax=ax)
    ax.set_title(country.upper())

fig.tight_layout()
fig.supxlabel(target.capitalize(), y=-0.03)
fig.supylabel(r"$\mathbb{P}(X \leq x)$", x=-0.02)
fig.savefig(figpath / f"{target}-ppc-by-country.pdf")

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=6, nrows=2, figsize=(8, 3))

for axrow, pol in zip(axes, labels["political"], strict=True):
    for ax, country in zip(axrow, labels["countries"], strict=True):
        plot_ppc(idata.sel(country=country, political=pol), ax=ax)

for ax, country in zip(axes[0], labels["countries"], strict=True):
    ax.set_title(country.upper())
    ax.set_xticks([])
for ax, pol in zip(axes[:, 0], labels["political"], strict=True):
    ax.set_ylabel(labels["political"][pol])

fig.supxlabel(target.capitalize(), y=0.05)
fig.supylabel(r"$\mathbb{P}(X \leq x)$", x=0.02, y=0.55)
fig.tight_layout()
fig.savefig(figpath / f"{target}-ppc-by-country-political.pdf")

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 3))
az.plot_bpv(idata, ax=ax, var_names=[target], kind="u_value")
ax.set_title(rf"{target.capitalize()}: $u$-values")
fig.tight_layout()
fig.savefig(figpath / f"{target}-bpv.pdf")

# %% ---------------------------------------------------------------------------------

waic = az.waic(idata)
print(waic)

# %% --------------------------------------------------------------------------------
