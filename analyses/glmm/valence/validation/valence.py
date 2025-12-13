# %% ---------------------------------------------------------------------------------

import arviz as az
import matplotlib as mpl  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn as sns  # noqa
import xarray as xr
from newsuse.data import DataFrame

from project import config, paths
from project.inference import set_xindex, waic_metrics
from project.plotting import ArvizLabeller

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)

target = "valence"
opts = config.glmm.valence.targets[target]
support = config.categorical[target]

figpath = paths.figures / "glmm" / "valence" / "validation"
figpath.mkdir(parents=True, exist_ok=True)

labels = {
    "countries": config.categorical.country,
    "political": dict(enumerate(config.categorical.political)),
    target: config.categorical[target],
}

data = DataFrame.from_(paths.final)

# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}.nc")
idata = set_xindex(idata, [opts.index_col, *opts.predictors.fixed, *opts.predictors.groups])

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
    mean: bool = False,
    support: list[int] = tuple(support),
    **kwargs,
) -> plt.Axes:
    """Plot posterior predictive check with mean observed value."""
    idata = idata.copy()
    if kwargs:
        idata.observed_data = idata.observed_data.sel(**kwargs)
        idata.posterior_predictive = idata.posterior_predictive.sel(**kwargs)
    if ax is None:
        ax = plt.gca()
    az.plot_ppc(idata, ax=ax, mean=mean, legend=False)
    xticks = np.array(support) + 0.5
    ax.set_xticks(xticks, labels=list(support))
    ax.set_xlabel(None)
    return ax


# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=3, figsize=(7, 3))
plot_ppc(idata, axes[0])
plot_ppc(idata, axes[1], political=1)
plot_ppc(idata, axes[2], political=0)

axes[0].set_title("Overall")
axes[1].set_title("Political")
axes[2].set_title("Other")

axes[0].legend()

fig.tight_layout()
fig.savefig(figpath / f"{target}-ppc.pdf")

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(7, 4))

for ax, country in zip(axes.flat, labels["countries"], strict=True):
    plot_ppc(idata, ax=ax, country=country)
    ax.set_title(country.upper())

axes[0, 0].legend()

fig.tight_layout()
fig.savefig(figpath / f"{target}-ppc-by-country.pdf")

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=6, nrows=2, figsize=(8, 3))

for axrow, pol in zip(axes, labels["political"], strict=True):
    for ax, country in zip(axrow, labels["countries"], strict=True):
        plot_ppc(idata, ax=ax, country=country, political=pol)

for ax, country in zip(axes[0], labels["countries"], strict=True):
    ax.set_title(country.upper())
    ax.set_xticks([])
for ax, pol in zip(axes[:, 0], labels["political"], strict=True):
    ax.set_ylabel(labels["political"][pol])

fig.supxlabel(target.capitalize(), y=0.05)
fig.tight_layout()
fig.savefig(figpath / f"{target}-ppc-by-country-political.pdf")

# %% ---------------------------------------------------------------------------------

obs_freqs = data[target].value_counts(normalize=True).sort_index()
waic = az.waic(idata)
metrics = waic_metrics(waic, null_elpd=np.log(obs_freqs).mean())
print(metrics)

# %% --------------------------------------------------------------------------------
