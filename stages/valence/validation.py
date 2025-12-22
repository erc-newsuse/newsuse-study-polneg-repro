# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from newsuse.data import DataFrame

from project import config, paths
from project.bayes import index_idata

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)

target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target (event): ").strip() or "event"

by = os.environ.get("BY")
if by is None:
    by = input("Enter grouping variable: ").strip() or ""

opts = config.glmm.valence.targets[target]
support = config.categorical[target]

figpath = paths.figures / "valence" / "validation" / target
figpath.mkdir(parents=True, exist_ok=True)

labels = {
    "country": config.categorical.country,
    "political": dict(enumerate(config.categorical.political)),
    target: config.categorical[target],
}
if by:
    labels[by] = config.categorical[by]

data = DataFrame.from_(paths.final)

# %% ---------------------------------------------------------------------------------

fname = f"{target}.nc" if not by else f"{target}-{by}.nc"
idata = az.from_netcdf(paths.glmm / "valence" / fname)
idata = index_idata(idata, ["key", *opts.predictors.fixed, *opts.predictors.groups])
terms_fixed = [t for t in idata.posterior.data_vars if "|" not in t]

# %% ---------------------------------------------------------------------------------

stats = az.summary(idata)
stats.describe()

# %% ---------------------------------------------------------------------------------

bad = stats.query("r_hat > 1.01").sort_values("r_hat", ascending=False)
bad.head(len(bad))

# %% ---------------------------------------------------------------------------------

axes = az.plot_trace(
    idata,
    var_names=terms_fixed,
    combined=True,
    figsize=(6, 6),
)
fig = axes.flatten()[0].figure
fig.tight_layout()

axes[0, 0].set_title("Posterior density", fontsize="x-large")
axes[0, 1].set_title("Trace plot", fontsize="x-large")

fig.savefig(figpath / f"{target}-trace.pdf")

# %% ---------------------------------------------------------------------------------

axes = az.plot_ess(
    idata,
    var_names=terms_fixed,
    figsize=(15, 15),
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
    var_names=terms_fixed,
    figsize=(18, 18),
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
        "num_pp_samples": 10,
        **kwargs,
    }
    az.plot_ppc(idata, ax=ax, **kwargs)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    return ax


# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=3, figsize=(7, 3))
plot_ppc(idata, ax=axes[0])
plot_ppc(idata.sel(political=0), ax=axes[1])
plot_ppc(idata.sel(political=1), ax=axes[2])

axes[0].set_title("Overall")
axes[1].set_title("Non-Political")
axes[2].set_title("Political")

fig.tight_layout()
fig.supxlabel(target.capitalize(), y=-0.03)
fig.supylabel(r"$\mathbb{P}(X \leq x)$", x=-0.02)
fig.savefig(figpath / f"{target}-ppc.pdf")

# %% ---------------------------------------------------------------------------------

bys = ["country"] if not by else [by]
for _by in bys:
    fig, axes = plt.subplots(
        ncols=(ncols := 3),
        nrows=(nrows := int(np.ceil(len(labels[_by]) / ncols))),
        figsize=(3 * ncols, 3 * nrows),
    )

    for ax, byval in zip(axes.flat, labels[_by], strict=True):
        sel = {_by: byval}
        plot_ppc(idata.sel(**sel), ax=ax)

    for ax, byval in zip(axes.flat, labels[_by], strict=True):
        ax.set_title(byval.upper())
        ax.set_xticks([])

    fig.supxlabel(target.capitalize(), y=0.05)
    fig.supylabel(r"$\mathbb{P}(X \leq x)$", x=0.02, y=0.55)
    fig.tight_layout()
    fig.savefig(figpath / f"{target}-ppc-by-{_by}.pdf")

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 3))
az.plot_bpv(idata, ax=ax, var_names=[target], kind="u_value")
ax.set_title(rf"{target.capitalize()}: $u$-values")
ax.set_ylim(0.8, 1.2)
fig.tight_layout()
fig.savefig(figpath / f"{target}-bpv.pdf")

# %% ---------------------------------------------------------------------------------

waic = az.waic(idata)
print(waic)

# %% --------------------------------------------------------------------------------
