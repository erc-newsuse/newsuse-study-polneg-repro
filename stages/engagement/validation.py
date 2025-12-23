# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from newsuse.data import DataFrame

from project import config, paths
from project.bayes import index_idata

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)

target = os.environ.get("TARGET")
if not target:
    target = input("Enter target (reactions): ").strip() or "reactions"
opts = config.glmm.engagement.targets[target]

figpath = paths.figures / "glmm" / "engagement" / "validation" / target
figpath.mkdir(parents=True, exist_ok=True)

labels = {
    "countries": config.categorical.country,
    "political": dict(enumerate(config.categorical.political)),
    "event": config.categorical.event,
    "sentiment": config.categorical.sentiment,
}

data = DataFrame.from_(paths.final)

# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "engagement" / f"{target}.nc")
idata = index_idata(idata, ["key", *opts.predictors.fixed, *opts.predictors.groups])
terms_fixed = [t for t in idata.posterior.data_vars if "|" not in t]

# %% ---------------------------------------------------------------------------------

stats = az.summary(idata)
stats.describe()

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 21))
bad = stats.query("r_hat > 1.01").sort_values("r_hat", ascending=False)
sns.scatterplot(
    data=bad.reset_index(),
    x="r_hat",
    y="index",
    ax=ax,
)
ax.set_xlabel(r"$\hat{r}$")

# %% ---------------------------------------------------------------------------------

axes = az.plot_trace(
    idata,
    var_names=terms_fixed,
    combined=True,
    figsize=(10, 30),
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
plot_ppc(idata.sel(political=1), ax=axes[1])
plot_ppc(idata.sel(political=0), ax=axes[2])

axes[0].set_title("Overall")
axes[1].set_title("Political")
axes[2].set_title("Other")

for ax in axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")

fig.tight_layout()
fig.supxlabel(target.capitalize(), y=-0.03)
fig.supylabel(r"$\mathbb{P}(X \leq x)$", x=-0.02)
fig.savefig(figpath / f"{target}-ppc.pdf")

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(7, 4))

for ax, country in zip(axes.flat, labels["countries"], strict=True):
    plot_ppc(idata.sel(country=country), ax=ax)
    ax.set_title(country.upper())

for ax in axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")

fig.tight_layout()
fig.supxlabel(target.capitalize(), y=-0.03)
fig.supylabel(r"$\mathbb{P}(X \leq x)$", x=-0.02)
fig.savefig(figpath / f"{target}-ppc-by-country.pdf")

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(7, 7))

for axrow, event in zip(axes, config.categorical.event, strict=True):
    for ax, sentiment in zip(axrow, config.categorical.sentiment, strict=True):
        plot_ppc(idata.sel(event=event, sentiment=sentiment), ax=ax)
        ax.set_title(f"Event = {event}, Sentiment = {sentiment}")
        ax.set_xscale("log")
        ax.set_yscale("log")

fig.tight_layout()
fig.supxlabel(target.capitalize(), y=-0.03)
fig.supylabel(r"$\mathbb{P}(X \leq x)$", x=-0.02)
fig.savefig(figpath / f"{target}-ppc-by-valence.pdf")

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 3))
az.plot_bpv(idata, ax=ax, var_names=[target], kind="u_value")
ax.set_title(rf"{target.capitalize()}: $u$-values")
fig.tight_layout()
fig.savefig(figpath / f"{target}-bpv.pdf")

# %% ---------------------------------------------------------------------------------

waic = az.waic(idata)
print(waic)

# %% ---------------------------------------------------------------------------------
