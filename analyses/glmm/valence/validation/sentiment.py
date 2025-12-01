# %% ---------------------------------------------------------------------------------

import arviz as az
import matplotlib as mpl  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn as sns  # noqa

from project import config, paths
from project.plotting import ArvizLabeller

az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)

target = "sentiment"

fixef_opts = {
    "var_names": "b_",
    "filter_vars": "like",
}

figpath = paths.figures / "glmm" / "valence" / "validation"
figpath.mkdir(parents=True, exist_ok=True)

# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}.nc")

# %% ---------------------------------------------------------------------------------

axes = az.plot_trace(
    idata, **fixef_opts, combined=True, figsize=(8, 24), labeller=ArvizLabeller()
)
fig = axes.flatten()[0].figure
fig.tight_layout()

for ax in axes[:, 0]:
    title = ax.get_title()
    ax.set_ylabel(title)
for ax in axes.flat:
    ax.set_title(None)

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

ax = az.plot_ppc(idata, mean=False)
ax.set_xticks([-0.5, 0.5, 1.5], labels=[-1, 0, 1])
ax.set_xlabel(target.capitalize())
fig = ax.figure
fig.tight_layout()

fig.savefig(figpath / f"{target}-ppc.pdf")

# %% ---------------------------------------------------------------------------------

ax = az.plot_bpv(idata, kind="u_value")
fig = ax.figure
ax.set_title(target.capitalize())
ax.set_xlabel("Data point index")
ax.set_ylabel("U-value")
fig.tight_layout()

fig.savefig(figpath / f"{target}-bpv.pdf")

# %% ---------------------------------------------------------------------------------
