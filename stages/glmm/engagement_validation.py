# %% ---------------------------------------------------------------------------------

import os
import re

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from project import config, paths
from project.bayes import rebuild_model

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)

TARGET = (
    os.environ.get("TARGET") or input("Enter target (reactions): ").strip() or "reactions"
)
opts = config.glmm.engagement.targets[TARGET]

figpath = paths.figures / "engagement" / "validation" / opts.response
figpath.mkdir(parents=True, exist_ok=True)

alpha = 1 - az.rcParams["stats.ci_prob"]

# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "engagement" / f"{opts.response}.nc")
model = rebuild_model(idata)

# %% ---------------------------------------------------------------------------------

terms_rx = {
    "fixed": [
        r"^threshold$",
        r"^(event)?:(sentiment)?:?(political)?:?(country)?$",
    ],
    "group (sd)": [r"\|.*_sigma$"],
    "group": [r"\|(outlet|country:year:month)$"],
}
terms_opts = {
    "var_names": sum([v for k, v in terms_rx.items() if k != "group"], []),
    "filter_vars": "regex",
}

terms_fixed = sorted(
    [
        t
        for t in idata.posterior.data_vars
        if any(re.match(rx, t) for rx in terms_rx["fixed"])
    ]
)

# %% ---------------------------------------------------------------------------------

model.build()
axes = model.plot_priors()
fig = axes.flatten()[0].figure
fig.tight_layout()
fig.savefig(figpath / f"{opts.response}-priors.pdf")
print(model)

# %% ---------------------------------------------------------------------------------

stats = {
    kind: az.summary(idata, var_names=terms, filter_vars="regex")
    for kind, terms in terms_rx.items()
}
stats["group (stats)"] = stats.pop("group").describe()[1:]

stats = pd.concat(stats, names=["kind"]).rename(
    index=lambda s: str(s).split("_", 1)[0], level=-1
)

stats.head(len(stats))

# %% ---------------------------------------------------------------------------------


def bold_row(row: pd.Series, *, threshold: float = 1.01) -> list[str]:
    if row[r"$\hat{R}$"] > threshold:
        return ["bfseries:--rwrap"] * len(row)
    return [""] * len(row)


# Print 'stat' nicely in LaTeX format
print(
    stats.pipe(lambda df: df[~df.index.duplicated(keep="first")])
    .rename(
        columns={
            "mean": "Mean",
            "sd": "SD",
            f"hdi_{(conf := alpha / 2 * 100):.1f}%": rf"HDI {conf:.1f}\%",
            f"hdi_{(100 - conf):.1f}%": rf"HDI {100 - conf:.1f}\%",
            "mcse_mean": "MCSE (mean)",
            "mcse_sd": "MCSE (SD)",
            "ess_bulk": "ESS (bulk)",
            "ess_tail": "ESS (tail)",
            "r_hat": r"$\hat{R}$",
        }
    )
    .rename(index=lambda s: s.replace("%", r"\%"), level=-1)
    .rename(index=lambda s: s.replace("_", r"\_"), level=-1)
    .style.apply(bold_row, axis=1)
    .format(precision=2, escape="latex")
    .to_latex(
        convert_css=False,
        multirow_align="t",
        hrules=True,
    )
)

# %% ---------------------------------------------------------------------------------

bad = az.summary(idata).query("r_hat > 1.01").sort_values("r_hat", ascending=False)
bad.head(len(bad))

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    figsize=(figsize := (24, 8)),
    nrows=2,
    ncols=len(terms_fixed),
)
axes = az.plot_trace(
    idata,
    var_names=terms_fixed,
    combined=True,
    figsize=figsize,
    legend=False,
    axes=axes.T,
)
fig = axes.flatten()[0].figure
fig.tight_layout()

fig.savefig(figpath / f"{opts.response}-trace.pdf")

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

fig.savefig(figpath / f"{opts.response}-ess.pdf")

# # %% ---------------------------------------------------------------------------------


# def plot_ppc(
#     idata: az.InferenceData,
#     ax: plt.Axes | None = None,
#     **kwargs,
# ) -> plt.Axes:
#     """Plot posterior predictive check with mean observed value."""
#     kwargs = {
#         "kind": "cumulative",
#         "legend": False,
#         "mean": False,
#         "num_pp_samples": 10,
#         **kwargs,
#     }
#     az.plot_ppc(idata, ax=ax, **kwargs)
#     ax.set_xlabel(None)
#     ax.set_ylabel(None)
#     return ax


# # %% ---------------------------------------------------------------------------------

# bys = ["country"]
# for _by in bys:
#     fig, axes = plt.subplots(
#         ncols=(ncols := len(labels[_by])),
#         nrows=(nrows := 2),
#         figsize=((height := 3) * ncols, height * nrows),
#     )
#     for axrow, political in zip(axes, [0, 1], strict=True):
#         for ax, byval in zip(axrow.flat, labels[_by], strict=True):
#             sel = {_by: byval, "political": political}
#             plot_ppc(idata.sel(**sel), ax=ax)
#             ax.set_xticks(labels[opts.response])
#             ax.set_xlabel(None)
#             ax.set_ylabel(None)
#             if ax in axes[0]:
#                 ax.set_title(labels[_by][byval])
#         axrow[0].set_ylabel(
#             "Non-Political" if political == 0 else "Political", fontsize="xx-large"
#         )
#     fig.supxlabel(opts.response.capitalize(), y=0.0, fontsize="xx-large")
#     fig.supylabel(r"$\mathbb{P}(X \leq x)$", x=0.01, y=0.55, fontsize="xx-large")
#     fig.tight_layout()
#     fig.savefig(figpath / f"{opts.response}-ppc-by-{_by}.pdf")

# # %% ---------------------------------------------------------------------------------

# fig, ax = plt.subplots(figsize=(7, 3))
# az.plot_bpv(idata, ax=ax, var_names=[opts.response], kind="u_value")
# ax.set_title(rf"{opts.response.capitalize()}: $u$-values")
# ax.set_ylim(0.8, 1.2)
# fig.tight_layout()
# fig.savefig(figpath / f"{opts.response}-bpv.pdf")

# # %% ---------------------------------------------------------------------------------

# waic = az.waic(idata)
# print(waic)

# # %% --------------------------------------------------------------------------------
