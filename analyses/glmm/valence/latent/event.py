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
from newsuse.data import DataFrame
from transformers import AutoModel

import project.model  # noqa
from project import config, paths
from project.inference import StatsAccessor, set_xindex
from project.model.ordinal import ordinal_probs
from project.plotting import annotate_ci, make_legend

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

alpha = 1 - az.rcParams["stats.ci_prob"]
conf = (1 - alpha) * 100
q0, q1 = alpha / 2, 1 - alpha / 2

target = "event"
support = [*config.categorical[target]]
opts = config.glmm.valence.targets[target]

figpath = paths.figures / "glmm" / "valence" / "latent"
figpath.mkdir(parents=True, exist_ok=True)

countries = config.categorical.country
political = dict(enumerate(config.categorical.political))

# %% Load the valence transformer ----------------------------------------------------

domain = "valence"
hyper = DataFrame.from_(paths.proc / f"{domain}-hyper.parquet")
best_model = hyper.loc[hyper.value.idxmax()].params_base
model = AutoModel.from_pretrained(paths.ml / "models" / domain / best_model)

biases = {t: head.ordinal.bias.detach().cpu().numpy() for t, head in model.heads.items()}[
    target
]


@xr.register_dataarray_accessor("stats")
class _StatsAccessor(StatsAccessor):
    alpha = config.inference.alpha


@xr.register_dataset_accessor("stats")
class _DatasetAccessor(StatsAccessor):
    alpha = config.inference.alpha


# %% ---------------------------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}-latent.nc")
idata = set_xindex(idata, [opts.index_col, *sum(opts.predictors.values(), start=[])])
epred = az.extract(idata, group="posterior_epred")
ppd = az.extract(idata, group="posterior_predictive")
# Average out week effects to focus on main effects
if "weekend" in epred.coords:
    epred = epred.stats.marginalize("weekend")

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
        marker_offset=0.25,
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
        marker_offset=0.35,
        digits=2,
        fontsize=6,
    )

legend = make_legend(fig, (0.06, 0.95), loc="upper left")
fig.legends = [legend]

ax.set_xticklabels([])
ax.set_xticks([*countries], [*countries.values()])

fig.tight_layout()
fig.savefig(figpath / f"{target}-latent-political-country.pdf")

# %% Derive expected posterior class probabilities -----------------------------------

coords = ppd.coords.copy()
coords[target] = config.categorical[target]

weight_cols = [
    "outlet",
]

probs = (
    xr.DataArray(
        np.swapaxes(ordinal_probs((ppd[target].values + biases[..., None, None]).T), 0, 1),
        coords=coords,
        dims=coords.dims,
    )
    .isel(sample=slice(0, None, 10))  # Subsample for memory reasons
    .to_dataframe(name="prob")["prob"]
    .groupby(["country", "political", "weekend", "outlet", "isotime", target, "draw"])
    .mean()
    .groupby(["country", "political", "outlet", "isotime", target, "draw"])
    .mean()
    .groupby(["country", "political", *weight_cols, target, "draw"])
    .mean()
    .reset_index()
    .groupby(["country"])
    .apply(
        lambda df: df.assign(weight=lambda d: 1 / len(d[weight_cols].drop_duplicates())),
        include_groups=False,
    )
    .reset_index("country", drop=False)
    .set_index(["political", "country", *weight_cols, target, "draw"])
)

# %% ---------------------------------------------------------------------------------

probs_political_country = probs.groupby(
    ["political", "country", "outlet", "draw", target]
).mean()

probs_country = probs_political_country.groupby(
    ["country", "outlet", "draw", target]
).mean()

# %% ---------------------------------------------------------------------------------

# diffs_overall = (
(
    probs_political_country.groupby(["outlet", "draw", "event"])
    .apply(
        lambda df: pd.Series(
            {"prob": df["prob"].diff().iloc[-1], "weight": df["weight"].mean()}
        )
    )
    .groupby(["event", "draw"])
    .apply(lambda df: np.average(df["prob"], weights=df["weight"]))
    .groupby("event")
    .quantile([q0, 0.5, q1])
    .unstack(-1)
    .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    .reset_index()
    # .groupby(target)
    # .apply(
    #     lambda df: pd.Series(
    #         np.quantile(
    #             df["prob"],
    #             [q0, 0.5, q1],
    #             weights=df["weight"],
    #             method="inverted_cdf",
    #         ),
    #         index=["lb", "median", "ub"],
    #     )
    # )
    # probs
    # .apply(weighted_mean, "prob", "weight")
    # .groupby(["draw", target])
    # .diff()
    # .dropna()
    # .droplevel(0)
    # .groupby([target])
    # .quantile([q0, 0.5, q1])
    # .unstack(-1)
    # .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    # .reset_index()
)

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))
df = (
    probs_political_country
    # .groupby(["political", "draw", target])
    # .apply(lambda df: np.average(df["prob"], weights=df["weight"]))
    # .groupby(["political", target])
    # .quantile([q0, 0.5, q1])
    # .unstack(-1)
    # .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    # .reset_index()
    .groupby(["political", target]).apply(
        lambda df: pd.Series(
            np.quantile(
                df["prob"],
                [q0, 0.5, q1],
                method="inverted_cdf",
                weights=df["weight"],
            ),
            index=["lb", "median", "ub"],
        )
    )
)
(
    so.Plot(
        df,
        x=target,
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

# %% TABLES --------------------------------------------------------------------------

az.summary(idata)

# %% ---------------------------------------------------------------------------------
