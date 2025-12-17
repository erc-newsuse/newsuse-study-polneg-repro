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
from newsuse.data import DataFrame
from transformers import AutoModel

import project.model  # noqa
from project import config, paths
from project.inference import StatsAccessor, set_xindex
from project.model.ordinal import ordinal_probs
from project.plotting import annotate_ci

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

figpath = paths.figures / "glmm" / "valence" / target
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

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}.nc")
idata = set_xindex(idata, [opts.index_col, *sum(opts.predictors.values(), start=[])])
epred = az.extract(idata, group="posterior_epred")
ppd = az.extract(idata, group="posterior_predictive")
# Average out week effects to focus on main effects
if "weekend" in epred.coords:
    epred = epred.stats.marginalize("weekend")

# %% Derive posterior expectation distributions for class probabilities --------------

coords = ppd.coords.copy()
coords[target] = config.categorical[target]
probs = (
    xr.DataArray(
        np.swapaxes(ordinal_probs((ppd[target].values + biases[..., None, None]).T), 0, 1),
        coords=coords,
        dims=coords.dims,
    )
    .to_dataframe(name="prob")
    .reset_index()
    .groupby(["country", "political", "draw", target])
    .sample(n=100, random_state=303)
    .reset_index(drop=True)
    .set_index(["country", "political", "draw", target])["prob"]
    .groupby(["political", "country", "draw", target])
    .mean()
)

# %% ---------------------------------------------------------------------------------

probs_overall = (
    probs.groupby(["political", target])
    .quantile([q0, 0.5, q1])
    .unstack(-1)
    .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    .reset_index()
)
probs_overall.insert(0, "country", "overall")

# %% ---------------------------------------------------------------------------------

probs_country = (
    probs.groupby(["country", "political", target])
    .quantile([q0, 0.5, q1])
    .unstack(-1)
    .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    .reset_index()
    .set_index("country")
    .loc[[*countries]]
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

probs_quantiles = pd.concat([probs_overall, probs_country], ignore_index=True)

# %% ---------------------------------------------------------------------------------

probs_diff = probs.groupby(["country", "draw", target]).diff().dropna().droplevel(0)

# %% ---------------------------------------------------------------------------------

eff_political = (
    probs_diff.groupby(target)
    .quantile([q0, 0.5, q1])
    .unstack(-1)
    .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

eff_political_country = (
    probs_diff.groupby(["country", target])
    .quantile([q0, 0.5, q1])
    .unstack(-1)
    .rename(columns={q0: "lb", 0.5: "median", q1: "ub"})
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6))
df = probs_overall
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

eff = eff_political.copy()
eff["anchor"] = df[["lb", "ub"]].mean(axis=1)
for value, row in eff.iterrows():
    value = int(value) - 1
    annotate_ci(
        ax,
        [value, row["anchor"]],
        row[["lb", "ub"]],
        prefix=r"$\Delta$ ",
        marker_offset=0.1,
        fontsize=7,
        zorder=100,
        show_box=False,
    )

ax.set_xlabel(None)
ax.set_ylabel(None)

ax.set_xlabel(target.capitalize(), fontsize="x-large")
ax.set_ylabel("Posterior class probability", fontsize="x-large")
ax.xaxis.set_ticks(support)

fig.legends.clear()
fig.tight_layout()
fig.savefig(figpath / f"{target}-effects-overall.pdf")

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(figsize=(9, 6), nrows=2, ncols=3)

for ax, country in zip(axes.flat, countries, strict=True):
    df = probs_country[probs_country.country == country]
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
    eff = eff_political_country.query("country == @country").reset_index(drop=True).copy()
    eff["anchor"] = df.groupby(target)[["lb", "ub"]]
    for value, row in eff.iterrows():
        value = int(value) - 1
        anchor = df.query(f"{target} == @value")["ub"].max()
        annotate_ci(
            ax,
            [value, anchor],
            row[["lb", "ub"]],
            prefix=r"$\Delta$ ",
            marker_offset=0.05,
            fontsize=7,
            zorder=100,
            show_box=False,
        )
    ax.set_title(config.categorical.country[country])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.xaxis.set_ticks(support)

fig.supylabel("Posterior class probability", fontsize="x-large")
fig.supxlabel(target.capitalize(), y=0.02, fontsize="x-large")
fig.legends.clear()

fig.tight_layout()
fig.savefig(figpath / f"{target}-effects-by-country.pdf")

# %% TABLES --------------------------------------------------------------------------

# %% Effects

tab = (
    pd.concat(
        [
            eff_political.assign(country="overall"),
            eff_political_country,
        ],
        ignore_index=True,
    )
    .set_index(["country", target])
    .loc[["overall", *countries]]
)
# Print 'tab' nicely as latex table
print(tab.to_latex(float_format="%.3f"))

# %% Model


def sanitize_param_name(name: str) -> dict[str, str]:
    import re

    name = re.sub(r"country(\w+)", r"\1", name)
    name = re.sub(r"mu(\d+)", r"$\\mu_{\1} \\mid$", name)
    name = name.replace("b_", r"$b$ ")
    name = re.sub(r"_+(Intercept|political)", r" \1", name)
    name = re.sub(r"(Intercept|political)_+", r"\1 ", name)
    name = re.sub(r"\s*\\mid\$\s*$", r"$", name)
    name = name.replace("$_", "$ ")
    name = name.replace("$b$", "[b]")
    name = name.replace("sd_", "[sd] ")
    name = name.removeprefix("Intercept ")
    if name.startswith("[sd]"):
        name = re.sub(r"\s*\\mid\$\s*Intercept", r"$", name)
    if not name.startswith("["):
        name = "[dist] " + name
    name = name.replace("[", "")
    name = name.replace("]", "")
    name = name.replace("Intercept", "us")
    name = re.sub(r"political$", "political:us", name)
    name = re.sub(r"sigma(\d+)", r"$\\sigma_{\1}$", name)
    name = re.sub(r"theta(\d+)", r"$\\theta_{\1}$", name)
    name = re.sub(r"\s*\\mid\s*", r"", name)
    group, param = name.split(" ", 1)
    comp = re.search(r"\s*\$\\(mu|sigma|theta)_(\{\d+\})\$\s*", param)
    comp = f"{comp.group(0).strip()}" if comp else None
    if comp:
        param = param.replace(comp, "").strip()
    return {"group": group, "comp": comp, "param": param}


model_tab = (
    az.summary(idata)
    .reset_index(names=["param"])[:-2]
    .assign(param=lambda df: df["param"].apply(sanitize_param_name))
    .assign(
        group=lambda df: df["param"].apply(lambda x: x["group"]),
        component=lambda df: df["param"].apply(lambda x: x["comp"]),
        term=lambda df: df["param"].apply(lambda x: x["param"]),
    )
    .drop(columns=["r_hat", "param"])
    .set_index(["group", "component", "term"])
    .groupby(level=["group", "component"])
    .apply(
        lambda df: df.sort_index(
            level="term", kind="stable", key=lambda x: x.str.count(":")
        ),
        include_groups=False,
    )
    .droplevel([0, 1])
    .loc[["b", "sd", "dist"]]
    .rename(
        columns={
            f"hdi_{alpha/2:.1%}": f"{alpha/2:.1%} HDI",
            f"hdi_{1 - alpha/2:.1%}": f"{1-alpha/2:.1%} HDI",
            "mcse_mean": "MCSE (mean)",
            "mcse_sd": "MCSE (sd)",
            "ess_bulk": "ESS (bulk)",
            "ess_tail": "ESS (tail)",
        }
    )
)

# Print nicely as latex table
print(model_tab.to_latex(float_format="%.3f"))

# %% ---------------------------------------------------------------------------------
