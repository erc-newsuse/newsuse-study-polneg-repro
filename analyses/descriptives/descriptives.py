# %% Prepare environment =============================================================
from collections.abc import Mapping

import matplotlib as mpl
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from newsuse.data import DataFrame

from project import config, paths

mpl.style.use(config.plotting.style)
mpl.rcParams.update(
    {
        k: v.make() if isinstance(v, Mapping) and "make!" in v else v
        for k, v in config.plotting.params.items()
    }
)

figpath = paths.figures / "descriptives"
figpath.mkdir(parents=True, exist_ok=True)

ci = f"{config.inference.conf*100:.0f}"
countries = config.countries.labels
colors = config.plotting.color


# %% Read data =======================================================================
dataset = (
    DataFrame.from_(paths.dataset)
    .rename(columns={"negativity": "negative"})
    .assign(likes=lambda df: df["interactions"])
)


# %% Basic information ===============================================================
#
# Post types' frequencies
#
(
    dataset.groupby(["type"])
    .size()
    .pipe(
        lambda s: pd.DataFrame(
            {
                "n": s,
                "p": np.round(s / s.sum() * 100, 1),
            }
        )
    )
    .sort_values("p", ascending=False)
)


# %% Outlet counts ===================================================================
n_outlets = (
    dataset.groupby(["country"])["name"]
    .nunique()
    .reset_index()
    .rename(columns={"name": "n_outlets"})
)

n_posts = dataset.groupby(["country"]).size().reset_index().rename(columns={0: "n_posts"})

table = n_outlets.merge(n_posts, on="country", how="left").set_index("country")

print(
    table.to_latex(
        escape=True,
        float_format="%.1f",
        formatters=[lambda x: f"{x:,}"] * table.shape[1],
        multicolumn_format="c",
    )
)


# %% Frequencies of political and negative news ======================================
def describe(data, *keys, **kwargs):
    data = data.copy()
    keys = list(keys)
    for k, v in kwargs.items():
        data[k] = v
        keys.append(k)
    posts = data.groupby(keys).size().reset_index(name="posts").set_index(keys)
    posts.insert(0, "outlets", data.groupby(keys)["pid"].nunique())
    posts["(%)"] = (posts["posts"] / posts["posts"].sum()) * 100
    freqs = data.groupby(keys)[["political", "negative"]].apply(
        lambda df: (df != "OTHER").mean()
    )
    freqs["negative | political"] = (
        data.query("political == 'POLITICAL'")
        .groupby(keys)["negative"]
        .apply(lambda s: (s != "OTHER").mean())
    )

    polneg = pd.concat(
        {
            "": posts,
            "% of": freqs * 100,
        },
        axis=1,
    )
    return polneg


polneg = pd.concat(
    [
        describe(dataset, country="overall"),
        describe(dataset, "country").loc[config.countries.order],
    ],
    axis=0,
)

print(
    polneg.rename(index={"overall": "Overall", **countries}).to_latex(
        escape=True,
        float_format="%.1f",
        formatters=[lambda x: f"{x:,}"] * polneg.shape[1],
        multicolumn_format="c",
    )
)


# %% Plot | Post types' frequencies ==================================================
keys = ["country", "political", "negative"]
counts = pd.concat(
    [dataset.assign(country="overall").groupby(keys).size(), dataset.groupby(keys).size()]
).to_frame("n")

data = (
    (counts / counts.groupby(level="country").sum())
    .reset_index()
    .assign(
        political=lambda df: df["political"].str.lower(),
        negative=lambda df: df["negative"].str.lower(),
    )
    .replace({"negative": {"other": "non-negative"}})
)

fig, axes = plt.subplot_mosaic(
    """
    AABCD
    AAEFG
    """,
    figsize=(8, 3),
)
palette = [colors.semantics[c] for c in ["negative", "other"]]

ax = axes["A"]
df = data.query("country == 'overall'")
spec = {"x": "political", "y": "n", "hue": "negative", "palette": palette}
sns.barplot(df, ax=ax, **spec)
ax.legend().set_label(None)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.set_ylim(0, 0.65)

gdata = (
    data.query("country != 'overall'")
    .set_index("country")
    .loc[[*countries]]
    .groupby(level="country", sort=False)
)
for gdf, ax in zip(gdata, "BCDEFG", strict=True):
    country, gdf = gdf
    ax = axes[ax]
    sns.barplot(gdf, ax=ax, **spec, legend=False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(countries[country])

for ax in axes.values():
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

for ax in "BCDEFG":
    axes[ax].sharex(axes["A"])
    axes[ax].sharey(axes["A"])
for ax in "CDFG":
    axes[ax].tick_params(labelleft=False)
for pair in ["BE", "CF", "DG"]:
    top, bot = pair
    axes[top].tick_params(labelbottom=False)

fig.supylabel("Post type frequency")
fig.tight_layout()
fig.savefig(figpath / "descriptives-post-types.pdf")


# %% Engagement statistics ===========================================================
keys = ["country", "political", "negative"]
metrics = ["likes", "comments", "shares"]


def IQR(s):
    return s.quantile(0.75) - s.quantile(0.25)


def Q1(s):
    return s.quantile(0.25)


def Q2(s):
    return s.quantile(0.5)


def Q3(s):
    return s.quantile(0.75)


parts = [
    dataset.assign(country="overall"),
    dataset.assign(country="overall", political="overall", negative="overall"),
    dataset.assign(political="overall", negative="overall"),
    dataset,
]
data = pd.concat(parts, ignore_index=True).assign(
    political=lambda df: df["political"].str.lower(),
    negativity=lambda df: df["negative"].str.lower(),
)

order = {"overall": 1, "other": 3}
inter = (
    data.groupby(["country", "political", "negativity"])[metrics]
    .agg(["mean", "std", "median", IQR])
    .sort_index(
        axis=0,
        level=["political", "negativity"],
        key=lambda idx: idx.map(lambda x: order.get(x, 2)),
    )
    .loc[["overall", *config.countries.order]]
    .rename(index={"overall": "Overall", **countries}, level="country")
)

# %% Engagement statistics | By country ==============================================
print(
    inter.drop("std", axis=1, level=1)
    .xs("overall", level="political")
    .xs("overall", level="negativity")
    .loc[[*countries.values(), "Overall"]]
    .to_latex(escape=True, float_format="%.1f")
)


# %% Engagement by country | CCDF plots ==============================================
data = dataset[[*keys, *metrics]].copy()
data = pd.concat([data.assign(country="overall"), data], ignore_index=True)
data[metrics] += 1

fig, axes = plt.subplots(ncols=len(metrics), figsize=(8, 2))

hue_order = ["overall", *countries]
palette = [colors.semantics[x] for x in hue_order]

for ax, metric in zip(axes.flat, metrics, strict=True):
    sns.ecdfplot(
        data,
        x=metric,
        hue="country",
        ax=ax,
        complementary=True,
        log_scale=True,
        hue_order=hue_order,
        palette=palette,
        legend=False,
    )
    ax.set_title(metric.title())
    ax.set_yscale("log")
    ax.set_xlabel(None)
    ax.set_ylabel(None)

axes.flatten()[0].set_ylabel("CCDF")
handles = [
    mpl.lines.Line2D([], [], lw=2, label=label, color=color)
    for label, color in zip(hue_order, palette, strict=True)
]
fig.legend(
    handles=handles,
    loc="center",
    framealpha=1.0,
    frameon=False,
    ncols=len(hue_order),
    bbox_to_anchor=(0.5, 0.0),
)
fig.tight_layout()
fig.savefig(figpath / "descriptives-engagement.pdf")


# %% Engagement by post type | CCDF plots ============================================
fig, axes = plt.subplots(ncols=len(metrics), figsize=(8, 2))

hue_order = ["OTHER", "NEGATIVE"]
palette = [colors.semantics[x.lower()] for x in hue_order]
df = data.query("country != 'overall'")

for ax, metric in zip(axes.flat, metrics, strict=True):
    sns.ecdfplot(
        df,
        x=metric,
        hue="negative",
        ax=ax,
        complementary=True,
        log_scale=True,
        hue_order=hue_order,
        palette=palette,
        legend=False,
    )
    ax.set_title(metric.title())
    ax.set_yscale("log")
    ax.set_xlabel(None)
    ax.set_ylabel(None)

axes.flatten()[0].set_ylabel("CCDF")
labels = [{"OTHER": "non-negative", "NEGATIVE": "negative"}[h] for h in hue_order]
handles = [
    mpl.lines.Line2D([], [], lw=2, label=label.lower(), color=color)
    for label, color in zip(labels, palette, strict=True)
]
axes.flatten()[0].legend(
    handles=handles,
    loc="best",
    framealpha=1.0,
    frameon=True,
)
fig.tight_layout()
fig.savefig(figpath / "descriptives-engagement-polneg.pdf")


# %% Outlet-level statistics =========================================================
(
    data.query("country == 'overall'")
    .groupby(["political", "negative"])[metrics]
    .agg(["mean", "median", IQR])
    .rename(str.title, axis=1, level=0)
    .rename(str.lower, axis=0, level=0)
    .rename(str.lower, axis=0, level=1)
    .rename_axis(["political", "negativity"], axis=0)
    .pipe(
        lambda df: print(
            df.to_latex(
                escape=True,
                float_format="%.1f",
                multicolumn_format="c",
            )
        )
    )
)

keys = ["country", "name"]
freq = (
    dataset.assign(
        political=lambda df: df["political"] != "OTHER",
        negative=lambda df: df["negative"] != "OTHER",
    )
    .groupby(keys)[["political", "negative"]]
    .mean()
    .transform(lambda x: x * 100)
)
freq.insert(0, "n", dataset.groupby(keys).size())
freq.columns = pd.MultiIndex.from_product([["posts"], freq.columns])

inter = dataset.groupby(keys)[metrics].agg(["mean", "median", IQR])

table = pd.concat([freq, inter], axis=1).loc[list(countries)].rename(index=countries)

print(table.to_latex(escape=True, float_format="%.1f"))

# %%
