# %% ---------------------------------------------------------------------------------

import os
from typing import Any

import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from newsuse.data import DataFrame

import project.model  # noqa
from project import config, paths

mpl.rcParams.update(config.plotting.params)

target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target name (event): ").strip() or "event"

support = np.asarray([*config.categorical[target]])

quality = ["low", "medium", "high"]
ideology = ["left", "center", "right"]
valence = (
    [*config.categorical.valence]
    if target == "valence"
    else ["negative", "neutral", "positive"]
)

domain = "valence"

figpath = paths.figures / "descriptives" / domain
figpath.mkdir(parents=True, exist_ok=True)


# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final).assign(
    country=lambda df: pd.Categorical(
        df["country"],
        categories=list(config.categorical.country),
    ),
    quality=lambda df: pd.Categorical(df["quality"], categories=quality, ordered=True),
    ideology=lambda df: pd.Categorical(df["ideology"], categories=ideology),
)

# %% ---------------------------------------------------------------------------------


def plot_frequencies(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str = "proportion",
    hue: str | None = None,
    palette: list[str] = config.plotting.color.political,
    text: bool = True,
    text_xshift: float | None = None,
    text_yshift: float = 0.05,
    fontsize: str = "small",
    **kwargs: Any,
) -> None:
    kwargs = {"palette": palette, **kwargs} if hue else {"color": "black", **kwargs}
    sns.barplot(data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    ax.set_title(x.capitalize())
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    cols = [x, y] if hue is None else [x, y, hue]
    if text_xshift is None:
        text_xshift = 2 if x == "valence" else 1
    if hue is not None and isinstance(data[hue].dtype, pd.CategoricalDtype):
        data = data.copy()
        data[hue] = data[hue].cat.codes
    for _, row in data[cols].iterrows():
        if text:
            ax.text(
                row[x] + text_xshift + (0.0 if hue is None else 0.2 * (-1 + 2 * row[hue])),
                row[y] + text_yshift,
                f"{row[y]:.1%}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                path_effects=[mpl.patheffects.withStroke(linewidth=3, foreground="white")],
            )


# %% --------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

ax = axes[0]
dist = data[target].value_counts(normalize=True).sort_index().reset_index()
plot_frequencies(ax, dist, target)
ax.set_title("Overall", fontsize="xx-large")
ax.set_ylabel(None)

for ax, (country, df) in zip(
    axes.flatten()[1:],
    data.groupby("country", observed=True),
    strict=True,
):
    dist = df[target].value_counts(normalize=True).sort_index().reset_index()
    plot_frequencies(ax, dist, target, legend=False)
    ax.set_title(config.categorical.country[country], fontsize="xx-large")

for ax in axes.flat:
    ax.set_xticks(support - support.min(), labels=valence)

title = "Combined valence" if target == "valence" else f"{target.capitalize()} valence"
fig.suptitle(title, fontsize="xx-large", x=0.0, y=0.95, ha="left")
fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies.pdf")

# %% --------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

ax = axes[0]
dist = pd.concat(
    {
        pol: data.query(f"political == {pol}")[[target]].value_counts(normalize=True)
        for pol in [0, 1]
    },
    names=["political"],
).reset_index()
plot_frequencies(ax, dist, target, hue="political", legend=True)
ax.set_title(None)
ax.set_ylabel(None)
legend = ax.get_legend()
ax.legend(
    title=None,
    frameon=False,
    handles=legend.legend_handles,
    labels=[
        "non-political" if int(h.get_label()) == 0 else "political"
        for h in legend.legend_handles
    ],
)

for ax, (_, df) in zip(
    axes.flatten()[1:],
    data.groupby("country", observed=True),
    strict=True,
):
    dist = pd.concat(
        {
            pol: df.query(f"political == {pol}")[[target]].value_counts(normalize=True)
            for pol in [0, 1]
        },
        names=["political"],
    ).reset_index()
    plot_frequencies(ax, dist, target, hue="political", legend=False)
    ax.set_title(None)

for ax in axes.flat:
    ax.set_xticks(support - support.min(), labels=valence)

fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies-political.pdf")

# %% ---------------------------------------------------------------------------------

for by in ["quality", "ideology"]:
    df = (
        data.groupby(["country", "political", by], observed=True)[target]
        .value_counts(normalize=True)
        .sort_index()
        .reset_index(name="proportion")
    )

    nrows = df["political"].nunique()
    ncols = len(config.categorical.country)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

    for ax, ((political, country), gdf) in zip(
        axes.flat,
        df.groupby(["political", "country"], observed=True),
        strict=True,
    ):
        plot_frequencies(
            ax,
            gdf,
            x=target,
            y="proportion",
            hue=by,
            palette=config.plotting.color[by],
            text_xshift=0.2,
            text_yshift=0.02,
            fontsize="x-small",
            legend=ax is axes.flatten()[0],
            text=False,
        )
        pol = "political" if political == 1 else "non-political"
        ax.set_title(f"{config.categorical.country[country]} | {pol}")
        if (legend := ax.get_legend()) is not None:
            legend.set_title(None)
            legend.set_frame_on(False)

    fig.tight_layout()
    fig.savefig(figpath / f"{target}-frequencies-by-{by}.pdf")

# %% ---------------------------------------------------------------------------------
