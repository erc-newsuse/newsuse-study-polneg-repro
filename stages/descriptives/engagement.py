# %% ---------------------------------------------------------------------------------

import warnings
from typing import Any

import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from newsuse.data import DataFrame

from project import config, paths

warnings.filterwarnings("ignore", category=UserWarning)

mpl.rcParams.update(config.plotting.params)

targets = ["event", "sentiment", "valence"]
quality = ["low", "medium", "high"]
ideology = ["left", "center", "right"]

figpath = paths.figures / "descriptives"
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


def plot_ecdf(data: pd.DataFrame, x: str, ax: mpl.axes.Axes, **kwargs: Any) -> None:
    """Plot empirical complementary cumulative distribution function (CCDF)."""
    if (hue := kwargs.get("hue")) and (palette := config.plotting.color.get(hue)):
        kwargs["palette"] = palette
    is_first_axis = ax is np.asarray(ax.figure.axes).flatten()[0]
    kwargs = {
        "complementary": True,
        "log_scale": (True, True),
        "legend": is_first_axis,
        **kwargs,
    }
    sns.ecdfplot(data, x=x, ax=ax, **kwargs)
    ax.set_xlabel(x.capitalize())
    if is_first_axis:
        ax.set_ylabel(r"$\mathbb{P}(X > x)$")
        if legend := ax.get_legend():
            legend.set_frame_on(False)
    else:
        ax.set_ylabel(None)


# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=3, figsize=(7, 2), sharex=True, sharey=True)

for ax, metric in zip(axes.flat, config.engagement, strict=True):
    plot_ecdf(data, metric, ax, hue="political", legend=False)
    ax.set_title(metric.capitalize())
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    if ax is axes[0]:
        ax.set_ylabel(r"$\mathbb{P}(X > x)$")
        # Add custom legend for political
        handles = [
            mpl.lines.Line2D([], [], color=color, label=label)
            for label, color in zip(
                config.categorical.political,
                config.plotting.color.political,
                strict=True,
            )
        ]
        ax.legend(handles=handles, title=None, frameon=False, fontsize="small")

fig.tight_layout()
fig.savefig(figpath / "engagement-ecdf.pdf")

# %% ---------------------------------------------------------------------------------

for by in ["country", "event", "sentiment"]:
    fig, axes = plt.subplots(ncols=3, figsize=(7, 2), sharex=True, sharey=True)
    for ax, metric in zip(axes.flat, config.engagement, strict=True):
        sns.boxplot(
            data=data,
            x=by,
            y=metric,
            hue="political",
            ax=ax,
            palette=config.plotting.color.political,
            fliersize=1,
            legend=False,
        )
        ax.set_ylim(0, data[metric].max() * 10**1.5)
        ax.set_yscale("symlog")
        ax.set_title(metric.capitalize())
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        if by == "country":
            xticks = [*config.categorical[by]]
            xtlabs = [*map(str.upper, config.categorical[by])]
        else:
            xticks = np.asarray([*config.categorical[by]]) + 1
            xtlabs = [*map(str.capitalize, config.categorical[by].values())]
        ax.set_xticks(xticks, xtlabs)
        if ax is axes[0]:
            ax.set_ylabel("Engagement")
        ax.tick_params(axis="y", labelsize="small")

    fig.supxlabel(by.capitalize(), fontsize="large", y=0.1)
    fig.tight_layout()
    fig.savefig(figpath / f"engagement-{by}.png")

# %% ---------------------------------------------------------------------------------
