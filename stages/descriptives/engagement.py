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

domain = "engagement"

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

fig, axes = plt.subplots(ncols=3, figsize=(7, 2), sharex=True, sharey=True)
for ax, metric in zip(axes.flat, config.engagement, strict=True):
    sns.boxplot(
        data=data,
        x="country",
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
    ax.set_xticks(
        [*config.categorical.country],
        [*map(str.upper, config.categorical.country)],
    )
    if ax is axes[0]:
        ax.set_ylabel("Engagement")

fig.tight_layout()
fig.savefig(figpath / "engagement-country.png")

# %% ---------------------------------------------------------------------------------

for valence in ["event", "sentiment"]:
    fig, axes = plt.subplots(ncols=3, figsize=(7, 2), sharex=True, sharey=True)
    for ax, metric in zip(axes.flat, config.engagement, strict=True):
        sns.boxplot(
            data=data,
            x=valence,
            y=metric,
            hue="political",
            ax=ax,
            palette=config.plotting.color.political,
            fliersize=1,
            legend=False,
        )
        ax.set_ylim(0, data[metric].max() * 10**1.5)
        ax.set_yscale("symlog")
        # ax.set_title(metric.capitalize())
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        if ax is axes[0]:
            ax.set_ylabel("Engagement")
        if ax is axes[1]:
            ax.set_xlabel(f"{valence.capitalize()}")
        ax.set_xticklabels(["negative", "neutral", "positive"])
    fig.tight_layout()
    fig.savefig(figpath / f"engagement-{valence}.png")

# %% ---------------------------------------------------------------------------------

for x in ["event", "sentiment", "valence"]:
    for by in ["quality", "ideology"]:
        for political_value, political in enumerate(config.categorical.political):
            fig, axes = plt.subplots(
                ncols=len(config.engagement),
                nrows=2,
                figsize=(4 * 4 / 3, 4),
                sharex="row",
                sharey="row",
            )
            df = data.query(f"political == {political_value}")
            palette = config.plotting.color.get(by)
            for ax, metric in zip(axes[0].flat, config.engagement, strict=True):
                plot_ecdf(df, metric, ax, hue=by, palette=palette)
                ax.set_title(metric.capitalize())
                ax.set_xlabel(None)
                ax.set_ylabel(None)
                if ax is axes[0, 0]:
                    ax.set_ylabel(r"$\mathbb{P}(X > x)$")
                # if ax is axes[0, 1]:
                #     ax.set_xlabel("Engagement")

            for ax, metric in zip(axes[1].flat, config.engagement, strict=True):
                sns.boxplot(
                    data=df,
                    x=x,
                    y=metric,
                    hue=by,
                    ax=ax,
                    palette=palette,
                    fliersize=1,
                    legend=False,
                )
                ax.set_ylim(0, data[metric].max() * 10**1.5)
                ax.set_yscale("symlog")
                ax.set_xlabel(None)
                ax.set_ylabel(None)
                if ax is axes[1, 0]:
                    ax.set_ylabel("Engagement")
                if ax is axes[1, 1]:
                    ax.set_xlabel(x.capitalize())

            fig.tight_layout()
            title = "non-political" if political == "other" else "political"
            fig.suptitle(f"{title.title()}", y=1.03, x=0.02, ha="left")
            fig.savefig(figpath / f"ecdf-{x}-{by}-political-{political}.png")

# %% ---------------------------------------------------------------------------------
