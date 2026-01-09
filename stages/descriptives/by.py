# %% ---------------------------------------------------------------------------------

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

factors = ["quality", "ideology"]
targets_valence = ["event", "sentiment"]
targets_engagement = [*config.engagement]

figpath = paths.figures / "descriptives" / "by"
figpath.mkdir(parents=True, exist_ok=True)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final).assign(
    country=lambda df: pd.Categorical(
        df["country"],
        categories=list(config.categorical.country),
    ),
    quality=lambda df: pd.Categorical(
        df["quality"],
        categories=list(config.categorical.quality),
        ordered=True,
    ),
    ideology=lambda df: pd.Categorical(
        df["ideology"],
        categories=list(config.categorical.ideology),
        ordered=True,
    ),
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

for target in targets_valence:
    for by in factors:
        factor = config.categorical[by]
        support = np.asarray([*config.categorical[target]])
        fig, axes = plt.subplots(ncols=len(factor), figsize=(7, 3))

        for ax, (group, df) in zip(
            axes.flat,
            data.groupby(by, observed=True),
            strict=True,
        ):
            dist = (
                df.groupby(["country", by], observed=True)[target]
                .value_counts(normalize=True)
                .sort_index()
                .groupby([target, by], observed=True)
                .mean()
                .reset_index()
            )
            plot_frequencies(ax, dist, target, legend=False)
            ax.set_title(group, fontsize="x-large")

        fig.suptitle(by.capitalize(), fontsize="xx-large", x=0.0, y=0.95, ha="left")
        fig.supxlabel(target.capitalize(), fontsize="xx-large", y=0.05)
        fig.tight_layout()
        fig.savefig(figpath / f"{target}-{by}.pdf")

# %% --------------------------------------------------------------------------------

for target in targets_valence:
    for by in factors:
        factor = config.categorical[by]
        support = np.asarray([*config.categorical[target]])
        fig, axes = plt.subplots(ncols=len(factor), figsize=(7, 3))

        for ax, (group, df) in zip(
            axes.flat,
            data.groupby(by, observed=True),
            strict=True,
        ):
            dist = (
                df.groupby(["country", by, "political"], observed=True)[target]
                .value_counts(normalize=True)
                .sort_index()
                .groupby([target, by, "political"], observed=True)
                .mean()
                .reset_index()
            )
            plot_frequencies(ax, dist, target, hue="political", legend=False)
            ax.set_title(group, fontsize="x-large")
        # Add custom legend for political in the first axis
        ax = axes.flatten()[0]
        handles = [
            mpl.patches.Patch(color=color, label=label)
            for label, color in zip(
                config.categorical.political,
                config.plotting.color.political,
                strict=True,
            )
        ]
        ax.legend(
            handles=handles,
            frameon=False,
        )
        fig.suptitle(by.capitalize(), fontsize="xx-large", x=0.0, y=0.95, ha="left")
        fig.supxlabel(target.capitalize(), fontsize="xx-large", y=0.05)
        fig.tight_layout()
        fig.savefig(figpath / f"{target}-{by}-political.pdf")

# %% ---------------------------------------------------------------------------------


def plot_distributions(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    palette: list[str] = config.plotting.color.political,
    **kwargs: Any,
) -> None:
    kwargs = {"palette": palette, **kwargs} if hue else {"color": "white", **kwargs}
    kwargs = {"fliersize": 1, **kwargs}
    sns.boxplot(data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    ax.set_xlabel(None)
    ax.set_ylabel(None)


# %% ---------------------------------------------------------------------------------

for valence in targets_valence:
    for target in targets_engagement:
        for by in factors:
            factor = config.categorical[by]
            fig, axes = plt.subplots(ncols=len(factor), figsize=(7, 3))

            for ax, (group, df) in zip(
                axes.flat,
                data.groupby(by, observed=True),
                strict=True,
            ):
                daily = (
                    df.groupby(
                        ["country", valence, by, "outlet", "year", "month", "day"],
                        observed=True,
                    )[target]
                    .mean()
                    .reset_index()
                )
                plot_distributions(ax, daily, valence, target, legend=False)
                ax.set_title(group, fontsize="x-large")
                ax.set_ylim(bottom=0)
                ax.set_yscale("symlog")

            fig.suptitle(by.capitalize(), fontsize="xx-large", x=0.0, y=0.95, ha="left")
            fig.supxlabel(valence.capitalize(), fontsize="xx-large", y=0.05)
            fig.supylabel(target.capitalize(), fontsize="xx-large", x=0.0)
            fig.tight_layout()
            fig.savefig(figpath / f"{target}-{valence}-{by}.pdf")

# %% ---------------------------------------------------------------------------------

for valence in targets_valence:
    for target in targets_engagement:
        for by in factors:
            factor = config.categorical[by]
            fig, axes = plt.subplots(ncols=len(factor), figsize=(7, 3))

            for ax, (group, df) in zip(
                axes.flat,
                data.groupby(by, observed=True),
                strict=True,
            ):
                daily = (
                    df.groupby(
                        [
                            "country",
                            "political",
                            valence,
                            by,
                            "outlet",
                            "year",
                            "month",
                            "day",
                        ],
                        observed=True,
                    )[target]
                    .mean()
                    .reset_index()
                )
                plot_distributions(
                    ax, daily, valence, target, hue="political", legend=False
                )
                ax.set_title(group, fontsize="x-large")
                ax.set_ylim(bottom=0)
                ax.set_yscale("symlog")

            # Add custom figure legend for political
            handles = [
                mpl.patches.Patch(color=color, label=label)
                for label, color in zip(
                    config.categorical.political,
                    config.plotting.color.political,
                    strict=True,
                )
            ]
            fig.legend(
                handles=handles,
                frameon=False,
                ncols=2,
                loc="center left",
                bbox_to_anchor=(0.05, 0.08),
            )

            fig.suptitle(by.capitalize(), fontsize="xx-large", x=0.0, y=0.95, ha="left")
            fig.supxlabel(valence.capitalize(), fontsize="xx-large", y=0.05)
            fig.supylabel(target.capitalize(), fontsize="xx-large", x=0.0)
            fig.tight_layout()
            fig.savefig(figpath / f"{target}-{valence}-{by}-political.pdf")

# %% ---------------------------------------------------------------------------------
