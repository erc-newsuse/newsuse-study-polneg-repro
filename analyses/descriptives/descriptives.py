# %% ---------------------------------------------------------------------------------

from typing import Any

import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from newsuse.data import DataFrame
from transformers import AutoModel

import project.model  # noqa
from project import config, paths

mpl.rcParams.update(config.plotting.params)

targets = ["event", "sentiment", "valence"]
quality = ["low", "medium", "high"]
ideology = ["left", "center", "right"]

figpath = paths.figures / "descriptives"
figpath.mkdir(parents=True, exist_ok=True)

domain = "valence"

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final).assign(
    country=lambda df: pd.Categorical(
        df["country"], categories=list(config.categorical.countries), ordered=True
    ),
    quality=lambda df: pd.Categorical(df["quality"], categories=quality, ordered=True),
)

# %% ---------------------------------------------------------------------------------

hyper = DataFrame.from_(paths.proc / f"{domain}-hyper.parquet")
best_model = hyper.loc[hyper.value.idxmax()].params_base
model = AutoModel.from_pretrained(paths.ml / "models" / domain / best_model)

biases = {
    target: head.ordinal.bias.detach().cpu().numpy() for target, head in model.heads.items()
}

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
    fontsize: str = "x-small",
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
    for _, row in data[cols].iterrows():
        if text:
            ax.text(
                row[x]
                + text_xshift
                + (0.0 if hue is None else 0.2 * (-1 + 2 * row["political"])),
                row[y] + text_yshift,
                f"{row[y]:.1%}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                path_effects=[mpl.patheffects.withStroke(linewidth=3, foreground="white")],
            )


# %% --------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

target = "event"
ax = axes[0]
dist = data[target].value_counts(normalize=True).sort_index().reset_index()
plot_frequencies(ax, dist, target)
ax.set_title("Overall")
ax.set_ylabel("Frequency")

for ax, (country, df) in zip(
    axes.flatten()[1:],
    data.groupby("country", observed=True),
    strict=True,
):
    dist = df[target].value_counts(normalize=True).sort_index().reset_index()
    plot_frequencies(ax, dist, target, legend=False)
    ax.set_title(config.categorical.countries[country])

fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies.pdf", dpi=300)

# %% --------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

target = "event"
ax = axes[0]
dist = data[[target, "political"]].value_counts(normalize=True).sort_index().reset_index()
plot_frequencies(ax, dist, target, hue="political")
ax.set_title("Overall")
ax.set_ylabel("Frequency")

for ax, (country, df) in zip(
    axes.flatten()[1:],
    data.groupby("country", observed=True),
    strict=True,
):
    dist = df[[target, "political"]].value_counts(normalize=True).sort_index().reset_index()
    plot_frequencies(ax, dist, target, hue="political", legend=False)
    ax.set_title(config.categorical.countries[country])

fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies-political.pdf", dpi=300)

# %% --------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

target = "sentiment"
ax = axes[0]
dist = data[target].value_counts(normalize=True).sort_index().reset_index()
plot_frequencies(ax, dist, target)
ax.set_title("Overall")
ax.set_ylabel("Frequency")

for ax, (country, df) in zip(
    axes.flatten()[1:],
    data.groupby("country", observed=True),
    strict=True,
):
    dist = df[target].value_counts(normalize=True).sort_index().reset_index()
    plot_frequencies(ax, dist, target, legend=False)
    ax.set_title(config.categorical.countries[country])

fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies.pdf", dpi=300)

# %% --------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

target = "sentiment"
ax = axes[0]
dist = data[[target, "political"]].value_counts(normalize=True).sort_index().reset_index()
plot_frequencies(ax, dist, target, hue="political")
ax.set_title("Overall")
ax.set_ylabel("Frequency")

for ax, (country, df) in zip(
    axes.flatten()[1:],
    data.groupby("country", observed=True),
    strict=True,
):
    dist = df[[target, "political"]].value_counts(normalize=True).sort_index().reset_index()
    plot_frequencies(ax, dist, target, hue="political", legend=False)
    ax.set_title(config.categorical.countries[country])

fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies-political.pdf", dpi=300)

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

target = "valence"
ax = axes[0]
dist = data[target].value_counts(normalize=True).sort_index().reset_index()
plot_frequencies(ax, dist, target)
ax.set_title("Overall")
ax.set_ylabel("Frequency")

for ax, (country, df) in zip(
    axes.flatten()[1:],
    data.groupby("country", observed=True),
    strict=True,
):
    dist = df[target].value_counts(normalize=True).sort_index().reset_index()
    plot_frequencies(ax, dist, target, legend=False)
    ax.set_title(config.categorical.countries[country])

fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies.pdf", dpi=300)

# %% --------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

target = "valence"
ax = axes[0]
dist = data[[target, "political"]].value_counts(normalize=True).sort_index().reset_index()
plot_frequencies(ax, dist, target, hue="political")
ax.set_title("Overall")
ax.set_ylabel("Frequency")

for ax, (country, df) in zip(
    axes.flatten()[1:],
    data.groupby("country", observed=True),
    strict=True,
):
    dist = df[[target, "political"]].value_counts(normalize=True).sort_index().reset_index()
    plot_frequencies(ax, dist, target, hue="political", legend=False)
    ax.set_title(config.categorical.countries[country])

fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies-political.pdf", dpi=300)


# %% Event latent valence ------------------------------------------------------------

fig, axes = plt.subplots(ncols=2, figsize=(7, 3))

target = "event"
ax = axes[0]
sns.kdeplot(
    data,
    x=f"{target}_latent",
    hue="political",
    ax=ax,
    fill=True,
    palette=config.plotting.color.political,
)
ax.set_xlabel(None)

for threshold in biases[target]:
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)

ax.text(
    x=biases[target].mean(),
    y=ax.get_ylim()[1] * 0.9,
    s="thresholds",
    ha="center",
    va="center",
    fontsize="medium",
    path_effects=[mpl.patheffects.withStroke(linewidth=3, foreground="white")],
)
for threshold in biases[target]:
    ytext = ax.get_ylim()[1] * 0.85
    ax.annotate(
        "",
        xy=(threshold, ytext * 0.9),
        xytext=(biases[target].mean(), ytext),
        arrowprops={"arrowstyle": "->", "color": "black"},
    )

ax = axes[1]
sns.boxplot(
    data,
    x=f"{target}_latent",
    y="country",
    hue="political",  # hue_order=list(config.categorical.countries),
    ax=ax,
    palette=config.plotting.color.political,
    legend=False,
)
ax.set_yticks(
    [*config.categorical.countries],
    labels=list(config.categorical.countries.values()),
)
ax.set_xlabel(None)
ax.set_ylabel(None)

fig.tight_layout()
fig.supxlabel(target.capitalize() + " latent valence", y=-0.05)
fig.savefig(figpath / f"{target}-latent-valence.pdf", dpi=300)

# %% Sentiment latent valence --------------------------------------------------------

fig, axes = plt.subplots(ncols=2, figsize=(7, 3))

target = "sentiment"
ax = axes[0]
sns.kdeplot(
    data,
    x=f"{target}_latent",
    hue="political",
    ax=ax,
    fill=True,
    palette=config.plotting.color.political,
)
ax.set_xlabel(None)

ax = axes[1]
sns.boxplot(
    data,
    x=f"{target}_latent",
    y="country",
    hue="political",
    ax=ax,
    palette=config.plotting.color.political,
    legend=False,
)
ax.set_yticks(
    [*config.categorical.countries],
    labels=list(config.categorical.countries.values()),
)
ax.set_xlabel(None)
ax.set_ylabel(None)

fig.tight_layout()
fig.supxlabel(target.capitalize() + " latent valence", y=-0.05)
fig.savefig(figpath / f"{target}-latent-valence.pdf", dpi=300)

# %% Expected total valence ----------------------------------------------------------------

fig, axes = plt.subplots(ncols=2, figsize=(7, 3))

target = "valence"
ax = axes[0]
sns.kdeplot(
    data,
    x=f"{target}_expected",
    hue="political",
    ax=ax,
    fill=True,
    palette=config.plotting.color.political,
    bw_adjust=2,
    clip=(-2, 2),
)
ax.set_xlabel(None)

ax = axes[1]
sns.boxplot(
    data,
    x=f"{target}_expected",
    y="country",
    hue="political",
    ax=ax,
    palette=config.plotting.color.political,
    legend=False,
)
ax.set_yticks(
    [*config.categorical.countries],
    labels=list(config.categorical.countries.values()),
)
ax.set_xlabel(None)
ax.set_ylabel(None)

fig.tight_layout()
fig.supxlabel(f"Expected {target}", y=-0.05)
fig.savefig(figpath / f"{target}-expected.pdf", dpi=300)


# %% Event and sentiment valence by quality ------------------------------------------

fig, axes = plt.subplot_mosaic(
    """
    AAABB
    AAADD
    AAAFF
    """,
    figsize=(9, 5),
)

ax = axes["A"]
sns.kdeplot(
    data,
    x="event_latent",
    y="sentiment_latent",
    hue="political",
    ax=ax,
    palette=config.plotting.color.political,
    alpha=0.7,
    fill=True,
)
for target, thresholds in biases.items():
    for threshold in thresholds:
        if target == "event":
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
        elif target == "sentiment":
            ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Event latent valence")
ax.set_ylabel("Sentiment latent valence")

for key, target in zip(list(axes)[1:], [*biases, "valence"], strict=True):
    ax = axes[key]
    y = target + ("_expected" if target == "valence" else "_latent")
    sns.boxplot(
        data,
        x="quality",
        y=y,
        hue="political",
        ax=ax,
        palette=config.plotting.color.political,
        legend=False,
    )
    ax.set_title(target.capitalize() if target != "valence" else "Expected valence")
    ax.set_ylabel(None)
    if key != "F":
        ax.set_xlabel(None)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    else:
        ax.set_xlabel("Outlet quality")

fig.tight_layout()
fig.savefig(figpath / "event-sentiment-valence-by-quality.pdf", dpi=300)

# %% ---------------------------------------------------------------------------------
