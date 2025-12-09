# %% ---------------------------------------------------------------------------------

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from newsuse.data import DataFrame

from project import config, paths

mpl.rcParams.update(config.plotting.params)

targets = ["event", "sentiment", "valence"]
quality = ["low", "medium", "high"]
ideology = ["left", "center", "right"]

figpath = paths.figures / "descriptives"
figpath.mkdir(parents=True, exist_ok=True)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final)

# %% ---------------------------------------------------------------------------------


def plot_frequencies(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str = "proportion",
    hue: str | None = None,
    palette: list[str] = config.plotting.color.political,
    **kwargs: Any,
) -> None:
    kwargs = {"palette": palette, **kwargs} if hue else {"color": palette[0], **kwargs}
    sns.barplot(data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    ax.set_title(x.capitalize())
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))


# %% --------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

target = "event"
ax = axes[0]
dist = data[target].value_counts(normalize=True).sort_index().reset_index()
plot_frequencies(ax, dist, target)
ax.set_title("Overall")
ax.set_ylabel("Frequency")

for ax, (country, df) in zip(axes.flatten()[1:], data.groupby("country"), strict=True):
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

for ax, (country, df) in zip(axes.flatten()[1:], data.groupby("country"), strict=True):
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

for ax, (country, df) in zip(axes.flatten()[1:], data.groupby("country"), strict=True):
    dist = df[target].value_counts(normalize=True).sort_index().reset_index()
    plot_frequencies(ax, dist, target, legend=False)
    ax.set_title(config.categorical.countries[country])

fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies.pdf", dpi=300)

# %% --------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=7, figsize=(21, 3))

target = "sentiment "
ax = axes[0]
dist = data[[target, "political"]].value_counts(normalize=True).sort_index().reset_index()
plot_frequencies(ax, dist, target, hue="political")
ax.set_title("Overall")
ax.set_ylabel("Frequency")

for ax, (country, df) in zip(axes.flatten()[1:], data.groupby("country"), strict=True):
    dist = df[[target, "political"]].value_counts(normalize=True).sort_index().reset_index()
    plot_frequencies(ax, dist, target, hue="political", legend=False)
    ax.set_title(config.categorical.countries[country])

fig.tight_layout()
fig.savefig(figpath / f"{target}-frequencies-political.pdf", dpi=300)


# %% ---------------------------------------------------------------------------------
