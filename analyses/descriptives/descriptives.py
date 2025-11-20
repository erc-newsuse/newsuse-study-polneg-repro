# %% ---------------------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from newsuse.data import DataFrame

from project import config, paths

mpl.rcParams.update(config.plotting.params)

targets = ["event", "sentiment", "valence"]

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final)

# %% ---------------------------------------------------------------------------------

dist = DataFrame({t: data[t].value_counts(normalize=True) for t in targets}).sort_index()

# %% ---------------------------------------------------------------------------------

interactions = data.groupby("valence")[["reactions", "comments", "shares"]].mean()

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(nrows=3, figsize=(7, 6))

for ax, col in zip(axes.flat, interactions.columns, strict=True):
    ax.plot(interactions.index, interactions[col], "-o")
    ax.set_title(f"Average {col.capitalize()}")

fig.supxlabel("Post Valence")
fig.supylabel("Average Count")
fig.tight_layout()

# %% Tables --------------------------------------------------------------------------

table = pd.pivot_table(
    data,
    index="event",
    columns="sentiment",
    aggfunc="size",
)

(table / table.sum().sum()).round(3)

# %% ---------------------------------------------------------------------------------
