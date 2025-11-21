# %% ---------------------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from newsuse.data import DataFrame

from project import config, paths

mpl.rcParams.update(config.plotting.params)

targets = ["event", "sentiment", "valence"]
quality = ["low", "medium", "high"]
ideology = ["left", "center", "right"]

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


def make_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.pivot_table(df, index="sentiment", columns="event", aggfunc="size")
        .pipe(lambda df: df / df.sum().sum())
        .fillna(0)
        .mul(100)
    )


make_pivot(data).round(1)

# %% By quality ----------------------------------------------------------------------

(data.groupby(["quality"]).apply(make_pivot, include_groups=False).round(1).loc[quality])

# %% By ideology ---------------------------------------------------------------------

(data.groupby(["ideology"]).apply(make_pivot, include_groups=False).round(1).loc[ideology])

# %% ---------------------------------------------------------------------------------
