# %% Setup -------------------------------------------------------------------------------

import pandas as pd
from newsuse.data import DataFrame

from project import config, paths

# %% Make data ---------------------------------------------------------------------------

data = pd.concat(
    [
        DataFrame.from_(paths.raw / "posts-eu.parquet", columns=config.dataset.usecols),
        DataFrame.from_(paths.raw / "posts-us.parquet", columns=config.dataset.usecols),
    ],
    axis=0,
    ignore_index=True,
)

textdata = data[["key", "text"]]
data = data.drop(columns="text")

idx = data.columns.tolist().index("timestamp")
data.insert(idx + 1, "weekday", data["timestamp"].dt.weekday)
data.insert(idx + 1, "day", data["timestamp"].dt.day)
data.insert(idx + 1, "month", data["timestamp"].dt.month)
data.insert(idx + 1, "year", data["timestamp"].dt.year)

idx = data.columns.tolist().index("name")
data.insert(idx + 1, "pid", data["country"] + "@" + data["name"])

loc = data.columns.tolist().index("likes") + 1
cols = [c for c in data if c.startswith("reactions_")]
data.insert(loc, "reactions", data[cols].sum(axis=1))
data.insert(loc + 1, "interactions", data[["likes", "reactions"]].max(axis=1))

# %% Save data ---------------------------------------------------------------------------

data.to_(paths.fulldata)
textdata.to_(paths.textdata)

# %% -------------------------------------------------------------------------------------
