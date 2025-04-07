# %% Setup -------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from newsuse.data import DataFrame

from project import config, paths

# %% Get raw data ------------------------------------------------------------------------

data = DataFrame.from_(paths.fulldata)

# %% Compute data quality statistics -----------------------------------------------------

quality = (
    DataFrame.from_(paths.daily)
    .groupby(["country", "name"])[["n"]]
    .mean()
    .reset_index()
    .rename(columns={"n": "rate"})
)
quality.index = pd.Series(quality["country"] + "@" + quality["name"])

keys = ["pid"]
cols = ["likes", "comments", "shares"]

Q = data.groupby(keys)[cols].quantile(q=np.arange(1, 10) / 10)
M = (
    Q.groupby(level=keys)[cols]
    .agg(lambda s: (((X := s.to_numpy())[1:] - X[:-1]) / (1 + X[:-1])).max())
    .sort_values("shares", ascending=False)
)

quality = (
    quality.merge(M, left_index=True, right_index=True)
    .assign(
        bad_rate=lambda df: df["rate"] < config.dataset.min_posts_per_day,
        bad_shares=lambda df: df["shares"] > config.dataset.max_decile_difference,
    )
    .assign(bad=lambda df: df["bad_rate"] | df["bad_shares"])
    .reset_index(names="pid")
)

# %% Get labels --------------------------------------------------------------------------

labels = DataFrame.from_(paths.labels).drop_duplicates("key", ignore_index=True)

# %% Make dataset ------------------------------------------------------------------------

dataset = data.query(f"~pid.isin({quality.query("bad").pid.tolist()})").merge(
    labels, how="left", on="key"
)

# %% Determine time bounds ---------------------------------------------------------------

start, end = (
    dataset.groupby(["country"])["timestamp"]
    .agg(["min", "max"])
    .agg({"min": "max", "max": "min"})
)

dataset = dataset[dataset["timestamp"].between(start, end)]

# %% Save dataset, quality, and bad sources ----------------------------------------------

dataset.to_(paths.dataset)
quality.to_(paths.quality)

# %% -------------------------------------------------------------------------------------
