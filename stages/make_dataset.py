# %% Setup -------------------------------------------------------------------------------

import numpy as np
from newsuse.data import DataFrame

from project import config, paths

# %% Get raw data --------------------------------------------------------------------

posts = DataFrame.from_(paths.posts)

# %% Compute data quality statistics -------------------------------------------------

quality = (
    DataFrame.from_(paths.daily)
    .groupby(["country", "name"])[["n"]]
    .mean()
    .reset_index()
    .rename(columns={"n": "rate"})
)

keys = ["country", "name"]
cols = ["reactions", "comments", "shares"]

Q = posts.groupby(keys)[cols].quantile(q=np.arange(1, 10) / 10)
M = (
    Q.groupby(level=keys)[cols]
    .agg(lambda s: (((X := s.to_numpy())[1:] - X[:-1]) / (1 + X[:-1])).max())
    .sort_values("shares", ascending=False)
)

# %% ---------------------------------------------------------------------------------

quality = (
    quality.set_index(keys)
    .merge(M, left_index=True, right_index=True)
    .reset_index()
    .assign(
        bad_rate=lambda df: df["rate"] < config.dataset.min_posts_per_day,
        bad_shares=lambda df: df["shares"] > config.dataset.max_decile_difference,
    )
    # .assign(bad=lambda df: df["bad_rate"] | df["bad_shares"])
    .assign(bad=lambda df: df["bad_rate"])
    .set_index(keys)
)

# %% Make dataset --------------------------------------------------------------------

dataset = (
    posts.set_index(keys)
    .pipe(lambda df: df[~df.index.isin(quality.query("bad").index)])
    .set_index("key", append=True)
    .reset_index(keys)
    .reset_index()
)

# %% Save dataset, quality, and bad sources ------------------------------------------

dataset.to_(paths.dataset)
quality.to_(paths.quality)

# %% ---------------------------------------------------------------------------------
