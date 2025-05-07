# %% Setup -------------------------------------------------------------------------------

from datetime import UTC, datetime, timedelta

import pandas as pd
from newsuse.data import DataFrame

from project import paths

# %% Get data ----------------------------------------------------------------------------

data = DataFrame.from_(
    paths.posts,
    columns=[
        "key",
        "country",
        "name",
        "timestamp",
    ],
)

# %% Compute daily publication counts ----------------------------------------------------

tsmin = data["timestamp"].min()
tsmax = data["timestamp"].max()
daterange = pd.date_range(
    datetime(tsmin.year, tsmin.month, 1, tzinfo=UTC),
    datetime(tsmax.year, tsmax.month + 1, 1, tzinfo=UTC) - timedelta(days=1),
    freq="D",
)

daily = (
    data[["country", "name", "timestamp"]]
    .set_index("timestamp")
    .groupby(["country", "name"])
    .apply(
        lambda df: (df.resample("D").size().to_frame("n").rename_axis("timestamp")),
        include_groups=False,
    )
    .reset_index()
    .query("n > 0")
    .groupby(["country", "name"])
    .apply(
        lambda df: (
            df.set_index("timestamp").reindex(
                pd.date_range(
                    df["timestamp"].min(), df["timestamp"].max(), freq="D", name="timestamp"
                )
            )
        )[["n"]],
        include_groups=False,
    )
    .reset_index()
    .assign(n=lambda df: df["n"].fillna(0).astype(int))
    .convert_dtypes()
)

# %% Save data ---------------------------------------------------------------------------

daily.to_(paths.proc / "daily.parquet")

# %% -------------------------------------------------------------------------------------
