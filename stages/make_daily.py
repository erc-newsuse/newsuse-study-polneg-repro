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
        "date",
    ],
)

# %% Compute daily publication counts ----------------------------------------------------

tmin = data["date"].min()
tmax = data["date"].max()
daterange = pd.date_range(
    datetime(tmin.year, tmin.month, 1, tzinfo=UTC),
    datetime(tmax.year, tmax.month + 1, 1, tzinfo=UTC) - timedelta(days=1),
    freq="D",
)

daily = (
    data[["country", "name", "date"]]
    .set_index("date")
    .groupby(["country", "name"])
    .apply(
        lambda df: (df.resample("D").size().to_frame("n").rename_axis("date")),
        include_groups=False,
    )
    .reset_index()
    .query("n > 0")
    .groupby(["country", "name"])
    .apply(
        lambda df: (
            df.set_index("date").reindex(
                pd.date_range(df["date"].min(), df["date"].max(), freq="D", name="date")
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
