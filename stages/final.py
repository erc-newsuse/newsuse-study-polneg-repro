# %% ---------------------------------------------------------------------------------

import numpy as np
from newsuse.data import DataFrame

from project import paths

# %% ---------------------------------------------------------------------------------

meta = DataFrame.from_(paths.outlet_meta)

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.dataset)
    .query("date < '2024-04-01'")
    .merge(DataFrame.from_(paths.labels, columns=["key", "event", "sentiment"]))
    .assign(
        year=lambda df: df["date"].dt.year,
        month=lambda df: df["date"].dt.month,
        day=lambda df: df["date"].dt.day,
        isotime=lambda df: df["date"]
        .dt.isocalendar()
        .pipe(lambda df: df["year"].astype(str) + ":" + df["week"].astype(str)),
    )[
        [
            "key",
            "name",
            "country",
            "year",
            "month",
            "day",
            "isotime",
            "reactions",
            "comments",
            "shares",
        ]
    ]
    .merge(meta, how="left", on=["country", "name"])
    .merge(
        DataFrame.from_(paths.labels, columns=["key", "political", "event", "sentiment"])
    )
    .assign(valence=lambda df: df[["event", "sentiment"]].sum(axis=1, skipna=False))
    .dropna(ignore_index=True)
    .assign(
        political=lambda df: np.where(df["political"] == "OTHER", 0, 1),
        outlet=lambda df: df["country"] + ":" + df["name"],
        time=lambda df: df["year"].astype(str)
        + ":"
        + df["month"].astype(str)
        + ":"
        + df["day"].astype(str),
    )
    .convert_dtypes()
)

# %% ---------------------------------------------------------------------------------

data.to_(paths.final)

# %% ---------------------------------------------------------------------------------
