# %% ---------------------------------------------------------------------------------

from newsuse.data import DataFrame

from project import paths

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.dataset)
    .merge(DataFrame.from_(paths.labels, columns=["key", "event", "sentiment"]))
    .assign(
        year=lambda df: df["date"].dt.year,
        month=lambda df: df["date"].dt.month,
        day=lambda df: df["date"].dt.day,
    )[
        [
            "key",
            "name",
            "country",
            "year",
            "month",
            "day",
            "reactions",
            "comments",
            "shares",
        ]
    ]
    .merge(
        DataFrame.from_(paths.labels, columns=["key", "political", "event", "sentiment"])
    )
    .assign(valence=lambda df: df[["event", "sentiment"]].sum(axis=1))
    .dropna(ignore_index=True)
)

# %% ---------------------------------------------------------------------------------

data.to_(paths.final)

# %% ---------------------------------------------------------------------------------
