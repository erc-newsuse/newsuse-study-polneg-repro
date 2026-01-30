# %% ---------------------------------------------------------------------------------

import pandas as pd
from newsuse.data import DataFrame

from project import config, paths

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(
    paths.final,
    columns=[
        "key",
        "country",
        "name",
        "quality",
        "ideology",
        "political",
        "event",
        "sentiment",
        "reactions",
        "comments",
        "shares",
    ],
)

# %% ---------------------------------------------------------------------------------

freqs = (
    pd.concat([data.assign(country="overall"), data])
    .groupby(["country", "political"])[["event", "sentiment"]]
    .value_counts(normalize=True)
    .sort_index()
    .reset_index(name="proportion")
    .assign(valence=lambda df: df[["event", "sentiment"]].sum(axis=1))
    .pipe(lambda df: df.set_index([c for c in df if c != "proportion"]))
    .pipe(lambda df: pd.concat({"prevalence": df}, axis=1))
)

# %% ---------------------------------------------------------------------------------

engagement = (
    pd.concat([data.assign(country="overall"), data])  # type: ignore
    .groupby(["country", "political", "event", "sentiment"])[
        ["reactions", "comments", "shares"]
    ]
    .describe()
    .filter(axis="columns", regex=r"\'(mean|std|\d+%)")
)

# %% ---------------------------------------------------------------------------------

table = freqs.merge(engagement, left_index=True, right_index=True)

# %% ---------------------------------------------------------------------------------

print(
    table.loc[["overall"]]
    .rename(lambda x: config.categorical.country.get(x, "Overall"), level="country")
    .rename(lambda x: "Political" if x else "Non-Political", level="political")
    .style.format(precision=2, escape="latex")
    .to_latex(
        hrules=True,
        multirow_align="t",
        multicol_align="c",
    )
)

# %% ---------------------------------------------------------------------------------

print(
    table.loc[[*config.categorical.country]]
    .rename(lambda x: config.categorical.country.get(x, "Overall"), level="country")
    .rename(lambda x: "Political" if x else "Non-Political", level="political")
    .style.format(precision=2, escape="latex")
    .to_latex(
        hrules=True,
        multirow_align="t",
        multicol_align="c",
    )
)

# %% ---------------------------------------------------------------------------------

groups = ["country", "quality", "ideology", "name"]
freqs = {
    "political": (
        data.groupby(groups)["political"]
        .value_counts(normalize=True)
        .sort_index()
        .unstack("political")
    ),
    "event": (
        data.groupby(groups)["event"]
        .value_counts(normalize=True)
        .sort_index()
        .unstack("event")
    ),
    "sentiment": (
        data.groupby(groups)["sentiment"]
        .value_counts(normalize=True)
        .sort_index()
        .unstack("sentiment")
    ),
}

freqs = pd.concat(freqs, names=["variable"], axis=1)

# %% ---------------------------------------------------------------------------------

engagement = (
    data.groupby(groups)[["reactions", "comments", "shares"]]
    .describe()
    .filter(axis="columns", regex=r"\'(\d+%)")
)

# %% ---------------------------------------------------------------------------------

tables = (
    freqs.drop(columns=("political", 0))
    .merge(engagement, left_index=True, right_index=True)
    .sort_index()
)

# %% ---------------------------------------------------------------------------------

print(
    tables.loc[[*config.categorical.country]]
    .rename(lambda x: x.capitalize() if x == "overall" else x.upper(), level="country")
    # .rename(lambda x: config.categorical.country.get(x, "Overall"), level="country")
    .style.format(precision=2, escape="latex")
    .to_latex(
        hrules=True,
        multirow_align="t",
        multicol_align="c",
    )
    .replace("%", r"\%")
)

# %% ---------------------------------------------------------------------------------
