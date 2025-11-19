# %% ---------------------------------------------------------------------------------

import pandas as pd
from newsuse.data import DataFrame

from project import paths

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.articles_sample).merge(
    DataFrame.from_(paths.dataset, columns=["key", "country", "political", "valence"]),
    on="key",
    how="left",
)

# %% ----------------------------------------------------------------------------------

consistency = (
    data.assign(
        political=lambda df: df["political"] == df["political_text"],
        valence=lambda df: df["valence"] == df["valence_text"],
    )
    .groupby(["country"])
    .agg(
        political=("political", "mean"),
        valence=("valence", "mean"),
    )
)

consistency.mean()

# %% ---------------------------------------------------------------------------------

(
    data.pipe(
        lambda df: pd.Series(
            {
                "political": df["political"].eq("POLITICAL").mean(),
                "political_text": df["political_text"].eq("POLITICAL").mean(),
                "valence": df["valence"].eq("NEGATIVE").mean(),
                "valence_text": df["valence_text"].eq("NEGATIVE").mean(),
            }
        )
    )
)

# %% ---------------------------------------------------------------------------------
