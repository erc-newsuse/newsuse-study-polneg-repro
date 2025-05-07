# %% ---------------------------------------------------------------------------------

import pandas as pd
from newsuse.data import DataFrame

from project import paths

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.articles_sample).merge(
    DataFrame.from_(paths.dataset, columns=["key", "country", "political", "negativity"]),
    on="key",
    how="left",
)

# %% ----------------------------------------------------------------------------------

consistency = (
    data.assign(
        political=lambda df: df["political"] == df["political_text"],
        negativity=lambda df: df["negativity"] == df["negativity_text"],
    )
    .groupby(["country"])
    .agg(
        political=("political", "mean"),
        negativity=("negativity", "mean"),
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
                "negativity": df["negativity"].eq("NEGATIVE").mean(),
                "negativity_text": df["negativity_text"].eq("NEGATIVE").mean(),
            }
        )
    )
)

# %% ---------------------------------------------------------------------------------
