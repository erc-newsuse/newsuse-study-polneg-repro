# %% ---------------------------------------------------------------------------------

import json

import pandas as pd
from newsuse.data import DataFrame

from project import paths
from project.gpt import ValenceClassification

DOMAIN = "valence"

# %% ---------------------------------------------------------------------------------

responses = DataFrame.from_(paths.gpt / f"{DOMAIN}-responses.jsonl.gz")[
    ["custom_id", "response"]
].pipe(
    lambda df: pd.concat(
        [
            DataFrame(
                df.pop("custom_id").str.split("__", expand=True).to_numpy(),
                columns=["key"],
            ),
            df,
        ],
        axis=1,
    )
)

# %% ---------------------------------------------------------------------------------

output = (
    responses.assign(
        output=lambda df: df.pop("response").map(
            lambda x: json.loads(x["body"]["output"][-1]["content"][0]["text"])
        )
    )
    .set_index(["key"])["output"]
    .map(ValenceClassification.model_validate)
    .map(ValenceClassification.model_dump)
    .pipe(lambda s: DataFrame(s.tolist(), index=s.index))
    .astype("int64[pyarrow]")
    .dropna()
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

output.to_(paths.gpt / f"{DOMAIN}.parquet")

# %% ---------------------------------------------------------------------------------
