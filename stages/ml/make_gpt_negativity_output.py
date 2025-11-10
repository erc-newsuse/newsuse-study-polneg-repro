# %% ---------------------------------------------------------------------------------

import json

import pandas as pd
from newsuse.data import DataFrame

from project import paths
from project.gpt import (
    EventClassification,
    SentimentClassification,
)

DOMAIN = "negativity"

output_models = {
    "event": EventClassification,
    "sentiment": SentimentClassification,
}

# %% ---------------------------------------------------------------------------------

responses = DataFrame.from_(paths.gpt / f"{DOMAIN}-responses.jsonl.gz")[
    ["custom_id", "response"]
].pipe(
    lambda df: pd.concat(
        [
            DataFrame(
                df.pop("custom_id").str.split("__", expand=True).to_numpy(),
                columns=["key", "target"],
            ),
            df,
        ],
        axis=1,
    )
)

# %% ---------------------------------------------------------------------------------

output = (
    responses.copy()
    .assign(
        output=lambda df: df.pop("response")
        .map(lambda x: json.loads(x["body"]["output"][-1]["content"][0]["text"]))
        .map(lambda x: list(x.values())[0])
    )
    .set_index(["key", "target"])
    .unstack("target")["output"]
    .astype("int64[pyarrow]")
)
output.columns = list(output_models.keys())
output = output.reset_index().dropna(ignore_index=True)

# %% ---------------------------------------------------------------------------------

output.to_(paths.gpt / f"{DOMAIN}.parquet")

# %% ---------------------------------------------------------------------------------
