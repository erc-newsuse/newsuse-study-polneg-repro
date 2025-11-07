# %% ---------------------------------------------------------------------------------

import json
import warnings

import numpy as np
import pandas as pd
from newsuse.data import DataFrame
from pydantic import ValidationError

from project import config, paths
from project.gpt import (
    EventClassification,
    SentimentClassification,
)

DOMAIN = "negativity"

opts = config.gpt[DOMAIN]

output_models = {
    "event": EventClassification,
    "sentiment": SentimentClassification,
}

# %% ---------------------------------------------------------------------------------

responses = DataFrame.from_(paths.gpt / f"{DOMAIN}-responses.jsonl.gz")[
    ["custom_id", "response"]
].rename(columns={"custom_id": "key"})

meta = (
    DataFrame.from_(paths.gpt / f"{DOMAIN}-requests.jsonl.gz")[
        ["key", "country", "params_id", "params"]
    ]
    .drop_duplicates(subset=["key", "params_id"], ignore_index=True)
    .assign(
        model=lambda df: df["params"].map(lambda x: x["model"]),
        reasoning=lambda df: df["params"].map(
            lambda x: x.get("reasoning", {}).get("effort")
        ),
    )
    .drop(columns=["params"])
)
meta["model"] = np.where(
    meta["reasoning"].notnull(),
    meta["model"] + "[" + meta.pop("reasoning") + "]",
    meta["model"],
)

# %% ---------------------------------------------------------------------------------

results = []
for key, response in responses[["key", "response"]].itertuples(index=False):
    text = response["body"]["output"][-1]["content"][0]["text"]
    try:
        result = json.loads(text)
        result = output_models[list(result)[0]].model_validate(result)
        results.append({"key": key, "output": result})
    except (json.JSONDecodeError, ValidationError) as exc:
        msg = f"Failed to process output '{text}' with error:\n{exc!r}"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
results = DataFrame(results)

# %% ---------------------------------------------------------------------------------

output = (
    responses.drop(columns=["response"])
    .merge(DataFrame(results), how="left", on="key")
    .dropna(ignore_index=True)
    .pipe(
        lambda df: pd.concat(
            [
                df["key"]
                .str.split("__", expand=True)
                .rename(columns={0: "key", 1: "params_id", 2: "target"}),
                df.drop(columns=["key"]),
            ],
            axis=1,
        )
    )
    .set_index(["key", "params_id", "target"])
    .map(lambda x: list(x.model_dump().values())[0])
    .unstack("target")
    .droplevel(0, axis=1)
    .reset_index()
    .merge(meta, how="left", on=["key", "params_id"])[
        ["key", "country", "params_id", "model", "event", "sentiment"]
    ]
)
output.columns = output.columns.tolist()

# %% ---------------------------------------------------------------------------------

output.to_(paths.gpt / f"{DOMAIN}.parquet")

# %% ---------------------------------------------------------------------------------
