# %% ---------------------------------------------------------------------------------

import json
import warnings
from functools import reduce

import numpy as np
import pandas as pd
from newsuse.data import DataFrame
from pydantic import ValidationError

from project import config, paths
from project.gpt import (
    EventClassification,
    NegativityClassification,
    SentimentClassification,
)

DOMAIN = "negativity"

opts = config.gpt[DOMAIN].experiment

output_models = {
    "event": EventClassification,
    "sentiment": SentimentClassification,
    "negativity": NegativityClassification,
}

# %% ---------------------------------------------------------------------------------

responses = DataFrame.from_(paths.gpt / f"{DOMAIN}-experiment-responses.jsonl.gz")[
    ["custom_id", "response"]
].rename(columns={"custom_id": "key"})

meta = (
    DataFrame.from_(paths.gpt / f"{DOMAIN}-experiment-requests.jsonl.gz")[
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
        target = key.rsplit("__", 1)[-1]
        result = json.loads(text)
        result = output_models[target].model_validate(result)
        results.append({"key": key, "output": result})
    except (json.JSONDecodeError, ValidationError) as exc:
        msg = f"failed to process output '{text}' with error:\n{exc!r}"
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
    .map(lambda x: x.model_dump())
    .assign(
        split=lambda df: df.index.get_level_values("target") != "negativity",
    )
    .set_index("split", append=True)
    .groupby(level=["key", "params_id", "split"])
    .apply(lambda df: (reduce(lambda x, y: x | y, df["output"], {})))
    .map(NegativityClassification.model_validate)
    .map(NegativityClassification.model_dump)
    .pipe(lambda s: DataFrame(s.tolist(), index=s.index))
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

output.to_(paths.gpt / f"{DOMAIN}-experiment.parquet")

# %% ---------------------------------------------------------------------------------
