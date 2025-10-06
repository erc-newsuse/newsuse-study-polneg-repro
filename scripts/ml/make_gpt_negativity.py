# %% ---------------------------------------------------------------------------------

import json
import warnings

from newsuse.data import DataFrame
from pydantic import ValidationError

from project import config, paths
from project.gpt import NegativityClassification

DOMAIN = "negativity"

opts = config.gpt[DOMAIN]

text_field_idx = idx = 1 if opts.model == "gpt-5" else 0

# %% ---------------------------------------------------------------------------------

output = DataFrame.from_(paths.gpt / f"{DOMAIN}-output.jsonl.gz")[
    ["custom_id", "country", "response"]
].rename(columns={"custom_id": "key"})

# %% ---------------------------------------------------------------------------------

results = []
for key, response in output[["key", "response"]].itertuples(index=False):
    text = response["body"]["output"][text_field_idx]["content"][0]["text"]
    try:
        result = json.loads(text)
        result = NegativityClassification.model_validate(result)
        results.append({"key": key, "output": result})
    except (json.JSONDecodeError, ValidationError) as exc:
        msg = f"Failed to process output '{text}' with error:\n{exc!r}"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
results = DataFrame(results)
proc = (
    output.drop(columns=["response"])
    .merge(DataFrame(results), how="left", on="key")
    .dropna(ignore_index=True)
)
records = []
for _, row in proc.iterrows():
    record = row.to_dict()
    record.update(record.pop("output").model_dump())
    records.append(record)

data = DataFrame(records)

# %% ---------------------------------------------------------------------------------

data.to_(paths.gpt / f"{DOMAIN}.parquet")

# %% ---------------------------------------------------------------------------------
