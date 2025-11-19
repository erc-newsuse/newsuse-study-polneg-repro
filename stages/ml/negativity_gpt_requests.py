# %% ---------------------------------------------------------------------------------

from copy import deepcopy

import numpy as np
import pandas as pd
from newsuse.data import DataFrame
from omegaconf import OmegaConf
from openai.lib._pydantic import to_strict_json_schema

from project import config, paths
from project.gpt import NegativityClassification

COMPLETED = "completed"
IN_PROGRESS = "in_progress"

domain = "negativity"
opts = config.gpt[domain].batch

rng = np.random.default_rng(opts.sample.seed)

# %% ---------------------------------------------------------------------------------

ground_truth = DataFrame.from_(
    paths.gpt / f"{domain}-ground-truth.parquet",
    columns=["key", "country", "title", "text"],
)

# %% ---------------------------------------------------------------------------------

sample = (
    DataFrame.from_(paths.dataset)[["key", "country"]]
    .merge(DataFrame.from_(paths.cls_political, columns=["key", "political"]), on="key")
    .merge(DataFrame.from_(paths.text))
    .dropna(ignore_index=True)
    .pipe(lambda df: df[df["text"].str.split().map(len) >= opts.sample.min_words])
    .dropna(ignore_index=True)
    .groupby(["country", "political"])
    .sample(n=opts.sample.size_per_group, random_state=rng)
    .pipe(
        lambda df: pd.concat([df, ground_truth], ignore_index=True).drop_duplicates(
            subset="key", keep="last"
        )
    )
    .reset_index(drop=True)
)

# %% Make requests -------------------------------------------------------------------

header = config.gpt.header
text_format = {
    "type": "json_schema",
    "name": "negativity",
    "strict": True,
    "schema": to_strict_json_schema(NegativityClassification),
}

with (paths.prompts / domain / "negativity.md").open() as fh:
    prompt = fh.read().strip()

# %% ---------------------------------------------------------------------------------

requests = []
params = deepcopy(OmegaConf.to_object(opts.params))
params.setdefault("text", {}).update({"format": text_format})
for key, row in sample.set_index("key").iterrows():
    text = [
        f"TITLE:\n{row.title}" if row.title else "",
        f"TEXT:\n{row.text}" if row.text else "",
    ]
    text = "\n\n".join(text).strip()
    if not text:
        continue
    body = {"input": text, "instructions": prompt, **params}
    request = {
        "custom_id": key,
        "country": row.country,
        **header,
        "body": body,
    }
    requests.append(request)

requests = DataFrame(requests)

# %% ---------------------------------------------------------------------------------

assert requests["custom_id"].is_unique, "Duplicate custom_id values found in requests."

# %% ---------------------------------------------------------------------------------

paths.gpt.mkdir(parents=True, exist_ok=True)
requests.to_(paths.gpt / "negativity-requests.jsonl.gz")

# %% ---------------------------------------------------------------------------------
