# %% ---------------------------------------------------------------------------------

from copy import deepcopy

import pandas as pd
from newsuse.data import DataFrame
from newsuse.dotpath import dotimport
from omegaconf import OmegaConf
from openai.lib._pydantic import to_strict_json_schema

from project import config, paths

COMPLETED = "completed"
IN_PROGRESS = "in_progress"

domain = "negativity"
opts = config.gpt[domain].batch

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
    .sample(n=opts.sample.size_per_group, random_state=opts.sample.seed)
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
    "name": "quality_assessment",
    "strict": True,
}

prompts = {}
for target in opts.targets:
    with (paths.prompts / domain / f"{target}.md").open() as fh:
        prompts[target] = fh.read().strip()

# %% ---------------------------------------------------------------------------------

requests = []
for target, target_opts in opts.targets.items():
    output_model = dotimport(f"project.gpt:{target.title()}Classification")
    request_params = deepcopy(OmegaConf.to_object(target_opts.params))
    tfrm = {**text_format, "schema": to_strict_json_schema(output_model)}
    request_params.setdefault("text", {}).update(format=tfrm)
    for key, row in sample.set_index("key").iterrows():
        text = [
            f"TITLE:\n{row.title}" if row.title else "",
            f"TEXT:\n{row.text}" if row.text else "",
        ]
        text = "\n\n".join(text).strip()
        if not text:
            continue
        body = {"input": text, "instructions": prompts[target], **request_params}
        request = {
            "custom_id": f"{key}__{target}",
            "key": key,
            "country": row.country,
            "target": target,
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
