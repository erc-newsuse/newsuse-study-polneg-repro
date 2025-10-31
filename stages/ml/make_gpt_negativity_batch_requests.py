# %% ---------------------------------------------------------------------------------


import pandas as pd
from newsuse.data import DataFrame
from openai.lib._pydantic import to_strict_json_schema

from project import config, paths
from project.gpt import NegativityClassification

COMPLETED = "completed"
IN_PROGRESS = "in_progress"

opts = config.gpt.negativity

# %% ---------------------------------------------------------------------------------

subsample_annotated = DataFrame.from_(paths.raw / "gpt-annotated-subsample.parquet")

# %% ---------------------------------------------------------------------------------

sample = (
    DataFrame.from_(paths.posts, columns=["key", "country"])
    .merge(DataFrame.from_(paths.cls_political, columns=["key", "political"]))
    .merge(DataFrame.from_(paths.text))
    .query("text.notnull() | title.notnull()")
    .query(
        "text.fillna('').str.len() + title.fillna('').str.len() "
        f">= {opts.sample.min_words}"
    )
    .reset_index(drop=True)
    .groupby(["country", "political"])
    .sample(n=opts.sample.size_per_group, random_state=opts.sample.seed)
    .reset_index(drop=True)
)

# %% ---------------------------------------------------------------------------------

sample = (
    pd.concat([sample, subsample_annotated])
    .drop_duplicates(subset=["key"])
    .sort_values(["country"], ignore_index=True)
)

# %% ---------------------------------------------------------------------------------

assert sample.key.is_unique, "Sample keys are not unique"

# %% ---------------------------------------------------------------------------------

with (paths.prompts / opts.prompt).open() as fh:
    instructions = fh.read().strip()

# %% Make requests -------------------------------------------------------------------

header = opts.header
body = {
    "instructions": instructions,
    "text": {
        "format": {
            "type": "json_schema",
            "name": "negativity_classification",
            "schema": to_strict_json_schema(NegativityClassification),
            "strict": True,
        },
    },
    "model": opts.model,
    **opts.configurations[opts.model],
}

requests = []
for key, row in sample.set_index("key").iterrows():
    title = row.title if pd.notnull(row.title) else ""
    if title:
        title = f"TITLE:\n{title}"
    content = row.text if pd.notnull(row.text) else ""
    if content:
        content = f"TEXT:\n{content}"
    text = (f"{title}\n\n{content}").strip()
    if not text:
        continue
    request = {
        "custom_id": key,
        "country": row.country,
        "political": row.political,
        **header,
        "body": {"input": text, **body},
    }
    requests.append(request)

requests = DataFrame(requests)

# %% ---------------------------------------------------------------------------------

paths.gpt.mkdir(parents=True, exist_ok=True)
requests.to_(paths.gpt / "negativity-requests.jsonl.gz")

# %% ---------------------------------------------------------------------------------
