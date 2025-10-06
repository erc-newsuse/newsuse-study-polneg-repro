# %% ---------------------------------------------------------------------------------


from newsuse.data import DataFrame
from openai.lib._pydantic import to_strict_json_schema

from project import config, paths
from project.gpt import NegativityClassification

COMPLETED = "completed"
IN_PROGRESS = "in_progress"

opts = config.gpt.negativity

# %% ---------------------------------------------------------------------------------

sample = (
    DataFrame.from_(paths.dataset, columns=["key", "country", "political"])
    .merge(DataFrame.from_(paths.text), on="key", how="left")
    .pipe(
        lambda df: df[df["text"].str.strip().str.split().map(len).ge(opts.sample.min_words)]
    )
    .query(f"country.isin({opts.sample.countries})")
    .query(f"political.isin({opts.sample.political})")
    .groupby(["country", "political"])
    .sample(n=opts.sample.size_per_group, random_state=opts.sample.seed)
    .reset_index(drop=True)
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
    # text = f"TITLE:\n{row.title}\n\nTEXT:\n{row.text}"
    text = f"TEXT:\n{row.text}"
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
