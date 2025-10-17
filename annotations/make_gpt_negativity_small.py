# %% ---------------------------------------------------------------------------------

from copy import deepcopy

import pandas as pd
from newsuse.data import DataFrame
from omegaconf import OmegaConf
from openai import OpenAI
from tqdm.auto import tqdm

from project import config, paths
from project.gpt import NegativityClassification

opts = config.gpt.negativity

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

assert sample.key.is_unique, "Sample keys are not unique"

# %% ---------------------------------------------------------------------------------

with (paths.prompts / opts.prompt).open() as fh:
    prompt = fh.read().strip()

client = OpenAI()
params = {
    "model": opts.model,
    **OmegaConf.to_object(opts.configurations[opts.model]),
}

# %% -----------------------------------------------------------------------------------

outputs = []
for key, row in tqdm(sample.set_index("key").iterrows(), total=len(sample)):
    content = f"TITLE:\n{row.title}\n\nTEXT:\n{row.text}"
    response = client.responses.parse(
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        **deepcopy(params),
        text_format=NegativityClassification,
        metadata={"key": key},
    )
    output_text_idx = 1 if opts.model.startswith("gpt-5") else 0
    output = {
        **response.metadata,
        **response.output[output_text_idx].content[0].parsed.model_dump(),  # type: ignore
    }
    outputs.append(output)

# %% ---------------------------------------------------------------------------------

output = DataFrame(outputs)

# %% ---------------------------------------------------------------------------------

output.to_(paths.gpt / "negativity-small.parquet")

# %% Make annotation workbook --------------------------------------------------------

workbook = sample.merge(output).merge(
    DataFrame.from_(paths.dataset, columns=["key", "name", "post_url"]), how="left"
)

annotators = ("Magdalena", "Erin", "Dominik", "Szymon")
with pd.ExcelWriter(paths.annotations / "gpt-small.xlsx") as writer:
    for annotator in annotators:
        workbook.to_(writer, sheet_name=annotator, index=False)

# %% ---------------------------------------------------------------------------------
