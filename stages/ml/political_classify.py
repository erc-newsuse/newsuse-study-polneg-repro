# %% ---------------------------------------------------------------------------------

import pandas as pd
from newsuse.data import DataFrame
from tqdm.auto import tqdm

from project import config, paths
from project.pipelines import KeyDataset, pipeline

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.text, columns=["key", "fulltext"])
    .dropna(ignore_index=True)
    .rename(columns={"fulltext": "text"})
)

# %% ---------------------------------------------------------------------------------

opts = config.classification.political
inference_opts = config.ml.inference

if opts.model.source == "huggingface":
    model_source = opts.model.name
else:
    model_source = paths.mldata / opts.model.name

# %% ---------------------------------------------------------------------------------

classifier = pipeline("text-classification", model_source, padding=True)
dataset = KeyDataset(data, "text")

# %% ---------------------------------------------------------------------------------

results = DataFrame(tqdm(classifier(dataset, **inference_opts), total=len(dataset)))

# %% ---------------------------------------------------------------------------------

results = pd.concat([data[["key"]], results], axis=1).rename(
    columns={"label": "political", "score": "political_score"}
)

# %% ---------------------------------------------------------------------------------

results.to_(paths.cls_political)

# %% ---------------------------------------------------------------------------------
