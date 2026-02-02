# %% ---------------------------------------------------------------------------------

import pandas as pd
from newsuse.data import DataFrame
from tqdm.auto import tqdm

from project import config, paths
from project.ml import KeyDataset, pipeline

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.text, columns=["key", "fulltext"])
    .dropna(ignore_index=True)
    .rename(columns={"fulltext": "text"})
)

# %% ---------------------------------------------------------------------------------

opts = config.ml.classifiers.political
classifier = pipeline(opts.task, opts.name, padding=True)
dataset = KeyDataset(data, "text")

# %% ---------------------------------------------------------------------------------

results = DataFrame(tqdm(classifier(dataset, **config.ml.inference), total=len(dataset)))

# %% ---------------------------------------------------------------------------------

results = pd.concat([data[["key"]], results], axis=1).rename(
    columns={"label": "political", "score": "political_score"}
)

# %% ---------------------------------------------------------------------------------

results.to_(paths.cls_political)

# %% ---------------------------------------------------------------------------------
