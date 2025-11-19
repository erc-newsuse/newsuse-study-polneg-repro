# %% ---------------------------------------------------------------------------------

import re

import datasets
import numpy as np
from datasets import Dataset, DatasetDict
from newsuse.data import DataFrame

from project import config, paths

domain = "valence"
dirpath = paths.ml / "datasets"

rng = np.random.default_rng(config.ml.dataset.valence.seed)

datasets.disable_caching()

rx_headers = re.compile(r"^(TITLE:|TEXT:)\n", re.MULTILINE)

# %% ---------------------------------------------------------------------------------

ground_truth = DataFrame.from_(paths.gpt / f"{domain}-ground-truth.parquet")

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.gpt / f"{domain}-requests.jsonl.gz")
    .rename(columns={"custom_id": "key"})
    .drop_duplicates(subset=["key", "country"], ignore_index=True)
    .assign(text=lambda df: df["body"].map(lambda s: s["input"]))[
        ["key", "country", "text"]
    ]
    .merge(DataFrame.from_(paths.gpt / f"{domain}.parquet"), on=["key"], how="inner")
    .dropna(ignore_index=True)
    .assign(text=lambda df: df["text"].str.replace(rx_headers, "", regex=True).str.strip())
    .assign(ground_truth=lambda df: df["key"].isin(ground_truth["key"]))
)

# %% ---------------------------------------------------------------------------------

index = data.index.to_numpy()
rng.shuffle(index)

# %% ---------------------------------------------------------------------------------

n_train = int(len(data) * config.ml.dataset.valence.training)
n_test = int(len(data) * config.ml.dataset.valence.testing)

index_train = index[:n_train]
index_test = index[n_train : n_train + n_test]
index_valid = index[n_train + n_test :]

# %% ---------------------------------------------------------------------------------

dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(data.loc[index_train].reset_index(drop=True)),
        "test": Dataset.from_pandas(data.loc[index_test].reset_index(drop=True)),
        "valid": Dataset.from_pandas(data.loc[index_valid].reset_index(drop=True)),
    }
)

# %% ---------------------------------------------------------------------------------

dataset.save_to_disk(dirpath / domain)

# %% ---------------------------------------------------------------------------------
