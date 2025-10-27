# %% ---------------------------------------------------------------------------------

import datasets
import numpy as np
from datasets import Dataset, DatasetDict
from newsuse.data import DataFrame

from project import config, paths

target = "negativity"
dirpath = paths.ml / "datasets"

rng = np.random.default_rng(config.ml.dataset.negativity.seed)

datasets.disable_caching()

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.gpt / f"{target}-requests.jsonl.gz")
    .rename(columns={"custom_id": "key"})
    .assign(text=lambda df: df["body"].map(lambda s: s["input"]))[
        ["key", "country", "text"]
    ]
    .merge(
        DataFrame.from_(paths.gpt / f"{target}.parquet"), on=["key", "country"], how="inner"
    )
    .dropna(ignore_index=True)
)

# %% ---------------------------------------------------------------------------------

index = data.index.to_numpy()
rng.shuffle(index)

# %% ---------------------------------------------------------------------------------

n_train = int(len(data) * config.ml.dataset.negativity.training)
n_test = int(len(data) * config.ml.dataset.negativity.testing)

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

dataset.save_to_disk(dirpath / target)

# %% ---------------------------------------------------------------------------------
