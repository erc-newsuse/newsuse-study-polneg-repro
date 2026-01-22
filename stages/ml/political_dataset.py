# %% ---------------------------------------------------------------------------------

import datasets
import numpy as np
from datasets import Dataset, DatasetDict
from newsuse.data import DataFrame

from project import config, paths

domain = "political"
dirpath = paths.ml / "datasets"

datasets.disable_caching()

rng = np.random.default_rng(config.ml.dataset[domain].seed)

# %% ---------------------------------------------------------------------------------

ground_truth = DataFrame.from_(
    paths.raw / f"{domain}-ground-truth.parquet",
    columns=["key", "country", "label", "text"],
).sample(frac=1.0, random_state=config.ml.dataset[domain].seed, replace=False)

# %% ---------------------------------------------------------------------------------

n_train = int(len(ground_truth) * config.ml.dataset[domain].training)
n_test = int(len(ground_truth) * config.ml.dataset[domain].testing)

dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(ground_truth[:n_train]),
        "test": Dataset.from_pandas(ground_truth[n_train : n_train + n_test]),
        "valid": Dataset.from_pandas(ground_truth[n_train + n_test :]),
    }
)

# %% ---------------------------------------------------------------------------------

dataset.save_to_disk(paths.ml / "datasets" / domain)

# %% ---------------------------------------------------------------------------------
