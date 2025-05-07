# %% Setup -------------------------------------------------------------------------------

from typing import Literal

import pandas as pd
from newsuse.cli import command
from newsuse.data import DataFrame
from newsuse.ml import pipeline

from project import config, paths


@command
class Args:
    model: Literal["political", "negativity"]


ARGS = Args()

paths = paths.__copy__(model=f"@ml/classifiers/{ARGS.model}")

# %% Get data and model ------------------------------------------------------------------

dataset = DataFrame.from_(paths.text)
classifier = pipeline("text-classification", paths.model)

# %% Classify ----------------------------------------------------------------------------

results = DataFrame(
    classifier(dataset, progress=True, batch_size=config.ml.inference.batch_size)
)

# %% Augment data ------------------------------------------------------------------------

data = (
    pd.concat([dataset.drop(columns="text"), results], axis=1)
    .rename({"label": ARGS.model, "score": f"{ARGS.model}_score"})
    .convert_dtypes(dtype_backend="pyarrow")
)

# %% Save data ---------------------------------------------------------------------------

data.to_(paths.proc / f"cls-{ARGS.model}.parquet")

# %% -------------------------------------------------------------------------------------
