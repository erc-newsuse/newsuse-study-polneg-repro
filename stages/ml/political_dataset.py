# %% ---------------------------------------------------------------------------------

from newsuse.data import DataFrame
from newsuse.ml import Dataset, DatasetDict

from project import config, paths

LABELS = [n.upper() for n in config.categorical.political]

paths.political.mkdir(parents=True, exist_ok=True)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.political / "data.parquet").merge(
    DataFrame.from_(paths.political / "text.parquet"), on=["key", "split"], how="left"
)

# %% ---------------------------------------------------------------------------------

dataset = data.pipe(
    lambda df: DatasetDict(
        {
            key: Dataset.from_pandas(gdf.reset_index(drop=True).drop(columns="split"))
            for key, gdf in df.groupby("split")
        }
    )
)

# %% ---------------------------------------------------------------------------------

info = dataset["train"].info
info.features["label"].names = LABELS

dataset = dataset.update_info(info)

# %% ---------------------------------------------------------------------------------

dataset.save_to_disk(paths.ml / "datasets" / "political")

# %% ---------------------------------------------------------------------------------
