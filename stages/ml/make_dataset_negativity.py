# %% ---------------------------------------------------------------------------------

from newsuse.data import DataFrame
from newsuse.ml import Dataset, DatasetDict

from project import config, paths

paths.negativity.mkdir(parents=True, exist_ok=True)

LABELS = list(config.annotations.negativity.labels)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.negativity / "data.parquet").merge(
    DataFrame.from_(paths.negativity / "text.parquet"), on=["key", "split"], how="left"
)

# %% ---------------------------------------------------------------------------------

dataset = data.pipe(
    lambda df: DatasetDict(
        {
            key: Dataset.from_pandas(gdf.reset_index(drop=True).drop(columns="split"))
            for key, gdf in df.groupby("split")
        }
    )
).class_encode_column("label")

# %% ---------------------------------------------------------------------------------

info = dataset["train"].info
info.features["label"].names = LABELS

dataset = dataset.update_info(info)

# %% ---------------------------------------------------------------------------------

# tweet = load_dataset("cardiffnlp/tweet_eval", "sentiment")

# %% ---------------------------------------------------------------------------------

dataset.save_to_disk(paths.ml / "datasets" / "negativity")

# %% ---------------------------------------------------------------------------------
