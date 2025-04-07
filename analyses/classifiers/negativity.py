# %% Prepare environment and main objects ============================================
from newsuse.ml import Dataset, Evaluator, pipeline

from project import config, paths

config["model"] = "negativity"
paths = paths.__copy__(
    model=f"@ml/classifiers/{config.model}",
    examples=f"@ml/datasets/{config.model}",
)

dataset = Dataset.from_disk(paths.examples)
classifier = pipeline("text-classification", paths.model)
performance = Evaluator(classifier)

# %% Performance on the validation set ===============================================
data = dataset["validation"].to_pandas()
performance(data)

# %% Performance by country
data.groupby("country").apply(performance, include_groups=False)


# %% Dataset statistics ==============================================================

# %% Training dataset
dataset["train"].to_pandas().groupby(["country", "label"]).size()

# %% Test dataset
dataset["test"].to_pandas().groupby(["country", "label"]).size()

# %% Validation dataset
dataset["validation"].to_pandas().groupby(["country", "label"]).size()

# %%
