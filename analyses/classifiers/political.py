# %% ---------------------------------------------------------------------------------

import datasets
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from project import config, paths
from project.pipelines import KeyDataset, pipeline

# %% ---------------------------------------------------------------------------------


def compute_metrics(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "f1": f1_score(df["true"], df["pred"], average="macro"),
            "acc1": accuracy_score(df["true"], df["pred"]),
        }
    )


# %% ---------------------------------------------------------------------------------

opts = config.classification.political
inference_opts = config.ml.inference

if opts.model.source == "huggingface":
    model_source = opts.model.name
else:
    model_source = paths.mldata / opts.model.name

# %% ---------------------------------------------------------------------------------

dataset = datasets.load_from_disk(paths.ml / "datasets" / "political")
pipe = pipeline("text-classification", model_source)

# %% ---------------------------------------------------------------------------------

testing = pd.concat(
    [
        dataset["test"].to_pandas().set_index(["country", "key"]),
    ]
)

results = pd.DataFrame(
    tqdm(pipe(KeyDataset(testing, "text"), batch_size=16), total=len(testing)),
)

# %% ---------------------------------------------------------------------------------

output = results.set_index(testing.index).sort_index()
data = (
    output[["label"]]
    .rename(columns={"label": "pred"})
    .combine_first(testing[["label"]].rename(columns={"label": "true"}))
)

# %% ---------------------------------------------------------------------------------

# %% ---------------------------------------------------------------------------------

perf_overall = data.pipe(compute_metrics).to_frame().T

perf_country = (
    data.groupby(["country"])
    .apply(compute_metrics)
    .rename(str.upper, axis=0, level="country")
    .reset_index()
)

perf_country  # noqa  # type: ignore

# %% ---------------------------------------------------------------------------------

print(
    pd.concat([perf_overall, perf_country])
    .fillna({"country": "Overall"})
    .set_index(["country"])
    .sort_index()
    .T.rename(
        {"f1": r"$F_1$", "o1": r"$O_1$", "acc1": r"Acc."},
    )[["Overall", *map(str.upper, config.categorical.country)]]
    .style.format(precision=3)
    .to_latex(hrules=True)
)

# %% ---------------------------------------------------------------------------------
