# %% ---------------------------------------------------------------------------------

import datasets
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd
import seaborn as sns  # noqa
from dlordinal.metrics import accuracy_off1
from newsuse.data import DataFrame
from scipy.stats import hmean
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import project.model  # noqa
from project import paths
from project.metrics import amae_score, o1_score
from project.pipelines import KeyDataset, pipeline

domain = "negativity"
targets = ["event", "sentiment"]

# %% ---------------------------------------------------------------------------------

hyper = DataFrame.from_(paths.proc / f"{domain}-hyper.parquet")
best_model = hyper.loc[hyper.value.idxmax()].params_base

# %% ---------------------------------------------------------------------------------

dataset = datasets.load_from_disk(paths.ml / "datasets" / domain)
model = AutoModel.from_pretrained(paths.ml / "models" / domain / best_model)
tokenizer = AutoTokenizer.from_pretrained(model.config.base_name_or_path)

# %% ---------------------------------------------------------------------------------

pipe = pipeline("text-multi-classification", model=model, tokenizer=tokenizer)

# %% ---------------------------------------------------------------------------------

testing = dataset["test"].to_pandas().set_index(["ground_truth", "country", "key"])
results = DataFrame(
    tqdm(pipe(KeyDataset(testing, "text"), batch_size=16), total=len(testing)),
)

# %% ---------------------------------------------------------------------------------

output = pd.concat(
    [
        DataFrame(results[t].tolist()).rename(columns={"label": t, "score": f"{t}_score"})
        for t in targets
    ],
    axis=1,
).set_index(testing.index)

# %% ---------------------------------------------------------------------------------

data = (
    output[["event", "sentiment"]]
    .merge(
        testing[["event", "sentiment"]],
        left_index=True,
        right_index=True,
        suffixes=("", "_t"),
    )
    .pipe(
        lambda df: pd.concat(
            {
                t: df[[t, f"{t}_t"]].rename(columns={t: "pred", f"{t}_t": "true"})
                for t in targets
            },
            axis=1,
            names=["target"],
        )
    )
    .stack(level=0, future_stack=True)
    .swaplevel("key", "target")
    .sort_index()
)

# %% ---------------------------------------------------------------------------------


def compute_metrics(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "f1": hmean(f1_score(df["true"], df["pred"], average="macro")),
            "o1": o1_score(df["true"], df["pred"]),
            "amae": amae_score(df["true"], df["pred"]),
            "acc1": accuracy_off1(df["true"], df["pred"]),
        }
    )


# %% ---------------------------------------------------------------------------------

perf = data.groupby("target").apply(compute_metrics)

print(perf.to_markdown(floatfmt=".3f"))

# %% ---------------------------------------------------------------------------------

perf_country = data.groupby(["country", "target"]).apply(compute_metrics)

# %% ---------------------------------------------------------------------------------

countries = data.index.get_level_values("country").unique()

fig, axes = plt.subplots(
    nrows=len(countries),
    ncols=len(targets),
    figsize=(7, 2 * len(countries)),
)

for axrow, country in zip(axes, countries, strict=True):
    axrow[0].set_title(country.upper())
    for (target, df), ax in zip(data.groupby("target"), axrow.flat, strict=True):
        odist = (
            df.xs(country, level="country")
            .xs(target, level="target")["pred"]
            .value_counts(normalize=True)
            .sort_index()
        )
        tdist = (
            df.xs(country, level="country")
            .xs(target, level="target")["true"]
            .value_counts(normalize=True)
            .sort_index()
        )
        df = (
            DataFrame({"output": odist, "target": tdist})
            .melt(ignore_index=False)
            .reset_index()
        )
        sns.barplot(
            data=df,
            x="index",
            y="value",
            hue="variable",
            ax=ax,
        )

fig.tight_layout()

# %% Validation on ground truth labels -----------------------------------------------

ground_truth = data.xs(True, level="ground_truth")

# %% ---------------------------------------------------------------------------------

perf_gt = ground_truth.groupby(level="target").apply(compute_metrics)

# %% ---------------------------------------------------------------------------------

countries = data.index.get_level_values("country").unique()

fig, axes = plt.subplots(
    nrows=len(countries),
    ncols=len(targets),
    figsize=(7, 2 * len(countries)),
)

for axrow, country in zip(axes, countries, strict=True):
    axrow[0].set_title(country.upper())
    for (target, df), ax in zip(ground_truth.groupby("target"), axrow.flat, strict=True):
        odist = (
            df.xs(country, level="country")
            .xs(target, level="target")["pred"]
            .value_counts(normalize=True)
            .sort_index()
        )
        tdist = (
            df.xs(country, level="country")
            .xs(target, level="target")["true"]
            .value_counts(normalize=True)
            .sort_index()
        )
        df = (
            DataFrame({"output": odist, "target": tdist})
            .melt(ignore_index=False)
            .reset_index()
        )
        sns.barplot(
            data=df,
            x="index",
            y="value",
            hue="variable",
            ax=ax,
        )

fig.tight_layout()

# %% ---------------------------------------------------------------------------------


def determine_valence(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            df.ge(1).any(axis=1) & df.ge(0).all(axis=1),
            1,
            np.where(
                df.le(-1).any(axis=1) & df.le(0).all(axis=1),
                -1,
                0,
            ),
        ),
        index=df.index,
        name="valence",
    )


def determine_strong_valence(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(df.ge(1).all(axis=1), 1, np.where(df.le(-1).all(axis=1), -1, 0)),
        index=df.index,
        name="strong_valence",
    )


# %% ---------------------------------------------------------------------------------

labels = (
    data.unstack("target")
    .stack(level=0, future_stack=True)
    .assign(
        valence=lambda df: determine_valence(df),
        strong_valence=lambda df: determine_strong_valence(df),
        additive=lambda df: df["event"] + df["sentiment"],
    )
    .unstack(level=-1)
)

# %% ---------------------------------------------------------------------------------

(data.groupby(["ground_truth", "target"]).apply(compute_metrics))

# %% ---------------------------------------------------------------------------------

labels["valence"].groupby("ground_truth").apply(compute_metrics)

# %% ---------------------------------------------------------------------------------

labels["strong_valence"].groupby("ground_truth").apply(compute_metrics)

# %% ---------------------------------------------------------------------------------

labels["additive"].groupby("ground_truth").apply(compute_metrics)

# %% ---------------------------------------------------------------------------------
