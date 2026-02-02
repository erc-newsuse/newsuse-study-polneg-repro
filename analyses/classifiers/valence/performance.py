# %% ---------------------------------------------------------------------------------

import datasets
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd
import seaborn as sns  # noqa
from dlordinal.metrics import accuracy_off1
from newsuse.data import DataFrame
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import project.ml  # noqa
from project import config, paths
from project.metrics import o1_score
from project.pipelines import KeyDataset, pipeline

domain = "valence"
targets = ["event", "sentiment"]

# %% ---------------------------------------------------------------------------------

dataset = datasets.load_from_disk(paths.ml / "datasets" / domain)
model = AutoModel.from_pretrained(paths.ml / "models" / domain / "best")
tokenizer = AutoTokenizer.from_pretrained(model.config.base_name_or_path)

# %% ---------------------------------------------------------------------------------

dset = (
    pd.concat(
        {split: dset.to_pandas() for split, dset in dataset.items()},
        names=["split"],
    )
    .reset_index("split")
    .reset_index(drop=True)
)
sizes = (
    dset.groupby(["split", "country"])
    .size()
    .swaplevel("split", "country")
    .loc[[*config.categorical.country]]
)

print(
    sizes.unstack("country")[[*config.categorical.country]]
    .rename(config.categorical.country, axis=1, level="country")
    .astype(str)
    .map(lambda x: rf"\num{{{x}}}")
    .style.to_latex(
        hrules=True,
    )
)

# %% ---------------------------------------------------------------------------------

pipe = pipeline("text-multi-classification", model=model, tokenizer=tokenizer)

# %% ---------------------------------------------------------------------------------

testing = pd.concat(
    [
        dataset["test"].to_pandas(),
        # dataset["valid"].to_pandas(),
    ]
).set_index(["ground_truth", "country", "key"])

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
    labels = [-1, 0, 1]
    return pd.Series(
        {
            "f1": f1_score(df["true"], df["pred"], average="macro", labels=labels),
            "o1": o1_score(df["true"], df["pred"], labels=labels),
            "acc1": accuracy_off1(df["true"], df["pred"], labels=labels),
        }
    )


# %% ---------------------------------------------------------------------------------

perf_overall = data.groupby("target").apply(compute_metrics).reset_index()

perf_country = (
    data.groupby(["country", "target"])
    .apply(compute_metrics)
    .rename(str.upper, axis=0, level="country")
    .reset_index()
)

perf_country  # noqa  # type: ignore

print(
    pd.concat([perf_overall, perf_country])
    .fillna({"country": "Overall"})
    .set_index(["country", "target"])
    .sort_index()
    .unstack("country")
    .stack(level=0, future_stack=True)
    .rename(
        {"f1": r"$F_1$", "o1": r"$O_1$", "acc1": r"$\pm 1$ Acc."},
    )[["Overall", *map(str.upper, config.categorical.country)]]
    .style.format(precision=3)
    .to_latex(hrules=True)
)

# %% Validation on ground truth labels -----------------------------------------------

ground_truth = (
    pd.concat(
        {
            split: ds.to_pandas()
            for split, ds in dataset.filter(lambda x: x["ground_truth"]).items()
        },
        names=["split"],
    )
    .reset_index("split")
    .reset_index(drop=True)
    .set_index(["split", "country", "key"])
    .sort_index()
)
labels = (
    DataFrame(
        tqdm(
            pipe(KeyDataset(ground_truth, "text"), batch_size=16), total=len(ground_truth)
        ),
    )
    .pipe(
        lambda df: pd.concat(
            [
                DataFrame(df[t].tolist()).rename(
                    columns={"label": t, "score": f"{t}_score"}
                )
                for t in targets
            ],
            axis=1,
        )
    )
    .set_index(ground_truth.index)
)

# %% ---------------------------------------------------------------------------------

data_gt = (
    labels[["event", "sentiment"]]
    .merge(
        ground_truth[["event", "sentiment"]],
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

perf_gt = data_gt.loc[["valid"]].groupby(level="target").apply(compute_metrics)
perf_gt.head()

# %% ---------------------------------------------------------------------------------
# Additional validation (LLM)

samples = {}
for model in ["llm", "transformer"]:
    samples[model] = (
        DataFrame.from_(paths.aux / f"{model}-sample.xlsx")[
            ["country", "key", "event", "sentiment", "event_mw", "sentiment_mw"]
        ]
        .assign(
            country=lambda df: df["country"].ffill(),
            event_mw=lambda df: df["event_mw"].combine_first(df["event"]),
            sentiment_mw=lambda df: df["sentiment_mw"].combine_first(df["sentiment"]),
        )
        .set_index(["country", "key"])
        .melt(ignore_index=False)
        .assign(
            target=lambda df: np.where(
                df["variable"].str.startswith("event"), "event", "sentiment"
            ),
            which=lambda df: np.where(df["variable"].str.endswith("_mw"), "true", "pred"),
        )
        .drop(columns=["variable"])
        .set_index(["target", "which"], append=True)
        .pipe(lambda df: df[~df.index.duplicated(keep="first")])["value"]
        .unstack("which")
        .sort_index()
        .astype(int)
        .map(lambda x: x if x in [-1, 0, 1] else np.nan)
        .dropna()
    )

# %% ---------------------------------------------------------------------------------

samples["llm"].groupby(level=["target"]).apply(compute_metrics)

# %% ---------------------------------------------------------------------------------

samples["transformer"].groupby(level=["target"]).apply(compute_metrics)

# %% ---------------------------------------------------------------------------------
