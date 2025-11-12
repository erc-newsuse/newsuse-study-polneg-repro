# %% ---------------------------------------------------------------------------------

import datasets
import matplotlib.pyplot as plt  # noqa
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

validation = dataset["valid"].to_pandas().set_index("key")
results = DataFrame(
    tqdm(pipe(KeyDataset(validation, "text"), batch_size=16), total=len(validation)),
)

# %% ---------------------------------------------------------------------------------

output = pd.concat(
    [
        DataFrame(results[t].tolist()).rename(columns={"label": t, "score": f"{t}_score"})
        for t in targets
    ],
    axis=1,
).set_index(validation.index)

# %% ---------------------------------------------------------------------------------

performance = DataFrame(
    [
        {
            "amae": amae_score(validation[t], output[t]),
            "f1": hmean(f1_score(validation[t], output[t], average="macro")),
            "o1": o1_score(validation[t], output[t]),
            "acc1": accuracy_off1(validation[t], output[t]),
        }
        for t in targets
    ],
    index=pd.Series(targets, name="target"),
)

print(performance.to_markdown(floatfmt=".3f"))

# %% Validation on ground truth labels -----------------------------------------------

ground_truth = DataFrame.from_(paths.gpt / f"{domain}-ground-truth.parquet").assign(
    text=lambda df: df.pop("title") + "\n\n" + df.pop("text")
)

# %% ---------------------------------------------------------------------------------

results_gt = DataFrame(
    tqdm(
        pipe(KeyDataset(ground_truth, "text"), batch_size=16),
        total=len(ground_truth),
    ),
)

# %% ---------------------------------------------------------------------------------

output_gt = pd.concat(
    [
        DataFrame(results_gt[t].tolist()).rename(
            columns={"label": t, "score": f"{t}_score"}
        )
        for t in targets
    ],
    axis=1,
).set_index(ground_truth.index)

# %% ---------------------------------------------------------------------------------

performance_gt = DataFrame(
    [
        {
            "amae": amae_score(ground_truth[t], output_gt[t]),
            "f1": hmean(f1_score(ground_truth[t], output_gt[t], average="macro")),
            "o1": o1_score(ground_truth[t], output_gt[t]),
            "acc1": accuracy_off1(ground_truth[t], output_gt[t]),
        }
        for t in targets
    ],
    index=pd.Series(targets, name="target"),
)

print(performance_gt.to_markdown(floatfmt=".3f"))

# %% ---------------------------------------------------------------------------------

countries = ground_truth["country"].unique()
output_gt["country"] = ground_truth["country"]

fig, axes = plt.subplots(
    nrows=len(countries),
    ncols=len(targets),
    figsize=(7, 2 * len(countries)),
)

for axrow, country in zip(axes, countries, strict=True):
    axrow[0].set_title(country.upper())
    for target, ax in zip(targets, axrow, strict=True):
        odist = (
            output_gt.loc[output_gt["country"] == country, target]
            .value_counts(normalize=True)
            .sort_index()
        )
        tdist = (
            ground_truth.loc[ground_truth["country"] == country, target]
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
            x=target,
            y="value",
            hue="variable",
            ax=ax,
        )

fig.tight_layout()

# %% ---------------------------------------------------------------------------------
