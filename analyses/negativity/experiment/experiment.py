# %% ---------------------------------------------------------------------------------

import matplotlib as mpl  # noqa
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa
import pandas as pd
import numpy as np  # noqa
from dlordinal.metrics import accuracy_off1
from newsuse.data import DataFrame
from sklearn.metrics import f1_score

from project import paths
from project.metrics import o1_score  # noqa

domain = "negativity"
targets = ["event", "sentiment"]
root = paths.root / "analyses" / domain
here = root / "experiment"

# %% ---------------------------------------------------------------------------------

ground_truth = (
    DataFrame.from_(paths.gpt / f"{domain}-ground-truth.parquet")
    .drop(columns=["text", "title"])
    .set_index(["key", "country"])
    .melt(ignore_index=False, var_name="target", value_name="label")
    .set_index("target", append=True)["label"]
)

# %% ---------------------------------------------------------------------------------

requests = (
    DataFrame.from_(paths.gpt / f"{domain}-experiment-requests.jsonl.gz")[
        ["key", "country", "params_id", "body", "target"]
    ]
    .assign(split=lambda df: df["target"] != "negativity")
    .assign(
        model=lambda df: df["body"].map(lambda x: x["model"]),
        reasoning=lambda df: df["body"].map(lambda x: x.get("reasoning", {}).get("effort")),
    )
    .assign(
        model=lambda df: np.where(
            df["reasoning"].notnull(),
            df["model"] + "[" + df["reasoning"] + "]",
            df["model"],
        )
    )
    .drop(columns=["body", "reasoning", "target"])
    .drop_duplicates(ignore_index=True)
)

# %% ---------------------------------------------------------------------------------

output = (
    DataFrame.from_(here / "experiment.parquet")
    .dropna()
    .merge(requests, on=["key", "params_id", "split"])
    .drop(columns=["params_id"])
    .set_index(["split", "model", "country", "key"])
    .astype("int64[pyarrow]")
    .dropna()
    .melt(ignore_index=False, var_name="target", value_name="label")
    .set_index("target", append=True)
    .reset_index()
    .merge(
        ground_truth.reset_index(),
        on=["key", "country", "target"],
        suffixes=("", "_t"),
    )
    .set_index(["split", "model", "country", "key", "target"])
    .sort_index()
)

# %% ---------------------------------------------------------------------------------

groups = output.index.names

f1_scores = output.groupby(level=groups).apply(
    lambda df: f1_score(df["label_t"], df["label"], average="macro")
)

# %% ---------------------------------------------------------------------------------

amae = []
for model in output.index.get_level_values("model").unique():
    model_output = output.loc[model]
    model_target = ground_truth.loc[model_output.index]
    scores = {"model": model}
    for target in targets:
        scores[target] = f1_score(model_output[target], model_target[target])
    amae.append(scores)

amae = DataFrame(amae).set_index("model")
amae_best = DataFrame(
    {"metric": "amae", "target": targets, "model": amae.idxmax(), "value": amae.max()}
)

# %% O1 score -------------------------------------------------------------------------

o1 = []
for model in output.index.get_level_values("model").unique():
    model_output = output.loc[model]
    model_target = ground_truth.loc[model_output.index]
    scores = {"model": model}
    for target in targets:
        scores[target] = f1_score(model_output[target], model_target[target])
    o1.append(scores)

o1 = DataFrame(o1).set_index("model")
o1_best = DataFrame(
    {"metric": "o1", "target": targets, "model": o1.idxmax(), "value": o1.max()}
)

# %% 1-off accuracy ------------------------------------------------------------------

acc = []
for model in output.index.get_level_values("model").unique():
    model_output = output.loc[model]
    model_target = ground_truth.loc[model_output.index]
    scores = {"model": model}
    for target in targets:
        scores[target] = accuracy_off1(model_output[target], model_target[target])
    acc.append(scores)

acc = DataFrame(acc).set_index("model")
acc_best = DataFrame(
    {"metric": "acc", "target": targets, "model": acc.idxmax(), "value": acc.max()}
)

# %% F1 score -------------------------------------------------------------------------

f1 = []
for model in output.index.get_level_values("model").unique():
    model_output = output.loc[model]
    model_target = ground_truth.loc[model_output.index]
    scores = {"model": model}
    for target in targets:
        scores[target] = f1_score(
            model_target[target],
            model_output[target],
            average="macro",
        )
    f1.append(scores)
f1 = DataFrame(f1).set_index("model")
f1_best = DataFrame(
    {"metric": "f1", "target": targets, "model": f1.idxmax(), "value": f1.max()}
)

# %% ---------------------------------------------------------------------------------

metrics = pd.concat(
    [amae_best, f1_best, o1_best, acc_best],
).set_index(["metric", "target", "model"])

print(metrics.reset_index().to_markdown(index=False, floatfmt=".3f"))

# %% Best models AMAE model by country -----------------------------------------------

f1_country = []
for model in amae_best["model"]:
    model_output = output.loc[model]
    model_target = ground_truth.loc[model_output.index]
    for country in model_output.index.get_level_values("country").unique():
        country_output = model_output.xs(country, level="country")
        country_target = model_target.xs(country, level="country")
        scores = {"model": model, "country": country}
        for target in targets:
            scores[target] = f1_score(
                country_output[target],
                country_target[target],
                average="macro",
            )
        f1_country.append(scores)

f1_country = DataFrame(f1_country).set_index(["model", "country"])
f1_overall = (
    f1_country.groupby("model")
    .mean()
    .assign(country="overall")
    .set_index("country", append=True)
)
f1_country = (
    pd.concat([f1_overall, f1_country])
    .swaplevel(axis=0)
    .loc[[*f1_country.index.get_level_values("country").unique(), "overall"]]
    .swaplevel(axis=0)
    .loc[f1_country.index.get_level_values("model").unique()]
)

# %% F1 score for events analysis ----------------------------------------------------

model = "gpt-5[medium]"
model_output = output.loc[model]
model_target = ground_truth.loc[model_output.index]

# %% Distributions of output labels --------------------------------------------------

output_dist = DataFrame(
    {
        t: model_output.groupby("country")[t].value_counts(normalize=True).sort_index()
        for t in targets
    }
)
target_dist = DataFrame(
    {
        t: model_target.groupby("country")[t].value_counts(normalize=True).sort_index()
        for t in targets
    }
)

dist = pd.concat({model: output_dist, "target": target_dist}).unstack(level=0)

dist.to_(here / "gpt-marginal-distributions.xlsx")

# %% ---------------------------------------------------------------------------------

countries = output_dist.index.get_level_values("country").unique()
fig, axes = plt.subplots(
    nrows=len(countries),
    ncols=len(targets),
    figsize=(7, 2 * len(countries)),
)

for axrow, country in zip(axes, countries, strict=True):
    axrow[0].set_ylabel(country.upper())
    axrow[1].set_ylabel(None)
    for ax, target in zip(axrow, targets, strict=True):
        df = (
            DataFrame(
                {
                    "target": target_dist.loc[country, target],
                    model: output_dist.loc[country, target],
                }
            )
            .melt(ignore_index=False)
            .reset_index(names="label")
        )
        sns.barplot(
            data=df,
            x="label",
            y="value",
            hue="variable",
            ax=ax,
            legend=False,
        )
        ax.set_xlabel(None)

for ax, target in zip(axes[0], targets, strict=True):
    ax.set_title(target.capitalize())

# Add custom legend
handles = [
    mpl.lines.Line2D(
        [], [], color=sns.color_palette()[i], marker="s", linestyle="", label=target
    )
    for i, target in enumerate(["target", model])
]
fig.legend(
    handles=handles,
    labels=(fig_labels := ["target", model]),
    loc="lower center",
    title="",
    ncols=len(fig_labels),
    bbox_to_anchor=(0.5, -0.02),
    frameon=False,
)

fig.tight_layout()


# %% ---------------------------------------------------------------------------------

scores = {}
for target in targets:
    df = pd.concat([model_target[target], model_output[target]], axis=1)
    scores[target] = (
        df.groupby("country")
        .apply(
            lambda df: pd.Series(
                f1_score(df.iloc[:, 0], df.iloc[:, 1], average=None),
                index=df.iloc[:, 0].unique(),
            )
        )
        .explode()
        .sort_index()
        .unstack()
    )

scores = pd.concat(scores, axis=1)

# %% ---------------------------------------------------------------------------------


def determine_valence(df: pd.DataFrame) -> pd.Series:
    return np.where(
        df[["event", "sentiment"]].ge(1).any(axis=1)
        & df[["event", "sentiment"]].ge(0).all(axis=1),
        1,
        np.where(
            df[["event", "sentiment"]].le(-1).any(axis=1)
            & df[["event", "sentiment"]].le(0).all(axis=1),
            -1,
            0,
        ),
    )


model_output = model_output.assign(
    negative_event=lambda df: (df["event"] == -1).astype("int64[pyarrow]"),
    negative_sentiment=lambda df: (df["sentiment"] == -1).astype("int64[pyarrow]"),
    # valence=lambda df: df["event"] + df["sentiment"],
    valence=lambda df: determine_valence(df),
)

model_target = model_target.assign(
    negative_event=lambda df: (df["event"] == -1).astype("int64[pyarrow]"),
    negative_sentiment=lambda df: (df["sentiment"] == -1).astype("int64[pyarrow]"),
    # valence=lambda df: df["event"] + df["sentiment"],
    valence=lambda df: determine_valence(df),
)

scores = {}
for target in ["negative_event", "negative_sentiment"]:
    df = pd.concat([model_target[target], model_output[target]], axis=1)
    scores[target] = (
        df.groupby("country")
        .apply(
            lambda df: pd.Series(
                f1_score(df.iloc[:, 0], df.iloc[:, 1], average=None),
                index=df.iloc[:, 0].unique(),
            )
        )
        .explode()
        .sort_index()
        .unstack()
    )

scores = pd.concat(scores, axis=1)

# %% ---------------------------------------------------------------------------------

target = "valence"
df = DataFrame({"target": model_target[target], "output": model_output[target]})

p5f1_raw = df.groupby("country").apply(
    lambda df: pd.Series(
        f1_score(
            df["target"],
            df["output"],  # type: ignore
            average=None,
        ),
        index=[-1, 0, 1],
        # index=[-2, -1, 0, 1, 2],
    )
)

p5f1 = df.groupby("country").apply(
    lambda df: pd.Series(
        f1_score(
            df["target"],
            df["output"],  # type: ignore
            average="weighted",
        ),
        index=["f1"],
    )
)

p5amae = df.groupby("country").apply(
    lambda df: pd.Series(
        f1_score(
            df["output"],
            df["target"],  # type: ignore
        ),
        index=["amae"],
    )
)

p5acc1 = df.groupby("country").apply(
    lambda df: pd.Series(
        accuracy_off1(
            df["output"],
            df["target"],  # type: ignore
        ),
        index=["acc1"],
    )
)

p5metrics = DataFrame(
    {
        "f1": p5f1["f1"],
        "amae": p5amae["amae"],
        "acc1": p5acc1["acc1"],
    }
)

metrics = pd.Series(
    {
        "f1": f1_score(df["target"], df["output"], average="macro"),
        "amae": f1_score(df["output"], df["target"]),
        "acc1": accuracy_off1(df["output"], df["target"]),
    }
)

# %% ---------------------------------------------------------------------------------

output_dist = (
    model_output.assign(
        # valence=lambda df: np.where(
        #     df["valence"] < 0, -1, np.where(df["valence"] > 0, 1, 0)
        # )
    )
    .groupby("country")["valence"]
    .value_counts(normalize=True)
    .sort_index()
    .reset_index(name=model)
)
target_dist = (
    model_target.assign(
        # valence=lambda df: np.where(
        #     df["valence"] < 0, -1, np.where(df["valence"] > 0, 1, 0)
        # )
    )
    .groupby("country")["valence"]
    .value_counts(normalize=True)
    .sort_index()
    .reset_index(name="target")
)
dist = (
    target_dist.merge(
        output_dist,
        on=["country", "valence"],
        how="outer",
    )
    .fillna(0.0)
    .melt(id_vars=["country", "valence"], var_name="type", value_name="proportion")
)

countries = dist["country"].unique()
nrows = countries.size
fig, axes = plt.subplots(
    nrows=nrows,
    figsize=(7, 2 * nrows),
)

for ax, country in zip(axes.flatten(), countries, strict=True):
    sns.barplot(
        x="valence",
        y="proportion",
        hue="type",
        data=dist.query(f"country.eq('{country}')"),
        legend=False,
        ax=ax,
    )
    ax.set_title(country.upper())
    ax.set_xlabel(None)
    ax.set_ylabel(None)

# Add custom legend
targets = ["target", model]
handles = [
    mpl.lines.Line2D(
        [], [], color=sns.color_palette()[i], marker="s", linestyle="", label=target
    )
    for i, target in enumerate(targets)
]
axes.flatten()[0].legend(
    handles=handles,
    labels=targets,
    loc="upper right",
    title="",
)

fig.supxlabel("Valence Score", y=0.01)
fig.supylabel("Proportion", x=0.01)
fig.tight_layout()

# %% ---------------------------------------------------------------------------------

labels = {}
for target in targets:
    labels[target] = model_output[target]
    labels[f"{target}_t"] = model_target[target]
labels = DataFrame(labels)

labels.to_(here / "labels-gpt-5-medium.xlsx")

# %% ---------------------------------------------------------------------------------
