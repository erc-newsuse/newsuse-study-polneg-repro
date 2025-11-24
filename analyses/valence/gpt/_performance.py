# %% ---------------------------------------------------------------------------------

import json
from functools import partial

import matplotlib as mpl  # noqa
import matplotlib.pyplot as plt  # noqa
import pandas as pd
import seaborn as sns  # noqa
from dlordinal.metrics import accuracy_off1
from newsuse.data import DataFrame
from sklearn.metrics import f1_score

from project import paths
from project.metrics import amae_score, o1_score

here = paths.root / "analyses" / "valence" / "gpt"

prompt = "prompt_6"
targets = ["event", "sentiment"]

metric_funs = {
    "f1": partial(f1_score, average="macro"),
    "o1": o1_score,
    "amae": amae_score,
    "acc1": accuracy_off1,
}

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(here / "Valence-GPT-Optimized.csv")[
        ["key", "country", "event", "sentiment", f"{prompt}_output"]
    ]
    .rename(columns={f"{prompt}_output": "output"})
    .assign(output=lambda df: df["output"].apply(json.loads))
    .assign(
        output=lambda df: df["output"].apply(
            lambda x: json.loads(x["output"][-1]["content"][-1]["text"])
        )
    )
    .assign(
        gpt_event=lambda df: df["output"].apply(lambda x: x["event"]),
        gpt_sentiment=lambda df: df["output"].apply(lambda x: x["sentiment"]),
    )
    .drop(columns=["output"])
)

# %% ---------------------------------------------------------------------------------


def compute_metrics(data: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        DataFrame(
            [
                {
                    "target": target,
                    **{
                        name: func(data[target], data[f"gpt_{target}"])
                        for name, func in metric_funs.items()
                    },
                }
                for target in targets
            ]
        )
        .set_index("target")
        .T
    )
    return metrics


# %% ---------------------------------------------------------------------------------

metrics = (
    pd.concat(
        [
            compute_metrics(data).assign(country="overall"),
            data.groupby("country")
            .apply(compute_metrics, include_groups=False)
            .reset_index(level="country"),
        ],
        axis=0,
    )
    .set_index("country", append=True)
    .swaplevel(0, 1)
    .reset_index(names=["country", "metric"])
    .set_index(["country", "metric"])
)

print(metrics.reset_index().to_markdown(floatfmt=".3f", index=False))

# %% ---------------------------------------------------------------------------------

distdata = data[["country", *targets, *(f"gpt_{t}" for t in targets)]].pipe(
    lambda df: pd.concat(
        [
            df[["country", *targets]].assign(type="true"),
            df[["country", *(f"gpt_{t}" for t in targets)]]
            .rename(columns={f"gpt_{t}": t for t in targets})
            .assign(type="gpt"),
        ]
    )
)

# %% ---------------------------------------------------------------------------------

dist = (
    distdata.groupby(["country", "type"])
    .apply(lambda df: DataFrame({t: df[t].value_counts(normalize=True) for t in targets}))
    .unstack(level="type")
)

# %% ---------------------------------------------------------------------------------

countries = dist.index.get_level_values("country").unique().tolist()

fig, axes = plt.subplots(
    nrows=len(countries),
    ncols=len(targets),
    figsize=(7, 2 * len(countries)),
)

for axrow, country in zip(axes, countries, strict=True):
    for target, ax in zip(targets, axrow, strict=True):
        df = dist.loc[country, target].melt(ignore_index=False).reset_index(names=["label"])
        sns.barplot(df, x="label", y="value", hue="type", ax=ax, legend=False)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        if ax in axes[0]:
            ax.set_title(target.capitalize())
    axrow[0].set_ylabel(country.upper())

handles = [
    mpl.lines.Line2D([], [], color=sns.color_palette()[i], label=label)
    for i, label in enumerate(["True", "GPT"])
]
axes.flatten()[0].legend(
    handles=handles,
    loc="best",
    frameon=False,
)

fig.supxlabel("Label", y=0.02)
fig.supylabel("Proportion", x=0.02)
fig.tight_layout()

# %% ---------------------------------------------------------------------------------
