# %% ---------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from newsuse.data import DataFrame
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

from project import config, paths

# %% ---------------------------------------------------------------------------------

dataset = DataFrame.from_(paths.dataset)
labels = DataFrame.from_(paths.labels)
sentiment = DataFrame.from_(paths.sentiment)

# %% ---------------------------------------------------------------------------------

data = (
    dataset.merge(sentiment[["key", "label"]], how="inner")
    .merge(labels[["key", "negativity"]], how="inner")
    .assign(
        negative=lambda df: (df["negativity"] == "NEGATIVE").astype(int),
        negative_sentiment=lambda df: (
            df["label"].isin(["Negative", "Very Negative"]).astype(int)
        ),
    )
    .rename(columns={"label": "sentiment"})
    .drop_duplicates(subset=["key"], ignore_index=True)
)

# %% ---------------------------------------------------------------------------------

pd.crosstab(
    data["sentiment"],
    data["negativity"],
    rownames=["predicted"],
    colnames=["true"],
    normalize="columns",
).loc[["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]]

# %% ---------------------------------------------------------------------------------

labels = ["non-negative", "negative"]
pred = np.where(data["sentiment"].isin(["Negative", "Very Negative"]), 1, 0)
true = np.where(data["negativity"] == "NEGATIVE", 1, 0)

print(classification_report(true, pred, target_names=labels, zero_division=0))

# %% ---------------------------------------------------------------------------------

cm = confusion_matrix(true, pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()

# %% ---------------------------------------------------------------------------------

(
    data.groupby(["country"]).apply(
        lambda df: f1_score(
            np.where(df["negativity"] == "NEGATIVE", 1, 0),
            np.where(df["sentiment"].isin(["Negative", "Very Negative"]), 1, 0),
        ),
        include_groups=False,
    )
)

# %% Sample examples manual analysis -------------------------------------------------

rng = np.random.default_rng(17)
sample = (
    data.groupby(["country", "negative", "sentiment"])[
        ["key", "country", "name", "timestamp", "negativity", "sentiment"]
    ]
    .apply(lambda df: df.sample(min(len(df), 10), random_state=rng))
    .merge(DataFrame.from_(paths.text), how="left", on="key")
)

loc = len(sample.columns) - 1
for annotator in reversed(config.sentiment.annotators):
    sample.insert(loc, annotator, pd.NA)

with pd.ExcelWriter(paths.proc / "sentiment-sample.xlsx") as writer:
    for country, df in sample.groupby("country"):
        df.to_excel(writer, sheet_name=country, index=False, freeze_panes=(1, 0))

# %% ---------------------------------------------------------------------------------
