# %% ---------------------------------------------------------------------------------


import pandas as pd
from newsuse.data import DataFrame

from project import paths

keycols = [
    "key",
    "country",
    "name",
    "political",
]

# %% ---------------------------------------------------------------------------------

sheets = DataFrame.from_(paths.annotations / "annotations.xlsx", sheet_name=None)

# %% ---------------------------------------------------------------------------------


def get_event_col(*columns: str) -> str | None:
    for col in columns:
        _col = col.lower()
        if (label := "event") in _col and _col != label:
            return col
    return None


def get_sentiment_col(*columns: str) -> str | None:
    for col in columns:
        _col = col.lower()
        if "sent" in _col and _col != "sentiment":
            return col
    return None


# %% ---------------------------------------------------------------------------------

data = {}
for annotator, df in sheets.items():
    event_col = get_event_col(*df.columns)
    sent_col = get_sentiment_col(*df.columns)
    remap = {}
    usecols = list(keycols)
    if event_col:
        remap[event_col] = "event"
        usecols.append(event_col)
    else:
        df["event_h"] = pd.NA
        usecols.append("event_h")
    if sent_col:
        remap[sent_col] = "sentiment"
        usecols.append(sent_col)
    else:
        df["sentiment_h"] = pd.NA
        usecols.append("sentiment_h")
    df = df[usecols].rename(columns=remap)
    # for col in ["event", "sentiment"]:
    #     df[col] = df[col].astype(int)
    data[annotator] = df.convert_dtypes()

data = pd.concat(data).reset_index(level=-1, drop=True).reset_index(names=["annotator"])

# %% ---------------------------------------------------------------------------------

majority = (
    data.groupby(keycols)[["event", "sentiment"]]
    .median()
    .reset_index()
    .dropna(ignore_index=True)
)

majority.to_(paths.annotations / "majority.xlsx", index=False)

# %% ---------------------------------------------------------------------------------

labels = (
    DataFrame.from_(paths.annotations / "gpt-small.xlsx")
    .pipe(
        lambda df: df.drop(
            columns=[c for c in ["title", "text", "post_url"] if c in df.columns]
        )
    )
    .pipe(lambda df: df.set_index([k for k in keycols if k in df.columns]))
    .replace({"negative": "-1", "neutral": "0", "positive": "1"})
    .dropna()
    .astype(int)
    .convert_dtypes()
)

# %% ---------------------------------------------------------------------------------

diffs = labels - majority.set_index(keycols)

# %% Distributions -------------------------------------------------------------------

print(diffs["event"].abs().value_counts().to_markdown())
print(diffs["sentiment"].abs().value_counts().to_markdown())

# %% ---------------------------------------------------------------------------------

perf = diffs.abs().div(2).mul(-1).add(1).groupby(level="country").mean().dropna()

print(perf.to_markdown())

# %% ---------------------------------------------------------------------------------

acc = (diffs == 0).groupby(level="country").mean().dropna()

print(acc.to_markdown())

# %% ---------------------------------------------------------------------------------
