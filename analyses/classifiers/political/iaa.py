# %% ---------------------------------------------------------------------------------


import pandas as pd
from newsuse.data import DataFrame
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef

from project import paths

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.raw / "political-ground-truth.parquet")
acols = data.filter(regex=r"^@", axis=1).columns.tolist()

# %% ---------------------------------------------------------------------------------

annotations = (
    data[["key", *acols]]
    .melt(id_vars=["key"], var_name="annotator", value_name="label")
    .dropna(ignore_index=True)
)

pairs = (
    annotations.pipe(lambda df: df.merge(df, on="key", suffixes=("1", "2")))
    .query("annotator1 != annotator2")
    .set_index(["annotator1", "annotator2", "key"])
    .sort_index()
)
# %% ---------------------------------------------------------------------------------

n_annotations = annotations.groupby(["key"]).size().describe()

n_pairs = pairs.groupby(["annotator1", "annotator2"]).size()

scores = pairs.groupby(["annotator1", "annotator2"]).apply(
    lambda df: pd.Series(
        {
            "Accuracy": accuracy_score(df["label1"], df["label2"]),
            "Matthew's Correlation Coefficient": matthews_corrcoef(
                df["label1"], df["label2"]
            ),
            "Cohen's Kappa": cohen_kappa_score(df["label1"], df["label2"]),
        }
    )
)

# %% ---------------------------------------------------------------------------------

average_scores = scores.mean().to_frame(name="Average").T
average_scores.pipe(lambda df: df.head(len(df)))

# %% ---------------------------------------------------------------------------------

print(
    average_scores.style.format(precision=3).to_latex(
        hrules=True,
    )
)

# %% ---------------------------------------------------------------------------------
