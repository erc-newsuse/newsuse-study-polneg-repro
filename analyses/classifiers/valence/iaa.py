# %% ---------------------------------------------------------------------------------


import pandas as pd
from dlordinal.metrics import accuracy_off1
from newsuse.data import DataFrame
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from project import paths

here = paths.root / "analyses" / "classifiers" / "valence" / "_auxiliary"

data = DataFrame.from_(paths.raw / "valence-ground-truth.json.gz")

# %% ---------------------------------------------------------------------------------

records = []

for _, row in data.iterrows():
    task_id = row["id"]
    for ann in row["annotations"]:
        if ann.get("was_cancelled"):
            continue

        annotator = ann["completed_by"]["email"]

        res_map = {}
        for res in ann["result"]:
            if "value" in res and "choices" in res["value"]:
                res_map[res["from_name"]] = int(res["value"]["choices"][0])

        if "event" in res_map and "sentiment" in res_map:
            records.append(
                {
                    "task_id": task_id,
                    "annotator": annotator,
                    "event": res_map["event"],
                    "sentiment": res_map["sentiment"],
                }
            )

# %% ---------------------------------------------------------------------------------

annotations = (
    pd.DataFrame(records)
    .set_index(["annotator", "task_id"])
    .melt(value_vars=["event", "sentiment"], var_name="target", ignore_index=False)
    .sort_index()
)

pairs = (
    annotations.reset_index()
    .pipe(lambda df: df.merge(df, on=["task_id", "target"], suffixes=("1", "2")))
    .query("annotator1 != annotator2")
    .set_index(["annotator1", "annotator2", "task_id", "target"])
    .sort_index()
)

#  %% ---------------------------------------------------------------------------------

n_annotations = (
    annotations.groupby(["task_id", "target"]).size().groupby("target").describe()
)

scores = pairs.groupby(["annotator1", "annotator2", "target"]).apply(
    lambda df: pd.Series(
        {
            "Off-1 Accuracy": accuracy_off1(df["value1"], df["value2"]),
            "Mathew's Correlation Coefficient": matthews_corrcoef(
                df["value1"], df["value2"]
            ),
            "Cohen's Kappa": cohen_kappa_score(df["value1"], df["value2"]),
        }
    )
)

# %% ---------------------------------------------------------------------------------

average_scores = scores.groupby(["target"]).mean()
average_scores.pipe(lambda df: df.head(len(df)))

# %% ---------------------------------------------------------------------------------

print(
    average_scores.style.format(precision=3).to_latex(
        hrules=True,
    )
)

# %% ---------------------------------------------------------------------------------
