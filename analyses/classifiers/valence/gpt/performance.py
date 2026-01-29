# %% -------------------------------------------------------------------------------

import pandas as pd
from dlordinal.metrics import accuracy_off1
from newsuse.data import DataFrame
from sklearn.metrics import cohen_kappa_score, f1_score

from project import paths
from project.metrics import o1_score

domain = "valence"

# %% ---------------------------------------------------------------------------------

ground_truth = DataFrame.from_(
    paths.gpt / f"{domain}-ground-truth.parquet",
    columns=["key", "country", "event", "sentiment"],
).melt(
    id_vars=["key", "country"],
    value_vars=["event", "sentiment"],
    var_name="target",
    value_name="true",
)

gpt = DataFrame.from_(paths.gpt / f"{domain}.parquet").melt(
    id_vars=["key"],
    value_vars=["event", "sentiment"],
    var_name="target",
    value_name="pred",
)

data = ground_truth.merge(gpt, on=["key", "target"], how="left")

# %% ---------------------------------------------------------------------------------


def compute_metrics(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "f1": f1_score(df["true"], df["pred"], average="macro"),
            "o1": o1_score(df["true"], df["pred"]),
            "acc1": accuracy_off1(df["true"], df["pred"]),
            "kappa": cohen_kappa_score(df["true"], df["pred"]),
        }
    )


# %% ---------------------------------------------------------------------------------

data.groupby(["target"]).apply(compute_metrics, include_groups=False)

# %% ---------------------------------------------------------------------------------

data.groupby(["country", "target"]).apply(compute_metrics, include_groups=False)

# %% ---------------------------------------------------------------------------------
