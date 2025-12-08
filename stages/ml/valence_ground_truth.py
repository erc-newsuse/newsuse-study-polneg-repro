# %% ---------------------------------------------------------------------------------

from newsuse.data import DataFrame

from project import paths

domain = "valence"

# %% ---------------------------------------------------------------------------------


def get_ground_truth(annotations: list[dict]) -> dict:
    return next(
        {r["from_name"]: int(r["value"]["choices"][0]) for r in ann["result"]}
        for ann in annotations
    )


ground_truth = (
    DataFrame.from_(paths.aux / f"{domain}-ground-truth.json.gz")
    .assign(
        key=lambda df: df["data"].map(lambda d: d["key"]),
        country=lambda df: df["data"].map(lambda d: d["country"]),
        title=lambda df: df["data"].map(lambda d: d.get("title", "")),
        text=lambda df: df["data"].map(lambda d: d.get("text", "")),
        labels=lambda df: df["annotations"].map(get_ground_truth),
    )
    .set_index(["key", "country", "title", "text"])["labels"]
    .pipe(lambda s: DataFrame(s.tolist(), index=s.index))
    .reset_index()
)

assert ground_truth.key.is_unique, "Ground truth keys are not unique"

# %% ---------------------------------------------------------------------------------

ground_truth.to_(paths.gpt / f"{domain}-ground-truth.parquet")

# %% ---------------------------------------------------------------------------------
