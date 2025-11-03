# %% ---------------------------------------------------------------------------------

from newsuse.data import DataFrame

from project import paths

here = paths.root / "analyses" / "negativity"

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.dataset, columns=["key", "country", "name", "post_url"])
    .merge(DataFrame.from_(paths.text))
    .merge(
        DataFrame.from_(paths.gpt / "negativity.parquet"),
        on=["key", "country"],
        how="inner",
    )
    .assign(
        nevent=lambda df: df["event"].eq(-1),
        nsentiment=lambda df: df["sentiment"].eq(-1),
    )
    .groupby(["country", "nevent", "nsentiment"])
    .sample(n=20, random_state=348024)
    .drop(columns=["nevent", "nsentiment"])
    .reset_index(drop=True)
)

# %% ---------------------------------------------------------------------------------

data.to_(here / "negativity-openai-dataset.jsonl")

# %% ---------------------------------------------------------------------------------
