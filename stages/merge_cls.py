# %% Setup ---------------------------------------------------------------------------

from newsuse.data import DataFrame

from project import paths

# %% Get data ------------------------------------------------------------------------

data = []
for path in paths.proc.glob("cls-*.parquet"):
    model = path.stem.removeprefix("cls-")
    df = DataFrame.from_(path).rename(columns={"label": model, "score": f"{model}_score"})
    data.append(df)

data, *other = data
for df in other:
    data = data.merge(df, how="left", on="key")

# %% ---------------------------------------------------------------------------------

data = data.assign(valence=lambda df: df["event"] + df["sentiment"])

# %% Save data -----------------------------------------------------------------------

data.to_(paths.proc / "cls.parquet")

# %% ---------------------------------------------------------------------------------
