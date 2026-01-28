# %% ---------------------------------------------------------------------------------

import arviz as az
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn.objects as so
import xarray as xr
from newsuse.data import DataFrame

from project import config, paths

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
opts = config.glmm.valence.targets["event"]
support = np.asarray([*config.categorical["valence"]])

analysis_name = "structural"

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))

predictors_fixed = [*opts.common]
predictors_groups = [*opts.group]

# %% VALENCE TABLES ------------------------------------------------------------------


def bold_row(row: pd.Series) -> list[str]:
    frm = []
    for target in ["event", "sentiment", "valence"]:
        part = row[target]
        if pd.isnull(part["sig"]) or not part["sig"]:
            frm.extend([""] * len(part))
        else:
            frm.extend(["bfseries:--rwrap"] * len(part))
    return frm


# %% ---------------------------------------------------------------------------------

meta = {
    "event": "valence",
    "sentiment": "valence",
    "valence": "valence",
}

valence_tables = {}

for table in ["posterior", "political", "valence"]:
    parts = {}
    for name, kind in meta.items():
        df = (
            DataFrame.from_(paths.tables / kind / f"{name}-{table}.tsv")
            .rename(columns={name: "valence", "contrast": "valence"})
            .pipe(
                lambda df: df.set_index(
                    [c for c in ["country", "political", "valence"] if c in df.columns]
                )
            )
        )
        parts[name] = df
    valence_tables[table] = (
        pd.concat(parts, axis=1)
        .sort_index()
        .loc[["overall", *config.categorical.country]]
        .rename(
            # lambda x: config.categorical.country.get(x, "Overall"),
            lambda x: x.capitalize() if x == "overall" else x.upper(),
            axis="index",
            level="country",
        )
        .rename(
            lambda x: f"~{x}" if x >= 0 else str(x),
            axis="index",
            level="valence",
        )
    )

# %% ---------------------------------------------------------------------------------

print(
    valence_tables["posterior"]
    .style.format(precision=3, escape="latex", na_rep="")
    .to_latex(
        convert_css=False,
        multirow_align="t",
        hrules=True,
    )
)

# %% ---------------------------------------------------------------------------------

print(
    valence_tables["political"].pipe(
        lambda df: (
            df.style.hide(
                df.filter(regex=r"\'sig\'|\'up\'", axis=1).columns.tolist(), axis=1
            )
            .apply(bold_row, axis=1)
            .format(precision=3, escape="latex", na_rep="")
            .to_latex(
                convert_css=False,
                multirow_align="t",
                hrules=True,
            )
        )
    )
)

# %% ---------------------------------------------------------------------------------

print(
    valence_tables["valence"].pipe(
        lambda df: (
            df.style.hide(
                df.filter(regex=r"\'sig\'|\'up\'", axis=1).columns.tolist(), axis=1
            )
            .apply(bold_row, axis=1)
            .format(precision=3, escape="latex", na_rep="")
            .to_latex(
                convert_css=False,
                multirow_align="t",
                hrules=True,
            )
        )
    )
)

# %% ENGAGEMENT TABLES ---------------------------------------------------------------

engagement_tables = {}

for table in [
    "volume-posterior",
    "volume-political",
    "volume-valence",
    "valence-posterior",
    "baseline-diffs",
    "rr-political",
    "rr-valence",
]:
    parts = {}
    for target in config.engagement:
        for valence in ["event", "sentiment", "valence"]:
            tabname = f"{target}-{valence}-{table}"
            if tabname.endswith("valence-posterior") or "valence-rr" in tabname:
                continue
            df = DataFrame.from_(paths.tables / "engagement" / f"{tabname}.tsv").rename(
                columns={"event": "valence", "sentiment": "valence"}
            )
            df = (
                df.set_index(df.columns.tolist()[: df.columns.get_loc("median")])
                .loc[["overall", *config.categorical.country]]
                .rename(
                    # lambda x: config.categorical.country.get(x, "Overall"),
                    lambda x: x.capitalize() if x == "overall" else x.upper(),
                    axis="index",
                    level="country",
                )
            )
            parts[(target, valence)] = df.rename(
                columns={
                    "median": "Me",
                    "lower": "LB",
                    "upper": "UB",
                }
            )
    if parts:
        engagement_tables[table] = (
            pd.concat(parts, axis=1)
            .pipe(lambda df: df.sort_index().loc[df.index.get_level_values(0).unique()])
            .rename(
                lambda x: f"~{x}" if x >= 0 else str(x),
                axis="index",
                level="valence",
            )
        )

# %% ---------------------------------------------------------------------------------


def _bold_row(row: pd.Series) -> list[str]:
    frm = []
    for target in config.engagement:
        for valence in ["event", "sentiment", "valence"]:
            if (key := (target, valence)) not in row.index:
                continue
            part = row[key]
            if pd.isnull(part["sig"]) or not part["sig"]:
                frm.extend([""] * len(part))
            else:
                frm.extend(["bfseries:--rwrap"] * len(part))
    return frm


for name, df in engagement_tables.items():
    print(f"\n\n# Table: {name}\n")
    df = df.style
    if "sig" in sum(df.columns, ()):
        df = df.apply(_bold_row, axis=1)
        df = df.hide(
            df.data.filter(regex=r"\'sig\'|\'up\'", axis=1).columns.tolist(), axis=1
        )
    print(
        df.format(precision=2, escape="latex", na_rep="").to_latex(
            convert_css=False,
            multirow_align="t",
            multicol_align="c",
            hrules=True,
        )
    )

# %% ---------------------------------------------------------------------------------
