# %% ---------------------------------------------------------------------------------

import os

import numpy as np
import pandas as pd
from newsuse.data import DataFrame

from project import config, paths
from project.rutils import df_to_r, make_formula, ro

ro.r("library(glmmTMB)")

# %% ---------------------------------------------------------------------------------

target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target name (reactions): ").strip() or "reactions"
opts = config.glmm.engagement.targets[target]

dirpath = paths.glmm / "engagement"
dirpath.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(opts.seed)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final).assign(
    country=lambda df: pd.Categorical(
        df["country"], categories=[*config.categorical.country]
    ),
    # Use unordered factors with 0 first for dummy coding with neutral as reference
    event=lambda df: pd.Categorical(
        df["event"],
        categories=[0, -1, 1],
        ordered=False,
    ),
    sentiment=lambda df: pd.Categorical(
        df["sentiment"],
        categories=[0, -1, 1],
        ordered=False,
    ),
    valence=lambda df: pd.Categorical(
        df["valence"], categories=[*config.categorical.valence], ordered=True
    ),
)[["key", target, *opts.predictors.fixed, *opts.predictors.groups]]

# %% ---------------------------------------------------------------------------------

kwargs = {
    "data": df_to_r(data),
    "formula": make_formula(opts.model.formula.format(target=target)),
    "ziformula": make_formula(opts.model.ziformula),
    "dispformula": make_formula(opts.model.dispformula),
    "family": ro.r(opts.model.family),
    "control": ro.ListVector(
        {
            "profile": opts.control.profile,
            "parallel": ro.ListVector(dict(opts.control.parallel)),
        }
    ),
}

# %% ---------------------------------------------------------------------------------

print(f"Fitting GLMM for '{target}' with {data.shape[0]} observations")
model = ro.r("glmmTMB")(**kwargs)

# %% ---------------------------------------------------------------------------------

assert (nobs := ro.r("nobs")(model)[0]) == len(
    data
), f"Fitted 'model' has {nobs} observations while 'data' has {len(data)}."

# %% ---------------------------------------------------------------------------------

print("Saving fitted 'glmmTMB' model as RDS file...")
ro.r["saveRDS"](model, str(dirpath / f"{target}.rds"))

# %% ---------------------------------------------------------------------------------
