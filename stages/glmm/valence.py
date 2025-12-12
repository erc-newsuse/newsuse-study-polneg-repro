# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import numpy as np
import pandas as pd
from newsuse.data import DataFrame
from omegaconf import OmegaConf

from project import config, paths
from project.brms import brm, ro

az.rcParams.update(config.arviz)

# %% ---------------------------------------------------------------------------------

target = os.environ.get("TARGET", "event")
opts = config.glmm.valence.targets[target]

dirpath = paths.glmm / "valence"
dirpath.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(opts.seed)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final).assign(
    country=lambda df: pd.Categorical(
        df["country"], categories=[*config.categorical.country]
    ),
    **{
        target: lambda df: pd.Categorical(
            df[target] - df[target].min(),
            categories=(cats := np.array([*config.categorical[target]])) - cats.min(),
            ordered=True,
        )
    },
)[["key", target, *opts.predictors.fixed, *opts.predictors.groups]]

# %% ---------------------------------------------------------------------------------

if (n := opts.model.get("subsample")) and n > 0:
    print(f"Subsampling to {n} data points for faster model fitting...")
    model_data = data.sample(n=n, random_state=rng)
else:
    model_data = data

# %% ---------------------------------------------------------------------------------

print(f"Fitting GLMM for '{target}' target with {model_data.shape[0]} observations")

kwargs = dict(OmegaConf.to_object(opts.solver))
kwargs["control"] = ro.ListVector(kwargs.get("control", {}))

model = brm(
    formula=opts.model.formula.format(target=target),
    data=model_data,
    prior=opts.model.get("prior"),
    family=ro.StrVector([opts.model.family, opts.model.link]),
    seed=int(rng.integers(0, 2**16 - 1)),
    **kwargs,
)

# %% ---------------------------------------------------------------------------------

assert (nobs := ro.r("nobs")(model.r)[0]) == len(
    data
), f"Fitted has {nobs} observations while data has {len(data)}!"

# %% ---------------------------------------------------------------------------------

print("Saving fitted 'brms' model as RDS file...")
ro.r["saveRDS"](model.r, str(dirpath / f"{target}-model.rds"))

# %% ---------------------------------------------------------------------------------
