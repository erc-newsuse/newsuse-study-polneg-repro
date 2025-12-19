# %% ---------------------------------------------------------------------------------

import os

import arviz as az
import numpy as np
import pandas as pd
from newsuse.data import DataFrame
from omegaconf import OmegaConf

from project import config, paths
from project.bayes import brm, brms_posterior
from project.rutils import ro

az.rcParams.update(config.arviz)

# Load 'brms' in the R process
ro.r("library(brms)")

# %% ---------------------------------------------------------------------------------

target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target name (event): ").strip() or "event"
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
        target: lambda df: df[f"{target}_latent"],
    },
)[["key", target, *opts.predictors.fixed, *opts.predictors.groups]]

# %% ---------------------------------------------------------------------------------

if (n := opts.model.get("subsample")) and n > 0:
    print(f"Subsampling to {n} data points for faster model fitting...")
    model_data = data.sample(n=n, random_state=rng)
else:
    model_data = data

# %% ---------------------------------------------------------------------------------


def make_kwargs(opts):
    kwargs = dict(OmegaConf.to_object(opts.solver))
    kwargs["formula"] = opts.model.formula.format(target=target)
    kwargs["data"] = model_data
    kwargs["prior"] = opts.model.get("prior")
    kwargs["family"] = ro.r(opts.model.family)
    kwargs["control"] = ro.ListVector(kwargs.get("control", {}))
    kwargs["threads"] = ro.r(f"threading({opts.solver.threads})")
    if opencl := opts.solver.get("opencl"):
        kwargs["opencl"] = ro.IntVector(opencl)
    if (opencl_ids := kwargs.pop("opencl_ids", None)) is not None:
        kwargs["opencl"] = ro.r(f"opencl({opencl_ids[0]}, {opencl_ids[1]})")
    return kwargs


# %% ---------------------------------------------------------------------------------

opts_advi = opts.copy()
opts_advi.update(solver=config.glmm.profiles.advi.solver.copy())

# %%

print(
    f"Initializing GLMM fit for '{target}' "
    f"using ADVI with {model_data.shape[0]} observations"
)
model = brm(**make_kwargs(opts_advi), seed=int(rng.integers(0, 2**16 - 1)))

# %% ---------------------------------------------------------------------------------

model = brms_posterior(model)
posterior = model.idata.posterior.stack(sample=["chain", "draw"])

# %% ---------------------------------------------------------------------------------

print(f"Fitting GLMM for '{target}' using NUTS...")
kwargs = make_kwargs(opts)
n_chains = kwargs.get("chains", 4)
n_samples = posterior.sizes["sample"]
sample_idx = rng.choice(n_samples, size=n_chains, replace=False)
init = ro.ListVector(
    {
        f"chain{i}": ro.ListVector(
            {
                k: v
                for k, v in posterior.isel(sample=idx).to_pandas().to_dict().items()
                if "[" not in k and "]" not in k and not k.startswith("lp")  # type: ignore
            }
        )
        for i, idx in enumerate(sample_idx)
    }
)
kwargs.update(init=init)
model = brm(**kwargs, seed=int(rng.integers(0, 2**16 - 1)))

# %% ---------------------------------------------------------------------------------

assert (nobs := ro.r("nobs")(model.r)[0]) == len(
    model_data
), f"Fitted 'model' has {nobs} observations while 'model_data' has {len(model_data)}."

# %% ---------------------------------------------------------------------------------

print("Saving fitted 'brms' model as RDS file...")
ro.r["saveRDS"](model.r, str(dirpath / f"{target}.rds"))

# %% ---------------------------------------------------------------------------------
