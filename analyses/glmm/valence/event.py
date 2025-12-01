# %% ---------------------------------------------------------------------------------


import arviz as az
import brmspy
import numpy as np
from brmspy.helpers.conversion import r_to_py  # noqa
from newsuse.data import DataFrame
from rpy2.robjects import globalenv
from rpy2.robjects import r as R

from project import config, paths

target = "event"
glmm_dir = paths.glmm / "valence"

opts = config.glmm.valence[target]
rng = np.random.default_rng(opts.seed)

R(
    """
library(brms)
library(emmeans)
"""
)

az.rcParams.update(config.arviz)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(paths.final)

# %% ---------------------------------------------------------------------------------

glmm = brmspy.FitResult(
    idata=az.from_netcdf(glmm_dir / f"{target}.nc"),
    r=R["readRDS"](str(glmm_dir / f"{target}.rds")),
)

globalenv["glmm"] = glmm.r

# %% ---------------------------------------------------------------------------------

emm = R("(emm <- emmeans(glmm, ~political, epred = TRUE))")

# %% ---------------------------------------------------------------------------------
