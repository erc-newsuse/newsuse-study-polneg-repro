# %% ---------------------------------------------------------------------------------

import arviz as az
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn.objects as so
import xarray as xr

from project import config, paths

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
targets = ["reactions", "comments", "shares"]
factors = ["quality", "ideology"]

analysis_name = "by"

figpath = paths.figures / analysis_name
tabpath = paths.tables / analysis_name
figpath.mkdir(parents=True, exist_ok=True)
tabpath.mkdir(parents=True, exist_ok=True)

terms = {
    "var_names": [r":?(quality|ideology)$"],
    "filter_vars": "regex",
}


def bold_row(row: pd.Series) -> list[str]:
    frm = []
    for target in targets:
        if target not in row.index:
            continue
        part = row[target]
        if pd.isnull(part["sig"]) or not part["sig"]:
            frm.extend([""] * len(part))
        else:
            frm.extend(["bfseries:--rwrap"] * len(part))
    return frm


# %% ---------------------------------------------------------------------------------

tables = {}
for by in factors:
    posteriors = {}
    for target in targets:
        idata = az.from_netcdf(paths.glmm / "engagement" / f"{target}-valence-{by}.nc")
        stats = az.summary(idata, **terms).filter(regex=r"^mean|hdi_[0-9.]+%|r_hat")
        stats.columns = ["mean", "lower", "upper", r"$\hat{R}$"]
        stats["sig"] = np.sign(stats["lower"]) == np.sign(stats["upper"])
        posteriors[target] = stats
    tables[by] = pd.concat(posteriors, names=["target"], axis=1)

# %% ---------------------------------------------------------------------------------

for by in factors:
    print(
        tables[by].pipe(
            lambda df: (
                df.style.hide(
                    df.filter(regex=r"\'sig\'|\'up\'", axis=1).columns.tolist(), axis=1
                )
                .apply(bold_row, axis=1)
                .format(precision=3, escape="latex", na_rep="")
                .to_latex(
                    convert_css=False,
                    multirow_align="t",
                    multicol_align="c",
                    hrules=True,
                )
            )
        )
    )

# %% ---------------------------------------------------------------------------------
