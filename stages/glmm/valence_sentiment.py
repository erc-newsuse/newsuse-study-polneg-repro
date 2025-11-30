# %% ---------------------------------------------------------------------------------


import bambi as bmb
import numpy as np
import pandas as pd
from newsuse.data import DataFrame

from project import config, paths
from project.inference import advi_trace_to_inference, make_priors

target = "sentiment"
output_dir = paths.glmm / "valence"
output_dir.mkdir(parents=True, exist_ok=True)

opts = config.glmm.valence[target]
rng = np.random.default_rng(opts.seed)

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.final)
    .pipe(
        lambda df: df.assign(
            political=pd.Categorical(df["political"], **config.categorical.political),
            event=pd.Categorical(df["event"], **config.categorical.valence),
            sentiment=pd.Categorical(df["sentiment"], **config.categorical.valence),
            outlet=df["country"] + ":" + df["name"],
            year=df["year"].astype(str),
            month=df["month"].astype(str),
            day=df["day"].astype(str),
        ).pipe(
            lambda df: df.assign(
                time=df["country"] + ":" + df["year"] + ":" + df["month"] + ":" + df["day"],
                country=pd.Categorical(
                    df["country"],
                    categories=list(config.categorical.countries),
                ),
            )
        )
    )
    .convert_dtypes()
)

# %% ---------------------------------------------------------------------------------

model = bmb.Model(
    formula=opts.formula.format(target=target),
    data=data,
    family=opts.family,
    priors=make_priors(opts.priors),
)
model.build()

# %% ---------------------------------------------------------------------------------

fit_opts = opts.fit.copy()
advi = model.fit(
    callbacks=[fit_opts.pop("callback").make()],
    random_seed=rng.integers(0, 2**32 - 1),
    **fit_opts,
)

# %% ---------------------------------------------------------------------------------

trace = advi.sample(random_state=rng, **opts.sample)
idata = advi_trace_to_inference(trace, model=model)

# %% ---------------------------------------------------------------------------------

model.predict(idata, kind="response", inplace=True, random_seed=rng)

# %% ---------------------------------------------------------------------------------

idata.to_netcdf(output_dir / f"{target}.nc")

# %% ---------------------------------------------------------------------------------
