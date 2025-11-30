# %% ---------------------------------------------------------------------------------

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
from newsuse.data import DataFrame
from pymc.backends.base import MultiTrace

from project import config, paths

target = "event"

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

sample = data.sample(n=int(1e4), random_state=42)

# %% ---------------------------------------------------------------------------------

formula = (
    f"{target} ~ "
    "0 + country * political "
    "+ (0 + political | outlet) "
    "+ (0 + political | country:year:month:day) "
).strip()

# %% ---------------------------------------------------------------------------------

model = bmb.Model(
    formula=formula,
    data=sample,
    family="cumulative",
    priors={
        # fixed effects
        r"common": bmb.Prior("Normal", mu=0, sigma=1),
        r"group_specific": bmb.Prior(
            "Normal",
            mu=0,
            sigma=bmb.Prior("HalfNormal", np.sqrt(np.pi / 2)),
        ),
    },
)
model.build()

# %% ---------------------------------------------------------------------------------

advi = model.fit(
    inference_method="vi",
    method="advi",
    progressbar=True,
)

# %% ---------------------------------------------------------------------------------


def advi_trace_to_inference(
    trace: MultiTrace, model: bmb.Model | None = None
) -> az.InferenceData:
    P = trace["p"]
    P = np.concatenate([np.zeros((*P.shape[:-1], 1)), P], axis=-1)
    P = np.concatenate([P, np.ones((*P.shape[:-1], 1))], axis=-1)
    P = np.diff(P, axis=-1)
    backend = trace.__dict__["_straces"][0]
    backend.var_shapes["p"] = P.shape[1:]
    backend.samples["p"] = P
    return pm.to_inference_data(trace, model=model.backend.model)


# %% ---------------------------------------------------------------------------------

trace = advi.sample(return_inferencedata=False)
idata = advi_trace_to_inference(trace, model=model)

# %% ---------------------------------------------------------------------------------

import matplotlib.pyplot as plt  # noqa

fig, ax = plt.subplots(figsize=(8, 6))

# %% ---------------------------------------------------------------------------------

bmb.interpret.plot_predictions(
    model,
    idata,
    "country",
    pps=True,
    use_hdi=True,
    prob=0.1,
    transforms={
        "event": lambda x: x == 2,
    },
    ax=ax,
)

# %% ---------------------------------------------------------------------------------

# multi_trace = advi.sample(draws=500, return_inferencedata=False)

# %% ---------------------------------------------------------------------------------

# idata = az.from_pymc3(
#     trace=multi_trace,
#     model=model.backend.model,
#     coords=model.backend.model.coords,
#     dims=model.backend.model.named_vars_to_dims,
# )

# %% ---------------------------------------------------------------------------------
