# %% ---------------------------------------------------------------------------------

import gc
from types import SimpleNamespace

import arviz as az
import matplotlib as mpl
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn.objects as so
import xarray as xr

from project import config, paths
from project.bayes import rebuild_model

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
opts = config.glmm.valence.targets["event"]
support = np.asarray([*config.categorical["valence"]])

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))

predictors_fixed = [*opts.common]
predictors_groups = [*opts.group]

rng = np.random.default_rng(opts.seed + 1717)

use_groups = ["posterior"]
opts.ppd.draws = 5

# %% ---------------------------------------------------------------------------------

print("Sampling posterior predictive distribution for event...")

event = SimpleNamespace()
event.idata = az.from_netcdf(paths.glmm / "valence" / "event.nc")
event.model = rebuild_model(event.idata)

print("Sampling posterior predictive distribution for sentiment...")

sentiment = SimpleNamespace()
sentiment.idata = az.from_netcdf(paths.glmm / "valence" / "sentiment.nc")
sentiment.model = rebuild_model(sentiment.idata)

for group in event.idata.groups():
    if group not in use_groups:
        del event.idata[group]
        del sentiment.idata[group]

# %% ---------------------------------------------------------------------------------

print("Sampling posterior predictive distribution for structural sentiment...")

structural = SimpleNamespace()
structural.idata = az.from_netcdf(paths.glmm / "valence" / "structural.nc")
structural.model = rebuild_model(structural.idata)

for group in structural.idata.groups():
    if group not in use_groups:
        del structural.idata[group]

# %% ---------------------------------------------------------------------------------

print("Generating general posterior predictive distribution of event and sentiment...")

grid = (
    # Generate a grid of `n` values per political-country combination
    # with randomly sampled random effects
    # We use the observed group levels to keep correlations
    # between event and sentiment random effects
    event.model.data.drop(columns=["key", "__obs__", "event"])
    .drop_duplicates(ignore_index=True)
    .groupby(["political", "country"], observed=True)
    .apply(
        lambda df: df.sample(n=100, random_state=rng, replace=True), include_groups=False
    )
    .droplevel(-1)
    .reset_index()
)

n_obs = len(grid)

ppd = {**opts.ppd}
event.ppd = (
    event.model.predict(
        event.idata.isel(draw=slice(ppd.pop("draws"))),
        data=grid,
        inplace=False,
        random_seed=rng,
        **ppd,
    )
    .posterior_predictive.pipe(lambda x: x - 1)
    .drop_vars("__obs__")
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in grid.items()})
    .to_dataframe()
    .reset_index()
    .assign(sample=lambda df: df.pop("draw") + df.pop("chain") * opts.ppd.draws)
    .rename(columns={"sample": "sample_event"})
)

ppd = {**opts.ppd}
sentiment.ppd = (
    sentiment.model.predict(
        sentiment.idata.isel(draw=slice(ppd.pop("draws"))),
        data=grid,
        inplace=False,
        random_seed=rng,
        **ppd,
    )
    .posterior_predictive.pipe(lambda x: x - 1)
    .drop_vars("__obs__")
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in grid.items()})
    .to_dataframe()
    .reset_index()
    .assign(sample=lambda df: df.pop("draw") + df.pop("chain") * opts.ppd.draws)
    .rename(columns={"sample": "sample_sentiment"})
)

ppd = {**opts.ppd}
structural.ppd = (
    structural.model.predict(
        structural.idata.isel(draw=slice(ppd.pop("draws"))),
        data=event.ppd,
        inplace=False,
        random_seed=rng,
        **ppd,
    )
    .posterior_predictive.pipe(lambda x: x - 1)
    .drop_vars("__obs__")
    .assign_coords({n: ("__obs__", c.to_numpy()) for n, c in event.ppd.items()})
    .to_dataframe()
    .reset_index()
    .assign(
        valence=lambda df: df[["event", "sentiment"]].sum(axis=1),
        sample=lambda df: df.pop("draw") + df.pop("chain") * opts.ppd.draws,
    )
    .rename(columns={"sample": "sample_structural"})
)

# %% ---------------------------------------------------------------------------------


def build_da(ppd: pd.DataFrame, target: str) -> xr.DataArray:
    obs_dim = [*opts.common, *opts.group]
    sample_dims = ppd.filter(like="sample_").columns.tolist()
    X = ppd.set_index(["__obs__", *sample_dims]).to_xarray()
    obs_data = ppd[["__obs__", *obs_dim]].drop_duplicates().set_index("__obs__")
    obs_coords = {n: ("__obs__", obs_data[n].to_numpy()) for n in obs_dim}
    X = X[target].assign_coords(obs_coords)
    return X


# %% ---------------------------------------------------------------------------------

print("Building dataset for valence posterior predictive distributions...")

dset = {
    "event": build_da(event.ppd, "event"),
    "sentiment": build_da(sentiment.ppd, "sentiment"),
    "sentiment_structural": build_da(structural.ppd, "sentiment"),
}

# %% ---------------------------------------------------------------------------------

del event
del sentiment
gc.collect()

# %% ---------------------------------------------------------------------------------

for response in config.engagement:
    for valence in ["event", "sentiment", "valence"]:
        print(f"Sampling posterior predictive distribution for {response} by {valence}...")

        idata = az.from_netcdf(paths.glmm / "engagement" / f"{response}-{valence}.nc")
        model = rebuild_model(idata)
        for group in idata.groups():
            if group not in use_groups:
                del idata[group]
        # kwargs = {**opts.ppd}
        kwargs = {**opts.ppd, "kind": "response_params"}
        ppd = (
            model.predict(
                idata.isel(draw=slice(kwargs.pop("draws"))),
                data=structural.ppd,
                inplace=False,
                random_seed=rng,
                **kwargs,
            )
            .posterior.drop_vars("__obs__")[["mu"]]
            .rename_vars({"mu": response})
            # .posterior_predictive.drop_vars("__obs__")
            .assign_coords(
                {n: ("__obs__", c.to_numpy()) for n, c in structural.ppd.items()}
            )
            .to_dataframe()
            .reset_index()
            .assign(sample=lambda df: df.pop("draw") + df.pop("chain") * opts.ppd.draws)
            .rename(columns={"sample": f"sample_{response}_{valence}"})
            .pipe(build_da, target=response)
        )
        dset[f"{response}_{valence}"] = ppd

# %% ---------------------------------------------------------------------------------

dset = xr.Dataset(dset)

# %% ---------------------------------------------------------------------------------

print("Saving posterior predictive distributions for all models...")

dset.to_netcdf(paths.glmm / "ppd.nc")

# %% ---------------------------------------------------------------------------------
