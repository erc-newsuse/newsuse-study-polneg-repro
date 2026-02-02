# %% ---------------------------------------------------------------------------------


import numpy as np
from newsuse.data import DataFrame
from transformers import AutoModel

from project import paths
from project.ml import ordinal_inverse, ordinal_probs

domain = "valence"

# %% ---------------------------------------------------------------------------------

meta = DataFrame.from_(paths.outlet_meta)

hyper = DataFrame.from_(paths.proc / f"{domain}-hyper.parquet")
best_model = hyper.loc[hyper.value.idxmax()].params_base
model = AutoModel.from_pretrained(paths.ml / "models" / domain / best_model)

biases = {
    target: head.ordinal.bias.detach().cpu().numpy() for target, head in model.heads.items()
}

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.dataset)
    .query("date < '2024-04-01'")
    .merge(DataFrame.from_(paths.labels, columns=["key", "event", "sentiment"]))
    .assign(
        year=lambda df: df["date"].dt.year,
        month=lambda df: df["date"].dt.month,
        day=lambda df: df["date"].dt.day,
        weekday=lambda df: df["date"].dt.weekday,
        weekend=lambda df: np.where(df["weekday"] >= 5, 1, 0),
        isotime=lambda df: df["date"]
        .dt.isocalendar()
        .pipe(lambda df: df["year"].astype(str) + ":" + df["week"].astype(str)),
    )[
        [
            "key",
            "name",
            "country",
            "year",
            "month",
            "day",
            "weekday",
            "weekend",
            "isotime",
            "reactions",
            "comments",
            "shares",
        ]
    ]
    .merge(meta, how="left", on=["country", "name"])
    .merge(DataFrame.from_(paths.labels))
    .assign(valence=lambda df: df[["event", "sentiment"]].sum(axis=1, skipna=False))
    # .dropna(ignore_index=True)
    .assign(
        political=lambda df: np.where(df["political"] == "OTHER", 0, 1),
        outlet=lambda df: df["country"] + ":" + df["name"],
        time=lambda df: df["year"].astype(str)
        + ":"
        + df["month"].astype(str)
        + ":"
        + df["day"].astype(str),
    )
    .convert_dtypes()
)

# %% Recover latent scores ----------------------------------------------------------

for target, bias in biases.items():
    probs = (
        data[[f"{target}_score_{v}" for v in ["negative", "neutral", "positive"]]]
        .astype(float)
        .to_numpy()
    )
    logits = ordinal_inverse(probs)
    latent = (logits - bias[None, ...]).mean(axis=-1)
    reconstructed_probs = ordinal_probs(latent[:, None] + bias)
    mask = ~np.isnan(probs).any(axis=1)
    assert np.allclose(
        probs[mask], reconstructed_probs[mask], rtol=1e-3, atol=1e-3
    ), f"reconstructed '{target}' probabilities do not match the original"
    data[f"{target}_latent"] = latent

# %% Compute expected valence -------------------------------------------------------

data["valence_expected"] = (
    data.filter(regex=r"^(event|sentiment)_score_.*$")
    .astype(float)
    .pipe(lambda df: (df.to_numpy() * np.array([-1, 0, 1] * 2)[None, :]).sum(axis=-1))
)

# %% ---------------------------------------------------------------------------------

(
    data.convert_dtypes()
    # .dropna(ignore_index=True)
    .to_(paths.final)
)

# %% ---------------------------------------------------------------------------------
