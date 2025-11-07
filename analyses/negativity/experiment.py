# %% ---------------------------------------------------------------------------------

from dlordinal.metrics import accuracy_off1
from newsuse.data import DataFrame
from sklearn.metrics import f1_score

from project import paths
from project.metrics import amae_score

domain = "negativity"
targets = ["event", "sentiment"]

# %% ---------------------------------------------------------------------------------

ground_truth = (
    DataFrame.from_(paths.gpt / f"{domain}-ground-truth.parquet")
    .drop(columns=["text", "title"])
    .set_index(["country", "key"])
)

output = (
    DataFrame.from_(paths.gpt / f"{domain}.parquet")[["key", "country", "model", *targets]]
    .dropna()
    .set_index(["model", "country", "key"])
    .sort_index()
)

# %% ---------------------------------------------------------------------------------

amae = []
for model in output.index.get_level_values("model").unique():
    model_output = output.loc[model]
    model_target = ground_truth.loc[model_output.index]
    scores = {"model": model}
    for target in targets:
        scores[target] = amae_score(model_output[target], model_target[target])
    amae.append(scores)

amae = DataFrame(amae).set_index("model")
amae_best = amae.idxmax()

# %% 1-off accuracy ------------------------------------------------------------------

acc = []
for model in output.index.get_level_values("model").unique():
    model_output = output.loc[model]
    model_target = ground_truth.loc[model_output.index]
    scores = {"model": model}
    for target in targets:
        scores[target] = accuracy_off1(model_output[target], model_target[target])
    acc.append(scores)

acc = DataFrame(acc).set_index("model")
acc_best = acc.idxmax()

# %% F1 score -------------------------------------------------------------------------

f1 = []
for model in output.index.get_level_values("model").unique():
    model_output = output.loc[model]
    model_target = ground_truth.loc[model_output.index]
    scores = {"model": model}
    for target in targets:
        scores[target] = f1_score(
            model_target[target],
            model_output[target],
            average="macro",
        )
    f1.append(scores)
f1 = DataFrame(f1).set_index("model")
f1_best = f1.idxmax()

# %% ---------------------------------------------------------------------------------
