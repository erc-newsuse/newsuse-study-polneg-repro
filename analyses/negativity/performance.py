# %% ---------------------------------------------------------------------------------

import datasets
import pandas as pd
from newsuse.data import DataFrame
from scipy.stats import hmean
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import project.model  # noqa
from project import paths
from project.metrics import amae_score
from project.pipelines import KeyDataset, pipeline

domain = "negativity"
targets = ["event", "sentiment"]

# %% ---------------------------------------------------------------------------------

hyper = DataFrame.from_(paths.proc / f"hyper-{domain}.parquet")
best_model = hyper.loc[hyper.value.idxmax()].params_base

# %% ---------------------------------------------------------------------------------

dataset = datasets.load_from_disk(paths.ml / "datasets" / domain)
model = AutoModel.from_pretrained(paths.ml / "models" / domain / best_model)
tokenizer = AutoTokenizer.from_pretrained(model.config.base_name_or_path)

# %% ---------------------------------------------------------------------------------

pipe = pipeline("text-multi-classification", model=model, tokenizer=tokenizer)

# %% ---------------------------------------------------------------------------------

validation = dataset["valid"].to_pandas().set_index("key")
results = DataFrame(
    tqdm(pipe(KeyDataset(validation, "text"), batch_size=16), total=len(validation)),
)

# %% ---------------------------------------------------------------------------------

output = pd.concat(
    [
        DataFrame(results[t].tolist()).rename(columns={"label": t, "score": f"{t}_score"})
        for t in targets
    ],
    axis=1,
).set_index(validation.index)

# %% ---------------------------------------------------------------------------------

performance = DataFrame(
    [
        {
            "amae": amae_score(validation[t], output[t]),
            "f1": hmean(f1_score(validation[t], output[t], average=None)),
        }
        for t in targets
    ],
    index=pd.Series(targets, name="target"),
)

# %% ---------------------------------------------------------------------------------
