# %% ---------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from newsuse.data import DataFrame
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import project.model  # noqa
from project import config, paths
from project.pipelines import KeyDataset, pipeline

domain = "valence"
targets = ["event", "sentiment"]

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.text)
    .assign(
        title=lambda df: df["title"].str.strip().fillna(""),
        text=lambda df: df["text"].str.strip().fillna(""),
    )
    .assign(
        text=lambda df: (
            (df.pop("title") + "\n\n" + df["text"]).str.strip().replace("", pd.NA)
        )
    )
    .dropna()
    .set_index("key")
)

hyper = DataFrame.from_(paths.proc / f"{domain}-hyper.parquet")
best_model = hyper.loc[hyper.value.idxmax()].params_base

# %% ---------------------------------------------------------------------------------

model = AutoModel.from_pretrained(paths.ml / "models" / domain / best_model)
tokenizer = AutoTokenizer.from_pretrained(model.config.base_name_or_path)
pipe = pipeline("text-multi-classification", model=model, tokenizer=tokenizer)

# %% ---------------------------------------------------------------------------------

data = data.sample(n=100000, random_state=303)
dset = KeyDataset(data[["text"]], "text")
bsize = config.ml.inference.batch_size
results = [*tqdm(pipe(dset, top_k=None, batch_size=bsize), total=len(dset))]

# %% ---------------------------------------------------------------------------------


def make_record(result: dict) -> dict:
    def make_label(label: int) -> str:
        return "negative" if label == -1 else "neutral" if label == 0 else "positive"

    def _make(target: str) -> dict:
        scores = {
            f"{target}_p_{make_label(r['label'])}": r["score"] for r in result[target]
        }
        idx = np.argmax([r["score"] for r in result[target]])
        return {target: result[target][idx]["label"], **scores}

    record = {}
    for t in targets:
        record.update(_make(t))
    return record


# %% ---------------------------------------------------------------------------------

records = DataFrame([make_record(r) for r in results])

# %% ---------------------------------------------------------------------------------

output = records.set_index(data.index).reset_index()

# %% ---------------------------------------------------------------------------------

output.to_(paths.cls_valence)

# %% ---------------------------------------------------------------------------------
