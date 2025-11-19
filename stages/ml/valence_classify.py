# %% ---------------------------------------------------------------------------------

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
    .assign(text=lambda df: df.pop("title") + "\n\n" + df["text"])
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

dset = KeyDataset(data[["text"]], "text")
bsize = config.ml.inference.batch_size
results = DataFrame(tqdm(pipe(dset, batch_size=bsize), total=len(dset)))

# %% ---------------------------------------------------------------------------------

output = (
    pd.concat(
        [
            DataFrame(results[t].tolist()).rename(
                columns={"label": t, "score": f"{t}_score"}
            )
            for t in targets
        ],
        axis=1,
    )
    .set_index(data.index)
    .reset_index()
)

# %% ---------------------------------------------------------------------------------

output.to_(paths.cls_valence)

# %% ---------------------------------------------------------------------------------
