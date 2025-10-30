# %% Setup ---------------------------------------------------------------------------

import datasets
from newsuse.data import DataFrame
from transformers import AutoConfig, AutoTokenizer, Trainer

from project import config, paths
from project.model import (
    NewsuseNegativityClassifier,
    NewsuseNegativityClassifierConfig,
    NewsuseNegativityEvaluator,
)

datasets.disable_caching()

domain = "negativity"

# %% ---------------------------------------------------------------------------------

params = (
    DataFrame.from_(paths.proc / f"hyper-{domain}.parquet")
    .pipe(
        lambda df: df.loc[
            df.value.idxmax(), [c for c in df.columns if c.startswith("params_")]
        ]
    )
    .rename(lambda s: s.removeprefix("params_"))
    .rename({"shared_n_layers": "num_shared_layers", "head_n_layers": "num_head_layers"})
    .to_dict()
)

model_params = {
    k: v
    for k, v in params.items()
    if k in NewsuseNegativityClassifierConfig.__init__.__annotations__
}
trainer_params = {k: v for k, v in params.items() if k not in model_params}

# %% ---------------------------------------------------------------------------------

model_config = config.ml.models[domain][model_params.pop("base")]
base_config = AutoConfig.from_pretrained(model_config.checkpoint, **model_config.base)
model_config = NewsuseNegativityClassifierConfig(
    base_config, **{**model_config.newsuse, **model_params}
)
model = NewsuseNegativityClassifier(model_config)
tokenizer = AutoTokenizer.from_pretrained(model_config.base_name_or_path)

# %% ---------------------------------------------------------------------------------

dataset = datasets.load_from_disk(paths.ml / "datasets" / domain).map(
    lambda d: tokenizer(d["text"], **config.ml.tokenize), batched=True
)

# %% Define test and eval datasets ---------------------------------------------------

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# %% ---------------------------------------------------------------------------------


def model_init() -> NewsuseNegativityClassifier:
    model = NewsuseNegativityClassifier(model_config)
    return model


args = {
    "label_names": model_config.targets,
}

trainer = Trainer(
    args=config.ml.training.arguments(paths.ml / "models" / domain, **args),
    model_init=model_init,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=NewsuseNegativityEvaluator(model_config),
    callbacks=[cb.make() for cb in config.ml.training.callbacks],
)

# %% Run training loop ---------------------------------------------------------------

history = trainer.train()

# %% ---------------------------------------------------------------------------------
