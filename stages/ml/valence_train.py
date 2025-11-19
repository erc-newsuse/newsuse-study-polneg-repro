# %% Setup ---------------------------------------------------------------------------

import datasets
from transformers import AutoTokenizer, Trainer

from project import config, paths
from project.model import (
    NewsuseValenceClassifier,
    NewsuseValenceClassifierConfig,
    NewsuseValenceEvaluator,
)

datasets.disable_caching()

domain = "valence"

# %% ---------------------------------------------------------------------------------

base_model_name = "distilbert/distilbert-base-multilingual-cased"
model_config = NewsuseValenceClassifierConfig(base_model_name)
model = NewsuseValenceClassifier(model_config)
tokenizer = AutoTokenizer.from_pretrained(model_config.base_name_or_path)

# %% ---------------------------------------------------------------------------------

dataset = datasets.load_from_disk(paths.ml / "datasets" / domain).map(
    lambda d: tokenizer(d["text"], **config.ml.tokenize), batched=True
)

# %% Define test and eval datasets ---------------------------------------------------

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# %% ---------------------------------------------------------------------------------


def model_init() -> NewsuseValenceClassifier:
    model = NewsuseValenceClassifier(model_config)
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
    compute_metrics=NewsuseValenceEvaluator(model_config),
    callbacks=[cb.make() for cb in config.ml.training.callbacks],
)

# %% Run training loop ---------------------------------------------------------------

history = trainer.train()

# %% ---------------------------------------------------------------------------------
