# %% Setup ---------------------------------------------------------------------------

import datasets
from transformers import AutoTokenizer, Trainer

from project import config, paths
from project.model import (
    NewsuseNegativityClassifier,
    NewsuseNegativityClassifierConfig,
    NewsuseNegativityEvaluator,
)

datasets.disable_caching()

target = "negativity"

# %% ---------------------------------------------------------------------------------

base_model_name = "distilbert/distilbert-base-multilingual-cased"
# base_model_name = "FacebookAI/xlm-roberta-large"
model_config = NewsuseNegativityClassifierConfig(base_model_name)
model = NewsuseNegativityClassifier(model_config)
tokenizer = AutoTokenizer.from_pretrained(model_config.base_name_or_path)

# %% ---------------------------------------------------------------------------------

dataset = datasets.load_from_disk(paths.ml / "datasets" / target).map(
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
    args=config.ml.training.arguments(paths.ml / "models" / target, **args),
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
