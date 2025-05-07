# %% Setup ---------------------------------------------------------------------------

from newsuse.ml import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Dataset,
    Trainer,
)

from project import config, paths

DOMAIN = "political"

config["model"] = config.ml.models[config.ml.models.use]
paths = paths.__copy__(
    model=f"@ml/classifiers/{DOMAIN}",
    dataset=f"@ml/datasets/{DOMAIN}",
)


tokenizer = AutoTokenizer.from_pretrained(config.model.base)

dataset = Dataset.from_disk(paths.dataset).tokenize(tokenizer, **config.model.tokenize)

# %% Define model and trainer ------------------------------------------------------------


def model_init():
    id2label = dict(enumerate(config.annotations[DOMAIN].labels))
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.base,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    return model


trainer = Trainer(
    args=config.model.training.arguments(paths.model),
    model_init=model_init,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=config.model.training.evaluation(),
    callbacks=[cb.make() for cb in config.model.training.callbacks],
)

# %% Run training loop -------------------------------------------------------------------

history = trainer.train()

# %% SAVE MODEL ==========================================================================

trainer.save_model(paths.model, remove_checkpoints=True)

# %% -------------------------------------------------------------------------------------
