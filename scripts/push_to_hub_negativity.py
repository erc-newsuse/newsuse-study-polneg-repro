# %% Setup ---------------------------------------------------------------------------

import os

import huggingface_hub
from newsuse.ml import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Dataset,
    Trainer,
)

from project import config, paths

# %% ---------------------------------------------------------------------------------

huggingface_hub.login(token=os.environ["HUGGINGFACE_HUB_UPLOAD_TOKEN"])

DOMAIN = "negativity"
USER = huggingface_hub.whoami()["name"]
MODELNAME = f"erc-newsuse-{DOMAIN}"

config["model"] = config.ml.models[config.ml.models.use]
paths = paths.__copy__(
    model=f"@ml/classifiers/{DOMAIN}",
    dataset=f"@ml/datasets/{DOMAIN}",
    card=f"@ml/cards/{DOMAIN}.md",
)

# %% ---------------------------------------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(paths.model)
tokenizer = AutoTokenizer.from_pretrained(paths.model)
dataset = Dataset.from_disk(paths.dataset).tokenize(tokenizer, **config.model.tokenize)

# %% Make trainer --------------------------------------------------------------------

trainer = Trainer(
    args=config.model.training.arguments(MODELNAME),
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=config.model.training.evaluation(),
    callbacks=[cb.make() for cb in config.model.training.callbacks],
)

# %% Push to hub ---------------------------------------------------------------------

trainer.push_to_hub()

# %% Push model card -----------------------------------------------------------------

card = huggingface_hub.ModelCard.load(paths.card)
card.push_to_hub(f"{USER}/{MODELNAME}")

# %% ---------------------------------------------------------------------------------
