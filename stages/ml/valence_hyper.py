# %% ---------------------------------------------------------------------------------

import os
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

import datasets
import numpy as np  # noqa
import optuna
import torch
from newsuse.data import DataFrame
from transformers import AutoConfig, AutoTokenizer, Trainer

from project import config, paths
from project.model import (
    NewsuseValenceClassifier,
    NewsuseValenceClassifierConfig,
    NewsuseValenceEvaluator,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
datasets.disable_caching()

domain = "valence"

# %% ---------------------------------------------------------------------------------

dataset = datasets.load_from_disk(paths.ml / "datasets" / domain)

# %% ---------------------------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    torch.cuda.empty_cache()  # Clear CUDA memory before each trial
    # Fetch hyperparameter combinations
    hyper = config.ml.hyper[domain]
    specs = [hyper.space.newsuse, hyper.space.args]
    spaces = [{}, {}]
    for spec, space in zip(specs, spaces, strict=True):
        for k, v in deepcopy(dict(spec)).items():
            method = getattr(trial, f"suggest_{v.pop('type')}")
            space[k] = method(k, **v)
    newsuse_space, args_space = spaces
    # Initialize model configs
    base = newsuse_space.pop("base")
    model_config = config.ml.models[domain][base]
    base_config = AutoConfig.from_pretrained(model_config.checkpoint, **model_config.base)
    # Build newsuse config
    newsuse_config = NewsuseValenceClassifierConfig(
        base_config, **{**model_config.newsuse, **newsuse_space}
    )
    # Initialize tokenizer and tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(newsuse_config.base_name_or_path)
    encoded = dataset.map(
        lambda d: tokenizer(d["text"], **model_config.tokenize),
        batched=True,
    )

    def model_init() -> NewsuseValenceClassifier:
        return NewsuseValenceClassifier(newsuse_config)

    with TemporaryDirectory() as tmpdir:
        dirpath = Path(tmpdir) / base
        # Run training
        args = model_config.training.arguments(
            dirpath, label_names=newsuse_config.targets, **args_space
        )
        trainer = Trainer(
            args=args,
            model_init=model_init,
            train_dataset=encoded["train"],
            eval_dataset=encoded["valid"],
            tokenizer=tokenizer,
            compute_metrics=NewsuseValenceEvaluator(newsuse_config),
            callbacks=[cb.make() for cb in model_config.training.callbacks],
        )
        trainer.train()
    # Get the best metric
    train_metric = "eval_" + trainer.args.metric_for_best_model.removeprefix("eval_")
    metrics = [m for m in trainer.state.log_history if train_metric in m]
    metrics.sort(key=lambda m: m[train_metric])
    nullscore = -1.0
    f1 = metrics[0]["eval_overall"] if metrics else nullscore
    # Save the model if best
    state = trial.study.trials_dataframe(attrs=["value"])
    best_value = state["value"].astype(float).fillna(nullscore).max()
    if f1 > best_value:
        path = paths.ml / "models" / domain / base
        if path.exists():
            rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(path)
        for checkpoint in path.glob("checkpoint-*"):
            rmtree(checkpoint)
    # Return out-of-sample F1 score
    return f1


# %% ---------------------------------------------------------------------------------

storage = f"sqlite:///{paths.root / 'optuna.db'}"
opts = {
    **config.ml.hyper[domain].study,
    "storage": storage,
}

# %% Optimize hyperparameters --------------------------------------------------------

study = optuna.create_study(**opts)
study.optimize(objective, **config.ml.hyper[domain].optimize)

# %% ---------------------------------------------------------------------------------

results = DataFrame(study.trials_dataframe())
results.to_(paths.valence_hyper)

# %% ---------------------------------------------------------------------------------
