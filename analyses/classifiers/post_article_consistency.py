# %% ---------------------------------------------------------------------------------

import pandas as pd
from newsuse.data import DataFrame
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import project.model  # noqa
from project import config, paths
from project.pipelines import pipeline

# %% ---------------------------------------------------------------------------------

sample = DataFrame.from_(paths.aux / "articles-sample.parquet")

# %% ---------------------------------------------------------------------------------

model = AutoModel.from_pretrained(paths.ml / "models" / "valence" / "best")
tokenizer = AutoTokenizer.from_pretrained(model.config.base_name_or_path)

# %% ---------------------------------------------------------------------------------

classifiers = {
    "political": pipeline(
        "text-classification", config.classification.political.model.name
    ),
    "valence": pipeline("text-multi-classification", model=model, tokenizer=tokenizer),
}

# %% ---------------------------------------------------------------------------------

labels = {}
for domain, classifier in classifiers.items():
    for col in ["article", "post"]:
        outputs = [classifier(t) for t in tqdm(sample[col], desc=f"{col} | {domain}")]
        if domain == "political":
            outputs = [{"political": x[0]["label"]} for x in outputs]  # type: ignore
        else:
            outputs = [{k: x[k]["label"] for k in ["event", "sentiment"]} for x in outputs]  # type: ignore
        outputs = pd.DataFrame(outputs)
        labels.setdefault(col, []).append(outputs)

# %% ---------------------------------------------------------------------------------

data = pd.concat({k: pd.concat(v, axis=1) for k, v in labels.items()}, names=["source"])

# %% ---------------------------------------------------------------------------------

metrics = []
for col in ["political", "event", "sentiment"]:
    df = data[col].unstack("source")
    x, y = df[["article", "post"]].values.T
    acc = accuracy_score(x, y)
    mcc = matthews_corrcoef(x, y)
    metrics.append(
        {"target": col, "Accuracy": acc, "Matthews Correlation Coefficient": mcc}
    )

metrics = pd.DataFrame(metrics)

# %% ---------------------------------------------------------------------------------

print(
    metrics.set_index("target")
    .style.format(precision=3, escape="latex")
    .to_latex(
        hrules=True,
    )
)

# %% ---------------------------------------------------------------------------------
