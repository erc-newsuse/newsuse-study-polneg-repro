# %% ---------------------------------------------------------------------------------

import json
import os
from typing import Any

import pandas as pd
from label_studio_sdk import LabelStudio
from newsuse.data import DataFrame
from rich.progress import track

from project import config, paths  # noqa

model_version = input("Enter model version/name: ")

here = paths.root / "analyses" / "valence"

# %% ---------------------------------------------------------------------------------

studio = LabelStudio(
    base_url=os.environ["LABEL_STUDIO_URL"],
    api_key=os.environ["LABEL_STUDIO_API_KEY"],
)
project = studio.projects.get(id=int(os.environ["LABEL_STUDIO_VALENCE_PROJECT_ID"]))
users = list(studio.users.list())
tasks = list(studio.tasks.list(project=project.id))

targets = list(project.parsed_label_config)

# %% ---------------------------------------------------------------------------------

gpt = (
    DataFrame.from_(here / "gpt" / "Valence-GPT-Optimized.csv")[["key", "prompt_6_output"]]
    .rename(columns={"prompt_6_output": "output"})
    .assign(output=lambda df: df["output"].apply(json.loads))
    .assign(
        output=lambda df: df["output"].apply(
            lambda x: json.loads(x["output"][-1]["content"][-1]["text"])
        )
    )
    .assign(
        event=lambda df: df["output"].apply(lambda x: x["event"]),
        sentiment=lambda df: df["output"].apply(lambda x: x["sentiment"]),
    )
    .drop(columns=["output"])
)

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(here / "polneg.json")
    .pipe(
        lambda df: pd.concat(
            [df[["id"]], DataFrame(df.data.tolist())],
            axis=1,
        )
    )
    .drop(columns=["event", "sentiment"])
    .merge(gpt, how="inner", on="key")
)

assert data.notnull().all().all(), "Data contains null values!"

# %% ---------------------------------------------------------------------------------


def make_annotation(
    label: str | int,
    from_name: str,
    to_name: str = "text",
) -> dict[str, Any]:
    return {
        "from_name": from_name,
        "to_name": to_name,
        "type": "choices",
        "value": {"choices": [str(label)]},
    }


# %% ---------------------------------------------------------------------------------

for task_id, row in track(
    data.set_index("id").iterrows(), total=len(data), description="Uploading predictions..."
):
    studio.predictions.create(
        task=task_id,
        model_version=model_version,
        result=[make_annotation(row[target], target) for target in targets],
    )

# %% ---------------------------------------------------------------------------------
