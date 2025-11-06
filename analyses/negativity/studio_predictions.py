# %% ---------------------------------------------------------------------------------

import os
from typing import Any

import pandas as pd
from label_studio_sdk import LabelStudio
from newsuse.data import DataFrame
from rich.progress import track

from project import config, paths  # noqa

MODEL_VERSION = "negativity-0.0"

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_("polneg.json").pipe(
    lambda df: pd.concat(
        [df[["id"]], DataFrame(df.data.tolist())],
        axis=1,
    )
)

# %% ---------------------------------------------------------------------------------

studio = LabelStudio(
    base_url=os.environ["LABEL_STUDIO_URL"],
    api_key=os.environ["LABEL_STUDIO_API_KEY"],
)
project = studio.projects.get(id=int(os.environ["LABEL_STUDIO_NEGATIVITY_PROJECT_ID"]))
users = list(studio.users.list())
tasks = list(studio.tasks.list(project=project.id))

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
        model_version=MODEL_VERSION,
        result=[
            make_annotation(row[target], target)
            for target in list(project.parsed_label_config)
        ],
    )

# %% ---------------------------------------------------------------------------------
