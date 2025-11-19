# %% ---------------------------------------------------------------------------------

import os

from label_studio_sdk import LabelStudio
from newsuse.data import DataFrame
from rich.progress import track

from project import paths
from project.metrics import f1_score

here = paths.root / "analyses" / "valence"

# %% ---------------------------------------------------------------------------------

studio = LabelStudio(
    base_url=os.environ["LABEL_STUDIO_URL"],
    api_key=os.environ["LABEL_STUDIO_API_KEY"],
)
project = studio.projects.get(id=int(os.environ["LABEL_STUDIO_VALENCE_PROJECT_ID"]))
tasks = list(studio.tasks.list(project=project.id))

# %% ----------------------------------------------------------------------------------


def parse_annotation(annotation: dict) -> dict:
    return {annotation["from_name"]: int(annotation["value"]["choices"][0])}


# %% ---------------------------------------------------------------------------------

records = []
for task in track(tasks, description="Parsing annotations..."):
    for ann in studio.annotations.list(id=task.id):
        record = {
            "id": task.id,
            "user": (ann.created_username.split("@", 1)[0].strip().split()[-1].strip()),
        }
        for result in ann.result or []:
            record.update(parse_annotation(result))
        records.append(record)

# %% ---------------------------------------------------------------------------------

records = DataFrame(records).set_index(["user", "id"]).astype("int64[pyarrow]")
annotators = records.index.get_level_values("user").unique().to_list()

# %% ---------------------------------------------------------------------------------

amae_scores = []
for a1 in annotators:
    for a2 in annotators:
        record = {"a1": a1, "a2": a2}
        df1 = records.loc[a1].dropna()
        df2 = records.loc[a2].dropna()
        id1 = df1.index.get_level_values("id")
        id2 = df2.index.get_level_values("id")
        mask = id1.isin(id2)
        if not mask.any():
            continue
        df1 = df1[mask]
        df2 = df2.loc[df1.index]
        for target in records:
            record[target] = f1_score(df1[target], df2[target])
        amae_scores.append(record)

amae = DataFrame(amae_scores).set_index(["a1", "a2"])

# %% ---------------------------------------------------------------------------------

(amae.unstack().T.to_xlsx(here / "annotators-amae.xlsx"))

# %% ---------------------------------------------------------------------------------
