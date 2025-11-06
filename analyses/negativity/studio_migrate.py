# %% ---------------------------------------------------------------------------------

import json
import os

from label_studio_sdk import LabelStudio
from newsuse.data import DataFrame
from rich.progress import track

from project import paths

here = paths.root / "analyses" / "negativity"

# %% ---------------------------------------------------------------------------------

studio = LabelStudio(
    base_url=os.environ["LABEL_STUDIO_URL"],
    api_key=os.environ["LABEL_STUDIO_API_KEY"],
)
project = studio.projects.get(id=int(os.environ["LABEL_STUDIO_NEGATIVITY_PROJECT_ID"]))
tasks = list(studio.tasks.list(project=project.id))
users = [u for u in studio.users.list() if u.organization_membership.active]

# %% ---------------------------------------------------------------------------------

mw_user = next(u for u in users if u.username == "magdalena.wojcieszak")

# %% ---------------------------------------------------------------------------------

mw = (
    DataFrame.from_(here / "Negativity-MW.csv")
    .query("country.eq('esp')")
    .reset_index(drop=True)[["key", "prompt_1_mw_event", "prompt_1_mw_sentiment"]]
    .rename(columns={"prompt_1_mw_event": "event", "prompt_1_mw_sentiment": "sentiment"})
)

# %% ---------------------------------------------------------------------------------

data = DataFrame.from_(here / "polneg.json")
cols = data.columns

# %% ---------------------------------------------------------------------------------

targets = ["event", "sentiment"]
df = (
    data.assign(key=lambda df: df["data"].map(lambda x: x["key"]))
    .set_index("key")
    .reset_index()
    .merge(mw, on="key", how="left")
    .dropna(subset=targets, ignore_index=True)
)

for _, row in track(df.iterrows(), total=len(df)):
    ann = row.annotations
    result = [
        {
            "from_name": target,
            "to_name": "text",
            "type": "choices",
            "value": {"choices": [json.loads(row[target])]},
        }
        for target in targets
    ]
    studio.annotations.create(
        id=row.id,
        result=result,
        completed_by=mw_user.id,
    )

# %% ---------------------------------------------------------------------------------
