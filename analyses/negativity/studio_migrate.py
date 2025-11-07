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

prefix = "prompt_1_ew"
user = next(u for u in users if u.username == "e.wertz")
country = "fr"

# %% ---------------------------------------------------------------------------------

user_data = (
    DataFrame.from_(here / "Negativity-OpenAI.csv")
    .query(f"country.eq('{country}')")
    .reset_index(drop=True)[["key", f"{prefix}_event", f"{prefix}_sentiment"]]
    .rename(columns={f"{prefix}_event": "event", f"{prefix}_sentiment": "sentiment"})
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
    .merge(user_data, on="key", how="left")
    .dropna(subset=targets, ignore_index=True)
)

# %% ---------------------------------------------------------------------------------

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
        completed_by=user.id,
    )

# %% ---------------------------------------------------------------------------------
