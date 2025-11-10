# %% ---------------------------------------------------------------------------------

import gzip
import json

from newsuse.data import DataFrame
from openai import OpenAI

from project import paths

client = OpenAI()

COMPLETED = "completed"
DOMAIN = "negativity"

# %% ---------------------------------------------------------------------------------

with gzip.open(paths.gpt / f"{DOMAIN}-jobs.jsonl.gz", "rt") as fh:
    batch_jobs = json.load(fh)


# %% ---------------------------------------------------------------------------------

output = []

if (responses_path := paths.gpt / f"{DOMAIN}-responses.jsonl.gz").exists():
    existing = DataFrame.from_(responses_path)
    output.extend(existing.to_dict(orient="records"))


incomplete = {}

for country, jobs in batch_jobs.items():
    for job in jobs.values():
        job = client.batches.retrieve(job["batch_id"])
        if job.status != COMPLETED:
            incomplete.setdefault(country, []).append(job)
            continue
        content = client.files.content(job.output_file_id).content
        lines = list(content.decode().splitlines())
        data = DataFrame([json.loads(line) for line in lines if line.strip()])
        data.insert(1, "country", country)
        output.extend(data.to_dict(orient="records"))

# %% ---------------------------------------------------------------------------------

output = (
    DataFrame(output)
    .drop_duplicates(subset=["custom_id"], keep="last")
    .sort_values(["country", "custom_id"], ignore_index=True)
)

# %% ---------------------------------------------------------------------------------

if output.error.notnull().any():
    mask = output.error.notnull()
    msg = (
        f"{mask.sum()} requests resulted in errors, " f"see output file '{responses_path}'"
    )
    raise RuntimeError(msg)

# %% ---------------------------------------------------------------------------------

output.to_(responses_path)

# %%  --------------------------------------------------------------------------------


def jsonify_batch_job(job):
    fields = ("id", "metadata", "status", "errors", "input_file_id", "output_file_id")
    dct = job.to_dict(exclude_defaults=True, exclude_none=True)
    dct = {k: v for k, v in dct.items() if k in fields}
    return dct


if incomplete:
    n_incomplete = sum(len(v) for v in incomplete.values())
    errmsg = f"Some ({n_incomplete}) jobs are not completed:\n"
    errmsg += json.dumps(incomplete, indent=4, default=jsonify_batch_job)
    raise RuntimeError(errmsg)

# %% ---------------------------------------------------------------------------------
