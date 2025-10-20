# %% ---------------------------------------------------------------------------------

import gzip
import io
import json
import sys
from itertools import batched

import numpy as np
from newsuse.data import DataFrame
from openai import OpenAI

from project import config, paths

COMPLETED = "completed"
IN_PROGRESS = "in_progress"

DOMAIN = "negativity"

opts = config.gpt[DOMAIN]

# %% ---------------------------------------------------------------------------------

requests = DataFrame.from_(paths.gpt / f"{DOMAIN}-requests.jsonl.gz")

# %% Check for existing output -------------------------------------------------------

if (output_path := paths.gpt / f"{DOMAIN}-output.jsonl.gz").exists():
    output = DataFrame.from_(output_path)
    mask = requests.custom_id.isin(output.custom_id)
    if mask.any():
        msg = (
            f"Output file '{output_path.name}' already contains responses to"
            f" {mask.sum()} requests out of {(~mask).sum()} in the current selection.\n"
            "Do you want to send only the remaining requests? (y/n): "
        )
        answer = input(msg).strip().lower()
        if answer == "n":
            print("Processing all, including requests with existing output...")
        elif answer == "y":
            msg = "Skipping already processed requests."
            print(msg)
            requests = requests[~mask]
        else:
            errmsg = f"Unexpected answer '{answer}', should be 'y' or 'n'"
            raise ValueError(errmsg)

# %% ---------------------------------------------------------------------------------

if requests.empty:
    print("No requests to process, exiting.")
    sys.exit(0)

# %% ---------------------------------------------------------------------------------

with (paths.prompts / opts.prompt).open() as fh:
    instructions = fh.read().strip()

client = OpenAI()

# %% ---------------------------------------------------------------------------------

requests_stats = (
    requests.groupby(["country"])
    .apply(
        lambda df: df.assign(idx=np.arange(len(df)) // opts.batch_size),
        include_groups=False,
    )
    .groupby(["country", "idx"])
    .size()
    .reset_index(name="n_requests")
    .to_dict(orient="records")
)

msg = "\nThe following jobs will be created:\n" + json.dumps(requests_stats, indent=4)
print(msg)
answer = input("Do you want to proceed? (y/n): ").strip().lower()
if answer == "n":
    print("Exiting without creating batch jobs.")
    sys.exit(1)
elif answer == "y":
    print("Proceeding to create batch jobs...")
else:
    errmsg = f"Unexpected answer '{answer}', should be 'y' or 'n'"
    raise ValueError(errmsg)

# %% ---------------------------------------------------------------------------------

header = opts.header
batch_jobs = {}

for country, data in requests.groupby("country"):
    for idx, batch_index in enumerate(batched(data.index, opts.batch_size)):
        batch_index = list(batch_index)
        batch = data.loc[batch_index]
        if batch.empty:
            continue
        for col in ("country", "political"):
            if col in batch.columns:
                batch.drop(columns=[col], inplace=True)
        # Use an in-memory bytes buffer instead of a temporary file
        buffer = io.BytesIO()
        # Write each request as a JSON line to the buffer
        for request in batch.to_dict(orient="records"):
            line = json.dumps(request).strip()
            buffer.write((line + "\n").encode())
        # Move the buffer's cursor to the beginning before reading
        buffer.seek(0)
        # Create the batch file using the in-memory buffer
        batch_file = client.files.create(
            file=buffer,
            purpose="batch",
        )
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint=header["url"],
            completion_window="24h",
            metadata={
                "description": f"[{country}|{idx}] negativity classification",
                "country": country,
                "idx": str(idx),
            },
        )
        batch_jobs.setdefault(country, {})[idx] = {
            "batch_id": batch_job.id,
            "batch_file_id": batch_file.id,
            "status": batch_job.status,
        }
# Report created batch jobs
print("\nCreated batch jobs:\n" + json.dumps(batch_jobs, indent=4))

# %% ---------------------------------------------------------------------------------

paths.gpt.mkdir(parents=True, exist_ok=True)
with gzip.open(paths.gpt / f"{DOMAIN}-jobs.jsonl.gz", "wt", encoding="utf-8") as fh:
    json.dump(batch_jobs, fh)

# %% ----------------------------------------------------------------------------------
