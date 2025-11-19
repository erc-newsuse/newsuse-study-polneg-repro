# %% ---------------------------------------------------------------------------------

import gzip
import io
import json
import sys
from itertools import batched

from newsuse.data import DataFrame
from openai import OpenAI

from project import config, paths

COMPLETED = "completed"
IN_PROGRESS = "in_progress"

domain = "valence"
opts = config.gpt[domain].batch

# %% ---------------------------------------------------------------------------------

requests = DataFrame.from_(paths.gpt / f"{domain}-experiment-requests.jsonl.gz")

# %% Check for existing responses ----------------------------------------------------

gpt_batch_responses_path = paths.gpt / f"{domain}-experiment-responses.jsonl.gz"
if gpt_batch_responses_path.exists():
    responses = DataFrame.from_(gpt_batch_responses_path)
    ids = list(
        zip(
            responses[k] if (k := "custom_id") in responses.columns else [],
            responses[k] if (k := "params_id") in responses.columns else [],
            strict=False,
        )
    )
    mask = requests[["custom_id", "params_id"]].map(lambda x: (x,)).sum(axis=1).isin(ids)
    if mask.any():
        msg = (
            f"File '{paths.gpt_batch_responses.name}' already contains responses to"
            f" {mask.sum()} requests out of {len(mask)} in the current selection.\n"
            "Do you want to send only the remaining requests? (y/n): "
        )
        answer = input(msg).strip().lower()
        if answer == "n":
            print("Processing all, including requests with existing responses...")
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

client = OpenAI()

# %% ---------------------------------------------------------------------------------

header = config.gpt.header
batch_jobs = {}

for params_id, data in requests.groupby("params_id"):
    for idx, batch_index in enumerate(batched(data.index, opts.batch_size)):
        batch_index = list(batch_index)
        batch = data.loc[batch_index]
        if batch.empty:
            continue
        # Use an in-memory bytes buffer instead of a temporary file
        buffer = io.BytesIO()
        country = list(batch.pop("country"))[0]
        params_id = list(batch.pop("params_id"))[0]
        for col in ["key", "target", "params"]:
            if col in batch.columns:
                del batch[col]
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
        metadata = {
            "description": f"[{country}|{idx}] article quality assessment input file",
            "country": country,
            "params_id": params_id,
            "idx": str(idx),
        }
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint=header["url"],
            completion_window="24h",
            metadata=metadata,
        )
        batch_jobs.setdefault(params_id, {})[idx] = {
            "batch_id": batch_job.id,
            "batch_file_id": batch_file.id,
            "status": batch_job.status,
        }

# Report created batch jobs
print("\nCreated batch jobs:\n" + json.dumps(batch_jobs, indent=4))

# %% ---------------------------------------------------------------------------------

paths.gpt.mkdir(parents=True, exist_ok=True)
with gzip.open(
    paths.gpt / f"{domain}-experiment-jobs.jsonl.gz", "wt", encoding="utf-8"
) as fh:
    json.dump(batch_jobs, fh)

# %% ----------------------------------------------------------------------------------
