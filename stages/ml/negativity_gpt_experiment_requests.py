# %% ---------------------------------------------------------------------------------

from copy import deepcopy
from itertools import product

import joblib
from newsuse.data import DataFrame
from newsuse.dotpath import dotimport
from omegaconf import OmegaConf
from openai.lib._pydantic import to_strict_json_schema

from project import config, paths

COMPLETED = "completed"
IN_PROGRESS = "in_progress"

domain = "negativity"
opts = config.gpt[domain].experiment

# %% ---------------------------------------------------------------------------------

ground_truth = DataFrame.from_(paths.gpt / f"{domain}-ground-truth.parquet")

# %% Build parametrs -----------------------------------------------------------------

parameters = []
for params in opts.params.values():
    params = OmegaConf.to_object(params)
    parameters.extend(
        dict(zip(params, values, strict=True)) for values in product(*params.values())
    )

msg = f"\nCreating requests for {len(parameters)} unique parameter sets."
print(msg)

# %% Make requests -------------------------------------------------------------------

header = config.gpt.header
text_format = {
    "type": "json_schema",
    "name": "quality_assessment",
    "strict": True,
}

prompts = {}
for prompt in opts.prompts:
    target = prompt.removesuffix(".md")
    with (paths.prompts / domain / prompt).open() as fh:
        prompts[target] = fh.read().strip()

# %% ---------------------------------------------------------------------------------

requests = []
for target in prompts:
    output_model = dotimport(f"project.gpt:{target.title()}Classification")
    for params in parameters:
        params_id = joblib.hash(tuple(params.items()))
        request_params = deepcopy(params)
        tfrm = {**text_format, "schema": to_strict_json_schema(output_model)}
        request_params.setdefault("text", {}).update(format=tfrm)
        for key, row in ground_truth.set_index("key").iterrows():
            text = [
                f"TITLE:\n{row.title}" if row.title else "",
                f"TEXT:\n{row.text}" if row.text else "",
            ]
            text = "\n\n".join(text).strip()
            if not text:
                continue
            body = {"input": text, "instructions": prompts[target], **request_params}
            request = {
                "custom_id": f"{key}__{params_id}__{target}",
                "key": key,
                "params_id": params_id,
                "params": params,
                "country": row.country,
                "target": target,
                **header,
                "body": body,
            }
            requests.append(request)

requests = DataFrame(requests)

# %% ---------------------------------------------------------------------------------

paths.gpt.mkdir(parents=True, exist_ok=True)
requests.to_(paths.gpt / "negativity-experiment-requests.jsonl.gz")

# %% ---------------------------------------------------------------------------------
