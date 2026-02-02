# %% Setup ---------------------------------------------------------------------------

import os
import shutil
from pathlib import Path

import huggingface_hub
from transformers import AutoTokenizer

from project import paths
from project.ml import NewsuseValenceClassifier

# %% ---------------------------------------------------------------------------------

huggingface_hub.login(token=os.environ["HUGGINGFACE_HUB_UPLOAD_TOKEN"])

DOMAIN = "valence"
USER = huggingface_hub.whoami()["name"]
MODELNAME = f"erc-newsuse-{DOMAIN}"

paths = paths.__copy__(
    model=f"@ml/models/{DOMAIN}/best",
    card=f"@ml/cards/{DOMAIN}.md",
)

# Path to the self-contained valence.py module
VALENCE_MODULE = Path(__file__).parent.parent.parent / "project" / "ml" / "valence.py"

# %% Load model ----------------------------------------------------------------------

model = NewsuseValenceClassifier.from_pretrained(paths.model)
tokenizer = AutoTokenizer.from_pretrained(paths.model)

# %% Configure for trust_remote_code -------------------------------------------------

# This tells HuggingFace which classes to load from which files
# when users load the model with trust_remote_code=True
model.config.auto_map = {
    "AutoConfig": "valence.NewsuseValenceClassifierConfig",
    "AutoModel": "valence.NewsuseValenceClassifier",
    "AutoModelForSequenceClassification": "valence.NewsuseValenceClassifier",
}

# Set custom pipeline class for the model
model.config.custom_pipelines = {
    "text-multi-classification": {
        "impl": "valence.TextMultiClassificationPipeline",
        "pt": ("valence.NewsuseValenceClassifier",),
        "tf": (),
    }
}

# Clear local path from config - this gets set to the local save directory
# by save_pretrained() and would cause errors when loading from Hub
model.config._name_or_path = "sztal/erc-newsuse-valence"

# %% Save locally with custom code ---------------------------------------------------

save_dir = Path(paths.model) / "hub_upload"
save_dir.mkdir(parents=True, exist_ok=True)

# Save model and tokenizer
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)

# Copy the self-contained valence.py module
shutil.copy(VALENCE_MODULE, save_dir / "valence.py")

# %% Push to hub ---------------------------------------------------------------------

api = huggingface_hub.HfApi()
api.create_repo(f"{USER}/{MODELNAME}", private=False, exist_ok=True)

commit_message = ""
while not commit_message.strip():
    commit_message = input(f"Enter commit message for new version of '{MODELNAME}': ")

api.upload_folder(
    folder_path=save_dir,
    repo_id=f"{USER}/{MODELNAME}",
    commit_message=commit_message,
)

# %% Push model card -----------------------------------------------------------------

card = huggingface_hub.ModelCard.load(paths.card)
card.push_to_hub(f"{USER}/{MODELNAME}")

# %% Cleanup -------------------------------------------------------------------------

shutil.rmtree(save_dir)

# %% ---------------------------------------------------------------------------------
