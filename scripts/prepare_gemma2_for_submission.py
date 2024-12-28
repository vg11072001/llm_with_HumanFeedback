import os
import torch

from transformers import AutoTokenizer

import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

from human_pref.inference.modeling_gemma2 import Gemma2ForSequenceClassification
from huggingface_hub import HfApi, login
from user_secrets import UserSecretsClient

secrets_client = UserSecretsClient()

os.environ['KAGGLE_USERNAME'] = secrets_client.get_secret("KAGGLE_USERNAME")
os.environ['KAGGLE_KEY'] = secrets_client.get_secret("KAGGLE_KEY")

api = KaggleApi()
api.authenticate()
login(secrets_client.get_secret("HF_TOKEN"))

save_path = "../uploads/m0"
checkpoint_path = "../output/stage1/m0/update_last.pth"
model_name_or_path = "google/gemma-2-9b-it"
repo_id = "hangloosevg/gemma2-human-feedbacks"
path = kagglehub.dataset_download("cdeotte/gemma2-9b-it-fp16")
weights_dir = path 

tokenizer= AutoTokenizer.from_pretrained(model_name_or_path)
model = Gemma2ForSequenceClassification.from_pretrained(
    weights_dir,
    torch_dtype=torch.float16,
)
state_dict = torch.load(checkpoint_path, "cpu")["model"]
for idx, layer in enumerate(model.model.layers):
    state_dict[f"model.layers.{idx}.mlp.gate_up_proj.weight"] = torch.cat(
        [
            state_dict[f"model.layers.{idx}.mlp.gate_proj.weight"],
            state_dict[f"model.layers.{idx}.mlp.up_proj.weight"],
        ],
        dim=0,
    )
print(model.load_state_dict(state_dict, strict=False))
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)


# Create API client
api = HfApi()

# Upload model and tokenizer to Hub
api.upload_folder(
    folder_path=save_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload fine-tuned Gemma2 model with human feedback"
)

print(f"Model successfully uploaded to https://huggingface.co/{repo_id}")
