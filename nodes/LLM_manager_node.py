import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download
import json



root_dir = os.path.dirname(os.path.dirname(__file__))
model_dir = os.path.join(root_dir, "models")


def download_model(repo_id):
    local_path = os.path.join(model_dir, repo_id.split("/")[-1])
    os.system(
        f"huggingface-cli download --resume-download {repo_id} --local-dir {local_path}"
    )


def load_model_names():
    with open(os.path.join(root_dir, "support_models.json")) as f:
        model_names = json.load(f)["model_names"]
    return model_names




if __name__ == "__main__":
    repo_id = "microsoft/Phi-3-mini-4k-instruct"
    download_model(repo_id)
