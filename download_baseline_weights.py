from huggingface_hub import list_repo_files, hf_hub_download
import os

repo_id = "ottogin/locality-diffusion-baselines"
base_dir = "data"

# List all files in the repo
all_files = list_repo_files(repo_id)
files_to_download = [f for f in all_files if f.startswith("models/baseline_unet/")]

for file in files_to_download:
    local_path = os.path.join(base_dir, file)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=file, local_dir=base_dir)
