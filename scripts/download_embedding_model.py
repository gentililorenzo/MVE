from huggingface_hub import snapshot_download
import os

local_model_path = "./models/stella_en_400M_v5"

snapshot_download(
    repo_id="NovaSearch/stella_en_400M_v5",
    local_dir=local_model_path,
    local_dir_use_symlinks=False,  # Do not download links-to but real files
    resume_download=True
)