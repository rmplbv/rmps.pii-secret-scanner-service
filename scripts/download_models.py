"""Download the Hugging Face models named in config.yaml into $HF_HOME.
Run at image build time so containers work fully offline."""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import load_settings  # noqa: E402

s = load_settings()
repos = [s.layers.transformer.model, s.layers.gliner.model]
for repo in repos:
    print("downloading", repo, flush=True)
    ignore = ["*.msgpack", "*.h5", "*.ot", "*onnx*", "flax_*", "tf_*", "rust_*", "training_args.bin"]
    files = HfApi().list_repo_files(repo)
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")  # same weights twice otherwise
    snapshot_download(repo, ignore_patterns=ignore)

# GLiNER loads the tokenizer/config of its backbone from the hub
from gliner import GLiNER  # noqa: E402

GLiNER.from_pretrained(s.layers.gliner.model)
print("done")
