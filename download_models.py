"""
Run with the glass interpreter:
    glass/bin/python download_models.py

All file paths below are resolved relative to THIS SCRIPT'S location
(not the current working directory), so it's safe to run from anywhere —
matches check_glass.py's convention, so whatever this script downloads is
exactly where check_glass.py will look for it.

- ZipDepth: checkpoints already bundled in ZipDepth/checkpoints/ — nothing to do.
- YOLO26s: ultralytics auto-downloads yolo26s.pt on first YOLO(...) call;
  pre-fetched here so first run on-device doesn't stall.
- Piper: vi_VN-vais1000-medium is the best-quality Vietnamese voice in the
  official rhasspy/piper-voices set (single-speaker, 63MB). Other Vietnamese
  options if you want to compare: vi_VN-25hours_single-low, vi_VN-vivos-x_low.
"""
from pathlib import Path
from huggingface_hub import snapshot_download
from ultralytics import YOLO
from piper.download_voices import download_voice

BASE_DIR = Path(__file__).resolve().parent

MODELS = [
    "vinai/PhoWhisper-tiny",
    "VietAI/envit5-translation",
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
]

for repo_id in MODELS:
    print(f"Downloading {repo_id} ...")
    snapshot_download(repo_id=repo_id)

print("Downloading yolo26s.pt ...")
YOLO(str(BASE_DIR / "yolo26s.pt"))

print("Downloading Piper vi_VN voice (vais1000, medium) ...")
download_voice("vi_VN-vais1000-medium", BASE_DIR / "voices")

print(
    "Done. HF models cached under ~/.cache/huggingface/hub/, "
    f"Piper voice under {BASE_DIR / 'voices'}/, "
    f"YOLO weights at {BASE_DIR / 'yolo26s.pt'}"
)
