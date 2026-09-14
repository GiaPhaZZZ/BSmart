#!/usr/bin/env bash
# ============================================================================
# glass — single env. ZipDepth, YOLO26, PhoWhisper, EnViT5, SmolVLM2, Piper TTS.
#
# Piper (OHF-Voice/piper1-gpl) replaces viet-tts: it's ONNX-runtime based, has
# no torch/numpy pin at all, and its own pyproject only asks for onnxruntime +
# a handful of small pure-Python deps. Verified with `uv`'s resolver against
# the same torch==2.14 / transformers<5 / ultralytics stack used below — zero
# downgrades, and it imports and runs fine in the same process as the rest.
# No second venv, no HTTP hop needed.
#
# NOTE: transformers is pinned <5 below. EnViT5-translation's 2022-era
# tokenizer files can't load on transformers v5's rewritten sentencepiece
# backend (fails with "'dict' object is not an instance of 'Sequence'").
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/glass}"
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Large wheels (torch is 500MB+) can outrun uv's default 30s timeout on slow
# or WSL-mounted-drive connections. Also worth running this script from a
# native Linux path (e.g. ~/glass) rather than /mnt/d/... — WSL's 9p bridge
# to Windows drives makes large file extraction far slower than the actual
# download.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

# --- venv --------------------------------------------------------------
uv venv glass --python 3.10
PY="glass/bin/python"

# --- torch -------------------------------------------------------------
# Default PyPI wheel bundles CUDA. On a headless/edge box with no GPU:
#   uv pip install --python "$PY" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PY" torch torchvision torchaudio

# --- YOLO26 (ultralytics) ----------------------------------------------
uv pip install --python "$PY" ultralytics

# --- shared transformers stack (PhoWhisper, EnViT5, SmolVLM2) ----------
# Pinned below v5: v5 shipped a rewritten sentencepiece tokenizer backend
# that can't load EnViT5's 2022-era tokenizer_config.json / special_tokens
# map (fails with "'dict' object is not an instance of 'Sequence'" deep in
# AddedToken merging). Unpinned, uv resolves the latest 5.x and this breaks.
uv pip install --python "$PY" \
    "transformers>=4.49,<5" \
    accelerate \
    sentencepiece \
    num2words \
    decord \
    soundfile \
    librosa \
    huggingface-hub \
    ctranslate2 \
    requests

# --- Piper TTS (replaces viet-tts) --------------------------------------
uv pip install --python "$PY" piper-tts

# --- ZipDepth (not on PyPI — install from source) -----------------------
if [ ! -d "ZipDepth" ]; then
    git clone https://github.com/fabiotosi92/ZipDepth
fi

# ZipDepth's own requirements.txt installs "the default PyTorch wheel,
# which on Linux ships with a bundled CUDA runtime" (see its README) — if
# it pins its own torch/torchvision/torchaudio versions, installing it
# as-is here would silently override the torch build chosen above,
# including the CPU-only override in the comment. Strip those three lines
# out and install everything else ZipDepth needs, so the torch already in
# this venv is left alone.
grep -viE '^(torch|torchvision|torchaudio)([[:space:]]*[=<>!~].*)?$' \
    ZipDepth/requirements.txt > ZipDepth/requirements.no-torch.txt || true
uv pip install --python "$PY" -r ZipDepth/requirements.no-torch.txt
uv pip install --python "$PY" -e ZipDepth

echo ""
echo "glass ready at: $PROJECT_ROOT/glass  (one env, everything installed)"
echo "Run: $PY /path/to/download_models.py   to pull PhoWhisper / EnViT5 / SmolVLM2 / YOLO26s / Piper vi_VN voice."
echo "Note: place download_models.py and check_glass.py in the same directory"
echo "      (e.g. $PROJECT_ROOT) — both resolve model paths relative to their own location."
