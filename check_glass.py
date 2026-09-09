"""
Sanity check: confirm every package imports and every model file/weight is
actually present and loadable — not just that pip said "Installed".

Run with the glass interpreter:
    glass/bin/python check_glass.py

All file paths below are resolved relative to THIS SCRIPT'S location
(not the current working directory), so it's safe to run from anywhere.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".cache" / "huggingface"))

ok, fail = [], []

def check(name, fn):
    try:
        fn()
        ok.append(name)
    except Exception as e:
        fail.append((name, str(e)))

# --- packages import ------------------------------------------------------
def _torch():
    import torch
    assert torch.__version__

def _ultralytics():
    import ultralytics

def _transformers():
    import transformers

def _huggingface_hub():
    import huggingface_hub

def _piper():
    import piper

check("import: torch", _torch)
check("import: ultralytics", _ultralytics)
check("import: transformers", _transformers)
check("import: huggingface_hub", _huggingface_hub)
check("import: piper", _piper)

# --- ZipDepth ---------------------------------------------------------
def _zipdepth():
    base = BASE_DIR / "ZipDepth"
    ckpts = list((base / "checkpoints").glob("*.pth"))
    assert ckpts, f"no .pth checkpoints found in {base / 'checkpoints'}"

    # ZipDepth isn't necessarily pip-installed as an importable "zipdepth"
    # package — the actual runner (test_depth.py) never imports it, it
    # shells out to scripts/infer.py instead. So try the import, but don't
    # fail the whole check just because that package isn't on the path —
    # fall back to confirming the script-based layout is present instead.
    try:
        import zipdepth  # noqa: F401
    except ImportError:
        infer_script = base / "scripts" / "infer.py"
        assert infer_script.exists(), (
            f"'zipdepth' package not importable, and {infer_script} not found either"
        )

check("ZipDepth (repo layout + checkpoints)", _zipdepth)

# --- YOLO26 -------------------------------------------------------------
def _yolo():
    from ultralytics import YOLO
    p = BASE_DIR / "yolo26s.pt"
    assert p.exists(), f"{p} not found — run download_models.py"
    YOLO(str(p))  # loads the weights, not just checks the file exists

check("YOLO26s weights load", _yolo)

# --- HF models: check the local cache has each repo, then actually load ---
def _hf_repo_cached(repo_id, loader):
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=repo_id, local_files_only=True)
    loader(path)

def _phowhisper():
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    _hf_repo_cached(
        "vinai/PhoWhisper-tiny",
        lambda p: (AutoProcessor.from_pretrained(p), AutoModelForSpeechSeq2Seq.from_pretrained(p)),
    )

def _envit5():
    import json
    import shutil
    import tempfile
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id="VietAI/envit5-translation", local_files_only=True)

    # tokenizers>=0.23 can't parse this model's (2022-era) fast-tokenizer JSON —
    # use_fast=False loads the slow sentencepiece-backed tokenizer instead.
    #
    # But this repo's special_tokens_map.json / tokenizer_config.json also has
    # "additional_special_tokens" saved as a bare dict instead of a list of
    # tokens. Even the slow tokenizer merges special tokens through the same
    # AddedToken-sequence code, so it fails with:
    #   'dict' object is not an instance of 'Sequence'
    # Work around it by copying the cached snapshot into a temp dir and
    # wrapping that field in a list there, leaving the real HF cache alone.
    patched_dir = Path(tempfile.mkdtemp(prefix="envit5_patched_"))
    for item in Path(path).iterdir():
        dest = patched_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    for fname in ("special_tokens_map.json", "tokenizer_config.json"):
        fpath = patched_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        ast = data.get("additional_special_tokens")
        if isinstance(ast, dict):
            data["additional_special_tokens"] = [ast]
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    AutoTokenizer.from_pretrained(str(patched_dir), use_fast=False)
    AutoModelForSeq2SeqLM.from_pretrained(path)

def _smolvlm2():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _hf_repo_cached(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        lambda p: (AutoProcessor.from_pretrained(p), AutoModelForImageTextToText.from_pretrained(p)),
    )

check("PhoWhisper-tiny loads", _phowhisper)
check("EnViT5 translation loads", _envit5)
check("SmolVLM2-256M-Video loads", _smolvlm2)

# --- Piper voice ----------------------------------------------------------
def _piper_voice():
    from piper import PiperVoice
    onnx = BASE_DIR / "voices" / "vi_VN-vais1000-medium.onnx"
    cfg = onnx.with_suffix(onnx.suffix + ".json")  # vi_VN-...onnx.json
    assert onnx.exists(), f"{onnx} not found"
    assert cfg.exists(), f"{cfg} not found (PiperVoice.load needs the matching .onnx.json config)"
    PiperVoice.load(str(onnx))

check("Piper vi_VN voice loads", _piper_voice)

# --- report -----------------------------------------------------------
print("\n=== OK ===")
for name in ok:
    print(f"  [x] {name}")

print("\n=== FAILED ===" if fail else "\n(no failures)")
for name, err in fail:
    print(f"  [ ] {name}\n      {err}")

print(f"\n{len(ok)}/{len(ok) + len(fail)} checks passed")