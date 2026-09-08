#!/usr/bin/env python3
"""
glass pipeline — offline blind-assistant demo, single image + single spoken question.

Flow:
    audio (vi, spoken question)  --[PhoWhisper CT2 int8]--> vi text
    vi text                      --[EnViT5 CT2 int8]------> en text
    image + en text (prompt)     --[SmolVLM2]--------------> en description
    en description                --[EnViT5 CT2 int8]------> vi description
    vi description                --[Piper TTS]-------------> out.wav

Run inside the glass venv (activate it first, or call glass/bin/python directly):

    source glass/glass/bin/activate
    python pipeline.py --image test/photo/pavement_5.webp --audio test/audio/mieu_ta_khung_canh.mp3

Persistent mode (load all models once, run many queries without reloading):

    python pipeline.py --serve
    # then feed it JSON lines on stdin, one query per line:
    {"image": "test/photo/pavement_5.webp", "audio": "test/audio/mieu_ta_khung_canh.mp3", "out": "out1.wav"}
    {"image": "test/photo/other.webp", "audio": "test/audio/other.mp3"}
    (Ctrl-D or an empty line to stop.)

Prerequisites (run once, before this script will work):

    ct2-transformers-converter --model vinai/PhoWhisper-tiny \\
        --output_dir phowhisper-ct2-int8 --quantization int8
    ct2-transformers-converter --model VietAI/envit5-translation \\
        --output_dir envit5-ct2-int8 --quantization int8

Note (WSL): use forward slashes for paths (test/photo/..., not test\\photo\\...) —
backslash is a Windows path separator and isn't interpreted that way on Linux/WSL.

Note (mp3 input): librosa/soundfile read mp3 via libsndfile>=1.1 or, as a
fallback, via ffmpeg (audioread backend). If loading the .mp3 raises an
error, install ffmpeg: sudo apt install ffmpeg
"""
import argparse
import json
import re
import shutil
import sys
import wave
from pathlib import Path

import ctranslate2
import librosa
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    WhisperProcessor,
)
from piper import PiperVoice

BASE_DIR = Path(__file__).resolve().parent

# ---------- Config ----------
ASR_MODEL_PATH = "vinai/PhoWhisper-tiny"           # for feature extractor + tokenizer only
VLM_MODEL_PATH = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
TRANSLATE_MODEL_PATH = "VietAI/envit5-translation"  # for tokenizer only

ASR_CT2_DIR = BASE_DIR / "phowhisper-ct2-int8"
TRANSLATE_CT2_DIR = BASE_DIR / "envit5-ct2-int8"
ENVIT5_TOKENIZER_CACHE = BASE_DIR / ".cache" / "envit5_patched"

PIPER_VOICE_PATH = BASE_DIR / "voices" / "vi_VN-vais1000-medium.onnx"

ASR_SAMPLE_RATE = 16000
MAX_SIDE = 384
MAX_NEW_TOKENS = 80    # ceiling/safety net, not the target length
MIN_NEW_TOKENS = 5     # don't force padding to reach a length
TRANSLATE_MAX_LENGTH = 256
ASR_MAX_LENGTH = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def log(msg: str):
    print(f"[glass] {msg}", file=sys.stderr)


# ---------- EnViT5 tokenizer patch (cached on disk, done once ever) ----------
# transformers>=5's rewritten sentencepiece backend can't parse EnViT5's
# 2022-era tokenizer files (fails with "'dict' object is not an instance of
# 'Sequence'" on additional_special_tokens). Load with use_fast=False from a
# patched copy of the cached snapshot. Patched once into ENVIT5_TOKENIZER_CACHE
# and reused after that, instead of re-downloading/re-patching every run.
def load_envit5_tokenizer(repo_id: str) -> AutoTokenizer:
    if ENVIT5_TOKENIZER_CACHE.exists():
        return AutoTokenizer.from_pretrained(str(ENVIT5_TOKENIZER_CACHE), use_fast=False)

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id)
    ENVIT5_TOKENIZER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = ENVIT5_TOKENIZER_CACHE.with_name(ENVIT5_TOKENIZER_CACHE.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    for item in Path(path).iterdir():
        dest = tmp_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

    for fname in ("special_tokens_map.json", "tokenizer_config.json"):
        fpath = tmp_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        ast = data.get("additional_special_tokens")
        if isinstance(ast, dict):
            data["additional_special_tokens"] = [ast]
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_dir.rename(ENVIT5_TOKENIZER_CACHE)
    return AutoTokenizer.from_pretrained(str(ENVIT5_TOKENIZER_CACHE), use_fast=False)


# ---------- Load models (once, at import/startup time) ----------
def _load_all_models():
    log("Loading PhoWhisper feature extractor + tokenizer (CT2 backend)...")
    if not ASR_CT2_DIR.exists():
        raise FileNotFoundError(
            f"{ASR_CT2_DIR} not found — run:\n"
            f"  ct2-transformers-converter --model {ASR_MODEL_PATH} "
            f"--output_dir {ASR_CT2_DIR.name} --quantization int8"
        )
    whisper_processor = WhisperProcessor.from_pretrained(ASR_MODEL_PATH)
    asr_model = ctranslate2.models.Whisper(str(ASR_CT2_DIR), device=DEVICE)

    log("Loading EnViT5 tokenizer + CT2 translator...")
    if not TRANSLATE_CT2_DIR.exists():
        raise FileNotFoundError(
            f"{TRANSLATE_CT2_DIR} not found — run:\n"
            f"  ct2-transformers-converter --model {TRANSLATE_MODEL_PATH} "
            f"--output_dir {TRANSLATE_CT2_DIR.name} --quantization int8"
        )
    translate_tokenizer = load_envit5_tokenizer(TRANSLATE_MODEL_PATH)
    translate_model = ctranslate2.Translator(str(TRANSLATE_CT2_DIR), device=DEVICE)

    log("Loading SmolVLM2 (vision-language)...")
    vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH)
    try:
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL_PATH, dtype=DTYPE, _attn_implementation="sdpa",
        ).to(DEVICE)
    except Exception as e:
        log(f"sdpa attention unavailable ({e!r}), falling back to eager.")
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL_PATH, dtype=DTYPE, _attn_implementation="eager",
        ).to(DEVICE)
    vlm_model.eval()

    log("Loading Piper voice (text-to-speech)...")
    if not PIPER_VOICE_PATH.exists():
        raise FileNotFoundError(
            f"{PIPER_VOICE_PATH} not found — run download_models.py first "
            "(it fetches the vi_VN-vais1000-medium voice into ./voices)."
        )
    piper_voice = PiperVoice.load(str(PIPER_VOICE_PATH))

    log("All models loaded.\n")
    return {
        "whisper_processor": whisper_processor,
        "asr_model": asr_model,
        "translate_tokenizer": translate_tokenizer,
        "translate_model": translate_model,
        "vlm_processor": vlm_processor,
        "vlm_model": vlm_model,
        "piper_voice": piper_voice,
    }


MODELS = None  # populated by main() / serve loop, exactly once


# ---------- Helpers ----------
def resize_image(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


def speech_to_text_vi(audio_path: Path) -> str:
    processor = MODELS["whisper_processor"]
    model = MODELS["asr_model"]

    audio, _ = librosa.load(str(audio_path), sr=ASR_SAMPLE_RATE, mono=True)
    features_np = processor.feature_extractor(
        audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="np"
    ).input_features.astype("float32")
    features = ctranslate2.StorageView.from_array(features_np)

    prompt = processor.tokenizer.convert_tokens_to_ids(
        ["<|startoftranscript|>", "<|vi|>", "<|transcribe|>", "<|notimestamps|>"]
    )
    results = model.generate(
        features, [prompt], beam_size=1, max_length=ASR_MAX_LENGTH
    )
    token_ids = results[0].sequences_ids[0]
    return processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def translate(text: str, src_lang: str) -> str:
    tokenizer = MODELS["translate_tokenizer"]
    translator = MODELS["translate_model"]

    prefixed = f"{src_lang}: {text}"
    input_ids = tokenizer.encode(prefixed, truncation=True, max_length=TRANSLATE_MAX_LENGTH)
    source_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    results = translator.translate_batch(
        [source_tokens], max_decoding_length=TRANSLATE_MAX_LENGTH
    )
    output_tokens = results[0].hypotheses[0]
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    return decoded.split(":", 1)[-1].strip()


def make_concise_prompt(user_prompt: str) -> str:
    return (
        f"{user_prompt} "
        "Answer in 1 simple, complete sentence (max ~30 words). "
        "Be concise but cover the most important details."
    )


def clean_truncated_text(text: str, was_truncated: bool) -> str:
    text = text.strip()
    if not was_truncated:
        return text
    sentence_ends = [m.end() for m in re.finditer(r"[.!?]", text)]
    if sentence_ends and sentence_ends[-1] > len(text) * 0.5:
        return text[: sentence_ends[-1]]
    trimmed = text.rsplit(" ", 1)[0] if " " in text else text
    return trimmed.rstrip(" .,;:") + "..."


@torch.inference_mode()
def describe_scene(image: Image.Image, prompt_en: str) -> str:
    vlm_processor = MODELS["vlm_processor"]
    vlm_model = MODELS["vlm_model"]

    image = resize_image(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_en},
            ],
        },
    ]

    text_prompt = vlm_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = vlm_processor(
        text=text_prompt, images=[image], return_tensors="pt"
    ).to(vlm_model.device, dtype=DTYPE)

    generated_ids = vlm_model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )

    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_len:]
    was_truncated = output_ids.shape[1] >= MAX_NEW_TOKENS

    text = vlm_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return clean_truncated_text(text, was_truncated)


def speak_vi(text: str, out_path: Path):
    piper_voice = MODELS["piper_voice"]
    with wave.open(str(out_path), "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)


# ---------- One end-to-end query ----------
def run_query(image_path: Path, audio_path: Path, out_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    img = Image.open(image_path)

    log(f"Transcribing {audio_path} (vi)...")
    prompt_vi = speech_to_text_vi(audio_path)
    print(f"[STT (vi)]      : {prompt_vi}")

    prompt_en = translate(prompt_vi, src_lang="vi")
    print(f"[Prompt (en)]   : {prompt_en}")

    result_en = describe_scene(img, make_concise_prompt(prompt_en))
    print(f"[SmolVLM (en)]  : {result_en}")

    result_vi = translate(result_en, src_lang="en")
    print(f"[Ket qua (vi)]  : {result_vi}")

    speak_vi(result_vi, out_path)
    log(f"Wrote spoken answer to {out_path}")


# ---------- Main ----------
def main():
    global MODELS

    parser = argparse.ArgumentParser(
        description="glass: photo + spoken vi question -> spoken vi answer"
    )
    parser.add_argument("--image", help="path to the photo, e.g. test/photo/pavement_5.webp")
    parser.add_argument("--audio", help="path to the spoken vi question, e.g. test/audio/mieu_ta_khung_canh.mp3")
    parser.add_argument("--out", default="out.wav", help="output wav path (default: out.wav)")
    parser.add_argument(
        "--serve", action="store_true",
        help="load all models once, then read JSON lines from stdin "
             '(each: {"image": "...", "audio": "...", "out": "..."}) '
             "until EOF/blank line — avoids reloading models per query.",
    )
    args = parser.parse_args()

    MODELS = _load_all_models()

    if args.serve:
        log("Serving. Send one JSON object per line on stdin, e.g.:")
        log('  {"image": "test/photo/pavement_5.webp", "audio": "test/audio/mieu_ta_khung_canh.mp3"}')
        log("Ctrl-D or a blank line to stop.")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                break
            try:
                req = json.loads(line)
                run_query(
                    Path(req["image"]),
                    Path(req["audio"]),
                    Path(req.get("out", "out.wav")),
                )
            except Exception as e:
                log(f"error handling request: {e!r}")
        return

    if not args.image or not args.audio:
        parser.error("--image and --audio are required unless --serve is used")

    run_query(Path(args.image), Path(args.audio), Path(args.out))


if __name__ == "__main__":
    main()