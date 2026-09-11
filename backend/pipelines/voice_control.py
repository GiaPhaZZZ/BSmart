#!/usr/bin/env python3
"""
Function0_voice_control.py — offline Vietnamese voice-command transcription
+ feature matching + spoken confirmation playback.

Purpose:
    1. Take a short spoken-Vietnamese audio clip (a voice command, e.g.
       "Mở tính năng 1, chatbot.") and transcribe it to Vietnamese text
       using the same PhoWhisper CT2 int8 ASR backend as the main glass
       pipeline.
    2. Match the transcribed text against a keyword list for each of the
       4 features (Tính năng 1..4) to figure out which feature the user
       is asking to activate.
    3. Print the matched feature and play the corresponding confirmation
       mp3 (activate_voice/Open_f1.mp3 .. Open_f4.mp3).

Usage (single query):
    python Function0_voice_control.py --audio test/audio/mo_tinh_nang.mp3

Persistent mode (load the model once, transcribe many clips without reloading):
    python Function0_voice_control.py --serve
    # then feed JSON lines on stdin, one per query:
    {"audio": "test/audio/mo_tinh_nang.mp3"}
    {"audio": "test/audio/khac.mp3"}
    (Ctrl-D or a blank line to stop.)

Prerequisites (same as pipeline.py — run once beforehand):
    ct2-transformers-converter --model vinai/PhoWhisper-tiny \\
        --output_dir phowhisper-ct2-int8 --quantization int8

Audio playback:
    Uses ffplay (part of ffmpeg, already a dependency of this project for
    mp3 decoding) to play the matched confirmation clip. If ffplay isn't
    on PATH, falls back to afplay (macOS) / PowerShell (Windows) /
    paplay,aplay,mpg123 (Linux).

Note (WSL): use forward slashes for paths (test/audio/..., not test\\audio\\...).

Note (mp3 input): librosa/soundfile read mp3 via libsndfile>=1.1 or, as a
fallback, via ffmpeg (audioread backend). If loading the .mp3 raises an
error, install ffmpeg: sudo apt install ffmpeg
"""
import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import ctranslate2
import librosa
from transformers import WhisperProcessor

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
MODELS_DIR = REPO_ROOT / "models"

# ---------- Config ----------
ASR_MODEL_PATH = "vinai/PhoWhisper-tiny"   # for feature extractor + tokenizer only
ASR_CT2_DIR = MODELS_DIR / "phowhisper-ct2-int8"
if not ASR_CT2_DIR.exists():
    ASR_CT2_DIR = REPO_ROOT / "phowhisper-ct2-int8"

ASR_SAMPLE_RATE = 16000
ASR_MAX_LENGTH = 200

DEVICE = "cpu"  # voice-command transcription is tiny/fast; keep it simple and portable

AUDIO_DIR = REPO_ROOT / "assets" / "audio"
if not AUDIO_DIR.exists():
    AUDIO_DIR = REPO_ROOT / "activate_voice"

# ---------- Feature keyword map ----------
# Each feature maps to a list of trigger words/phrases. Matching is done on
# accent-stripped, lowercased text, so entries here don't need every accent
# variant — but keeping them written normally is fine and more readable.
FEATURE_KEYWORDS = {
    1: {
        "name": "Chatbot / Hỏi đáp",
        "sound": AUDIO_DIR / "Open_f1.mp3",
        "keywords": [
            "một", "1", "chatbot", "hỏi đáp", "hoi dap", "miêu tả",
            "mieu ta", "trả lời tự động", "tra loi tu dong", "thứ nhất",
            "thu nhat",
        ],
    },
    2: {
        "name": "Tự động / Hướng dẫn đi bộ",
        "sound": AUDIO_DIR / "Open_f2.mp3",
        "keywords": [
            "hai", "2", "tự động", "tu dong", "autopilot", "hướng dẫn",
            "huong dan", "đi bộ", "di bo", "thứ hai", "thu hai",
        ],
    },
    3: {
        "name": "Dẫn đường / Chỉ đường / Bản đồ",
        "sound": AUDIO_DIR / "Open_f3.mp3",
        "keywords": [
            "ba", "3", "thứ ba", "thu ba", "dẫn đường", "dan duong",
            "chỉ đường", "chi duong", "bản đồ", "ban do",
        ],
    },
    4: {
        "name": "Chụp ảnh",
        "sound": AUDIO_DIR / "Open_f4.mp3",
        "keywords": [
            "bốn", "bon", "4", "thứ tư", "thu tu", "chụp ảnh", "chup anh",
        ],
    },
}


def log(msg: str):
    print(f"[voice_control] {msg}", file=sys.stderr)


# ---------- Text normalization + matching ----------
def normalize_text(s: str) -> str:
    """Lowercase, strip Vietnamese diacritics, drop punctuation, collapse spaces."""
    s = s.lower()
    s = s.replace("đ", "d")  # đ has no combining-mark decomposition in NFKD
    nfkd = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_feature(text: str):
    """
    Match transcribed text against FEATURE_KEYWORDS.
    Returns (feature_id_or_None, matched_keywords_list, all_scores_dict).
    Matching is whole-word/phrase based (accent-insensitive) to avoid short
    keywords like "ba" or "hai" matching inside unrelated words.
    """
    norm_text = normalize_text(text)
    padded_text = f" {norm_text} "

    scores = {}
    matched = {}
    for fid, info in FEATURE_KEYWORDS.items():
        hits = [kw for kw in info["keywords"] if f" {normalize_text(kw)} " in padded_text]
        if hits:
            scores[fid] = len(hits)
            matched[fid] = hits

    if not scores:
        return None, [], scores

    best_fid = max(
        scores,
        key=lambda fid: (scores[fid], max(len(normalize_text(kw)) for kw in matched[fid])),
    )
    return best_fid, matched[best_fid], scores


# ---------- Audio playback ----------
def play_audio(path: Path):
    """Play an mp3 via whatever offline player is available on this system."""
    if not path.exists():
        log(f"Warning: sound file not found, skipping playback: {path}")
        return

    system = platform.system()
    candidates = []
    if shutil.which("ffplay"):
        candidates.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])
    if system == "Darwin" and shutil.which("afplay"):
        candidates.append(["afplay", str(path)])
    if system == "Windows":
        candidates.append([
            "powershell", "-c",
            f"(New-Object Media.SoundPlayer '{path}').PlaySync();",
        ])
    if system == "Linux":
        for player in ("paplay", "aplay", "mpg123"):
            if shutil.which(player):
                candidates.append([player, str(path)])

    if not candidates:
        log(
            "No audio player found (tried ffplay/afplay/paplay/aplay/mpg123/"
            f"PowerShell). Install ffmpeg (for ffplay) or play manually: {path}"
        )
        return

    for cmd in candidates:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception as e:
            log(f"playback via {cmd[0]} failed: {e!r}")

    log(f"All playback attempts failed for {path}")


# ---------- Load model (once, at startup) ----------
def _load_asr():
    log("Loading PhoWhisper feature extractor + tokenizer (CT2 backend)...")
    if not ASR_CT2_DIR.exists():
        raise FileNotFoundError(
            f"{ASR_CT2_DIR} not found — run:\n"
            f"  ct2-transformers-converter --model {ASR_MODEL_PATH} "
            f"--output_dir {ASR_CT2_DIR.name} --quantization int8"
        )
    whisper_processor = WhisperProcessor.from_pretrained(ASR_MODEL_PATH)
    asr_model = ctranslate2.models.Whisper(str(ASR_CT2_DIR), device=DEVICE)
    log("ASR model loaded.\n")
    return whisper_processor, asr_model


MODELS = None  # populated by main(), exactly once


# ---------- Core transcription ----------
def speech_to_text_vi(audio_path: Path) -> str:
    """Transcribe a Vietnamese spoken-command audio clip to Vietnamese text."""
    whisper_processor, asr_model = MODELS

    audio, _ = librosa.load(str(audio_path), sr=ASR_SAMPLE_RATE, mono=True)
    features_np = whisper_processor.feature_extractor(
        audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="np"
    ).input_features.astype("float32")
    features = ctranslate2.StorageView.from_array(features_np)

    prompt = whisper_processor.tokenizer.convert_tokens_to_ids(
        ["<|startoftranscript|>", "<|vi|>", "<|transcribe|>", "<|notimestamps|>"]
    )
    results = asr_model.generate(
        features, [prompt], beam_size=1, max_length=ASR_MAX_LENGTH
    )
    token_ids = results[0].sequences_ids[0]
    return whisper_processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


# ---------- One end-to-end query ----------
def run_query(audio_path: Path) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    log(f"Transcribing {audio_path} (vi)...")
    command_vi = speech_to_text_vi(audio_path)
    print(f"[Command (vi)]  : {command_vi}")

    fid, hits, scores = match_feature(command_vi)
    if fid is None:
        print("[Match]         : không nhận diện được tính năng nào (no matching feature)")
        return command_vi

    info = FEATURE_KEYWORDS[fid]
    print(f"[Match]         : Tính năng {fid} — {info['name']}  (matched: {', '.join(hits)})")
    if len(scores) > 1:
        others = {k: v for k, v in scores.items() if k != fid}
        log(f"other candidates: {others}")

    play_audio(info["sound"])
    return command_vi


# ---------- Main ----------
def main():
    global MODELS

    parser = argparse.ArgumentParser(
        description="Function0: spoken vi voice command -> vi text (for command matching)"
    )
    parser.add_argument("--audio", help="path to the spoken vi command, e.g. test/audio/mo_tinh_nang.mp3")
    parser.add_argument(
        "--serve", action="store_true",
        help="load the ASR model once, then read JSON lines from stdin "
             '(each: {"audio": "..."}) until EOF/blank line — avoids reloading per query.',
    )
    args = parser.parse_args()

    MODELS = _load_asr()

    if args.serve:
        log("Serving. Send one JSON object per line on stdin, e.g.:")
        log('  {"audio": "test/audio/mo_tinh_nang.mp3"}')
        log("Ctrl-D or a blank line to stop.")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                break
            try:
                req = json.loads(line)
                run_query(Path(req["audio"]))
            except Exception as e:
                log(f"error handling request: {e!r}")
        return

    if not args.audio:
        parser.error("--audio is required unless --serve is used")

    run_query(Path(args.audio))


if __name__ == "__main__":
    main()