#!/usr/bin/env python3
"""
BSmart Feature 1 pipeline.

Flow:
    audio (vi)       --[PhoWhisper CT2 int8]--> Vietnamese question
    Vietnamese text  --[EnViT5 CT2 int8]------> English question
    image + English  --[SmolVLM2 GGUF via llama-server]--> English answer
    English answer   --[EnViT5 CT2 int8]------> Vietnamese answer
    Vietnamese text  --[Piper TTS]------------> WAV audio

SmolVLM2 is intentionally served outside this Python process by llama.cpp's
`llama-server`; Python calls its OpenAI-compatible `/v1/chat/completions`
endpoint. This keeps the production backend aligned with the GGUF prototype
without keeping a second standalone script.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _optional_path(value: Optional[str]) -> Optional[Path]:
    if value is None or value.strip() == "":
        return None
    return Path(value).expanduser()


def _env_path(name: str, default: Path) -> Path:
    return _optional_path(os.getenv(name)) or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _detect_ct2_device() -> str:
    explicit = os.getenv("BSMART_CT2_DEVICE")
    if explicit:
        return explicit
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _default_llama_server_bin() -> Path:
    override = _optional_path(os.getenv("BSMART_LLAMA_SERVER_BIN"))
    if override is not None:
        return override

    names = ["llama-server.exe", "llama-server"] if os.name == "nt" else ["llama-server"]
    candidates = []
    for name in names:
        candidates.extend(
            [
                REPO_ROOT / "llama.cpp" / "build" / "bin" / name,
                REPO_ROOT / "llama.cpp" / "build" / "examples" / "server" / name,
                REPO_ROOT / "llama.cpp" / "build" / "tools" / "server" / name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


# ---------- Config ----------
ASR_MODEL_PATH = os.getenv("BSMART_ASR_MODEL_ID", "vinai/PhoWhisper-tiny")
TRANSLATE_MODEL_PATH = os.getenv("BSMART_TRANSLATE_MODEL_ID", "VietAI/envit5-translation")

ASR_CT2_DIR = _env_path("BSMART_ASR_CT2_DIR", MODELS_DIR / "phowhisper-ct2-int8")
if not ASR_CT2_DIR.exists():
    ASR_CT2_DIR = _env_path("BSMART_ASR_CT2_DIR", REPO_ROOT / "phowhisper-ct2-int8")

TRANSLATE_CT2_DIR = _env_path("BSMART_TRANSLATE_CT2_DIR", MODELS_DIR / "envit5-ct2-int8")
if not TRANSLATE_CT2_DIR.exists():
    TRANSLATE_CT2_DIR = _env_path("BSMART_TRANSLATE_CT2_DIR", REPO_ROOT / "envit5-ct2-int8")

ENVIT5_TOKENIZER_CACHE = _env_path(
    "BSMART_ENVIT5_TOKENIZER_CACHE",
    REPO_ROOT / ".cache" / "envit5_patched",
)

PIPER_VOICE_PATH = _env_path(
    "BSMART_PIPER_VOICE_PATH",
    MODELS_DIR / "voices" / "vi_VN-vais1000-medium.onnx",
)
if not PIPER_VOICE_PATH.exists():
    PIPER_VOICE_PATH = _env_path(
        "BSMART_PIPER_VOICE_PATH",
        REPO_ROOT / "voices" / "vi_VN-vais1000-medium.onnx",
    )

LLAMA_SERVER_BIN = _default_llama_server_bin()
LLAMA_HF_REPO = os.getenv("BSMART_LLAMA_HF_REPO", "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF")
LLAMA_MODEL_GGUF = _optional_path(os.getenv("BSMART_LLAMA_MODEL_GGUF"))
LLAMA_MMPROJ_GGUF = _optional_path(os.getenv("BSMART_LLAMA_MMPROJ_GGUF"))
LLAMA_SERVER_HOST = os.getenv("BSMART_LLAMA_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = _env_int("BSMART_LLAMA_PORT", 8085)
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
LLAMA_AUTO_START = _env_bool("BSMART_LLAMA_AUTO_START", True)
LLAMA_CONTEXT_SIZE = _env_int("BSMART_LLAMA_CONTEXT_SIZE", 2048)
LLAMA_STARTUP_TIMEOUT_S = _env_float("BSMART_LLAMA_STARTUP_TIMEOUT_S", 60.0)
LLAMA_HEALTH_TIMEOUT_S = _env_float("BSMART_LLAMA_HEALTH_TIMEOUT_S", 2.0)
LLAMA_REQUEST_TIMEOUT_S = _env_float("BSMART_LLAMA_REQUEST_TIMEOUT_S", 120.0)
LLAMA_OPENAI_MODEL_NAME = os.getenv("BSMART_LLAMA_OPENAI_MODEL", "SmolVLM2-256M")

ASR_SAMPLE_RATE = _env_int("BSMART_ASR_SAMPLE_RATE", 16000)
MAX_SIDE = _env_int("BSMART_VQA_MAX_SIDE", 384)
MAX_NEW_TOKENS = _env_int("BSMART_VQA_MAX_NEW_TOKENS", 80)
TRANSLATE_MAX_LENGTH = _env_int("BSMART_TRANSLATE_MAX_LENGTH", 256)
ASR_MAX_LENGTH = _env_int("BSMART_ASR_MAX_LENGTH", 200)
DEVICE = _detect_ct2_device()


class Feature1PipelineError(RuntimeError):
    """Base class for expected Feature 1 runtime failures."""


class InvalidImageError(Feature1PipelineError):
    """The uploaded image could not be decoded."""


class EmptyTranscriptError(Feature1PipelineError):
    """ASR completed but returned no usable text."""


class LlamaServerError(Feature1PipelineError):
    """Base class for llama-server failures."""


class LlamaServerUnavailable(LlamaServerError):
    """llama-server is down and could not be started."""


class LlamaServerTimeout(LlamaServerError):
    """llama-server startup or inference exceeded the configured timeout."""


class LlamaServerResponseError(LlamaServerError):
    """llama-server returned an invalid or failing response."""


@dataclass
class StageTimings:
    stages: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    def get(self, name: str) -> float:
        return self.stages.get(name, 0.0)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.stages)


def log(msg: str) -> None:
    print(f"[glass] {msg}", file=sys.stderr)


def load_envit5_tokenizer(repo_id: str):
    """Load EnViT5 tokenizer from a patched local cache."""
    if ENVIT5_TOKENIZER_CACHE.exists():
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(ENVIT5_TOKENIZER_CACHE), use_fast=False)

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

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


_asr_models = None
_translation_models = None
_piper_voice = None
_vlm_server_info: Optional[Dict[str, Any]] = None
_llama_proc: Optional[subprocess.Popen] = None
_llama_log_handle = None
_atexit_registered = False

_asr_lock = threading.Lock()
_translation_lock = threading.Lock()
_piper_lock = threading.Lock()
_vlm_lock = threading.Lock()


def get_asr_model():
    """Lazy-load PhoWhisper CT2 ASR components."""
    global _asr_models
    if _asr_models is not None:
        return _asr_models

    with _asr_lock:
        if _asr_models is not None:
            return _asr_models

        if not ASR_CT2_DIR.exists():
            raise FileNotFoundError(
                f"PhoWhisper CT2 model not found: {ASR_CT2_DIR}. Run "
                f"`ct2-transformers-converter --model {ASR_MODEL_PATH} "
                f"--output_dir {ASR_CT2_DIR.name} --quantization int8`."
            )

        import ctranslate2
        from transformers import WhisperProcessor

        log(f"Loading PhoWhisper CT2 ASR from {ASR_CT2_DIR} on {DEVICE}...")
        processor = WhisperProcessor.from_pretrained(ASR_MODEL_PATH)
        model = ctranslate2.models.Whisper(str(ASR_CT2_DIR), device=DEVICE)
        _asr_models = {"processor": processor, "model": model, "model_id": ASR_MODEL_PATH}
        log("PhoWhisper CT2 ASR loaded.")
        return _asr_models


def get_translation_model():
    """Lazy-load EnViT5 CT2 translation components."""
    global _translation_models
    if _translation_models is not None:
        return _translation_models

    with _translation_lock:
        if _translation_models is not None:
            return _translation_models

        if not TRANSLATE_CT2_DIR.exists():
            raise FileNotFoundError(f"EnViT5 CT2 model not found: {TRANSLATE_CT2_DIR}")

        import ctranslate2

        log(f"Loading EnViT5 CT2 translator from {TRANSLATE_CT2_DIR} on {DEVICE}...")
        tokenizer = load_envit5_tokenizer(TRANSLATE_MODEL_PATH)
        translator = ctranslate2.Translator(str(TRANSLATE_CT2_DIR), device=DEVICE)
        _translation_models = {
            "tokenizer": tokenizer,
            "translator": translator,
            "model_id": TRANSLATE_MODEL_PATH,
            "max_length": TRANSLATE_MAX_LENGTH,
        }
        log("EnViT5 CT2 translator loaded.")
        return _translation_models


def get_piper_voice():
    """Lazy-load Piper Vietnamese voice."""
    global _piper_voice
    if _piper_voice is not None:
        return _piper_voice

    with _piper_lock:
        if _piper_voice is not None:
            return _piper_voice

        if not PIPER_VOICE_PATH.exists():
            raise FileNotFoundError(
                f"Piper voice not found: {PIPER_VOICE_PATH}. Run backend/scripts/download_models.py."
            )

        from piper import PiperVoice

        log(f"Loading Piper voice from {PIPER_VOICE_PATH}...")
        _piper_voice = PiperVoice.load(str(PIPER_VOICE_PATH))
        log("Piper voice loaded.")
        return _piper_voice


def _parse_quantization(model_ref: Optional[str]) -> str:
    if not model_ref:
        return "auto"
    name = Path(model_ref).name
    match = re.search(r"(?:^|-)(IQ\d_[A-Z0-9_]+|Q\d(?:_[A-Z0-9]+)+|F16|F32)(?:\.gguf|$|:)", name)
    if match:
        return match.group(1)
    if ":" in model_ref:
        suffix = model_ref.rsplit(":", 1)[1].strip()
        if suffix:
            return suffix
    return "auto"


def _llama_mode_and_model(
    hf_repo: Optional[str],
    model_gguf: Optional[Path],
    mmproj_gguf: Optional[Path],
) -> Dict[str, Any]:
    if hf_repo:
        return {
            "mode": "hf_repo",
            "model": hf_repo,
            "mmproj": None,
            "quantization": _parse_quantization(hf_repo),
        }

    return {
        "mode": "local_gguf",
        "model": str(model_gguf) if model_gguf else None,
        "mmproj": str(mmproj_gguf) if mmproj_gguf else None,
        "quantization": _parse_quantization(str(model_gguf) if model_gguf else None),
    }


def _runtime_info(
    *,
    server_bin: Path = LLAMA_SERVER_BIN,
    host: str = LLAMA_SERVER_HOST,
    port: int = LLAMA_SERVER_PORT,
    hf_repo: Optional[str] = LLAMA_HF_REPO,
    model_gguf: Optional[Path] = LLAMA_MODEL_GGUF,
    mmproj_gguf: Optional[Path] = LLAMA_MMPROJ_GGUF,
    auto_start: bool = LLAMA_AUTO_START,
) -> Dict[str, Any]:
    model_info = _llama_mode_and_model(hf_repo, model_gguf, mmproj_gguf)
    return {
        **model_info,
        "url": f"http://{host}:{port}",
        "server_bin": str(server_bin),
        "auto_start": auto_start,
        "context_size": LLAMA_CONTEXT_SIZE,
        "startup_timeout_s": LLAMA_STARTUP_TIMEOUT_S,
        "request_timeout_s": LLAMA_REQUEST_TIMEOUT_S,
        "openai_model": LLAMA_OPENAI_MODEL_NAME,
    }


def _llama_server_is_up(url: str, timeout_s: float = LLAMA_HEALTH_TIMEOUT_S) -> bool:
    try:
        import requests

        resp = requests.get(f"{url}/health", timeout=timeout_s)
        return resp.status_code == 200
    except Exception:
        return False


def llama_server_is_up(url: Optional[str] = None) -> bool:
    """Public health probe. This never auto-starts llama-server."""
    return _llama_server_is_up(url or LLAMA_SERVER_URL)


def _open_llama_log():
    global _llama_log_handle
    if _llama_log_handle is not None and not _llama_log_handle.closed:
        return _llama_log_handle

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    _llama_log_handle = open(log_dir / "llama-server.log", "ab", buffering=0)
    return _llama_log_handle


def _stop_llama_server() -> None:
    global _llama_proc, _llama_log_handle
    if _llama_proc is not None and _llama_proc.poll() is None:
        log("Stopping llama-server...")
        _llama_proc.terminate()
        try:
            _llama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _llama_proc.kill()
    _llama_proc = None

    if _llama_log_handle is not None and not _llama_log_handle.closed:
        _llama_log_handle.close()
    _llama_log_handle = None


def _build_llama_command(
    server_bin: Path,
    host: str,
    port: int,
    hf_repo: Optional[str],
    model_gguf: Optional[Path],
    mmproj_gguf: Optional[Path],
) -> list[str]:
    if hf_repo:
        return [
            str(server_bin),
            "-hf",
            hf_repo,
            "--host",
            host,
            "--port",
            str(port),
            "-c",
            str(LLAMA_CONTEXT_SIZE),
        ]

    if not model_gguf or not Path(model_gguf).exists():
        raise LlamaServerUnavailable(
            "SmolVLM2 GGUF model file is not configured or does not exist. "
            "Set BSMART_LLAMA_HF_REPO or BSMART_LLAMA_MODEL_GGUF."
        )
    if not mmproj_gguf or not Path(mmproj_gguf).exists():
        raise LlamaServerUnavailable(
            "SmolVLM2 mmproj GGUF file is not configured or does not exist. "
            "Set BSMART_LLAMA_HF_REPO or BSMART_LLAMA_MMPROJ_GGUF."
        )

    return [
        str(server_bin),
        "-m",
        str(model_gguf),
        "--mmproj",
        str(mmproj_gguf),
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        str(LLAMA_CONTEXT_SIZE),
    ]


def ensure_llama_server(
    server_bin: Path = LLAMA_SERVER_BIN,
    host: str = LLAMA_SERVER_HOST,
    port: int = LLAMA_SERVER_PORT,
    url: Optional[str] = None,
    hf_repo: Optional[str] = LLAMA_HF_REPO,
    model_gguf: Optional[Path] = LLAMA_MODEL_GGUF,
    mmproj_gguf: Optional[Path] = LLAMA_MMPROJ_GGUF,
    auto_start: bool = LLAMA_AUTO_START,
) -> Dict[str, Any]:
    """Ensure a quantized SmolVLM2 llama-server is healthy."""
    global _llama_proc, _atexit_registered

    runtime = _runtime_info(
        server_bin=server_bin,
        host=host,
        port=port,
        hf_repo=hf_repo,
        model_gguf=model_gguf,
        mmproj_gguf=mmproj_gguf,
        auto_start=auto_start,
    )
    server_url = url or runtime["url"]
    started = time.perf_counter()

    if _llama_proc is not None and _llama_proc.poll() is not None:
        log("Previously launched llama-server is no longer running; will attempt restart.")
        _llama_proc = None

    if _llama_server_is_up(server_url):
        return {**runtime, "healthy": True, "started_by_backend": False, "load_seconds": 0.0}

    if not auto_start:
        raise LlamaServerUnavailable(
            f"llama-server is not healthy at {server_url} and auto-start is disabled."
        )

    if not Path(server_bin).exists():
        raise LlamaServerUnavailable(
            f"llama-server binary not found: {server_bin}. Build llama.cpp or set "
            "BSMART_LLAMA_SERVER_BIN."
        )

    cmd = _build_llama_command(Path(server_bin), host, port, hf_repo, model_gguf, mmproj_gguf)
    log(f"Launching llama-server at {server_url}: {' '.join(cmd)}")
    log_handle = _open_llama_log()
    _llama_proc = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle)
    if not _atexit_registered:
        atexit.register(_stop_llama_server)
        _atexit_registered = True

    deadline = time.perf_counter() + LLAMA_STARTUP_TIMEOUT_S
    while time.perf_counter() < deadline:
        if _llama_proc.poll() is not None:
            raise LlamaServerUnavailable(
                "llama-server exited before becoming healthy. "
                f"See {REPO_ROOT / 'logs' / 'llama-server.log'}."
            )
        if _llama_server_is_up(server_url):
            load_seconds = time.perf_counter() - started
            return {
                **runtime,
                "healthy": True,
                "started_by_backend": True,
                "load_seconds": load_seconds,
            }
        time.sleep(0.5)

    raise LlamaServerTimeout(
        f"llama-server did not become healthy within {LLAMA_STARTUP_TIMEOUT_S:.1f}s at {server_url}."
    )


def get_vlm_model() -> Dict[str, Any]:
    """Lazy-connect to SmolVLM2 GGUF via llama-server."""
    global _vlm_server_info
    if _vlm_server_info is not None and _llama_server_is_up(_vlm_server_info["url"]):
        return _vlm_server_info

    with _vlm_lock:
        if _vlm_server_info is not None and _llama_server_is_up(_vlm_server_info["url"]):
            return _vlm_server_info

        _vlm_server_info = ensure_llama_server()
        return _vlm_server_info


def get_llama_runtime_info(check_health: bool = False) -> Dict[str, Any]:
    info = _runtime_info()
    if _vlm_server_info:
        info.update(_vlm_server_info)
    if check_health:
        info["healthy"] = _llama_server_is_up(info["url"])
    return info


def configure_llama_server(
    *,
    server_bin: Optional[str] = None,
    hf_repo: Optional[str] = None,
    model_gguf: Optional[str] = None,
    mmproj_gguf: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """Override llama-server settings before the first request."""
    global LLAMA_SERVER_BIN, LLAMA_HF_REPO, LLAMA_MODEL_GGUF, LLAMA_MMPROJ_GGUF
    global LLAMA_SERVER_HOST, LLAMA_SERVER_PORT, LLAMA_SERVER_URL, _vlm_server_info

    if server_bin:
        LLAMA_SERVER_BIN = Path(server_bin).expanduser()
    if hf_repo is not None:
        LLAMA_HF_REPO = hf_repo.strip() or None
    if model_gguf:
        LLAMA_MODEL_GGUF = Path(model_gguf).expanduser()
    if mmproj_gguf:
        LLAMA_MMPROJ_GGUF = Path(mmproj_gguf).expanduser()
    if host:
        LLAMA_SERVER_HOST = host
    if port is not None:
        LLAMA_SERVER_PORT = port

    LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
    _vlm_server_info = None


def validate_image_file(image_path: Path) -> None:
    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(image_path) as img:
            img.verify()
    except UnidentifiedImageError as e:
        raise InvalidImageError(f"Invalid image file: {image_path}") from e
    except Exception as e:
        raise InvalidImageError(f"Image could not be decoded: {e}") from e


def resize_image(img, max_side: int = MAX_SIDE):
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    return img


def _image_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def speech_to_text_vi(
    audio_path: Path,
    *,
    models: Optional[Dict[str, Any]] = None,
    timings: Optional[StageTimings] = None,
) -> str:
    """Transcribe Vietnamese audio with PhoWhisper CT2 INT8."""
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    import ctranslate2
    import librosa

    active_timings = timings or StageTimings()
    if models is None:
        with active_timings.track("asr_model_load"):
            models = get_asr_model()
    processor = models["processor"]
    model = models["model"]

    with active_timings.track("audio_preprocessing"):
        audio, _ = librosa.load(str(audio_path), sr=ASR_SAMPLE_RATE, mono=True)
        features_np = processor.feature_extractor(
            audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="np"
        ).input_features.astype("float32")
        features = ctranslate2.StorageView.from_array(features_np)

    with active_timings.track("stt"):
        prompt = processor.tokenizer.convert_tokens_to_ids(
            ["<|startoftranscript|>", "<|vi|>", "<|transcribe|>", "<|notimestamps|>"]
        )
        results = model.generate(features, [prompt], beam_size=1, max_length=ASR_MAX_LENGTH)
        token_ids = results[0].sequences_ids[0]
        transcript = processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    return transcript


def translate(
    text: str,
    src_lang: str,
    *,
    models: Optional[Dict[str, Any]] = None,
    timings: Optional[StageTimings] = None,
    timing_key: Optional[str] = None,
) -> str:
    if src_lang not in {"vi", "en"}:
        raise ValueError(f"Unsupported translation source language: {src_lang}")
    if not text or not text.strip():
        raise ValueError("Cannot translate empty text")

    active_timings = timings or StageTimings()
    if models is None:
        with active_timings.track("translation_model_load"):
            models = get_translation_model()
    tokenizer = models["tokenizer"]
    translator = models["translator"]
    max_length = models["max_length"]

    key = timing_key or ("vi_to_en" if src_lang == "vi" else "en_to_vi")
    with active_timings.track(key):
        prefixed = f"{src_lang}: {text.strip()}"
        input_ids = tokenizer.encode(prefixed, truncation=True, max_length=max_length)
        source_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        results = translator.translate_batch([source_tokens], max_decoding_length=max_length)
        output_tokens = results[0].hypotheses[0]
        output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
        translated = decoded.split(":", 1)[-1].strip()

    if not translated:
        raise RuntimeError("EnViT5 returned empty translation")
    return translated


def make_concise_prompt(user_prompt: str) -> str:
    return (
        f"{user_prompt.strip()} "
        "Answer in one concise, complete English sentence. "
        "Start with 'There is', 'There are', or 'No,' when appropriate. "
        "Do not invent details not visible in the image."
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


def run_vlm_question(
    image_path: Path,
    question_en: str,
    *,
    timings: Optional[StageTimings] = None,
) -> str:
    """Run SmolVLM2 GGUF through llama-server for one image/question pair."""
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not question_en or not question_en.strip():
        raise ValueError("question_en must not be empty")

    from PIL import Image

    active_timings = timings or StageTimings()
    with active_timings.track("image_preparation"):
        with Image.open(image_path) as raw_image:
            image = resize_image(raw_image)
            data_url = _image_to_data_url(image)

    payload = {
        "model": LLAMA_OPENAI_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer in 1 simple, complete sentence (max ~30 words). "
                    "Be concise but cover the most important visible details."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": make_concise_prompt(question_en)},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": MAX_NEW_TOKENS,
    }

    with active_timings.track("vlm"):
        vlm = get_vlm_model()
        try:
            import requests

            resp = requests.post(
                f"{vlm['url']}/v1/chat/completions",
                json=payload,
                timeout=LLAMA_REQUEST_TIMEOUT_S,
            )
        except requests.Timeout as e:
            raise LlamaServerTimeout(
                f"SmolVLM2 GGUF inference timed out after {LLAMA_REQUEST_TIMEOUT_S:.1f}s"
            ) from e
        except requests.RequestException as e:
            raise LlamaServerUnavailable(f"llama-server request failed: {e}") from e

        if not resp.ok:
            body = resp.text[:500]
            raise LlamaServerResponseError(
                f"llama-server returned HTTP {resp.status_code}: {body}"
            )

        try:
            result = resp.json()
            choice = result["choices"][0]
            text = choice["message"]["content"]
        except Exception as e:
            raise LlamaServerResponseError(f"Invalid llama-server JSON response: {e}") from e

        answer = clean_truncated_text(text, choice.get("finish_reason") == "length")
        if not answer:
            raise LlamaServerResponseError("SmolVLM2 GGUF returned an empty answer")
        return answer


def speak_vi(text: str, out_path: Path, *, timings: Optional[StageTimings] = None) -> None:
    active_timings = timings or StageTimings()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _piper_voice is None:
        with active_timings.track("piper_model_load"):
            voice = get_piper_voice()
    else:
        voice = get_piper_voice()
    with active_timings.track("tts"):
        with wave.open(str(out_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)


def run_feature1(
    image_path: Path,
    *,
    question_vi: Optional[str] = None,
    audio_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    return_audio: bool = False,
    timings: Optional[StageTimings] = None,
) -> Dict[str, Any]:
    active_timings = timings or StageTimings()

    validate_image_file(image_path)

    if question_vi is None and audio_path is not None:
        question_vi = speech_to_text_vi(audio_path, timings=active_timings)

    question_vi = (question_vi or "").strip()
    if not question_vi:
        raise EmptyTranscriptError("PhoWhisper returned an empty question transcript")

    question_en = translate(question_vi, "vi", timings=active_timings, timing_key="vi_to_en")
    answer_en = run_vlm_question(image_path, question_en, timings=active_timings)
    answer_vi = translate(answer_en, "en", timings=active_timings, timing_key="en_to_vi")

    audio_path_written = None
    if return_audio or out_path is not None:
        target = out_path or (REPO_ROOT / "out.wav")
        speak_vi(answer_vi, target, timings=active_timings)
        audio_path_written = str(target)

    return {
        "question": question_vi,
        "answer": answer_vi,
        "question_en": question_en,
        "answer_en": answer_en,
        "model": get_llama_runtime_info(check_health=False)["model"],
        "translation_model": TRANSLATE_MODEL_PATH,
        "audio_path": audio_path_written,
    }


def build_timing_payload(timings: StageTimings, total_seconds: float) -> Dict[str, Any]:
    stages = timings.snapshot()
    main_keys = [
        "asr_model_load",
        "translation_model_load",
        "piper_model_load",
        "audio_preprocessing",
        "stt",
        "vi_to_en",
        "image_preparation",
        "vlm",
        "en_to_vi",
        "tts",
    ]
    known = sum(stages.get(key, 0.0) for key in main_keys)
    total = max(total_seconds, known)
    other = max(0.0, total - known)
    model_load = (
        stages.get("asr_model_load", 0.0)
        + stages.get("translation_model_load", 0.0)
        + stages.get("piper_model_load", 0.0)
    )
    return {
        "model_load_s": model_load,
        "asr_model_load_s": stages.get("asr_model_load", 0.0),
        "translation_model_load_s": stages.get("translation_model_load", 0.0),
        "piper_model_load_s": stages.get("piper_model_load", 0.0),
        "audio_preprocessing_s": stages.get("audio_preprocessing", 0.0),
        "stt_s": stages.get("stt", 0.0),
        "vi_to_en_s": stages.get("vi_to_en", 0.0),
        "image_preparation_s": stages.get("image_preparation", 0.0),
        "vlm_s": stages.get("vlm", 0.0),
        "en_to_vi_s": stages.get("en_to_vi", 0.0),
        "tts_s": stages.get("tts", 0.0),
        "other_network_s": other,
        "total_s": total,
        "raw_stages_s": stages,
    }


def format_latency_report(timing_payload: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Audio prep:        {timing_payload.get('audio_preprocessing_s', 0.0):.2f} s",
            f"STT:               {timing_payload.get('stt_s', 0.0):.2f} s",
            f"VI->EN:            {timing_payload.get('vi_to_en_s', 0.0):.2f} s",
            f"Image prep:        {timing_payload.get('image_preparation_s', 0.0):.2f} s",
            f"VLM:               {timing_payload.get('vlm_s', 0.0):.2f} s",
            f"EN->VI:            {timing_payload.get('en_to_vi_s', 0.0):.2f} s",
            f"TTS:               {timing_payload.get('tts_s', 0.0):.2f} s",
            f"Model load:        {timing_payload.get('model_load_s', 0.0):.2f} s",
            f"Other/network:     {timing_payload.get('other_network_s', 0.0):.2f} s",
            "--------------------------------",
            f"TOTAL:             {timing_payload.get('total_s', 0.0):.2f} s",
        ]
    )


def _load_all_models(
    *,
    llama_server_bin: Optional[str] = None,
    llama_hf_repo: Optional[str] = None,
    llama_model_gguf: Optional[str] = None,
    llama_mmproj_gguf: Optional[str] = None,
    llama_host: Optional[str] = None,
    llama_port: Optional[int] = None,
) -> Dict[str, Any]:
    configure_llama_server(
        server_bin=llama_server_bin,
        hf_repo=llama_hf_repo if llama_hf_repo is not None else LLAMA_HF_REPO,
        model_gguf=llama_model_gguf,
        mmproj_gguf=llama_mmproj_gguf,
        host=llama_host,
        port=llama_port,
    )
    return {
        "asr": get_asr_model(),
        "translation": get_translation_model(),
        "vlm": get_vlm_model(),
        "piper_voice": get_piper_voice(),
    }


MODELS = None


def _run_cli_query(image_path: Path, audio_path: Path, out_path: Path) -> Dict[str, Any]:
    timings = StageTimings()
    started = time.perf_counter()
    result = run_feature1(
        image_path,
        audio_path=audio_path,
        out_path=out_path,
        return_audio=True,
        timings=timings,
    )
    timing_payload = build_timing_payload(timings, time.perf_counter() - started)
    print(f"[STT (vi)]      : {result['question']}")
    print(f"[Prompt (en)]   : {result['question_en']}")
    print(f"[SmolVLM (en)]  : {result['answer_en']}")
    print(f"[Ket qua (vi)]  : {result['answer']}")
    print(format_latency_report(timing_payload))
    log(f"Wrote spoken answer to {out_path}")
    return {**result, "timings": timing_payload}


def main() -> None:
    global MODELS

    parser = argparse.ArgumentParser(
        description="BSmart Feature 1: photo + spoken Vietnamese question -> spoken Vietnamese answer"
    )
    parser.add_argument("--image", help="path to the photo, e.g. test/photo/pavement_5.webp")
    parser.add_argument("--audio", help="path to the spoken Vietnamese question")
    parser.add_argument("--out", default="out.wav", help="output wav path (default: out.wav)")
    parser.add_argument(
        "--serve",
        action="store_true",
        help='load all models once, then read JSON lines: {"image": "...", "audio": "...", "out": "..."}',
    )
    parser.add_argument("--llama-server-bin", default=None)
    parser.add_argument("--llama-hf-repo", default=LLAMA_HF_REPO)
    parser.add_argument("--llama-model", default=None)
    parser.add_argument("--llama-mmproj", default=None)
    parser.add_argument("--llama-host", default=LLAMA_SERVER_HOST)
    parser.add_argument("--llama-port", type=int, default=LLAMA_SERVER_PORT)
    args = parser.parse_args()

    use_local_files = bool(args.llama_model and args.llama_mmproj)
    MODELS = _load_all_models(
        llama_server_bin=args.llama_server_bin,
        llama_hf_repo=None if use_local_files else args.llama_hf_repo,
        llama_model_gguf=args.llama_model if use_local_files else None,
        llama_mmproj_gguf=args.llama_mmproj if use_local_files else None,
        llama_host=args.llama_host,
        llama_port=args.llama_port,
    )

    if args.serve:
        log('Serving. Send one JSON object per line, e.g. {"image": "...", "audio": "..."}')
        for line in sys.stdin:
            line = line.strip()
            if not line:
                break
            try:
                req = json.loads(line)
                _run_cli_query(
                    Path(req["image"]),
                    Path(req["audio"]),
                    Path(req.get("out", "out.wav")),
                )
            except Exception as e:
                log(f"error handling request: {e!r}")
        return

    if not args.image or not args.audio:
        parser.error("--image and --audio are required unless --serve is used")

    _run_cli_query(Path(args.image), Path(args.audio), Path(args.out))


if __name__ == "__main__":
    main()
