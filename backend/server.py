#!/usr/bin/env python3
"""
BSmart FastAPI backend.

This server keeps the mobile-facing API stable while delegating Feature 1 model
work to backend.pipelines.visual_qa:
  - PhoWhisper CT2 INT8 for ASR when audio is posted to /qa
  - EnViT5 CT2 INT8 for VI<->EN translation
  - SmolVLM2 quantized GGUF through llama.cpp llama-server
  - Piper TTS when an audio response is requested
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MODELS_DIR = REPO_ROOT / "models"

for p in [str(BASE_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipelines import visual_qa as f1  # noqa: E402

app = FastAPI(
    title="BSmart AI Backend",
    description="Multimodal AI Backend for Smart Glasses (ASR, Visual QA, Autopilot Guide)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_asr_model():
    """Lazy-load PhoWhisper CT2 ASR through the Feature 1 pipeline module."""
    return f1.get_asr_model()


def get_vlm_model():
    """Lazy-connect to SmolVLM2 GGUF through llama-server."""
    return f1.get_vlm_model()


def get_translation_model():
    """Lazy-load EnViT5 CT2 translation through the Feature 1 pipeline module."""
    return f1.get_translation_model()


def speech_to_text_vi(audio_path: Path, timings: Optional[f1.StageTimings] = None) -> str:
    return f1.speech_to_text_vi(audio_path, timings=timings)


def translate_text(
    text: str,
    src_lang: str,
    timings: Optional[f1.StageTimings] = None,
    timing_key: Optional[str] = None,
) -> str:
    return f1.translate(text, src_lang, timings=timings, timing_key=timing_key)


def run_vlm_question(
    image_path: Path,
    question_en: str,
    timings: Optional[f1.StageTimings] = None,
) -> str:
    return f1.run_vlm_question(image_path, question_en, timings=timings)


def synthesize_speech_vi(
    text: str,
    out_path: Path,
    timings: Optional[f1.StageTimings] = None,
) -> None:
    f1.speak_vi(text, out_path, timings=timings)


async def _save_upload(
    upload: UploadFile,
    default_name: str,
    tmp_paths: list[Path],
    timings: f1.StageTimings,
) -> Path:
    suffix = Path(upload.filename or default_name).suffix or Path(default_name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = Path(tmp.name)
        with timings.track("request_io"):
            tmp.write(await upload.read())
    tmp_paths.append(path)
    return path


def _timing_payload(timings: f1.StageTimings, request_started: float) -> dict:
    return f1.build_timing_payload(timings, time.perf_counter() - request_started)


def _log_timing(prefix: str, payload: dict) -> None:
    print(f"{prefix}\n{f1.format_latency_report(payload)}")


def _raise_feature1_error(
    *,
    error: Exception,
    stage: str,
    timings: f1.StageTimings,
    request_started: float,
) -> None:
    payload = _timing_payload(timings, request_started)
    _log_timing(f"[Backend][Feature1] failed stage={stage}", payload)

    if isinstance(error, f1.InvalidImageError):
        status_code = 400
        code = "invalid_image"
    elif isinstance(error, f1.EmptyTranscriptError):
        status_code = 422
        code = "empty_transcript"
    elif isinstance(error, f1.LlamaServerTimeout):
        status_code = 504
        code = "qa_inference_timeout"
    elif isinstance(error, f1.LlamaServerUnavailable):
        status_code = 503
        code = "llama_server_unavailable"
    elif isinstance(error, f1.LlamaServerResponseError):
        status_code = 502
        code = "llama_server_bad_response"
    else:
        status_code = 500
        code = "feature1_stage_failed"

    raise HTTPException(
        status_code=status_code,
        detail={
            "error": code,
            "stage": stage,
            "message": str(error),
            "timings": payload,
        },
    ) from error


def _delete_temp_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                os.remove(path)
        except OSError:
            pass


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint. This probes llama-server without starting it."""
    llama_info = f1.get_llama_runtime_info(check_health=True)
    return {
        "status": "ok",
        "service": "BSmart AI Backend",
        "models": {
            "phowhisper_int8": f1.ASR_CT2_DIR.exists(),
            "envit5_int8": f1.TRANSLATE_CT2_DIR.exists(),
            "piper_vi_voice": f1.PIPER_VOICE_PATH.exists(),
            "yolo26s": (MODELS_DIR / "yolo26s.pt").exists() or (BASE_DIR / "yolo26s.pt").exists(),
            "smolvlm2_gguf": {
                "mode": llama_info["mode"],
                "model": llama_info["model"],
                "mmproj": llama_info["mmproj"],
                "quantization": llama_info["quantization"],
                "server_bin": llama_info["server_bin"],
                "url": llama_info["url"],
                "healthy": llama_info.get("healthy", False),
                "auto_start": llama_info["auto_start"],
                "request_timeout_s": llama_info["request_timeout_s"],
                "startup_timeout_s": llama_info["startup_timeout_s"],
            },
        },
    }


@app.post("/transcribe")
async def transcribe_endpoint(
    audio: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """Speech-to-text endpoint for real PhoWhisper CT2 inference."""
    file_to_process = audio or audio_file
    if not file_to_process:
        raise HTTPException(status_code=400, detail={"error": "missing_audio"})

    timings = f1.StageTimings()
    request_started = time.perf_counter()
    tmp_paths: list[Path] = []

    try:
        tmp_audio_path = await _save_upload(file_to_process, "recording.wav", tmp_paths, timings)
        try:
            transcribed_text = speech_to_text_vi(tmp_audio_path, timings=timings)
        except Exception as e:
            print(f"[Backend Error] Transcribe failed: {e}")
            traceback.print_exc()
            _raise_feature1_error(
                error=e,
                stage="stt",
                timings=timings,
                request_started=request_started,
            )

        from pipelines import voice_control as f0

        fid, hits, _ = f0.match_feature(transcribed_text)
        payload = _timing_payload(timings, request_started)
        _log_timing("[Backend][Transcribe] latency", payload)
        return {
            "text": transcribed_text,
            "matched_feature": fid,
            "matched_keywords": hits,
            "timings": payload,
        }
    finally:
        _delete_temp_files(tmp_paths)


@app.post("/qa")
async def qa_endpoint(
    image: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    return_audio: bool = Form(False),
):
    """
    Feature 1 endpoint.

    Existing mobile contract is preserved:
      image + question -> JSON text answer.

    Full backend pipeline is also supported:
      image + audio -> STT -> VQA -> Piper WAV, returned as base64 in JSON.
    """
    request_started = time.perf_counter()
    timings = f1.StageTimings()
    tmp_paths: list[Path] = []

    try:
        if image is None:
            raise HTTPException(status_code=400, detail={"error": "missing_image"})

        tmp_image_path = await _save_upload(image, "image.jpg", tmp_paths, timings)
        try:
            f1.validate_image_file(tmp_image_path)
        except Exception as e:
            print(f"[Backend Error] Image validation failed: {e}")
            if not isinstance(e, f1.InvalidImageError):
                traceback.print_exc()
            _raise_feature1_error(
                error=e,
                stage="image_preparation",
                timings=timings,
                request_started=request_started,
            )

        audio_upload = audio or audio_file
        question_vi = (question or "").strip()
        audio_was_input = audio_upload is not None

        if not question_vi and audio_upload is not None:
            tmp_audio_path = await _save_upload(audio_upload, "question.wav", tmp_paths, timings)
            try:
                question_vi = speech_to_text_vi(tmp_audio_path, timings=timings).strip()
            except Exception as e:
                print(f"[Backend Error] Question STT failed: {e}")
                if not isinstance(e, f1.Feature1PipelineError):
                    traceback.print_exc()
                _raise_feature1_error(
                    error=e,
                    stage="stt",
                    timings=timings,
                    request_started=request_started,
                )
            if not question_vi:
                _raise_feature1_error(
                    error=f1.EmptyTranscriptError("PhoWhisper returned an empty question transcript"),
                    stage="stt",
                    timings=timings,
                    request_started=request_started,
                )

        if not question_vi:
            raise HTTPException(status_code=400, detail={"error": "missing_question_or_audio"})

        try:
            question_en = translate_text(
                question_vi,
                src_lang="vi",
                timings=timings,
                timing_key="vi_to_en",
            )
        except Exception as e:
            print(f"[Backend Error] Question translation failed: {e}")
            traceback.print_exc()
            _raise_feature1_error(
                error=e,
                stage="question_vi_to_en",
                timings=timings,
                request_started=request_started,
            )

        try:
            answer_en = run_vlm_question(tmp_image_path, question_en, timings=timings)
        except Exception as e:
            print(f"[Backend Error] QA inference failed: {e}")
            if not isinstance(e, f1.LlamaServerError):
                traceback.print_exc()
            _raise_feature1_error(
                error=e,
                stage="vlm",
                timings=timings,
                request_started=request_started,
            )

        try:
            answer_vi = translate_text(
                answer_en,
                src_lang="en",
                timings=timings,
                timing_key="en_to_vi",
            )
        except Exception as e:
            print(f"[Backend Error] Answer translation failed: {e}")
            traceback.print_exc()
            _raise_feature1_error(
                error=e,
                stage="answer_en_to_vi",
                timings=timings,
                request_started=request_started,
            )

        audio_base64 = None
        if return_audio or audio_was_input:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_answer:
                tmp_answer_path = Path(tmp_answer.name)
            tmp_paths.append(tmp_answer_path)
            try:
                synthesize_speech_vi(answer_vi, tmp_answer_path, timings=timings)
                audio_base64 = base64.b64encode(tmp_answer_path.read_bytes()).decode("ascii")
            except Exception as e:
                print(f"[Backend Error] Piper TTS failed: {e}")
                if not isinstance(e, f1.Feature1PipelineError):
                    traceback.print_exc()
                _raise_feature1_error(
                    error=e,
                    stage="tts",
                    timings=timings,
                    request_started=request_started,
                )

        timing_payload = _timing_payload(timings, request_started)
        _log_timing("[Backend][Feature1] latency", timing_payload)
        response = {
            "success": True,
            "question": question_vi,
            "answer": answer_vi,
            "question_en": question_en,
            "answer_en": answer_en,
            "model": f1.get_llama_runtime_info(check_health=False)["model"],
            "translation_model": f1.TRANSLATE_MODEL_PATH,
            "timings": timing_payload,
        }
        if audio_base64 is not None:
            response["audio_base64"] = audio_base64
            response["audio_mime"] = "audio/wav"
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Backend Error] QA failed: {e}")
        traceback.print_exc()
        _raise_feature1_error(
            error=e,
            stage="feature1",
            timings=timings,
            request_started=request_started,
        )
    finally:
        _delete_temp_files(tmp_paths)


@app.post("/navigate")
async def navigate_endpoint(image: UploadFile = File(...)):
    """
    Feature 3 endpoint.

    This remains intentionally separate from the Feature 1 GGUF integration.
    """
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = Path(tmp.name)

    try:
        yolo_path = MODELS_DIR / "yolo26s.pt"
        if not yolo_path.exists():
            yolo_path = BASE_DIR / "yolo26s.pt"

        if yolo_path.exists():
            try:
                from ultralytics import YOLO

                yolo = YOLO(str(yolo_path))
                results = yolo(str(tmp_path))
                boxes = results[0].boxes
                labels = (
                    [results[0].names[int(cls)] for cls in boxes.cls.cpu().numpy()]
                    if len(boxes) > 0
                    else []
                )
                if labels:
                    warning = f"Lưu ý, phát hiện {', '.join(set(labels))} ở phía trước."
                else:
                    warning = "Đường đi phía trước thoáng."
                return {"warning": warning, "detected_objects": labels}
            except Exception as yolo_err:
                print(f"[Backend Warning] YOLO error: {yolo_err}")

        return {"warning": "Phía trước đường trống, di chuyển an toàn."}
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn

    print("Starting BSmart Backend Server on http://0.0.0.0:8000 ...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
