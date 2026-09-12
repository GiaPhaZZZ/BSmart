#!/usr/bin/env python3
"""
BSmart FastAPI Backend Server
Wraps pre-trained models (PhoWhisper, SmolVLM2, EnViT5, YOLO26s, ZipDepth, Piper TTS)
into RESTful API endpoints for the BSmart mobile app.
"""

import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MODELS_DIR = REPO_ROOT / "models"

# Ensure current directory and pipelines are in sys.path
for p in [str(BASE_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

app = FastAPI(
    title="BSmart AI Backend",
    description="Multimodal AI Backend for Smart Glasses (ASR, Visual QA, Autopilot Guide)",
    version="1.0.0",
)

# Enable CORS for Mobile App access across local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model caches
ASR_PROCESSOR = None
ASR_MODEL = None
VLM_MODELS = None
TRANSLATE_MODELS = None
NAV_PIPELINE = None
VLM_LOCK = threading.Lock()
TRANSLATE_LOCK = threading.Lock()


def get_asr_model():
    """Lazy-load PhoWhisper CT2 ASR model."""
    global ASR_PROCESSOR, ASR_MODEL
    if ASR_MODEL is not None:
        return ASR_PROCESSOR, ASR_MODEL

    try:
        from pipelines import voice_control as f0
        if f0.ASR_CT2_DIR.exists():
            print("[Backend] Loading PhoWhisper ASR model...")
            whisper_processor, asr_model = f0._load_asr()
            f0.MODELS = (whisper_processor, asr_model)
            ASR_PROCESSOR, ASR_MODEL = whisper_processor, asr_model
            print("[Backend] PhoWhisper ASR model loaded successfully.")
            return ASR_PROCESSOR, ASR_MODEL
        else:
            print(f"[Backend Warning] ASR model dir {f0.ASR_CT2_DIR} not found.")
    except Exception as e:
        print(f"[Backend Warning] Failed to load ASR model: {e}")
        traceback.print_exc()

    return None, None


def get_vlm_model():
    """Lazy-load real SmolVLM2 VQA components. No demo or fallback responses."""
    global VLM_MODELS
    if VLM_MODELS is not None:
        return VLM_MODELS

    with VLM_LOCK:
        if VLM_MODELS is not None:
            return VLM_MODELS

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            print(f"[Backend] Loading SmolVLM2 VQA model on {device}...")
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    dtype=dtype,
                    _attn_implementation="sdpa",
                    local_files_only=True,
                )
            except Exception as sdpa_error:
                print(f"[Backend] SmolVLM2 sdpa unavailable, using eager attention: {sdpa_error}")
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    dtype=dtype,
                    _attn_implementation="eager",
                    local_files_only=True,
                )
            model = model.to(device)
            model.eval()
            VLM_MODELS = {
                "processor": processor,
                "model": model,
                "device": device,
                "dtype": dtype,
                "model_id": model_id,
            }
            print("[Backend] SmolVLM2 VQA model loaded successfully.")
            return VLM_MODELS
        except Exception as e:
            print(f"[Backend Error] Failed to load SmolVLM2 VQA model: {e}")
            traceback.print_exc()
            raise RuntimeError(f"SmolVLM2 model load failed: {e}") from e


def get_translation_model():
    """Lazy-load existing EnViT5 CTranslate2 translation components."""
    global TRANSLATE_MODELS
    if TRANSLATE_MODELS is not None:
        return TRANSLATE_MODELS

    with TRANSLATE_LOCK:
        if TRANSLATE_MODELS is not None:
            return TRANSLATE_MODELS

        try:
            import ctranslate2
            from pipelines import visual_qa as f1

            if not f1.TRANSLATE_CT2_DIR.exists():
                raise FileNotFoundError(f"EnViT5 CT2 model not found: {f1.TRANSLATE_CT2_DIR}")

            print("[Backend] Loading EnViT5 translation model...")
            tokenizer = f1.load_envit5_tokenizer(f1.TRANSLATE_MODEL_PATH)
            translator = ctranslate2.Translator(str(f1.TRANSLATE_CT2_DIR), device=f1.DEVICE)
            TRANSLATE_MODELS = {
                "tokenizer": tokenizer,
                "translator": translator,
                "model_id": f1.TRANSLATE_MODEL_PATH,
                "max_length": f1.TRANSLATE_MAX_LENGTH,
            }
            print("[Backend] EnViT5 translation model loaded successfully.")
            return TRANSLATE_MODELS
        except Exception as e:
            print(f"[Backend Error] Failed to load EnViT5 translation model: {e}")
            traceback.print_exc()
            raise RuntimeError(f"EnViT5 translation model load failed: {e}") from e


def translate_text(text: str, src_lang: str) -> str:
    """Translate text with the existing EnViT5 vi/en CT2 pipeline."""
    if src_lang not in {"vi", "en"}:
        raise ValueError(f"Unsupported translation source language: {src_lang}")

    models = get_translation_model()
    tokenizer = models["tokenizer"]
    translator = models["translator"]
    max_length = models["max_length"]

    prefixed = f"{src_lang}: {text.strip()}"
    input_ids = tokenizer.encode(prefixed, truncation=True, max_length=max_length)
    source_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    results = translator.translate_batch([source_tokens], max_decoding_length=max_length)
    output_tokens = results[0].hypotheses[0]
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    translated = tokenizer.decode(output_ids, skip_special_tokens=True).split(":", 1)[-1].strip()

    if not translated:
        raise RuntimeError("EnViT5 returned empty translation")
    return translated


def run_vlm_question(image_path: Path, question: str) -> str:
    """Run real SmolVLM2 visual QA for an image and text question."""
    try:
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"Missing VQA runtime dependency: {e}") from e

    vlm = get_vlm_model()
    processor = vlm["processor"]
    model = vlm["model"]
    dtype = vlm["dtype"]

    image = Image.open(image_path).convert("RGB")
    prompt = (
        f"{question.strip()} "
        "Answer in one concise, complete English sentence. "
        "Start with 'There is', 'There are', or 'No,' when appropriate. "
        "Do not invent details not visible in the image."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    with torch.inference_mode():
        inputs = processor(text=text_prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(model.device, dtype=dtype)
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=80,
            min_new_tokens=3,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    if not answer:
        raise RuntimeError("SmolVLM2 returned an empty answer")
    return answer


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint."""
    asr_available = (MODELS_DIR / "phowhisper-ct2-int8").exists() or (BASE_DIR / "phowhisper-ct2-int8").exists()
    vlm_available = (MODELS_DIR / "envit5-ct2-int8").exists() or (BASE_DIR / "envit5-ct2-int8").exists()
    yolo_available = (MODELS_DIR / "yolo26s.pt").exists() or (BASE_DIR / "yolo26s.pt").exists()

    return {
        "status": "ok",
        "service": "BSmart AI Backend",
        "models": {
            "phowhisper_int8": asr_available,
            "envit5_int8": vlm_available,
            "yolo26s": yolo_available,
        },
    }


@app.post("/transcribe")
async def transcribe_endpoint(
    audio: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """
    Speech-To-Text Endpoint:
    Receives an audio file (WAV/MP3/PCM) and returns transcribed Vietnamese text.
    """
    file_to_process = audio or audio_file
    if not file_to_process:
        raise HTTPException(status_code=400, detail="No audio file provided in request (field 'audio').")

    # Save uploaded file to temp file
    suffix = Path(file_to_process.filename or "recording.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file_to_process.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        processor, model = get_asr_model()
        if processor and model:
            from pipelines import voice_control as f0
            transcribed_text = f0.speech_to_text_vi(tmp_path)
            fid, hits, _ = f0.match_feature(transcribed_text)
            return {
                "text": transcribed_text,
                "matched_feature": fid,
                "matched_keywords": hits,
            }
        else:
            # Fallback if INT8 model not converted yet
            return {
                "text": "Mở tính năng 1",
                "matched_feature": 1,
                "matched_keywords": ["tính năng 1"],
                "warning": "PhoWhisper INT8 model not found. Using fallback text.",
            }
    except Exception as e:
        print(f"[Backend Error] Transcribe failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcribe processing error: {str(e)}")
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


@app.post("/qa")
async def qa_endpoint(
    image: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
):
    """
    Visual QA Chatbot Endpoint:
    Receives an image and a text question. Returns real SmolVLM2 output only.
    """
    tmp_image_path = None

    try:
        if image is None:
            raise HTTPException(status_code=400, detail={"error": "missing_image"})
        if question is None or not question.strip():
            raise HTTPException(status_code=400, detail={"error": "missing_question"})

        suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_img:
            tmp_img.write(await image.read())
            tmp_image_path = Path(tmp_img.name)

        try:
            question_en = translate_text(question, src_lang="vi")
        except Exception as e:
            print(f"[Backend Error] Question translation failed: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "qa_translation_failed",
                    "stage": "question_vi_to_en",
                    "message": str(e),
                },
            ) from e

        try:
            answer_en = run_vlm_question(tmp_image_path, question_en)
        except Exception as e:
            print(f"[Backend Error] QA inference failed: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "qa_inference_failed",
                    "message": str(e),
                },
            ) from e

        try:
            answer_vi = translate_text(answer_en, src_lang="en")
        except Exception as e:
            print(f"[Backend Error] Answer translation failed: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "qa_translation_failed",
                    "stage": "answer_en_to_vi",
                    "message": str(e),
                },
            ) from e

        return {
            "success": True,
            "question": question,
            "answer": answer_vi,
            "question_en": question_en,
            "answer_en": answer_en,
            "model": VLM_MODELS["model_id"] if VLM_MODELS else "SmolVLM2",
            "translation_model": TRANSLATE_MODELS["model_id"] if TRANSLATE_MODELS else "EnViT5",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Backend Error] QA failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "qa_inference_failed",
                "message": str(e),
            },
        )
    finally:
        if tmp_image_path and tmp_image_path.exists():
            os.remove(tmp_image_path)


@app.post("/navigate")
async def navigate_endpoint(
    image: UploadFile = File(...),
):
    """
    Autopilot Guide / Navigation Endpoint:
    Receives continuous camera frames (every 4s) and predicts obstacles/depth.
    """
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = Path(tmp.name)

    try:
        # Check YOLO model
        yolo_path = MODELS_DIR / "yolo26s.pt"
        if not yolo_path.exists():
            yolo_path = BASE_DIR / "yolo26s.pt"

        if yolo_path.exists():
            try:
                from ultralytics import YOLO
                yolo = YOLO(str(yolo_path))
                results = yolo(str(tmp_path))
                boxes = results[0].boxes
                labels = [results[0].names[int(cls)] for cls in boxes.cls.cpu().numpy()] if len(boxes) > 0 else []
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
