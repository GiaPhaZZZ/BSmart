# 👓 BSmart — Multimodal AI Pipeline for Smart Glasses

A lightweight, high-performance multimodal AI system designed for smart glasses to assist visually impaired individuals with **voice interaction, visual question answering, and autonomous navigation assistance**.

---

## ✨ System Architecture & Features

| Feature | Model Stack | Operational Mode | Description |
| :--- | :--- | :--- | :--- |
| **Function 0 — Voice Control** | `PhoWhisper-tiny ONNX` + Android SpeechRecognizer fallback | On-Demand | Speech-to-text voice command recognition & feature activation |
| **Function 1 — Visual QA Chatbot** | `SmolVLM2 ONNX` + Android TTS | One-Shot | Offline multimodal Q&A on captured environment images via spoken questions |
| **Function 2 — Photo Capture** | ESP32-S3 camera / phone camera fallback | One-Shot | Capture and save a local JPEG image, then return to `IDLE` |
| **Function 3 — Navigation / Obstacle Awareness** | `YOLO26s ONNX` + `ZipDepth ONNX` + Android TTS | Continuous (4s/frame) | Real-time object detection, relative depth warning, and optional destination-aware guidance |
| **FastAPI Backend Server** | `FastAPI` + `Uvicorn` | Dev / Reference | REST API wrapper for local workstation testing; not required for the offline Android runtime |
| **Mobile App (React Native)** | `ONNX Runtime Android` + Native Kotlin modules | Standalone Offline | On-device inference, BLE control, voice UI, and smart-glasses state machine |

---

## 📁 Repository Structure

```text
BSmart/
├── backend/                     # Python FastAPI Backend & AI Pipelines
│   ├── server.py                # FastAPI REST server for workstation/reference testing
│   ├── pipelines/               # Core multimodal processing pipelines
│   │   ├── voice_control.py     # Voice command recognition & activation
│   │   ├── visual_qa.py         # Visual QA multimodal chatbot pipeline
│   │   └── autopilot.py         # Reference obstacle detection & depth guidance
│   ├── scripts/                 # Utility, setup & diagnostic scripts
│   │   ├── check_glass.py       # 11/11 Sanity & inference verifier
│   │   ├── download_models.py   # Pretrained AI models downloader
│   │   ├── setup_glass.sh
│   │   ├── start_backend.bat
│   │   └── start_backend.sh
│   └── requirements.txt         # Python dependencies
│
├── mobile_app/                  # React Native mobile application
│   ├── android/                 # Android project with ONNX Runtime and native Kotlin modules
│   ├── src/                     # UI components, state machine & inference services
│   └── scripts/                 # Icon generation & dependency patches
│
├── firmware/                    # Smart Glasses ESP32-S3 hardware code
├── ai_core/                     # ONNX model export & packaging scripts
├── assets/                      # App icons (icon.svg) & sound clips (Open_f1..f4.mp3)
├── docs/                        # Architecture & documentation
├── test/                        # Test photos & audio clips
└── start_backend.bat            # One-click Windows backend server startup
```

---

## 🚀 Quick Setup & Installation Guide

### 1. Environment Activation
```bash
# Clone the repository
git clone https://github.com/GiaPhaZZZ/BSmart.git
cd BSmart

# Activate Virtual Environment (WSL / Ubuntu / Windows)
source ./glass/bin/activate  # or ./glass/glass/bin/activate
```

### 2. Download Pretrained Models
Download all required baseline weights (`yolo26s.pt`, `PhoWhisper`, `EnViT5`, `SmolVLM2`, `Piper vi_VN`):
```bash
python backend/scripts/download_models.py
```

### 3. INT8 Model Optimization (Must-Do for CPU Speed)
Convert speech & translation models to CTranslate2 INT8 format for real-time CPU performance:
```bash
# Convert PhoWhisper ASR
ct2-transformers-converter --model vinai/PhoWhisper-tiny --output_dir models/phowhisper-ct2-int8 --quantization int8

# Convert EnViT5 Translation
ct2-transformers-converter --model VietAI/envit5-translation --output_dir models/envit5-ct2-int8 --quantization int8
```

### 4. Verify System Readiness (11/11 Sanity Check)
Run the diagnostic script to verify that all libraries and models load correctly:
```bash
python backend/scripts/check_glass.py
```
*Expected Output: `11/11 checks passed` with no errors.*

---

## 🌐 FastAPI Backend Server API

Start the backend server to serve the Mobile App and Smart Glasses:

```bash
# On Windows
start_backend.bat

# On Linux / WSL / Terminal
python backend/server.py
```
*Server runs at: `http://0.0.0.0:8000`*

### API Specification

#### 1. `GET /health`
* **Description**: Server & model availability status check.
* **Response**: `200 OK`
  ```json
  {
    "status": "ok",
    "service": "BSmart AI Backend",
    "models": {
      "phowhisper_int8": true,
      "envit5_int8": true,
      "yolo26s": true
    }
  }
  ```

#### 2. `POST /navigate`
* **Description**: Autopilot camera frame obstacle & terrain analysis.
* **Request**: `multipart/form-data` (`image`: JPEG/PNG file)
* **Response**: `200 OK`
  ```json
  {
    "warning": "Lưu ý, phát hiện car ở phía trước.",
    "detected_objects": ["car"]
  }
  ```

#### 3. `POST /transcribe`
* **Description**: Spoken command recognition.
* **Request**: `multipart/form-data` (`audio`: WAV/MP3 file)
* **Response**: `200 OK`
  ```json
  {
    "text": "mở tính năng 1.",
    "matched_feature": 1,
    "matched_keywords": ["tính năng 1"]
  }
  ```

#### 4. `POST /qa`
* **Description**: Visual QA Chatbot (Image + Spoken/Text Question -> Answer).
* **Request**: `multipart/form-data` (`image`: file, `audio`: optional file, `question`: optional text)
* **Response**: `200 OK`
  ```json
  {
    "question": "Phía trước có gì?",
    "answer": "Một con phố với hàng cây và những toà nhà dưới bầu trời trong xanh."
  }
  ```

---

## 📱 Mobile App & Offline Standalone APK Build

### 1. Package ONNX Models for Mobile
Export lightweight models to Android assets:
```bash
python ai_core/package_models.py
```

### 2. Build Debug APK (Standalone Offline)
```powershell
cd mobile_app/android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/BSmart.apk
```

### 3. Build Release APK (Standalone)
```bash
cd mobile_app
npx react-native build-android --mode=release
```
*Output APK location:* `mobile_app/android/app/build/outputs/apk/release/app-release.apk`

### Android Runtime Notes

- ONNX models are packaged in Android assets, then streamed with a 64 KB buffer into the app sandbox at runtime: `filesDir/onnx_models/<model filename>`.
- Model files are cached and only recopied when the extracted file size differs from the asset size.
- ONNX Runtime sessions are created from file paths, not in-memory `ByteArray`s, to avoid heap spikes when loading large models such as `smolvlm2_decoder.onnx`.
- Sessions are lazy-loaded per active feature and inactive groups can be released through `releaseObjectDetectionModels()`, `releaseVlmModels()`, and `releaseAsrModels()`.
- `android:largeHeap="true"` is not used as the primary memory fix.

### Voice Command Aliases

Voice command matching is accent-insensitive and supports common ASR variants:

| Feature | Accepted examples |
| :--- | :--- |
| Feature 1 — QA | `1`, `số 1`, `một`, `số một`, `tính năng 1`, `tính năng một`, `tính năng số 1`, `tính năng số một`, `hỏi đáp`, `gpt` |
| Feature 2 — Capture | `2`, `số 2`, `hai`, `số hai`, `tính năng 2`, `tính năng hai`, `tính năng số 2`, `tính năng số hai`, `tính năng hay`, `tính năng số hay`, `chụp ảnh` |
| Feature 3 — Navigation | `3`, `số 3`, `ba`, `số ba`, `tính năng 3`, `tính năng ba`, `tính năng số 3`, `tính năng số ba`, `tìm đường`, `dẫn đường` |

---

## 🔌 ESP32-S3 Firmware Flashing

BSmart includes a one-click flashing script that automatically detects the ESP32-S3 COM port, builds from current source when release artifacts are missing or stale, and flashes validated binaries (`bootloader.bin`, `partitions.bin`, and `firmware.bin`).

```bash
# Make sure the ESP32 is connected via USB, then run at project folder:
python flash_firmware.py

# Select the port explicitly when more than one ESP32 adapter is present
python flash_firmware.py --port COM7

# Always rebuild from source before flashing
python flash_firmware.py --rebuild
```

The generated `release/firmware/manifest.json` records source metadata and SHA-256 hashes for all three binaries. A failed build never falls back to older artifacts.
*Note: The script automatically installs required dependencies (`esptool`, `pyserial`) if they are missing.*

---

## 🧪 CLI Feature Execution Examples

```bash
# Function 0: Voice Command Test
python backend/pipelines/voice_control.py --audio test/audio/mo_tinh_nang.mp3

# Function 1: Visual QA Chatbot Test
python backend/pipelines/visual_qa.py --image test/photo/road.jpg --audio test/audio/mieu_ta_khung_canh.mp3 --out out.wav

# Function 3 reference: Autopilot Navigation Test
python backend/pipelines/autopilot.py --image test/photo/road.jpg --audio guide.wav
```
