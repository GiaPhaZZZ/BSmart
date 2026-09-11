# 👓 BSmart — Multimodal AI Pipeline for Smart Glasses

A lightweight, high-performance multimodal AI system designed for smart glasses to assist visually impaired individuals with **voice interaction, visual question answering, and autonomous navigation assistance**.

---

## ✨ System Architecture & Features

| Feature | Model Stack | Operational Mode | Description |
| :--- | :--- | :--- | :--- |
| **Function 0 — Voice Control** | `PhoWhisper-tiny (CT2 INT8)` | On-Demand | Speech-to-text voice command recognition & feature activation |
| **Function 1 — Visual QA Chatbot** | `PhoWhisper` + `EnViT5` + `SmolVLM2` + `Piper TTS` | One-Shot | Multimodal Q&A on captured environment images via spoken questions |
| **Function 2 — Autopilot Guide** | `YOLO26s` + `ZipDepth` + `Piper TTS` | Continuous (1-5s) | Real-time object detection, 3-column terrain depth estimation & audio warning |
| **FastAPI Backend Server** | `FastAPI` + `Uvicorn` | Service | RESTful API server bridging Smart Glasses & Mobile App |
| **Mobile App (React Native)** | `ONNX Runtime Native C++` | Standalone / Hybrid | On-device inference & voice interface for smart glasses |

---

## 📁 Repository Structure

```text
BSmart/
├── backend/                     # Python FastAPI Backend & AI Pipelines
│   ├── server.py                # FastAPI REST server for Mobile App integration
│   ├── pipelines/               # Core multimodal processing pipelines
│   │   ├── voice_control.py     # Voice command recognition & activation
│   │   ├── visual_qa.py         # Visual QA multimodal chatbot pipeline
│   │   └── autopilot.py         # Autopilot obstacle detection & depth guidance
│   ├── scripts/                 # Utility, setup & diagnostic scripts
│   │   ├── check_glass.py       # 11/11 Sanity & inference verifier
│   │   ├── download_models.py   # Pretrained AI models downloader
│   │   ├── setup_glass.sh
│   │   ├── start_backend.bat
│   │   └── start_backend.sh
│   └── requirements.txt         # Python dependencies
│
├── mobile_app/                  # React Native mobile application
│   ├── android/                 # Android project with ONNX Runtime Native
│   ├── src/                     # UI components, state machine & inference services
│   └── scripts/                 # Icon generation & dependency patches
│
├── firmware/                    # Smart Glasses ESP32-S3 hardware code
├── ai_core/                     # ONNX model export & packaging scripts
├── models/                      # Pretrained weights & model checkpoints (Git Ignored)
├── assets/                      # App icons (icon.svg) & sound clips (Open_f1..f4.mp3)
├── release/                     # Packaged standalone APK outputs (BSmart.apk)
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

### 2. Build Release APK (Standalone)
```bash
cd mobile_app
npx react-native build-android --mode=release
```
*Output APK location:* `mobile_app/android/app/build/outputs/apk/release/app-release.apk`

---

## 🔌 ESP32-S3 Firmware Flashing

BSmart includes a one-click flashing script that automatically detects the ESP32-S3 COM port and flashes the release binaries (`bootloader.bin`, `partitions.bin`, and `firmware.bin`).

```bash
# Make sure the ESP32 is connected via USB, then run at project folder:
python flash_firmware.py
```
*Note: The script automatically installs required dependencies (`esptool`, `pyserial`) if they are missing.*

---

## 🧪 CLI Feature Execution Examples

```bash
# Function 0: Voice Command Test
python backend/pipelines/voice_control.py --audio test/audio/mo_tinh_nang.mp3

# Function 1: Visual QA Chatbot Test
python backend/pipelines/visual_qa.py --image test/photo/road.jpg --audio test/audio/mieu_ta_khung_canh.mp3 --out out.wav

# Function 2: Autopilot Navigation Test
python backend/pipelines/autopilot.py --image test/photo/road.jpg --audio guide.wav
```
