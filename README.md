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
├── Function0_voice_control.py   # Voice command recognition & activation
├── Function1_chatbot.py         # Visual QA multimodal chatbot pipeline
├── Funtion2_autopilot.py        # Autopilot obstacle detection & depth guidance
├── server.py                    # FastAPI REST server for Mobile App integration
├── download_models.py           # Pretrained AI models downloader
├── check_glass.py               # 11/11 Sanity & inference verifier
├── start_backend.bat            # One-click Windows backend server startup
│
├── ai_core/                     # ONNX model export & packaging scripts
│   ├── export_phowhisper_onnx.py
│   ├── export_smolvlm2_onnx.py
│   ├── export_yolo_zipdepth_onnx.py
│   └── package_models.py
│
├── mobile_app/                  # React Native mobile application
│   ├── android/                 # Android project with ONNX Runtime Native
│   ├── src/                     # React Native UI & On-Device inference modules
│   └── App.tsx
│
├── firmware/                    # Smart Glasses ESP32 hardware code
├── ZipDepth/                    # Monocular depth estimation repository & checkpoints
├── voices/                      # Piper TTS Vietnamese voice assets (.onnx)
└── test/                        # Test photos & audio clips
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
python download_models.py
```

### 3. INT8 Model Optimization (Must-Do for CPU Speed)
Convert speech & translation models to CTranslate2 INT8 format for real-time CPU performance:
```bash
# Convert PhoWhisper ASR
ct2-transformers-converter --model vinai/PhoWhisper-tiny --output_dir phowhisper-ct2-int8 --quantization int8

# Convert EnViT5 Translation
ct2-transformers-converter --model VietAI/envit5-translation --output_dir envit5-ct2-int8 --quantization int8
```

### 4. Verify System Readiness (11/11 Sanity Check)
Run the diagnostic script to verify that all libraries and models load correctly:
```bash
python check_glass.py
```
*Expected Output: `11/11 checks passed` with no errors.*

---

## 🌐 FastAPI Backend Server API

Start the backend server to serve the Mobile App and Smart Glasses:

```bash
# On Windows
start_backend.bat

# On Linux / WSL / Terminal
python server.py
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

## 🧪 CLI Feature Execution Examples

```bash
# Function 0: Voice Command Test
python Function0_voice_control.py --audio test/audio/mo_tinh_nang.mp3

# Function 1: Visual QA Chatbot Test
python Function1_chatbot.py --image test/photo/road.jpg --audio test/audio/mieu_ta_khung_canh.mp3 --out out.wav

# Function 2: Autopilot Navigation Test
python Funtion2_autopilot.py --image test/photo/road.jpg --audio guide.wav
```
