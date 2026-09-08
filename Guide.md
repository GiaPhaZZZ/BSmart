# Smart Glasses

A lightweight multimodal AI pipeline for smart glasses, designed to provide **voice interaction, visual question answering, and navigation assistance for blind users**.

---

## ✨ Features

| Function                           | Description                                                                                                   | Mode       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------- |
| **Function 0 — Voice Control**     | Recognizes user voice commands and activates the corresponding function                                       | On-demand  |
| **Function 1 — Visual QA Chatbot** | Answers questions about a captured image using voice input                                                    | One-shot   |
| **Function 2 — Autopilot Guide**   | Periodically captures images, detects obstacles/environment, estimates distance, and generates audio warnings | Continuous |
| **Function 3**                     | Under development                                                                                             | —          |

---

# 🛠️ Requirements

### Operating System

* Ubuntu / WSL2
* Python
* `uv`

### Recommended

* NVIDIA GPU with CUDA support for accelerated inference
* Internet connection for downloading models

---

# 🚀 Installation

## 1. Enter Ubuntu / WSL

If you are using Windows with WSL:

```bash
wsl
```

If you are already inside Ubuntu/WSL, skip this step.

---

## 2. Run the Setup Script

From the project root directory:

```bash
bash set_up.bash
```

The setup script installs the required Python environment and project dependencies.

> **Note:** The exact dependencies installed by `set_up.bash` should be kept in the script itself rather than manually duplicated in this README.

---

## 3. Download Required Models

After the environment has been created:

```bash

source ./glass/bin/activate #Remember to check the path is /glass or /glass/glass

python download_models.py
```

This downloads the models required by the Smart Glasses pipeline.

---

## 4. Install FFmpeg

FFmpeg is required for audio processing.

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Verify the installation:

```bash
ffmpeg -version
```

---

## 5. Verify the Installation

Run:

```bash
python check_glass.py
```

If the script completes successfully without missing-package, missing-model, or configuration errors, the installation is considered complete.

---

# 🔄 Activate the Environment

Every time a new WSL terminal is opened, activate the project environment:

```bash
source ./glass/glass/bin/activate
```

You should activate the environment **before running any Smart Glasses function**.

For example:

```bash
source ./glass/glass/bin/activate

python Function0_voice_control.py --audio test/audio/mo_tinh_nang.mp3
```

---

# ⚡ Must do: INT8 Model Optimization

The project can use CTranslate2 INT8 models to reduce model size and potentially improve inference efficiency.

## PhoWhisper

```bash
ct2-transformers-converter \
    --model vinai/PhoWhisper-tiny \
    --output_dir phowhisper-ct2-int8 \
    --quantization int8
```

## EnvIT5

```bash
ct2-transformers-converter \
    --model VietAI/envit5-translation \
    --output_dir envit5-ct2-int8 \
    --quantization int8
```

After verifying that the INT8 versions work correctly, the original model files can be removed if additional disk space is required.

> **Important:** Do not delete the original models until the INT8 versions have been tested successfully.

---

# 🎙️ Function 0 — Voice Function Activation

Voice control allows the user to activate different Smart Glasses functions using spoken commands.

### Pipeline

```text
User Audio
    ↓
Speech-to-Text
    ↓
Command Recognition
    ↓
Function Matching
    ↓
Activate Function
    ↓
Audio Feedback
```

The system transcribes the user's audio and matches the resulting command against the available functions.

### Run

```bash
python Function0_voice_control.py \
    --audio test/audio/mo_tinh_nang.mp3
```

---

# 💬 Function 1 — Visual QA Chatbot

The Visual QA Chatbot allows the user to ask questions about an image using voice input.

### Pipeline

```text
Image + User Audio
        ↓
   Speech-to-Text
        ↓
Vietnamese → English
        ↓
      SmolVLM
        ↓
  Generated Answer
        ↓
English → Vietnamese
        ↓
   Text-to-Speech
        ↓
    Audio Output
```

### Workflow

1. Load one image.
2. Load the user's audio question.
3. Transcribe the audio.
4. Translate the Vietnamese question into English.
5. Send the image and question to SmolVLM.
6. Generate an answer.
7. Translate the answer back into Vietnamese.
8. Convert the answer into speech.
9. Save the resulting audio.

### Run

```bash
python Function1_chatbot.py \
    --image test/photo/stair_2.jpg \
    --audio test/audio/mieu_ta_khung_canh.mp3 \
    --out out.wav
```

### Output

The generated response is saved as:

```text
out.wav
```

---

# 🦯 Function 2 — Autopilot Guide

The Autopilot Guide continuously analyzes the surrounding environment and provides audio guidance for blind users.

### Pipeline

```text
Camera
   ↓
Capture Image
   ↓
Object Detection
   +
Depth Estimation
   ↓
Environment / Terrain Analysis
   ↓
Warning Generation
   ↓
Text-to-Speech
   ↓
Audio Warning
   ↓
Capture Next Image
   ↺
```

### Current Behavior

The autopilot mode:

* Captures approximately **one image every 5 seconds**.
* Detects relevant objects.
* Estimates object distance.
* Analyzes terrain/environment conditions.
* Generates navigation warnings.
* Converts warnings into speech.
* Repeats continuously until the user exits the function.

### Run

```bash
python Function2_autopilot.py \
    --image test/photo/japan.jpeg \
    --audio guide.wav
```

---

# 🧪 Test Data

The `test/` directory contains sample images and audio files for testing the system.

```text
test/
├── audio/
│   ├── mo_tinh_nang.mp3
│   └── mieu_ta_khung_canh.mp3
│
└── photo/
    ├── stair_2.jpg
    └── japan.jpeg
```

The `activate_voice/` directory contains predefined audio commands used for activating or switching between functions.

Example:

```text
activate_voice/
├── ...
```

---

# 📁 Project Structure

The project is organized approximately as follows:

```text
project/
│
├── set_up.bash
├── download_models.py
├── check_glass.py
│
├── Function0_voice_control.py
├── Function1_chatbot.py
├── Function2_autopilot.py
│
├── test/
│   ├── audio/
│   └── photo/
│
├── activate_voice/
│
├── glass/
│   └── glass/
│
└── ...
```

---

# 🖼️ Supported Input Formats

The system should support common image formats:

```text
.jpg
.jpeg
.png
.webp
```

Audio input should support common formats such as:

```text
.mp3
.wav
```

Additional formats may be supported depending on the underlying libraries and models.

---

# 🔍 Installation Verification

A complete installation should follow this sequence:

```bash
# Enter WSL if using Windows
wsl

# Run setup
bash set_up.bash

source ./glass/bin/activate #Check for the path is /glass or /glass/glass

# Download models
python download_models.py

# Install system audio dependency
sudo apt update
sudo apt install -y ffmpeg

# Verify installation
python check_glass.py
```

If `check_glass.py` finishes without errors, the environment is ready.

---

# ⚠️ Troubleshooting

## Environment not activated

If Python cannot find project dependencies:

```bash
source ./glass/glass/bin/activate
```

Then verify:

```bash
which python
python --version
```

---

## FFmpeg not found

Install FFmpeg:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Then verify:

```bash
ffmpeg -version
```

---

## Model not found

Run:

```bash
python download_models.py
```

Then run:

```bash
python check_glass.py
```

---

# 🚧 Function 3

**Status:** Under development.

Additional Smart Glasses functionality will be added in future versions.

---

# 📝 Notes

* Run commands from the **project root directory** unless otherwise specified.
* Activate the virtual environment before running the Python functions.
* Do not delete original model files until their INT8 versions have been verified.
* Keep test images and audio in the `test/` directory.
* The INT8 conversion step is optional and is primarily intended to reduce model storage requirements and improve deployment efficiency.

---

# 📌 Quick Start

For an already configured environment:

```bash
# Activate environment
source ./glass/glass/bin/activate

# Voice control
python Function0_voice_control.py \
    --audio test/audio/mo_tinh_nang.mp3

# Visual QA
python Function1_chatbot.py \
    --image test/photo/stair_2.jpg \
    --audio test/audio/mieu_ta_khung_canh.mp3 \
    --out out.wav

# Autopilot
python Function2_autopilot.py \
    --image test/photo/japan.jpeg \
    --audio guide.wav
```
