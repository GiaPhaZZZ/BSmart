#pragma once

// =============================================================================
// Camera and Microphone Pin Definitions for BSmart Glasses (ESP32-S3)
// =============================================================================

// Choose hardware profile:
// 1: CAMERA_MODEL_XIAO_ESP32S3 (Seeed Studio XIAO ESP32S3 Sense with onboard PDM mic)
// 2: CAMERA_MODEL_ESP32S3_CAM_INMP441 (ESP32-S3 CAM with external I2S INMP441 mic)
#define CAMERA_MODEL_ESP32S3_CAM_INMP441

#if defined(CAMERA_MODEL_XIAO_ESP32S3)
  // Camera DVP pins for Seeed Studio XIAO ESP32S3 Sense
  #define PWDN_GPIO_NUM     -1
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM     10
  #define SIOD_GPIO_NUM     40  // Camera SDA / SIOD
  #define SIOC_GPIO_NUM     39  // Camera SCL / SIOC

  #define Y9_GPIO_NUM       48
  #define Y8_GPIO_NUM       11
  #define Y7_GPIO_NUM       12
  #define Y6_GPIO_NUM       14
  #define Y5_GPIO_NUM       16
  #define Y4_GPIO_NUM       18
  #define Y3_GPIO_NUM       17
  #define Y2_GPIO_NUM       15
  #define VSYNC_GPIO_NUM    38
  #define HREF_GPIO_NUM     47
  #define PCLK_GPIO_NUM     13

  // Onboard PDM MEMS Microphone on XIAO ESP32S3 Sense expansion board
  #define USE_PDM_MIC       1
  #define I2S_MIC_CLK_IO    42  // PDM CLK (ws_io_num)
  #define I2S_MIC_DATA_IO   41  // PDM DATA (data_in_num)

  // External I2S DAC Speaker (MAX98357A) for XIAO ESP32S3
  #define I2S_SPK_BCLK_IO   8   // BCLK (D8)
  #define I2S_SPK_LRC_IO    9   // LRC / Word Select (D9)
  #define I2S_SPK_DOUT_IO   7   // DIN / Data Out (D7)

#elif defined(CAMERA_MODEL_ESP32S3_CAM_INMP441)
  // Standard ESP32-S3 CAM (e.g. Freenove / Ai-Thinker ESP32-S3 CAM)
  #define PWDN_GPIO_NUM     -1
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM     15
  #define SIOD_GPIO_NUM     4
  #define SIOC_GPIO_NUM     5

  #define Y9_GPIO_NUM       16
  #define Y8_GPIO_NUM       17
  #define Y7_GPIO_NUM       18
  #define Y6_GPIO_NUM       12
  #define Y5_GPIO_NUM       10
  #define Y4_GPIO_NUM       8
  #define Y3_GPIO_NUM       9
  #define Y2_GPIO_NUM       11
  #define VSYNC_GPIO_NUM    6
  #define HREF_GPIO_NUM     7
  #define PCLK_GPIO_NUM     13

  // External I2S INMP441 Microphone (Standard I2S)
  #define USE_PDM_MIC       0
  #define I2S_MIC_SCK_IO    1   // BCLK
  #define I2S_MIC_WS_IO     2   // LRC / WS
  #define I2S_MIC_DIN_IO    3   // SD / DIN

  // External I2S DAC Speaker (MAX98357A) for ESP32-S3 CAM
  #define I2S_SPK_BCLK_IO   47  // BCLK
  #define I2S_SPK_LRC_IO    48  // LRC / Word Select
  #define I2S_SPK_DOUT_IO   21  // DIN / Data Out
#endif

// Physical Button Pin (Action / Record button - active LOW with internal pull-up)
#define BUTTON_RECORD_PIN   0
