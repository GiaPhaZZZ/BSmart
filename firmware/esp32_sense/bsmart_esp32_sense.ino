/**
 * =============================================================================
 * BSmart Smart Glasses Firmware for ESP32-S3 Sense (FW-01, FW-02, FW-03)
 * Core Platform: ESP32-S3 (Seeed Studio XIAO ESP32S3 Sense / ESP32-S3 CAM)
 *
 * Implements:
 *   - FW-01: BLE GATT Server, Connection management, Physical Button Handler
 *   - FW-02: OV2640/OV5640 Camera Capture & Zero-Allocation Chunked BLE Stream
 *   - FW-03: I2S/PDM Microphone 16kHz 16-bit Mono Recording & Safe BLE Stream
 *
 * Technical Contracts:
 *   - Compatible with mobile_app/src/services/ble/BlePlxService.ts
 *   - Service UUID:    0000180D-0000-1000-8000-00805F9B34FB
 *   - Button Char:     00002A37-0000-1000-8000-00805F9B34FB (Notify)
 *   - Image Char:      00002A38-0000-1000-8000-00805F9B34FB (Notify)
 *   - Audio In Char:   00002A39-0000-1000-8000-00805F9B34FB (Notify)
 *   - Audio Out Char:  00002A3A-0000-1000-8000-00805F9B34FB (Write)
 * =============================================================================
 */

#include "camera_pins.h"
#include "driver/i2s.h"
#include "esp_camera.h"
#include "mbedtls/base64.h"
#include <Arduino.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

// --- BLE UUIDs ---
#define SERVICE_UUID "0000180D-0000-1000-8000-00805F9B34FB"
#define BUTTON_CHAR_UUID "00002A37-0000-1000-8000-00805F9B34FB"
#define IMAGE_CHAR_UUID "00002A38-0000-1000-8000-00805F9B34FB"
#define AUDIO_IN_CHAR_UUID "00002A39-0000-1000-8000-00805F9B34FB"
#define AUDIO_OUT_CHAR_UUID "00002A3A-0000-1000-8000-00805F9B34FB"

#define BLE_DEVICE_NAME "BSmart_Glasses"

// --- Audio Configuration (FW-03) ---
#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
// 120 bytes PCM (60 samples @ 16-bit) encodes to exactly 160 chars in Base64
// (Multiple of 3 bytes) Perfectly fits inside standard BLE MTU packet (<180
// bytes) without packet dropping
#define I2S_READ_CHUNK_LEN 120

// --- Button Timing Constraints (FW-04) ---
#define SHORT_PRESS_MAX_MS 300
#define DEBOUNCE_DELAY_MS 30

// BLE Server & Characteristics pointers
BLEServer *pServer = nullptr;
BLECharacteristic *pButtonChar = nullptr;
BLECharacteristic *pImageChar = nullptr;
BLECharacteristic *pAudioInChar = nullptr;
BLECharacteristic *pAudioOutChar = nullptr;

bool deviceConnected = false;
bool oldDeviceConnected = false;

// Button state machine variables
bool isButtonPressed = false;
bool isRecordingAudio = false;
unsigned long buttonPressStartTime = 0;
unsigned long lastDebounceTime = 0;
int lastButtonState = HIGH;

// Auto 4s frame timer for navigation mode
unsigned long lastFrameCaptureTime = 0;
bool autoCaptureEnabled = false;

// Forward declaration
void captureAndSendImage();

// =============================================================================
// BLE Callbacks & Remote Command Processing
// =============================================================================
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    deviceConnected = true;
    Serial.println("[BLE] Mobile App connected!");
  }

  void onDisconnect(BLEServer *server) override {
    deviceConnected = false;
    autoCaptureEnabled = false;
    isRecordingAudio = false;
    Serial.println("[BLE] Mobile App disconnected. Resetting state and "
                   "restarting advertising...");
  }
};

class AudioOutCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pChar) override {
    std::string value = pChar->getValue();
    if (value.empty())
      return;

    // Support control commands sent via App
    if (value == "NAV_START") {
      autoCaptureEnabled = true;
      lastFrameCaptureTime = millis() - 4000; // Trigger immediate first frame
      Serial.println(
          "[App Command] Navigation mode started (Auto 4s capture ON)");
    } else if (value == "NAV_STOP") {
      autoCaptureEnabled = false;
      Serial.println(
          "[App Command] Navigation mode stopped (Auto capture OFF)");
    } else if (value == "CAPTURE") {
      Serial.println("[App Command] Single capture requested");
      captureAndSendImage();
    } else {
      // Streamed TTS audio payload
      Serial.printf("[BLE Audio Out] Received %u bytes audio data from App\n",
                    (unsigned int)value.length());
    }
  }
};

// =============================================================================
// Camera Initialization & Capture (FW-02)
// =============================================================================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // PSRAM check & buffer optimization
  if (psramFound()) {
    Serial.println("[Camera] PSRAM detected. Using QVGA high quality.");
    config.frame_size = FRAMESIZE_QVGA; // 320x240 for optimal speed (<25KB)
    config.jpeg_quality = 12;           // 0-63, lower is higher quality
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    Serial.println(
        "[Camera] No PSRAM detected. Using QVGA with 1 framebuffer in SRAM.");
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 14;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[Camera] Init failed with error 0x%x\n", err);
    return false;
  }

  Serial.println("[Camera] OV2640/OV5640 initialized successfully.");
  return true;
}

/**
 * Capture frame and stream chunked Base64 packets over BLE.
 * Optimized with ZERO large heap allocation:
 * Slices raw JPEG buffer directly into 120-byte chunks (multiple of 3),
 * encodes each chunk into 160 Base64 characters on stack,
 * and formats packet header: "seq:total:payload_base64".
 */
void captureAndSendImage() {
  if (!deviceConnected || !pImageChar)
    return;

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[Camera] Failed to capture frame buffer");
    return;
  }

  size_t rawLen = fb->len;
  Serial.printf("[Camera] Frame captured: %u bytes JPEG\n",
                (unsigned int)rawLen);

  // Raw chunk size must be multiple of 3 (120 bytes -> 160 chars Base64 with no
  // padding)
  const size_t rawChunkSize = 120;
  int totalChunks = (rawLen + rawChunkSize - 1) / rawChunkSize;

  Serial.printf("[Camera] Streaming %d chunks to BLE client...\n", totalChunks);

  // Stack buffer for Base64 encoded payload + header (no dynamic heap
  // fragmentation)
  char base64Buf[200];
  char packetBuf[240];

  for (int seq = 0; seq < totalChunks; seq++) {
    if (!deviceConnected)
      break;

    size_t offset = seq * rawChunkSize;
    size_t chunkRawBytes =
        (offset + rawChunkSize > rawLen) ? (rawLen - offset) : rawChunkSize;

    // Encode directly from raw JPEG frame buffer
    size_t encodedLen = 0;
    int ret =
        mbedtls_base64_encode((unsigned char *)base64Buf, sizeof(base64Buf),
                              &encodedLen, fb->buf + offset, chunkRawBytes);

    if (ret != 0) {
      Serial.printf("[Camera] Base64 encoding error on chunk %d\n", seq);
      continue;
    }
    base64Buf[encodedLen] = '\0';

    // Format protocol: "seq:total:payload_base64"
    int packetLen = snprintf(packetBuf, sizeof(packetBuf), "%d:%d:%s", seq,
                             totalChunks, base64Buf);

    pImageChar->setValue((uint8_t *)packetBuf, packetLen);
    pImageChar->notify();

    // 15ms interval between packets to ensure reliable GATT notify buffer
    // clearance
    delay(15);
  }

  esp_camera_fb_return(fb);
  Serial.println("[Camera] Image transmission complete.");
}

// =============================================================================
// I2S / PDM Microphone Initialization & Recording (FW-03)
// =============================================================================
bool initI2SMicrophone() {
#if USE_PDM_MIC
  // Configuration for Onboard PDM MEMS Microphone (Seeed XIAO ESP32S3 Sense)
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};

  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_PIN_NO_CHANGE,
      .ws_io_num = I2S_MIC_CLK_IO, // GPIO 42 (PDM CLK)
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_MIC_DATA_IO // GPIO 41 (PDM DATA)
  };
#else
  // Configuration for External I2S Microphone (INMP441)
  i2s_config_t i2s_config = {.mode =
                                 (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
                             .sample_rate = SAMPLE_RATE,
                             .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
                             .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
                             .communication_format = I2S_COMM_FORMAT_STAND_I2S,
                             .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
                             .dma_buf_count = 4,
                             .dma_buf_len = 256,
                             .use_apll = false,
                             .tx_desc_auto_clear = false,
                             .fixed_mclk = 0};

  i2s_pin_config_t pin_config = {.bck_io_num = I2S_MIC_SCK_IO,
                                 .ws_io_num = I2S_MIC_WS_IO,
                                 .data_out_num = I2S_PIN_NO_CHANGE,
                                 .data_in_num = I2S_MIC_DIN_IO};
#endif

  esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("[I2S] Driver install failed: 0x%x\n", err);
    return false;
  }

  err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("[I2S] Pin configuration failed: 0x%x\n", err);
    return false;
  }

  Serial.println(
      "[I2S] Microphone initialized successfully (16kHz, 16-bit Mono).");
  return true;
}

/**
 * Record audio chunk while user holds the button and stream over BLE.
 * Uses 120 bytes PCM -> 160 chars Base64 on stack (<180 bytes MTU packet).
 */
void streamAudioChunk() {
  if (!deviceConnected || !pAudioInChar)
    return;

  uint8_t rawPcm[I2S_READ_CHUNK_LEN];
  size_t bytesRead = 0;

  esp_err_t result = i2s_read(I2S_PORT, rawPcm, sizeof(rawPcm), &bytesRead,
                              20 / portTICK_PERIOD_MS);
  if (result == ESP_OK && bytesRead > 0) {
    char base64Buf[200];
    size_t encodedLen = 0;

    int ret =
        mbedtls_base64_encode((unsigned char *)base64Buf, sizeof(base64Buf),
                              &encodedLen, rawPcm, bytesRead);

    if (ret == 0) {
      pAudioInChar->setValue((uint8_t *)base64Buf, encodedLen);
      pAudioInChar->notify();
    }
  }
}

// =============================================================================
// Button & Event Management (FW-01 / FW-04)
// =============================================================================
void sendButtonEvent(const char *eventCode) {
  if (deviceConnected && pButtonChar) {
    pButtonChar->setValue((uint8_t *)eventCode, strlen(eventCode));
    pButtonChar->notify();
    Serial.printf("[Button] Sent event: %s\n", eventCode);
  }
}

void handleButton() {
  int reading = digitalRead(BUTTON_RECORD_PIN);

  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY_MS) {
    // Button pressed (Active LOW)
    if (reading == LOW && !isButtonPressed) {
      isButtonPressed = true;
      buttonPressStartTime = millis();
    }
    // Button held past threshold -> PTT HOLD (Start Recording)
    else if (reading == LOW && isButtonPressed && !isRecordingAudio) {
      if ((millis() - buttonPressStartTime) > SHORT_PRESS_MAX_MS) {
        isRecordingAudio = true;
        sendButtonEvent("01"); // BUTTON_HOLD
        Serial.println("[Button] HOLD detected -> Voice recording started");
      }
    }
    // Button released
    else if (reading == HIGH && isButtonPressed) {
      unsigned long pressDuration = millis() - buttonPressStartTime;
      isButtonPressed = false;

      if (isRecordingAudio) {
        isRecordingAudio = false;
        sendButtonEvent("00"); // BUTTON_RELEASE
        Serial.println("[Button] RELEASE detected -> Voice recording stopped");

        // Automatically capture & send scene photo for QA or Image Capture
        // features
        Serial.println("[Action] Capturing scene photo for App query...");
        captureAndSendImage();
      } else if (pressDuration <= SHORT_PRESS_MAX_MS) {
        sendButtonEvent(
            "02"); // BUTTON_SHORT_PRESS (Emergency cancel / Return Home)
        autoCaptureEnabled =
            false; // Cancel ongoing navigation loop on short press
        Serial.println("[Button] SHORT_PRESS detected -> Emergency Cancel / "
                       "Reset to IDLE");
      }
    }
  }

  lastButtonState = reading;
}

// =============================================================================
// Setup & Main Loop
// =============================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n==========================================");
  Serial.println("  BSmart Smart Glasses ESP32-S3 Firmware  ");
  Serial.println("==========================================");

  // 1. Initialize Button Pin
  pinMode(BUTTON_RECORD_PIN, INPUT_PULLUP);

  // 2. Initialize Camera
  initCamera();

  // 3. Initialize Microphone
  initI2SMicrophone();

  // 4. Initialize BLE GATT Server
  BLEDevice::init(BLE_DEVICE_NAME);
  // Request higher BLE MTU to maximize throughput for chunked payloads
  BLEDevice::setMTU(517);

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Button Characteristic (Notify)
  pButtonChar = pService->createCharacteristic(
      BUTTON_CHAR_UUID,
      BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ);
  pButtonChar->addDescriptor(new BLE2902());

  // Image Characteristic (Notify)
  pImageChar = pService->createCharacteristic(
      IMAGE_CHAR_UUID,
      BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ);
  pImageChar->addDescriptor(new BLE2902());

  // Audio In Characteristic (Notify - Mic stream)
  pAudioInChar = pService->createCharacteristic(
      AUDIO_IN_CHAR_UUID,
      BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ);
  pAudioInChar->addDescriptor(new BLE2902());

  // Audio Out Characteristic (Write - App TTS or Commands)
  pAudioOutChar = pService->createCharacteristic(
      AUDIO_OUT_CHAR_UUID,
      BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  pAudioOutChar->setCallbacks(new AudioOutCallbacks());

  pService->start();

  // Start Advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("[BLE] Advertising started. Waiting for BSmart Mobile App...");
}

void loop() {
  // Handle physical button events
  handleButton();

  // Stream microphone audio while user is holding button (Push-to-talk)
  if (isRecordingAudio) {
    streamAudioChunk();
  }

  // Periodic navigation capture (every 4s) when enabled
  if (autoCaptureEnabled && deviceConnected) {
    if (millis() - lastFrameCaptureTime >= 4000) {
      lastFrameCaptureTime = millis();
      captureAndSendImage();
    }
  }

  // Handle BLE disconnect & reconnect advertising
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    Serial.println("[BLE] Advertising restarted.");
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }

  delay(5);
}
