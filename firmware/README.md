# Firmware Directory (ESP32-S3)

Thư mục này chứa mã nguồn C/C++ nạp cho vi điều khiển **ESP32-S3** (ESP32-S3 Sense / ESP32-S3 CAM):

- `esp32_cam/`: Code nạp qua Arduino IDE hoặc ESP-IDF / PlatformIO.
  - Thu thập hình ảnh từ camera (OV2640 / OV5640).
  - Thu âm từ I2S microphone (INMP441).
  - Stream hình ảnh / âm thanh qua Wi-Fi (WebSocket / HTTP Server) hoặc BLE tới Mobile App.
