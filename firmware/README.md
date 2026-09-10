# Firmware BSmart (ESP32-S3 Sense / ESP32-S3 CAM)

Thư mục này chứa mã nguồn C/C++ nạp cho vi điều khiển **ESP32-S3** trang bị trên kính thông minh BSmart:

## Cấu trúc thư mục

* [`esp32_sense/bsmart_esp32_sense.ino`](esp32_sense/bsmart_esp32_sense.ino): Mã nguồn chính tích hợp:
  * **FW-01:** BLE GATT Server (`BSmart_Glasses`), quản lý kết nối và ngắt nút bấm (Hold, Release, Short-press).
  * **FW-02:** Điều khiển camera OV2640/OV5640, chụp JPEG (320x240 QVGA), mã hóa Base64 và phân mảnh gói tin BLE (`seq:total:payload`).
  * **FW-03:** Driver I2S Microphone (INMP441) thu âm 16kHz 16-bit Mono khi nhấn giữ nút Push-to-talk.
* [`esp32_sense/camera_pins.h`](esp32_sense/camera_pins.h): Định nghĩa sơ đồ chân GPIO cho Seeed XIAO ESP32S3 Sense và ESP32-S3 CAM/EYE.

## Hướng dẫn nạp vi điều khiển qua Arduino IDE

1. **Cài đặt Board ESP32:**
   * Trong Arduino IDE: `File` -> `Preferences` -> thêm URL: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   * `Tools` -> `Board` -> `Boards Manager...` -> Tìm và cài đặt `esp32` (phiên bản `>= 2.0.14`).
2. **Chọn Board & Cấu hình:**
   * **Board:** `XIAO_ESP32S3` hoặc `ESP32S3 Dev Module`.
   * **PSRAM:** `OPI PSRAM` (Bắt buộc bật để hỗ trợ Camera Framebuffer).
   * **Flash Size:** `8MB` (hoặc `16MB`).
   * **Upload Speed:** `921600`.
3. **Cài đặt Thư viện:**
   * Đảm bảo đã có thư viện `esp32-camera` (mặc định đi kèm trong ESP32 Arduino core).
4. **Nạp code:**
   * Mở file [`esp32_sense/bsmart_esp32_sense.ino`](esp32_sense/bsmart_esp32_sense.ino) và nhấn **Upload**.
