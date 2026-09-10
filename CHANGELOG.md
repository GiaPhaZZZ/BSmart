# CHANGELOG

All notable changes to BSmart will be documented in this file.

---

## [0.0.1] — 2026-09-10 (Early Preview)

Bản phát hành sớm nội bộ, phục vụ mục đích demo và kiểm thử.

### Firmware (ESP32-S3 Sense)
- Mã nguồn C++ hoàn chỉnh cho kính BSmart (`firmware/esp32_sense/`)
- BLE GATT Server: truyền ảnh JPEG, audio PCM, sự kiện nút bấm
- Camera OV2640: chụp ảnh JPEG 320×240 QVGA theo yêu cầu và tự động 4s (Navigation)
- Mic I2S (INMP441): ghi âm 16kHz mono 16-bit, stream Base64 qua BLE
- Nút bấm: phân biệt Hold (PTT), Release, Short-press (hủy khẩn cấp)
- Nhận và thực thi lệnh điều khiển từ App: `NAV_START`, `NAV_STOP`, `CAPTURE`, `CMD:PLAY_F*`

### Android App (React Native)
- State machine 6 trạng thái: `IDLE`, `LISTENING`, `PROCESSING`, `FEATURE_1_QA`, `FEATURE_2_CAPTURE`, `FEATURE_3_NAVIGATION`
- BLE auto-connect với `react-native-ble-plx`, hỗ trợ MockBleService cho emulator
- AI on-device đầy đủ (offline 100%): PhoWhisper-tiny, SmolVLM2-256M, YOLO26s, ZipDepth
- Blind-First UX: `BlindTouchArea`, TalkBack, âm bíp ToneGenerator, rung haptic
- Android Foreground Service với `PARTIAL_WAKE_LOCK` để duy trì kết nối khi tắt màn hình
- Tự động lưu ảnh chụp (Tính năng 2) vào bộ nhớ điện thoại

### DevOps
- CI/CD GitHub Actions: tự động build và publish APK + firmware khi push tag `v*`

### Lưu ý
- **Bản early preview** — chưa qua kiểm thử tích hợp phần cứng đầy đủ (HW-01 ⏳)
- BLE Audio Pipe (kính phát âm thanh) chưa hoàn thiện (FW-06 ⚠️); app fallback phát qua loa điện thoại
- APK là bản **debug unsigned** — cần bật "Cài từ nguồn không xác định" trên Android để cài

### Hướng dẫn cài đặt nhanh
1. **Firmware:** Tải `bsmart_firmware_v0.0.1.bin` → nạp vào ESP32-S3 bằng `esptool` hoặc PlatformIO
2. **App Android:** Tải `app-debug.apk` → bật Unknown Sources → cài đặt bình thường
