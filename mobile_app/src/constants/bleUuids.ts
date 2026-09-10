/**
 * BSmart Constants: BLE UUIDs
 * NOTE: These are placeholder UUIDs. Replace with firmware-provided UUIDs
 * once the ESP32 firmware specification is available.
 * See: docs/requirement.md §12
 */
export const BLE_UUIDS = {
  // These UUIDs are synchronized with firmware/esp32_sense/bsmart_esp32_sense.ino (lines 32-36).
  // DO NOT change without updating the firmware defines accordingly.
  SERVICE_UUID: '0000180D-0000-1000-8000-00805F9B34FB',
  BUTTON_CHARACTERISTIC_UUID: '00002A37-0000-1000-8000-00805F9B34FB',
  IMAGE_CHARACTERISTIC_UUID: '00002A38-0000-1000-8000-00805F9B34FB',
  AUDIO_IN_CHARACTERISTIC_UUID: '00002A39-0000-1000-8000-00805F9B34FB',
  AUDIO_OUT_CHARACTERISTIC_UUID: '00002A3A-0000-1000-8000-00805F9B34FB',
};

// BLE scan timeout (ms)
export const BLE_SCAN_TIMEOUT_MS = 10000;

// Device name prefix to identify BSmart glasses
export const BLE_DEVICE_NAME_PREFIX = 'BSmart';
