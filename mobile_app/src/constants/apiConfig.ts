/**
 * BSmart Constants: API Configuration
 * 
 * Ứng dụng hiện tại đã chuyển sang kiến trúc 100% Standalone (Offline-first).
 * Tính năng Navigation (Feature 3) dùng trực tiếp Mapbox SDK và AI inference on-device.
 * 
 * Các tính năng ASR/VLM cũ (Feature 1/2) qua BLE vẫn có fallback gọi API backend nếu 
 * chưa có module ONNX native. Ở đây giữ lại API_BASE_URL để code không lỗi.
 */

export const API_BASE_URL = 'https://reprimand-ambiguous-founder.ngrok-free.dev';

// Optional route mode only. Leave blank for obstacle-awareness demos.
// Do not commit personal or production Mapbox tokens into source.
export const MAPBOX_ACCESS_TOKEN = '';

export const API_ENDPOINTS = {
  TRANSCRIBE: `${API_BASE_URL}/transcribe`,
  QA: `${API_BASE_URL}/qa`,
};

export const API_TIMEOUT_MS = 1500;
export const API_VQA_TIMEOUT_MS = 90000;

export type Feature1VlmMode = 'server' | 'on_device_onnx' | 'on_device_gguf';

// Default stays server-backed so the experimental Android GGUF path cannot
// change production Feature 1 behavior until it has real device evidence.
export const FEATURE1_VLM_MODE: Feature1VlmMode = 'on_device_gguf';

export const FEATURE1_GGUF_VLM = {
  modelPath: '',
  mmprojPath: '',
  contextSize: 2048,
  threads: 2,
  maxTokens: 48,
  temperature: 0,
};
