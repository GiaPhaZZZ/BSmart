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

export const API_ENDPOINTS = {
  TRANSCRIBE: `${API_BASE_URL}/transcribe`,
  QA: `${API_BASE_URL}/qa`,
};

export const API_TIMEOUT_MS = 1500;
export const API_VQA_TIMEOUT_MS = 90000;
