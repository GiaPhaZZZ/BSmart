/**
 * BSmart Constants: API Configuration
 * Set API_BASE_URL via environment or update this file to point to your server.
 */

// Self-hosted AI backend — change IP/port to match your deployment
export const API_BASE_URL = 'http://192.168.1.100:8000';

export const API_ENDPOINTS = {
  TRANSCRIBE: `${API_BASE_URL}/transcribe`,
  QA: `${API_BASE_URL}/qa`,
};

// Timeout in milliseconds for API calls
export const API_TIMEOUT_MS = 30000;
