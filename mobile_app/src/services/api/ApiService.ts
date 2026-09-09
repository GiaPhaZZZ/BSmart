/**
 * BSmart API Service
 * Handles cloud API calls: /transcribe and /qa
 * See: docs/requirement.md §2.3, SKILL.md §4.B
 */

import { API_ENDPOINTS, API_TIMEOUT_MS } from '../../constants/apiConfig';

export interface TranscribeResult {
  text: string;
}

export interface QAResult {
  answer: string;
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number = API_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Send audio to /transcribe endpoint
 * @param audioBase64 - base64 encoded WAV/PCM audio
 * @returns transcribed text
 */
export async function transcribeAudio(
  audioBase64: string,
): Promise<TranscribeResult> {
  const formData = new FormData();

  // Convert base64 to blob
  const audioBlob = {
    uri: `data:audio/wav;base64,${audioBase64}`,
    name: 'recording.wav',
    type: 'audio/wav',
  } as unknown as Blob;

  formData.append('audio', audioBlob);

  const response = await fetchWithTimeout(API_ENDPOINTS.TRANSCRIBE, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Transcribe failed: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  return { text: data.text ?? '' };
}

/**
 * Send image + question to /qa endpoint
 * @param imageBase64 - base64 encoded JPEG image
 * @param question - text question in Vietnamese
 * @returns answer text
 */
export async function askQA(
  imageBase64: string,
  question: string,
): Promise<QAResult> {
  const formData = new FormData();

  const imageBlob = {
    uri: `data:image/jpeg;base64,${imageBase64}`,
    name: 'image.jpg',
    type: 'image/jpeg',
  } as unknown as Blob;

  formData.append('image', imageBlob);
  formData.append('question', question);

  const response = await fetchWithTimeout(API_ENDPOINTS.QA, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`QA failed: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  return { answer: data.answer ?? '' };
}
