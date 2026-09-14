/**
 * BSmart API Service
 * Handles cloud API calls: /transcribe and /qa
 * See: docs/requirement.md §2.3, SKILL.md §4.B
 */

import {
  API_ENDPOINTS,
  API_TIMEOUT_MS,
  API_VQA_TIMEOUT_MS,
} from '../../constants/apiConfig';

export interface TranscribeResult {
  text: string;
}

export interface QAResult {
  answer: string;
  question?: string;
  question_en?: string;
  answer_en?: string;
  model?: string;
  timings?: {
    model_load_s?: number;
    asr_model_load_s?: number;
    translation_model_load_s?: number;
    piper_model_load_s?: number;
    audio_preprocessing_s?: number;
    stt_s?: number;
    vi_to_en_s?: number;
    image_preparation_s?: number;
    vlm_s?: number;
    en_to_vi_s?: number;
    tts_s?: number;
    other_network_s?: number;
    total_s?: number;
  };
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

  console.log('[Feature1][ServerVQA] POST /qa request', {
    endpoint: API_ENDPOINTS.QA,
    imageBytesApprox: Math.round((imageBase64.length * 3) / 4),
    question,
    timeoutMs: API_VQA_TIMEOUT_MS,
  });

  const requestStartedAt = Date.now();
  const response = await fetchWithTimeout(
    API_ENDPOINTS.QA,
    {
      method: 'POST',
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
      body: formData,
    },
    API_VQA_TIMEOUT_MS,
  );
  const clientElapsedMs = Date.now() - requestStartedAt;

  console.log('[Feature1][ServerVQA] HTTP response', {
    status: response.status,
    ok: response.ok,
    elapsedMs: clientElapsedMs,
  });

  if (!response.ok) {
    let details = '';
    try {
      details = JSON.stringify(await response.json());
    } catch {
      details = await response.text().catch(() => '');
    }
    throw new Error(`QA failed: ${response.status} ${response.statusText} ${details}`.trim());
  }

  const data = await response.json();
  if (!data?.answer) {
    throw new Error('QA failed: response missing answer');
  }

  if (data.timings) {
    const backendTotalMs = Number(data.timings.total_s) * 1000;
    const networkApiOverheadMs = Number.isFinite(backendTotalMs)
      ? Math.max(0, clientElapsedMs - backendTotalMs)
      : clientElapsedMs;
    console.log('[Feature1][ServerVQA] Latency', {
      clientTotalMs: clientElapsedMs,
      backendTimings: data.timings,
      networkApiOverheadMs,
    });
  }

  console.log('[Feature1][ServerVQA] Parsed response', {
    answer: data.answer,
    question_en: data.question_en,
    answer_en: data.answer_en,
    model: data.model,
  });

  return {
    answer: data.answer,
    question: data.question,
    question_en: data.question_en,
    answer_en: data.answer_en,
    model: data.model,
    timings: data.timings,
  };
}
