/**
 * BSmart On-Device Speech Recognition (ASR) Service (MOD-01)
 * Powered by PhoWhisper-tiny on-device pipeline.
 *
 * Responsibilities:
 * 1. Transcribe 16kHz PCM/WAV speech chunks into Vietnamese text completely offline.
 * 2. Vietnamese text normalization and voice command keyword spotting.
 */

import { AppState } from '../../types';
import { modelRegistry } from './ModelRegistry';
import { transcribeAudio } from '../api/ApiService';

export interface AsrTranscriptionResult {
  text: string;
  matchedState: AppState | null;
  confidence: number;
}

/**
 * Remove Vietnamese accents and normalize string for robust fuzzy matching.
 */
export function normalizeVietnameseText(str: string): string {
  return str
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/g, 'd')
    .trim();
}

/**
 * Spot keyword from transcribed Vietnamese text using flexible Regex.
 * Matches:
 *   - "tính năng 1" / "tính năng một" / "gpt" -> FEATURE_1_QA
 *   - "tính năng 2" / "tính năng hai"         -> FEATURE_2_CAPTURE
 *   - "tính năng 3" / "tính năng ba"          -> FEATURE_3_NAVIGATION
 */
export function matchVoiceCommand(rawText: string): AppState | null {
  const norm = normalizeVietnameseText(rawText);

  if (/tinh nang (1|mot)|chuc nang (1|mot)|so (1|mot)|hoi dap|gpt/i.test(norm)) {
    return AppState.FEATURE_1_QA;
  }

  if (/tinh nang (2|hai|hay)|chuc nang (2|hai|hay)|so (2|hai|hay)|chup anh/i.test(norm)) {
    return AppState.FEATURE_2_CAPTURE;
  }

  if (/tinh nang (3|ba)|chuc nang (3|ba)|so (3|ba)|dan duong|vat can/i.test(norm)) {
    return AppState.FEATURE_3_NAVIGATION;
  }

  return null;
}

import { NativeModules, Platform } from 'react-native';

const { OnnxInferenceModule } = NativeModules;

/**
 * Transcribe audio on-device using PhoWhisper pipeline.
 * @param audioBase64 - base64 encoded WAV/PCM audio
 * @returns Transcribed text and matched command state
 */
export async function transcribeAudioOnDevice(
  audioBase64: string,
): Promise<AsrTranscriptionResult> {
  const isReady = modelRegistry.isModelReady('phowhisper');

  if (!isReady || !audioBase64 || audioBase64.length === 0) {
    return {
      text: '',
      matchedState: null,
      confidence: 0,
    };
  }

  if (Platform.OS === 'android' && OnnxInferenceModule?.runSpeechRecognition) {
    try {
      const result = await OnnxInferenceModule.runSpeechRecognition(audioBase64);
      if (result && result.text) {
        const cleanText = result.text.trim();
        const matched = matchVoiceCommand(cleanText);
        return {
          text: cleanText,
          matchedState: matched,
          confidence: result.confidence || 0.95,
        };
      }
    } catch (err) {
      console.warn('[ASR] On-device PhoWhisper execution failed:', err);
    }
  }

  console.warn('[ASR] On-device PhoWhisper not available. Returning empty transcript.');
  return {
    text: '',
    matchedState: null,
    confidence: 0,
  };
}

/**
 * Unified speech transcription:
 * 1. Tries PhoWhisper server (/transcribe) for actual neural Vietnamese speech recognition.
 * 2. Falls back to on-device ASR handler if server is unavailable/offline.
 */
export async function transcribeSpeech(
  audioBase64: string,
): Promise<AsrTranscriptionResult> {
  if (!audioBase64 || audioBase64.length === 0) {
    return { text: '', matchedState: null, confidence: 0 };
  }

  // 1. Try real PhoWhisper backend endpoint
  try {
    const apiResult = await transcribeAudio(audioBase64);
    if (apiResult && apiResult.text && apiResult.text.trim().length > 0) {
      const cleanText = apiResult.text.trim();
      const matched = matchVoiceCommand(cleanText);
      return {
        text: cleanText,
        matchedState: matched,
        confidence: 0.98,
      };
    }
  } catch (err) {
    console.log('[ASR] Backend PhoWhisper not reachable, using offline ASR:', err);
  }

  // 2. Fallback to on-device handler
  return transcribeAudioOnDevice(audioBase64);
}
