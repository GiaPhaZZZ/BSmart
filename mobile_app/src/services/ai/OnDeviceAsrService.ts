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
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

const VOICE_COMMAND_ALIASES: Record<AppState, string[]> = {
  [AppState.IDLE]: [],
  [AppState.LISTENING]: [],
  [AppState.PROCESSING]: [],
  [AppState.FEATURE_1_QA]: [
    '1',
    'so 1',
    'mot',
    'so mot',
    'tinh nang 1',
    'tinh nang mot',
    'tinh nang so mot',
    'tinh nang so 1',
    'chuc nang 1',
    'chuc nang mot',
    'chuc nang so 1',
    'chuc nang so mot',
    'hoi dap',
    'gpt',
  ],
  [AppState.FEATURE_2_CAPTURE]: [
    '2',
    'hai',
    'so 2',
    'so hai',
    'chup anh',
    'chup',
    'tinh nang 2',
    'tinh nang so 2',
    'tinh nang hai',
    'tinh nang so hai',
    'tinh nang hay',
    'tinh nang so hay',
    'chuc nang 2',
    'chuc nang hai',
    'chuc nang hay',
    'chuc nang so 2',
    'chuc nang so hai',
    'chuc nang so hay',
    'so hay',
  ],
  [AppState.FEATURE_3_NAVIGATION]: [
    '3',
    'ba',
    'so 3',
    'so ba',
    'tinh nang 3',
    'tinh nang so 3',
    'tinh nang ba',
    'tinh nang so ba',
    'chuc nang 3',
    'chuc nang ba',
    'chuc nang so 3',
    'chuc nang so ba',
    'tim duong',
    'dan duong',
    'vat can',
    'canh bao',
  ],
};

function containsAlias(normText: string, alias: string): boolean {
  const escapedAlias = alias.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(^|\\s)${escapedAlias}(?=\\s|$)`).test(normText);
}

/**
 * Spot keyword from transcribed Vietnamese text using accent-insensitive aliases.
 * Matches:
 *   - "tính năng 1" / "tính năng một" / "gpt" -> FEATURE_1_QA
 *   - "tính năng 2" / "tính năng hai"         -> FEATURE_2_CAPTURE
 *   - "tính năng 3" / "tính năng ba"          -> FEATURE_3_NAVIGATION
 */
export function matchVoiceCommand(rawText: string): AppState | null {
  const norm = normalizeVietnameseText(rawText);

  return [
    AppState.FEATURE_1_QA,
    AppState.FEATURE_2_CAPTURE,
    AppState.FEATURE_3_NAVIGATION,
  ].find(state =>
    VOICE_COMMAND_ALIASES[state].some(alias => containsAlias(norm, alias)),
  ) ?? null;
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
 * Unified speech transcription — on-device PRIMARY (no cloud fallback).
 * Architecture: On-device PhoWhisper ONNX inference is the only production path.
 * Returns structured failure if model unavailable or inference fails.
 */
export async function transcribeSpeech(
  audioBase64: string,
): Promise<AsrTranscriptionResult> {
  if (!audioBase64 || audioBase64.length === 0) {
    return { text: '', matchedState: null, confidence: 0 };
  }
  return transcribeAudioOnDevice(audioBase64);
}
