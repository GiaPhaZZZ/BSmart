/**
 * BSmart TTS Service
 * Converts Vietnamese text to speech and sends to glasses via BLE.
 * Uses Android Text-to-Speech (react-native's Tts module or Speech).
 *
 * For MVP: uses React Native's built-in Speech API (expo-speech or react-native-tts).
 * Since neither is installed, we implement a simple mock TTS that logs output.
 * Replace with react-native-tts when adding to package.json.
 *
 * See: docs/requirement.md §2.2 (TTS engine: Android Text-to-Speech)
 */

import { IBleService } from '../../types';

let currentSpeakRef: (() => void) | null = null;

/**
 * Speak text via device TTS and transmit audio to glasses.
 * In MVP: logs to console and uses a placeholder.
 * In production: generate TTS audio → base64 → sendAudio via BLE.
 */
export async function speakViaBle(
  text: string,
  bleService: IBleService,
): Promise<void> {
  // Cancel current speech if any
  stopSpeaking();

  console.log('[TTS] Speak:', text);

  // TODO: When react-native-tts is added:
  // 1. Tts.speak(text, { language: 'vi-VN' })
  // 2. Capture audio output
  // 3. bleService.sendAudio(base64AudioData)
  //
  // For MVP demo: simulate with a delay representing speech duration
  const wordCount = text.split(' ').length;
  const estimatedDurationMs = Math.max(1000, wordCount * 300);

  await new Promise<void>((resolve, reject) => {
    let cancelled = false;
    currentSpeakRef = () => {
      cancelled = true;
      reject(new Error('TTS cancelled'));
    };

    setTimeout(() => {
      if (!cancelled) {
        currentSpeakRef = null;
        resolve();
      }
    }, estimatedDurationMs);
  });

  // Simulate sending audio to glasses
  // In real implementation: bleService.sendAudio(ttsAudioBase64)
  await bleService.sendAudio('MOCK_TTS_AUDIO_PLACEHOLDER');
}

/**
 * Stop any currently playing TTS
 */
export function stopSpeaking(): void {
  if (currentSpeakRef) {
    currentSpeakRef();
    currentSpeakRef = null;
  }
}

/**
 * Speak text, stopping any existing speech first (for urgent warnings)
 */
export async function speakUrgent(
  text: string,
  bleService: IBleService,
): Promise<void> {
  stopSpeaking();
  return speakViaBle(text, bleService);
}
