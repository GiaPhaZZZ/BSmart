/**
 * BSmart TTS Service
 * Converts Vietnamese text to speech using Android Text-to-Speech
 * via react-native-tts.
 *
 * MVP note: Audio plays on the phone speaker (device TTS output).
 * Full BLE audio piping (phone → glasses speaker) requires firmware
 * audio chunking protocol (currently BLOCKED — see docs/requirement.md §12).
 *
 * Interface is preserved so this can be mocked in tests without changes
 * to the state machine.
 *
 * See: docs/requirement.md §2.2
 */

import Tts from 'react-native-tts';
import { IBleService } from '../../types';

const TTS_LANGUAGE = 'vi-VN';
const TTS_RATE = 0.5; // Slightly slower than default for clarity

let ttsInitialized = false;
let initPromise: Promise<void> | null = null;
let currentUtteranceId: string | number | null = null;
let currentCancelRef: (() => void) | null = null;

/**
 * Initialize TTS engine once. Safe to call multiple times.
 */
async function ensureTtsReady(): Promise<void> {
  if (ttsInitialized) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      await Tts.getInitStatus();
    } catch (e: any) {
      // Engine not installed — request install and wait
      if (e?.code === 'no_engine') {
        await Tts.requestInstallEngine();
        await Tts.getInitStatus();
      } else {
        throw e;
      }
    }

    await Tts.setDefaultLanguage(TTS_LANGUAGE);
    await Tts.setDefaultRate(TTS_RATE);
    ttsInitialized = true;
    console.log('[TTS] Initialized, language:', TTS_LANGUAGE);
  })();

  return initPromise;
}

/**
 * Speak text via Android TTS.
 * Plays on phone speaker (MVP).
 * BLE audio forwarding to glasses is BLOCKED pending firmware audio protocol.
 *
 * @param text      Vietnamese text to speak
 * @param bleService  BLE service reference (reserved for future BLE audio pipe)
 */
export async function speakViaBle(
  text: string,
  _bleService: IBleService,
): Promise<void> {
  stopSpeaking();

  console.log('[TTS] Speak:', text);

  try {
    await ensureTtsReady();
  } catch (e) {
    console.error('[TTS] Init failed:', e);
    // Fall through: still attempt to speak, Android may recover
  }

  await new Promise<void>((resolve, reject) => {
    let settled = false;

    const onFinish = (event: { utteranceId: string | number }) => {
      if (event.utteranceId !== currentUtteranceId) return;
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };

    const onError = (event: { utteranceId: string | number }) => {
      if (event.utteranceId !== currentUtteranceId) return;
      if (settled) return;
      settled = true;
      cleanup();
      // Resolve (not reject) on TTS error to avoid crashing state machine
      console.warn('[TTS] Error during utterance:', event.utteranceId);
      resolve();
    };

    const onCancel = (event: { utteranceId: string | number }) => {
      if (event.utteranceId !== currentUtteranceId) return;
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error('TTS cancelled'));
    };

    function cleanup() {
      Tts.removeEventListener('tts-finish', onFinish);
      Tts.removeEventListener('tts-error', onError);
      Tts.removeEventListener('tts-cancel', onCancel);
      currentCancelRef = null;
      currentUtteranceId = null;
    }

    Tts.addEventListener('tts-finish', onFinish);
    Tts.addEventListener('tts-error', onError);
    Tts.addEventListener('tts-cancel', onCancel);

    currentUtteranceId = Tts.speak(text);

    // Store cancel hook for stopSpeaking()
    currentCancelRef = () => {
      if (!settled) {
        settled = true;
        cleanup();
        Tts.stop();
        reject(new Error('TTS cancelled'));
      }
    };
  });

  // TODO (BLOCKED): When firmware audio protocol is defined,
  // replace above TTS speak with:
  //   1. Capture TTS output as PCM/WAV
  //   2. Encode to base64
  //   3. bleService.sendAudio(base64) using BLE chunking protocol
  // Until then, audio plays on the phone speaker.
}

/**
 * Stop currently playing TTS immediately.
 */
export function stopSpeaking(): void {
  if (currentCancelRef) {
    currentCancelRef();
    currentCancelRef = null;
  } else {
    // Best-effort stop even if no active ref
    Tts.stop().catch(() => {});
  }
}

/**
 * Speak text with highest priority — cancels existing speech first.
 */
export async function speakUrgent(
  text: string,
  bleService: IBleService,
): Promise<void> {
  stopSpeaking();
  return speakViaBle(text, bleService);
}
