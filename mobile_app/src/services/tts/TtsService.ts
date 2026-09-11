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

import { NativeModules } from 'react-native';
import Tts from 'react-native-tts';
import { IBleService, BleConnectionState } from '../../types';

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
 * Speak text via Android TTS and transmit audio data to glasses over BLE.
 * Dual-output:
 *   1. Transmits audio data to ESP32 glasses (AUDIO_OUT GATT characteristic).
 *   2. Concurrently plays on phone speaker via Android TTS.
 *
 * @param text      Vietnamese text to speak
 * @param bleService  BLE service reference for streaming data to glasses
 */
export async function speakViaBle(
  text: string,
  bleService?: IBleService,
): Promise<void> {
  stopSpeaking();

  console.log('[TTS] Speak:', text);

  // 1. Transmit synthesized PCM audio data (Direction B) to glasses via BLE
  if (bleService && bleService.getConnectionState() === BleConnectionState.CONNECTED) {
    (async () => {
      try {
        const nativeModule = NativeModules?.AudioPlayerModule;
        if (nativeModule && typeof nativeModule.synthesizeSpeechToPcm === 'function') {
          const pcmBase64 = await nativeModule.synthesizeSpeechToPcm(text);
          if (pcmBase64 && pcmBase64.length > 0) {
            await bleService.sendAudio(pcmBase64);
            return;
          }
        }
      } catch (synthErr) {
        console.warn('[TTS] Synthesis to PCM warning, falling back to text payload:', synthErr);
      }
      // Fallback: send text command if PCM synthesis unavailable
      await bleService.sendAudio(text);
    })().catch(err => {
      console.warn('[TTS] Failed to stream audio to glasses via BLE:', err);
    });
  }

  try {
    await ensureTtsReady();
  } catch (e) {
    console.error('[TTS] Init failed:', e);
    // Fall through: still attempt to speak, Android may recover
  }

  await new Promise<void>((resolve) => {
    let settled = false;

    // Safety timeout: guarantees the state machine NEVER hangs waiting for TTS
    const safetyTimeout = setTimeout(() => {
      if (!settled) {
        settled = true;
        cleanup();
        resolve();
      }
    }, Math.max(3500, text.length * 150));

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
      console.warn('[TTS] Error during utterance:', event.utteranceId);
      resolve();
    };

    const onCancel = (event: { utteranceId: string | number }) => {
      if (event.utteranceId !== currentUtteranceId) return;
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };

    let subFinish: { remove?: () => void } | null = null;
    let subError: { remove?: () => void } | null = null;
    let subCancel: { remove?: () => void } | null = null;

    function cleanup() {
      clearTimeout(safetyTimeout);
      try {
        subFinish?.remove?.();
      } catch (e) {
        // ignore
      }
      try {
        subError?.remove?.();
      } catch (e) {
        // ignore
      }
      try {
        subCancel?.remove?.();
      } catch (e) {
        // ignore
      }
      subFinish = null;
      subError = null;
      subCancel = null;
      currentCancelRef = null;
      currentUtteranceId = null;
    }

    try {
      subFinish = Tts.addEventListener('tts-finish', onFinish) as any;
      subError = Tts.addEventListener('tts-error', onError) as any;
      subCancel = Tts.addEventListener('tts-cancel', onCancel) as any;
    } catch (e) {
      console.warn('[TTS] addEventListener error:', e);
    }

    try {
      currentUtteranceId = Tts.speak(text);
    } catch (speakErr) {
      console.warn('[TTS] Tts.speak error:', speakErr);
      cleanup();
      resolve();
      return;
    }

    // Store cancel hook for stopSpeaking()
    currentCancelRef = () => {
      if (!settled) {
        settled = true;
        cleanup();
        Tts.stop().catch(() => {});
        resolve();
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
