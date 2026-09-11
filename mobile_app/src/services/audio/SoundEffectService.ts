/**
 * BSmart Sound Effect Service
 * Plays pre-recorded Vietnamese feature activation audio clips (from activate_voice/).
 *
 * Mappings:
 *   - feature1 (Hỏi đáp / Chatbot) -> open_f1.mp3
 *   - feature2 (Chụp ảnh)         -> open_f4.mp3
 *   - feature3 (Dẫn đường/Auto)   -> open_f2.mp3
 *
 * Protocol:
 *   - If connected via BLE, sends audio trigger command to peripheral.
 *   - Concurrently plays sound on phone speaker via native AudioPlayerModule.
 *   - Falls back to TTS if native sound player is unavailable.
 */

import { NativeModules, Vibration } from 'react-native';
import { IBleService, BleConnectionState } from '../../types';
import { speakViaBle, stopSpeaking } from '../tts/TtsService';

const { AudioPlayerModule } = NativeModules;

export type FeatureKey = 'feature1' | 'feature2' | 'feature3';

export interface FeatureSoundConfig {
  soundName: string;
  fallbackText: string;
  bleCommand: string;
}

export const FEATURE_SOUND_MAP: Record<FeatureKey, FeatureSoundConfig> = {
  feature1: {
    soundName: 'open_f1',
    fallbackText: 'Đã vào tính năng 1, hỏi đáp, đã sẵn sàng',
    bleCommand: 'CMD:PLAY_F1',
  },
  feature2: {
    soundName: 'open_f2',
    fallbackText: 'Đã vào tính năng 2, chụp ảnh',
    bleCommand: 'CMD:PLAY_F2',
  },
  feature3: {
    soundName: 'open_f3',
    fallbackText: 'Đã vào tính năng 3, chế độ dẫn đường',
    bleCommand: 'CMD:PLAY_F3',
  },
};

/**
 * Play the activation sound for the designated feature.
 * Sends signal to BLE peripheral if connected, and plays via phone speaker.
 */
export async function playFeatureActivationSound(
  feature: FeatureKey,
  bleService?: IBleService,
): Promise<void> {
  // Stop any active TTS or sound effect first
  stopSpeaking();
  stopFeatureSound();

  const config = FEATURE_SOUND_MAP[feature];
  if (!config) return;

  // 1. If BLE peripheral is connected, send command to glasses
  if (bleService && bleService.getConnectionState() === BleConnectionState.CONNECTED) {
    try {
      await bleService.sendAudio(config.bleCommand);
    } catch (e) {
      console.warn('[SoundEffect] Error sending BLE audio trigger to glasses:', e);
    }
  }

  // 2. Play the audio clip on phone speaker
  try {
    if (AudioPlayerModule?.playSound) {
      await AudioPlayerModule.playSound(config.soundName);
      return;
    }
  } catch (e) {
    console.warn(`[SoundEffect] Native playSound failed for ${config.soundName}, fallback to TTS:`, e);
  }

  // 3. Fallback to TTS if native player is not available (e.g. Unit tests, emulator)
  if (bleService) {
    await speakViaBle(config.fallbackText, bleService);
  }
}

/**
 * Stop any ongoing feature activation audio clip immediately.
 */
export function stopFeatureSound(): void {
  try {
    if (AudioPlayerModule?.stopSound) {
      AudioPlayerModule.stopSound().catch(() => {});
    }
  } catch (e) {
    // Ignore error
  }
}

/**
 * Play walkie-talkie start beep and trigger haptic vibration when user holds button to speak.
 */
export function playPttStartFeedback(haptic = true): void {
  if (haptic) {
    try {
      Vibration.vibrate(50);
    } catch {
      // Ignore vibration error on unsupported platforms
    }
  }
  try {
    if (AudioPlayerModule?.playBeep) {
      AudioPlayerModule.playBeep('start').catch(() => {});
    }
  } catch {
    // Ignore
  }
}

/**
 * Play walkie-talkie release click and trigger haptic vibration when user releases button.
 */
export function playPttEndFeedback(haptic = true): void {
  if (haptic) {
    try {
      Vibration.vibrate(30);
    } catch {
      // Ignore
    }
  }
  try {
    if (AudioPlayerModule?.playBeep) {
      AudioPlayerModule.playBeep('stop').catch(() => {});
    }
  } catch {
    // Ignore
  }
}

/**
 * Play emergency cancel feedback (double haptic + cancel tone).
 */
export function playCancelFeedback(haptic = true): void {
  if (haptic) {
    try {
      Vibration.vibrate([0, 40, 50, 40]);
    } catch {
      // Ignore
    }
  }
  try {
    if (AudioPlayerModule?.playBeep) {
      AudioPlayerModule.playBeep('cancel').catch(() => {});
    }
  } catch {
    // Ignore
  }
}
