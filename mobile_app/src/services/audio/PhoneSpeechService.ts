/**
 * BSmart Phone Speech Service
 * Interfaces with native PhoneSpeechModule to enable Vietnamese voice command
 * recognition and QA questioning via the phone's built-in microphone.
 *
 * Used when:
 * - Glasses hardware is not yet connected (Standalone mode).
 * - Glasses hardware does not have a functional microphone.
 * - Testing voice pipeline directly on phone.
 */

import { NativeModules, PermissionsAndroid, Platform } from 'react-native';

const { PhoneSpeechModule } = NativeModules;

export interface PhoneSpeechResult {
  text: string;
}

/**
 * Request RECORD_AUDIO runtime permission on Android.
 */
export async function requestMicrophonePermission(): Promise<boolean> {
  if (Platform.OS !== 'android') return true;

  try {
    const hasPermission = await PermissionsAndroid.check(
      PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
    );
    if (hasPermission) return true;

    const status = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
      {
        title: 'Quyền sử dụng Micro',
        message:
          'BSmart cần quyền truy cập Micro để nhận diện giọng nói tiếng Việt khi bạn nhấn giữ màn hình.',
        buttonPositive: 'Cho phép',
        buttonNegative: 'Từ chối',
      },
    );

    return status === PermissionsAndroid.RESULTS.GRANTED;
  } catch (err) {
    console.warn('[PhoneSpeechService] Permission request error:', err);
    return false;
  }
}

/**
 * Check if Phone Speech Recognition is available on this device.
 */
export async function isPhoneSpeechAvailable(): Promise<boolean> {
  if (!PhoneSpeechModule?.isAvailable) return false;
  try {
    return await PhoneSpeechModule.isAvailable();
  } catch {
    return false;
  }
}

/**
 * Start listening via the phone's microphone.
 */
export async function startPhoneListening(): Promise<boolean> {
  const granted = await requestMicrophonePermission();
  if (!granted) {
    console.warn('[PhoneSpeechService] Microphone permission not granted');
    return false;
  }

  if (!PhoneSpeechModule?.startListening) {
    console.warn('[PhoneSpeechService] Native PhoneSpeechModule not linked');
    return false;
  }

  try {
    await PhoneSpeechModule.startListening();
    return true;
  } catch (err) {
    console.warn('[PhoneSpeechService] startListening error:', err);
    return false;
  }
}

/**
 * Stop listening and retrieve transcribed Vietnamese text.
 */
export async function stopPhoneListening(): Promise<PhoneSpeechResult> {
  if (!PhoneSpeechModule?.stopListening) {
    return { text: '' };
  }

  try {
    // Wait 400ms to ensure the trailing words are fully captured before stopping
    await new Promise<void>(resolve => setTimeout(resolve, 400));
    
    const result = await PhoneSpeechModule.stopListening();
    const text = typeof result?.text === 'string' ? result.text.trim() : '';
    return { text };
  } catch (err) {
    console.warn('[PhoneSpeechService] stopListening error:', err);
    return { text: '' };
  }
}

/**
 * Cancel current listening session.
 */
export async function cancelPhoneListening(): Promise<void> {
  if (!PhoneSpeechModule?.cancelListening) return;
  try {
    await PhoneSpeechModule.cancelListening();
  } catch (err) {
    console.warn('[PhoneSpeechService] cancelListening error:', err);
  }
}
