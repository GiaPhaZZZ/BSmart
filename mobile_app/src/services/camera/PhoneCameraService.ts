/**
 * BSmart Phone Camera Service
 * Interfaces with native PhoneCameraModule to capture photos via phone's rear camera
 * when smart glasses are disconnected or camera is absent.
 */

import { NativeModules, PermissionsAndroid, Platform } from 'react-native';

const { PhoneCameraModule } = NativeModules;

/**
 * Request CAMERA runtime permission on Android.
 */
export async function requestCameraPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') return true;

  try {
    const hasPermission = await PermissionsAndroid.check(
      PermissionsAndroid.PERMISSIONS.CAMERA,
    );
    if (hasPermission) return true;

    const status = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.CAMERA,
      {
        title: 'Quyền sử dụng Camera',
        message:
          'BSmart cần quyền truy cập Camera để quan sát không gian phía trước khi không kết nối với kính thông minh.',
        buttonPositive: 'Cho phép',
        buttonNegative: 'Từ chối',
      },
    );

    return status === PermissionsAndroid.RESULTS.GRANTED;
  } catch (err) {
    console.warn('[PhoneCameraService] Permission request error:', err);
    return false;
  }
}

/**
 * Check if Phone Camera is available.
 */
export async function isPhoneCameraAvailable(): Promise<boolean> {
  if (!PhoneCameraModule?.isCameraAvailable) return false;
  try {
    return await PhoneCameraModule.isCameraAvailable();
  } catch {
    return false;
  }
}

/**
 * Capture a single photo frame from the phone's rear camera.
 * Returns Base64 encoded JPEG string, or null on failure.
 */
export async function capturePhonePhoto(): Promise<string | null> {
  const granted = await requestCameraPermission();
  if (!granted) {
    console.warn('[PhoneCameraService] Camera permission denied');
    return null;
  }

  if (!PhoneCameraModule?.capturePhoto) {
    console.warn('[PhoneCameraService] Native PhoneCameraModule not linked');
    return null;
  }

  try {
    const base64 = await PhoneCameraModule.capturePhoto();
    if (typeof base64 === 'string' && base64.length > 0) {
      return base64;
    }
    return null;
  } catch (err) {
    console.warn('[PhoneCameraService] capturePhoto error:', err);
    return null;
  }
}
