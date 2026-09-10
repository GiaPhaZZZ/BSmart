/**
 * BSmart Background Foreground Service Controller
 * Prevents Android Doze Mode and keeps BLE/AI alive when the screen is off.
 */

import { NativeModules, Platform } from 'react-native';

const { ForegroundServiceModule } = NativeModules;

export async function startBackgroundService(): Promise<boolean> {
  if (Platform.OS !== 'android' || !ForegroundServiceModule) {
    return false;
  }
  try {
    await ForegroundServiceModule.startService();
    return true;
  } catch (error) {
    console.warn('[ForegroundService] Failed to start background service:', error);
    return false;
  }
}

export async function stopBackgroundService(): Promise<boolean> {
  if (Platform.OS !== 'android' || !ForegroundServiceModule) {
    return false;
  }
  try {
    await ForegroundServiceModule.stopService();
    return true;
  } catch (error) {
    console.warn('[ForegroundService] Failed to stop background service:', error);
    return false;
  }
}
