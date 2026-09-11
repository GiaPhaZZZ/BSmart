/**
 * BSmart BLE Permission Service
 * Requests necessary Android runtime permissions for BLE scanning and connecting.
 */

import { PermissionsAndroid, Platform } from 'react-native';

export async function requestBlePermissions(): Promise<boolean> {
  if (Platform.OS !== 'android') {
    return true;
  }

  const apiLevel = Platform.Version;

  try {
    if (typeof apiLevel === 'number' && apiLevel >= 31) {
      // Android 12+ (API 31+) permissions
      const results = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      ]);

      const scanGranted =
        results[PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN] ===
        PermissionsAndroid.RESULTS.GRANTED;
      const connectGranted =
        results[PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT] ===
        PermissionsAndroid.RESULTS.GRANTED;

      return scanGranted && connectGranted;
    } else {
      // Android 6 to 11
      const locationGranted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        {
          title: 'Quyền vị trí để quét Bluetooth',
          message:
            'BSmart cần quyền vị trí để phát hiện và kết nối với Kính AI qua Bluetooth Low Energy.',
          buttonPositive: 'Đồng ý',
          buttonNegative: 'Từ chối',
        },
      );

      return locationGranted === PermissionsAndroid.RESULTS.GRANTED;
    }
  } catch (error) {
    console.warn('[BlePermissionService] Error requesting BLE permissions:', error);
    return false;
  }
}

/**
 * Request all essential runtime permissions (BLE, Audio, Camera) on app launch
 * so that blind users don't encounter blocking permission dialogs while holding the touch area.
 */
export async function requestAllAppPermissions(): Promise<boolean> {
  if (Platform.OS !== 'android') return true;

  try {
    const apiLevel = Platform.Version;
    const permissionsToRequest: string[] = [
      PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
      PermissionsAndroid.PERMISSIONS.CAMERA,
    ];

    if (typeof apiLevel === 'number' && apiLevel >= 31) {
      permissionsToRequest.push(
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      );
    } else {
      permissionsToRequest.push(PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION);
    }

    const results = await PermissionsAndroid.requestMultiple(permissionsToRequest as any);
    return (
      results[PermissionsAndroid.PERMISSIONS.RECORD_AUDIO] ===
      PermissionsAndroid.RESULTS.GRANTED
    );
  } catch (err) {
    console.warn('[BlePermissionService] Error requesting all app permissions:', err);
    return false;
  }
}
