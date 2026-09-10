/**
 * BSmart BLE Service Factory
 * Toggle USE_MOCK_BLE to switch between mock and real hardware.
 */

import { IBleService } from '../../types';
import { MockBleService } from './MockBleService';
import { BlePlxService } from './BlePlxService';

// Default to Real BLE (false) so that the Bluetooth button actually opens BLE connection.
// Can be toggled at runtime in settings or for emulator testing.
let _useMockBle = false;

export const USE_MOCK_BLE = false;

let _bleServiceInstance: IBleService | null = null;

export function isUsingMockBle(): boolean {
  return _useMockBle;
}

export function setUseMockBle(enabled: boolean): IBleService {
  if (_useMockBle !== enabled || !_bleServiceInstance) {
    _useMockBle = enabled;
    _bleServiceInstance = enabled ? new MockBleService() : new BlePlxService();
  }
  return _bleServiceInstance;
}

export function getBleService(): IBleService {
  if (!_bleServiceInstance) {
    _bleServiceInstance = _useMockBle ? new MockBleService() : new BlePlxService();
  }
  return _bleServiceInstance;
}

/**
 * Get typed mock service for simulation helpers (only when Mock BLE is active)
 */
export function getMockBleService(): MockBleService | null {
  const svc = getBleService();
  return svc instanceof MockBleService ? svc : null;
}

export { BleAutoConnectService } from './BleAutoConnectService';
