/**
 * BSmart BLE Service Factory
 * Toggle USE_MOCK_BLE to switch between mock and real hardware.
 */

import { IBleService } from '../../types';
import { MockBleService } from './MockBleService';
import { BlePlxService } from './BlePlxService';

// Set to false when real ESP32 hardware is available
export const USE_MOCK_BLE = true;

let _bleServiceInstance: IBleService | null = null;

export function getBleService(): IBleService {
  if (!_bleServiceInstance) {
    _bleServiceInstance = USE_MOCK_BLE ? new MockBleService() : new BlePlxService();
  }
  return _bleServiceInstance;
}

/**
 * Get typed mock service for simulation helpers (only when USE_MOCK_BLE is true)
 */
export function getMockBleService(): MockBleService | null {
  const svc = getBleService();
  return svc instanceof MockBleService ? svc : null;
}

export { BleAutoConnectService } from './BleAutoConnectService';
