import { BlePlxService } from '../src/services/ble/BlePlxService';
import { BleConnectionState, BleEvent } from '../src/types';

// Mock react-native-ble-plx
jest.mock('react-native-ble-plx', () => {
  return {
    BleManager: jest.fn().mockImplementation(() => {
      return {
        startDeviceScan: jest.fn(),
        stopDeviceScan: jest.fn(),
      };
    }),
  };
});

describe('BlePlxService Unit Tests', () => {
  let service: BlePlxService;

  beforeEach(() => {
    service = new BlePlxService();
  });

  test('initial connection state should be DISCONNECTED', () => {
    expect(service.getConnectionState()).toBe(BleConnectionState.DISCONNECTED);
  });

  test('button event listener registers and unsubscribes cleanly', () => {
    const callback = jest.fn();
    const unsubscribe = service.onButtonEvent(callback);
    expect(typeof unsubscribe).toBe('function');
    unsubscribe();
  });

  test('image received listener registers cleanly', () => {
    const callback = jest.fn();
    const unsubscribe = service.onImageReceived(callback);
    expect(typeof unsubscribe).toBe('function');
    unsubscribe();
  });

  test('chunked image packet reassembly parses sequences correctly', () => {
    const receivedImages: string[] = [];
    service.onImageReceived(img => receivedImages.push(img));

    // Simulate chunked packet stream: "0:2:CHUNK_A", "1:2:CHUNK_B"
    (service as any).parseImageChunkPacket('0:2:BASE64_PART_1_');
    (service as any).parseImageChunkPacket('1:2:BASE64_PART_2');

    expect(receivedImages).toHaveLength(1);
    expect(receivedImages[0]).toBe('BASE64_PART_1_BASE64_PART_2');
  });

  test('button packets parse HOLD, RELEASE, and SHORT_PRESS', () => {
    const events: BleEvent[] = [];
    service.onButtonEvent(evt => events.push(evt));

    (service as any).parseButtonPacket('AQ=='); // HOLD
    (service as any).parseButtonPacket('AA=='); // RELEASE
    (service as any).parseButtonPacket('Ag=='); // SHORT_PRESS

    expect(events).toHaveLength(3);
    expect(events[0].type).toBe('BUTTON_HOLD');
    expect(events[1].type).toBe('BUTTON_RELEASE');
    expect(events[2].type).toBe('BUTTON_SHORT_PRESS');
  });
});
