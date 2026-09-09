import { BleAutoConnectService } from '../src/services/ble/BleAutoConnectService';
import { BleConnectionState, IBleService } from '../src/types';
import { Vibration } from 'react-native';

jest.useFakeTimers();

describe('BleAutoConnectService Unit Tests', () => {
  let mockBleService: jest.Mocked<IBleService>;
  let autoConnect: BleAutoConnectService;
  let connectionCallback: ((state: BleConnectionState) => void) | null = null;

  beforeEach(() => {
    jest.clearAllMocks();
    connectionCallback = null;

    mockBleService = {
      connect: jest.fn().mockResolvedValue(undefined),
      disconnect: jest.fn().mockResolvedValue(undefined),
      sendAudio: jest.fn().mockResolvedValue(undefined),
      onButtonEvent: jest.fn().mockReturnValue(() => {}),
      onImageReceived: jest.fn().mockReturnValue(() => {}),
      onAudioReceived: jest.fn().mockReturnValue(() => {}),
      onConnectionStateChange: jest.fn().mockImplementation(cb => {
        connectionCallback = cb;
        return () => {
          connectionCallback = null;
        };
      }),
      getConnectionState: jest.fn().mockReturnValue(BleConnectionState.DISCONNECTED),
    };

    autoConnect = new BleAutoConnectService(mockBleService, {
      reconnectIntervalMs: 2000,
      enableHaptics: true,
      enableVoiceAnnouncement: false, // avoid TTS native call in unit tests
    });
  });

  afterEach(() => {
    autoConnect.stop();
  });

  test('start() activates auto-connect and listens to connection state', () => {
    expect(autoConnect.isActive()).toBe(false);
    autoConnect.start();
    expect(autoConnect.isActive()).toBe(true);
    expect(mockBleService.onConnectionStateChange).toHaveBeenCalled();
  });

  test('schedules reconnect attempt when disconnected', () => {
    autoConnect.start();

    // Fast-forward initial connect timer (1000ms)
    jest.advanceTimersByTime(1000);

    expect(mockBleService.connect).toHaveBeenCalled();
  });

  test('triggers haptics when connected and stops timer', () => {
    autoConnect.start();

    if (connectionCallback) {
      connectionCallback(BleConnectionState.CONNECTED);
    }

    expect(Vibration.vibrate).toHaveBeenCalledWith([0, 150, 100, 150]);
  });

  test('triggers haptics on disconnect and schedules reconnect', () => {
    autoConnect.start();

    if (connectionCallback) {
      connectionCallback(BleConnectionState.DISCONNECTED);
    }

    expect(Vibration.vibrate).toHaveBeenCalledWith(400);

    // Fast forward interval (2000ms)
    jest.advanceTimersByTime(2000);
    expect(mockBleService.connect).toHaveBeenCalled();
  });

  test('stop() deactivates service cleanly', () => {
    autoConnect.start();
    expect(autoConnect.isActive()).toBe(true);
    autoConnect.stop();
    expect(autoConnect.isActive()).toBe(false);
  });
});
