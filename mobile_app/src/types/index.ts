/**
 * BSmart Shared Types & State Enums
 */

export enum AppState {
  IDLE = 'IDLE',
  LISTENING = 'LISTENING',
  PROCESSING = 'PROCESSING',
  FEATURE_1_QA = 'FEATURE_1_QA',
  FEATURE_2_CAPTURE = 'FEATURE_2_CAPTURE',
  FEATURE_3_NAVIGATION = 'FEATURE_3_NAVIGATION',
}

export enum BleConnectionState {
  DISCONNECTED = 'Disconnected',
  CONNECTING = 'Connecting',
  CONNECTED = 'Connected',
  BLUETOOTH_OFF = 'BluetoothOff',
}

export interface BleEvent {
  type: 'BUTTON_HOLD' | 'BUTTON_RELEASE' | 'BUTTON_SHORT_PRESS';
}

export interface BleImageChunk {
  sequence: number;
  total: number;
  data: string; // base64
}

export interface BleAudioChunk {
  sequence: number;
  total: number;
  data: string; // base64
}

export interface DetectedObject {
  class: string;
  confidence: number;
  x: number; // center x relative to image width (0-1)
  y: number; // center y relative to image height (0-1)
  width: number; // relative width (0-1)
  height: number; // relative height (0-1)
  depthScore: number; // relative depth (0=near, 1=far)
}

export interface NavigationWarning {
  objectClass: string;
  position: 'Trái' | 'Giữa' | 'Phải';
  distance: 'Gần' | 'Xa';
  cooldownKey: string;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'error' | 'warn';
}

export interface IBleService {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  sendAudio(audioData: string): Promise<void>; // base64
  onButtonEvent(callback: (event: BleEvent) => void): () => void;
  onImageReceived(callback: (imageBase64: string) => void): () => void;
  onAudioReceived(callback: (audioBase64: string) => void): () => void;
  onConnectionStateChange(
    callback: (state: BleConnectionState) => void,
  ): () => void;
  getConnectionState(): BleConnectionState;
  getConnectedDeviceName?(): string | null;
  enableBluetooth?(): Promise<boolean>;
  isBluetoothEnabled?(): Promise<boolean>;
}
