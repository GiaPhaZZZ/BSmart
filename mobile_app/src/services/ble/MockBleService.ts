/**
 * BSmart Mock BLE Service
 * Simulates glasses hardware for development and testing without physical device.
 * See: SKILL.md §4.A, §5
 */

import {
  BleConnectionState,
  BleEvent,
  IBleService,
} from '../../types';

type ButtonCallback = (event: BleEvent) => void;
type ImageCallback = (imageBase64: string) => void;
type AudioCallback = (audioBase64: string) => void;
type ConnectionCallback = (state: BleConnectionState) => void;

// 1x1 transparent PNG as placeholder mock image (base64)
const MOCK_IMAGE_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

export class MockBleService implements IBleService {
  private connectionState: BleConnectionState = BleConnectionState.DISCONNECTED;
  private buttonCallbacks: Set<ButtonCallback> = new Set();
  private imageCallbacks: Set<ImageCallback> = new Set();
  private audioCallbacks: Set<AudioCallback> = new Set();
  private connectionCallbacks: Set<ConnectionCallback> = new Set();
  private autoImageInterval: ReturnType<typeof setInterval> | null = null;

  async connect(): Promise<void> {
    this.setConnectionState(BleConnectionState.CONNECTING);
    // Simulate connection delay
    await new Promise<void>(resolve => setTimeout(() => resolve(), 800));
    this.setConnectionState(BleConnectionState.CONNECTED);
    console.log('[MockBLE] Connected');
  }

  async disconnect(): Promise<void> {
    this.stopAutoImage();
    this.setConnectionState(BleConnectionState.DISCONNECTED);
    console.log('[MockBLE] Disconnected');
  }

  async sendAudio(audioData: string): Promise<void> {
    console.log('[MockBLE] sendAudio called, data length:', audioData.length);
    // In mock: simulate glasses receiving audio (no-op)
  }

  async isBluetoothEnabled(): Promise<boolean> {
    return true;
  }

  async enableBluetooth(): Promise<boolean> {
    this.setConnectionState(BleConnectionState.DISCONNECTED);
    return true;
  }

  onButtonEvent(callback: ButtonCallback): () => void {
    this.buttonCallbacks.add(callback);
    return () => this.buttonCallbacks.delete(callback);
  }

  onImageReceived(callback: ImageCallback): () => void {
    this.imageCallbacks.add(callback);
    return () => this.imageCallbacks.delete(callback);
  }

  onAudioReceived(callback: AudioCallback): () => void {
    this.audioCallbacks.add(callback);
    return () => this.audioCallbacks.delete(callback);
  }

  onConnectionStateChange(callback: ConnectionCallback): () => void {
    this.connectionCallbacks.add(callback);
    // Immediately call with current state
    callback(this.connectionState);
    return () => this.connectionCallbacks.delete(callback);
  }

  getConnectionState(): BleConnectionState {
    return this.connectionState;
  }

  getConnectedDeviceName(): string | null {
    if (this.connectionState !== BleConnectionState.CONNECTED) return null;
    return 'BSmart Glasses (Giả lập)';
  }

  // --- Mock-only simulation helpers ---

  /**
   * Simulate holding the Record button (triggers LISTENING)
   */
  simulateButtonHold(): void {
    this.emit(this.buttonCallbacks, { type: 'BUTTON_HOLD' });
    console.log('[MockBLE] Button HOLD simulated');
  }

  /**
   * Simulate releasing the Record button (ends recording)
   */
  simulateButtonRelease(): void {
    this.emit(this.buttonCallbacks, { type: 'BUTTON_RELEASE' });
    console.log('[MockBLE] Button RELEASE simulated');
  }

  /**
   * Simulate short press (cancel / return to IDLE)
   */
  simulateShortPress(): void {
    this.emit(this.buttonCallbacks, { type: 'BUTTON_SHORT_PRESS' });
    console.log('[MockBLE] Button SHORT_PRESS simulated');
  }

  /**
   * Simulate receiving a camera image from glasses
   */
  simulateImage(base64?: string): void {
    this.emit(this.imageCallbacks, base64 ?? MOCK_IMAGE_BASE64);
    console.log('[MockBLE] Image simulated');
  }

  /**
   * Start auto-sending mock images every intervalMs (for Feature 3 testing)
   */
  startAutoImage(intervalMs: number = 4000): void {
    this.stopAutoImage();
    this.autoImageInterval = setInterval(() => {
      this.simulateImage();
    }, intervalMs);
    console.log('[MockBLE] Auto image started, interval:', intervalMs);
  }

  /**
   * Stop auto image sending
   */
  stopAutoImage(): void {
    if (this.autoImageInterval !== null) {
      clearInterval(this.autoImageInterval);
      this.autoImageInterval = null;
      console.log('[MockBLE] Auto image stopped');
    }
  }

  private setConnectionState(state: BleConnectionState): void {
    this.connectionState = state;
    this.emit(this.connectionCallbacks, state);
  }

  private emit<T>(callbacks: Set<(val: T) => void>, value: T): void {
    callbacks.forEach(cb => {
      try {
        cb(value);
      } catch (e) {
        console.error('[MockBLE] Callback error:', e);
      }
    });
  }
}

// Singleton mock instance
export const mockBleService = new MockBleService();
