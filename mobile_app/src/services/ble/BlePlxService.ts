/**
 * BSmart Real BLE Service (react-native-ble-plx)
 * Implements IBleService for ESP32-S3 GATT server integration.
 * See: docs/requirement.md §2.1 & §12
 */

import { Linking, Platform } from 'react-native';
import { BleManager, Device, Subscription } from 'react-native-ble-plx';
import {
  BleConnectionState,
  BleEvent,
  IBleService,
} from '../../types';
import {
  BLE_UUIDS,
  BLE_SCAN_TIMEOUT_MS,
  BLE_DEVICE_NAME_PREFIX,
} from '../../constants/bleUuids';
import { stringToBase64 } from '../audio/audioUtils';

type ButtonCallback = (event: BleEvent) => void;
type ImageCallback = (imageBase64: string) => void;
type AudioCallback = (audioBase64: string) => void;
type ConnectionCallback = (state: BleConnectionState) => void;

interface ImageChunkBuffer {
  totalChunks: number;
  receivedCount: number;
  chunks: Map<number, string>;
  timestamp: number;
}

export class BlePlxService implements IBleService {
  private manager: BleManager;
  private device: Device | null = null;
  private connectionState: BleConnectionState = BleConnectionState.DISCONNECTED;

  private buttonCallbacks: Set<ButtonCallback> = new Set();
  private imageCallbacks: Set<ImageCallback> = new Set();
  private audioCallbacks: Set<AudioCallback> = new Set();
  private connectionCallbacks: Set<ConnectionCallback> = new Set();

  private imageBuffer: ImageChunkBuffer | null = null;
  private monitorSubscriptions: Subscription[] = [];

  constructor() {
    this.manager = new BleManager();
    this.initBluetoothStateListener();
  }

  private initBluetoothStateListener(): void {
    try {
      if (typeof this.manager?.onStateChange === 'function') {
        this.manager.onStateChange(state => {
          if (state === 'PoweredOff') {
            this.setConnectionState(BleConnectionState.BLUETOOTH_OFF);
          } else if (state === 'PoweredOn') {
            if (this.connectionState === BleConnectionState.BLUETOOTH_OFF) {
              this.setConnectionState(BleConnectionState.DISCONNECTED);
            }
          }
        }, true);
      }
    } catch (e) {
      console.warn('[BlePlxService] Error initializing Bluetooth state listener:', e);
    }
  }

  async isBluetoothEnabled(): Promise<boolean> {
    try {
      const state = await this.manager.state();
      return state === 'PoweredOn';
    } catch {
      return false;
    }
  }

  async enableBluetooth(): Promise<boolean> {
    try {
      if (Platform.OS === 'android') {
        // Race manager.enable() with a 1500ms timeout.
        // On Android 12+, react-native-ble-plx's enable() lacks Activity context
        // and hangs waiting indefinitely for adapter state without launching the prompt.
        let enabledViaManager = false;
        try {
          const enablePromise = this.manager.enable().then(() => true).catch(() => false);
          const timeoutPromise = new Promise<boolean>(resolve =>
            setTimeout(() => resolve(false), 1500),
          );
          enabledViaManager = await Promise.race([enablePromise, timeoutPromise]);
        } catch {
          enabledViaManager = false;
        }

        if (enabledViaManager) {
          return true;
        }

        // Direct fallback: Open Android Bluetooth Settings for instant 1-tap toggle
        try {
          await Linking.sendIntent('android.settings.BLUETOOTH_SETTINGS');
          return true;
        } catch {
          await Linking.openSettings();
          return true;
        }
      }
      return false;
    } catch (err) {
      console.warn('[BlePlxService] Could not enable Bluetooth:', err);
      return false;
    }
  }

  async connect(): Promise<void> {
    const isEnabled = await this.isBluetoothEnabled();
    if (!isEnabled) {
      this.setConnectionState(BleConnectionState.BLUETOOTH_OFF);
      throw new Error('Bluetooth trên điện thoại đang tắt. Vui lòng bật Bluetooth.');
    }

    this.setConnectionState(BleConnectionState.CONNECTING);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.manager.stopDeviceScan();
        this.setConnectionState(BleConnectionState.DISCONNECTED);
        reject(new Error('BLE scan timeout: No BSmart glasses found'));
      }, BLE_SCAN_TIMEOUT_MS);

      this.manager.startDeviceScan(null, null, async (error, scannedDevice) => {
        if (error) {
          clearTimeout(timeout);
          this.setConnectionState(BleConnectionState.DISCONNECTED);
          reject(error);
          return;
        }

        if (
          scannedDevice &&
          (scannedDevice.name?.startsWith(BLE_DEVICE_NAME_PREFIX) ||
            scannedDevice.localName?.startsWith(BLE_DEVICE_NAME_PREFIX))
        ) {
          this.manager.stopDeviceScan();
          clearTimeout(timeout);

          try {
            this.device = await scannedDevice.connect();
            await this.device.discoverAllServicesAndCharacteristics();
            this.setConnectionState(BleConnectionState.CONNECTED);
            this.setupMonitors();
            resolve();
          } catch (connectError) {
            this.setConnectionState(BleConnectionState.DISCONNECTED);
            reject(connectError);
          }
        }
      });
    });
  }

  async disconnect(): Promise<void> {
    this.monitorSubscriptions.forEach(sub => sub.remove());
    this.monitorSubscriptions = [];

    if (this.device) {
      try {
        await this.device.cancelConnection();
      } catch {
        // ignore disconnect errors
      }
      this.device = null;
    }
    this.setConnectionState(BleConnectionState.DISCONNECTED);
  }

  async sendAudio(audioData: string): Promise<void> {
    if (!this.device) {
      throw new Error('BLE not connected');
    }

    // Ensure payload is encoded as base64 string for react-native-ble-plx
    const isBase64 =
      audioData.length > 0 &&
      audioData.length % 4 === 0 &&
      /^[A-Za-z0-9+/=]+$/.test(audioData.trim());
    const base64Payload = isBase64 ? audioData : stringToBase64(audioData);

    // Chunk audio payload into MTU-friendly sizes (~160 chars Base64 = 120 bytes = 60 16-bit PCM samples)
    const chunkSize = 160;
    const totalChunks = Math.ceil(base64Payload.length / chunkSize);

    for (let i = 0; i < totalChunks; i++) {
      const chunkPayload = base64Payload.slice(i * chunkSize, (i + 1) * chunkSize);
      await this.device.writeCharacteristicWithResponseForService(
        BLE_UUIDS.SERVICE_UUID,
        BLE_UUIDS.AUDIO_OUT_CHARACTERISTIC_UUID,
        chunkPayload,
      );
    }
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
    callback(this.connectionState);
    return () => this.connectionCallbacks.delete(callback);
  }

  getConnectionState(): BleConnectionState {
    return this.connectionState;
  }

  getConnectedDeviceName(): string | null {
    if (this.connectionState !== BleConnectionState.CONNECTED) return null;
    return this.device?.name || this.device?.localName || 'BSmart Glasses';
  }

  private setupMonitors(): void {
    if (!this.device) return;

    // 1. Monitor Button Characteristic
    const buttonSub = this.device.monitorCharacteristicForService(
      BLE_UUIDS.SERVICE_UUID,
      BLE_UUIDS.BUTTON_CHARACTERISTIC_UUID,
      (error, characteristic) => {
        if (error || !characteristic?.value) return;
        this.parseButtonPacket(characteristic.value);
      },
    );
    this.monitorSubscriptions.push(buttonSub);

    // 2. Monitor Image Characteristic (Chunked JPEG stream)
    const imageSub = this.device.monitorCharacteristicForService(
      BLE_UUIDS.SERVICE_UUID,
      BLE_UUIDS.IMAGE_CHARACTERISTIC_UUID,
      (error, characteristic) => {
        if (error || !characteristic?.value) return;
        this.parseImageChunkPacket(characteristic.value);
      },
    );
    this.monitorSubscriptions.push(imageSub);

    // 3. Monitor Audio-In Characteristic (Mic Recording from Glasses)
    const audioSub = this.device.monitorCharacteristicForService(
      BLE_UUIDS.SERVICE_UUID,
      BLE_UUIDS.AUDIO_IN_CHARACTERISTIC_UUID,
      (error, characteristic) => {
        if (error || !characteristic?.value) return;
        this.emit(this.audioCallbacks, characteristic.value);
      },
    );
    this.monitorSubscriptions.push(audioSub);
  }

  /**
   * Parse button press event packets from ESP32
   * Format: '01' = HOLD, '00' = RELEASE, '02' = SHORT_PRESS
   */
  private parseButtonPacket(base64Data: string): void {
    const raw = base64Data.trim();
    if (raw === 'AQ==' || raw === '01') {
      this.emit(this.buttonCallbacks, { type: 'BUTTON_HOLD' });
    } else if (raw === 'AA==' || raw === '00') {
      this.emit(this.buttonCallbacks, { type: 'BUTTON_RELEASE' });
    } else if (raw === 'Ag==' || raw === '02') {
      this.emit(this.buttonCallbacks, { type: 'BUTTON_SHORT_PRESS' });
    }
  }

  /**
   * Reassemble chunked JPEG image packets
   * If single payload: directly emit image
   * If chunked payload: reassemble sequence 0..N-1
   * Stale buffer (e.g. dropped BLE chunk) is discarded after IMAGE_BUFFER_TIMEOUT_MS.
   */
  private static readonly IMAGE_BUFFER_TIMEOUT_MS = 5000;

  private parseImageChunkPacket(base64Chunk: string): void {
    if (!base64Chunk.includes(':')) {
      this.emit(this.imageCallbacks, base64Chunk);
      return;
    }

    const parts = base64Chunk.split(':');
    if (parts.length < 3) {
      this.emit(this.imageCallbacks, base64Chunk);
      return;
    }

    const seq = parseInt(parts[0], 10);
    const total = parseInt(parts[1], 10);
    const payload = parts.slice(2).join(':');

    const now = Date.now();
    if (
      this.imageBuffer &&
      now - this.imageBuffer.timestamp > BlePlxService.IMAGE_BUFFER_TIMEOUT_MS
    ) {
      // Error concealment: If we have at least 80% of chunks, try to decode it anyway
      if (this.imageBuffer.chunks.size > this.imageBuffer.totalChunks * 0.8) {
        console.warn('[BlePlx] Image buffer timed out, but emitting partial frame.');
        let fullBase64 = '';
        for (let i = 0; i < this.imageBuffer.totalChunks; i++) {
          fullBase64 += this.imageBuffer.chunks.get(i) ?? '';
        }
        this.emit(this.imageCallbacks, fullBase64);
      } else {
        console.warn('[BlePlx] Image buffer timed out (dropped chunks). Resetting.');
      }
      this.imageBuffer = null;
    }

    if (seq === 0 || !this.imageBuffer) {
      this.imageBuffer = {
        totalChunks: total,
        receivedCount: 0,
        chunks: new Map(),
        timestamp: now,
      };
    }

    this.imageBuffer.chunks.set(seq, payload);
    this.imageBuffer.receivedCount++;

    // Check if image complete or mostly complete
    if (this.imageBuffer.chunks.size >= total) {
      let fullBase64 = '';
      for (let i = 0; i < total; i++) {
        fullBase64 += this.imageBuffer.chunks.get(i) ?? '';
      }
      this.imageBuffer = null;
      this.emit(this.imageCallbacks, fullBase64);
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
        console.error('[BlePlx] Callback execution error:', e);
      }
    });
  }
}
