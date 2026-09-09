/**
 * BSmart Real BLE Service (react-native-ble-plx)
 * Implements IBleService for ESP32-S3 GATT server integration.
 * See: docs/requirement.md §2.1 & §12
 */

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
  }

  async connect(): Promise<void> {
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

  async sendAudio(audioBase64: string): Promise<void> {
    if (!this.device) {
      throw new Error('BLE not connected');
    }

    // Chunk audio payload into MTU-friendly sizes (~180 bytes per chunk)
    const chunkSize = 180;
    const totalChunks = Math.ceil(audioBase64.length / chunkSize);

    for (let i = 0; i < totalChunks; i++) {
      const chunkPayload = audioBase64.slice(i * chunkSize, (i + 1) * chunkSize);
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
   */
  private parseImageChunkPacket(base64Chunk: string): void {
    // If not starting with chunk header, treat as complete base64 image
    if (!base64Chunk.includes(':')) {
      this.emit(this.imageCallbacks, base64Chunk);
      return;
    }

    // Header protocol format: "seq:total:payload_base64"
    const parts = base64Chunk.split(':');
    if (parts.length < 3) {
      this.emit(this.imageCallbacks, base64Chunk);
      return;
    }

    const seq = parseInt(parts[0], 10);
    const total = parseInt(parts[1], 10);
    const payload = parts.slice(2).join(':');

    if (seq === 0 || !this.imageBuffer) {
      this.imageBuffer = {
        totalChunks: total,
        receivedCount: 0,
        chunks: new Map(),
        timestamp: Date.now(),
      };
    }

    this.imageBuffer.chunks.set(seq, payload);
    this.imageBuffer.receivedCount++;

    // Check if image complete
    if (this.imageBuffer.chunks.size === total) {
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
