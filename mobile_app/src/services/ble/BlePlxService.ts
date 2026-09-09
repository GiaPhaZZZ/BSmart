/**
 * BSmart Real BLE Service (stub)
 * Uses react-native-ble-plx to communicate with ESP32-S3 GATT server.
 *
 * NOTE: This is a structural stub. Full implementation requires:
 * 1. Actual BLE UUIDs from firmware team (see src/constants/bleUuids.ts)
 * 2. Confirmed chunk format from firmware specification
 * 3. Physical hardware testing
 *
 * Until firmware UUIDs are provided, use MockBleService.
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

export class BlePlxService implements IBleService {
  private manager: BleManager;
  private device: Device | null = null;
  private connectionState: BleConnectionState = BleConnectionState.DISCONNECTED;

  private buttonCallbacks: Set<ButtonCallback> = new Set();
  private imageCallbacks: Set<ImageCallback> = new Set();
  private audioCallbacks: Set<AudioCallback> = new Set();
  private connectionCallbacks: Set<ConnectionCallback> = new Set();

  private imageChunks: Map<number, string> = new Map();
  private imageChunkTotal: number = 0;
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
        reject(new Error('BLE scan timeout'));
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
          scannedDevice.name?.startsWith(BLE_DEVICE_NAME_PREFIX)
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
        // ignore
      }
      this.device = null;
    }
    this.setConnectionState(BleConnectionState.DISCONNECTED);
  }

  async sendAudio(audioBase64: string): Promise<void> {
    if (!this.device) {
      throw new Error('BLE not connected');
    }
    // TODO: Chunk audioBase64 and write to AUDIO_OUT_CHARACTERISTIC_UUID
    // Chunk format TBD with firmware team
    await this.device.writeCharacteristicWithResponseForService(
      BLE_UUIDS.SERVICE_UUID,
      BLE_UUIDS.AUDIO_OUT_CHARACTERISTIC_UUID,
      audioBase64,
    );
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

    // Monitor button characteristic
    const buttonSub = this.device.monitorCharacteristicForService(
      BLE_UUIDS.SERVICE_UUID,
      BLE_UUIDS.BUTTON_CHARACTERISTIC_UUID,
      (error, characteristic) => {
        if (error || !characteristic?.value) return;
        this.handleButtonData(characteristic.value);
      },
    );
    this.monitorSubscriptions.push(buttonSub);

    // Monitor image characteristic
    const imageSub = this.device.monitorCharacteristicForService(
      BLE_UUIDS.SERVICE_UUID,
      BLE_UUIDS.IMAGE_CHARACTERISTIC_UUID,
      (error, characteristic) => {
        if (error || !characteristic?.value) return;
        this.handleImageChunk(characteristic.value);
      },
    );
    this.monitorSubscriptions.push(imageSub);

    // Monitor audio-in characteristic
    const audioSub = this.device.monitorCharacteristicForService(
      BLE_UUIDS.SERVICE_UUID,
      BLE_UUIDS.AUDIO_IN_CHARACTERISTIC_UUID,
      (error, characteristic) => {
        if (error || !characteristic?.value) return;
        this.handleAudioChunk(characteristic.value);
      },
    );
    this.monitorSubscriptions.push(audioSub);
  }

  private handleButtonData(base64Value: string): void {
    // TODO: Parse actual button packet format from firmware spec
    // Placeholder: interpret '01' = HOLD, '00' = RELEASE, '02' = SHORT_PRESS
    // Simplified: treat the raw base64 string as a command token for demo
    // In production: decode base64 bytes and parse the firmware packet format
    const cmd = base64Value.trim();
    if (cmd === 'AQ==' || cmd === '01') {
      this.emit(this.buttonCallbacks, { type: 'BUTTON_HOLD' });
    } else if (cmd === 'AA==' || cmd === '00') {
      this.emit(this.buttonCallbacks, { type: 'BUTTON_RELEASE' });
    } else if (cmd === 'Ag==' || cmd === '02') {
      this.emit(this.buttonCallbacks, { type: 'BUTTON_SHORT_PRESS' });
    }
  }

  private handleImageChunk(base64Value: string): void {
    // TODO: Parse chunk format: type(1) + sequence(2) + total(2) + payload
    // This is a placeholder — real parsing depends on firmware chunk format
    // For now, treat each notification as a complete image
    this.emit(this.imageCallbacks, base64Value);
  }

  private handleAudioChunk(base64Value: string): void {
    // TODO: Parse audio chunk format from firmware spec
    this.emit(this.audioCallbacks, base64Value);
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
        console.error('[BlePlx] Callback error:', e);
      }
    });
  }
}
