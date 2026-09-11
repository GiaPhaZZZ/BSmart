/**
 * BSmart BLE Auto-Connect & Background Reconnect Service
 * Automatically reconnects to ESP32-S3 smart glasses when in range
 * and provides auditory + haptic feedback for visually impaired users.
 * See: docs/requirement.md §2.1, §13 (APP-10)
 */

import { Vibration } from 'react-native';
import { BleConnectionState, IBleService } from '../../types';
import { speakViaBle, stopSpeaking } from '../tts/TtsService';

export const RECONNECT_INTERVAL_MS = 5000;
export const DEFAULT_MAX_RETRIES = 2;

export interface AutoConnectConfig {
  reconnectIntervalMs?: number;
  maxRetries?: number;
  enableHaptics?: boolean;
  enableVoiceAnnouncement?: boolean;
}

export class BleAutoConnectService {
  private bleService: IBleService;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private isAutoConnectActive = false;
  private isAttemptingConnect = false;
  private hasConnectedOnce = false;
  private retryCount = 0;
  private config: Required<AutoConnectConfig>;
  private logCallback?: (msg: string) => void;
  private unsubConnection?: () => void;

  constructor(bleService: IBleService, config?: AutoConnectConfig) {
    this.bleService = bleService;
    this.config = {
      reconnectIntervalMs: config?.reconnectIntervalMs ?? RECONNECT_INTERVAL_MS,
      maxRetries: config?.maxRetries ?? DEFAULT_MAX_RETRIES,
      enableHaptics: config?.enableHaptics ?? true,
      enableVoiceAnnouncement: config?.enableVoiceAnnouncement ?? true,
    };
  }

  /**
   * Start background auto-connect monitoring
   */
  start(onLog?: (msg: string) => void): void {
    if (this.isAutoConnectActive) return;
    this.isAutoConnectActive = true;
    this.retryCount = 0;
    this.logCallback = onLog;

    this.log('Auto-Connect Service started');

    this.unsubConnection = this.bleService.onConnectionStateChange(state => {
      this.handleStateChange(state);
    });

    // Initial connect attempt if not connected
    if (this.bleService.getConnectionState() === BleConnectionState.DISCONNECTED) {
      this.scheduleReconnect(1000);
    }
  }

  /**
   * Stop background auto-connect monitoring
   */
  stop(): void {
    this.isAutoConnectActive = false;
    this.cancelScheduledReconnect();
    if (this.unsubConnection) {
      this.unsubConnection();
      this.unsubConnection = undefined;
    }
    this.log('Auto-Connect Service stopped');
  }

  /**
   * Handle BLE connection state transitions
   */
  private handleStateChange(state: BleConnectionState): void {
    if (!this.isAutoConnectActive) return;

    if (state === BleConnectionState.CONNECTED) {
      this.cancelScheduledReconnect();
      this.isAttemptingConnect = false;
      this.hasConnectedOnce = true;
      this.retryCount = 0;
      this.log('Glasses connected successfully');

      // Haptic confirmation: double pulse for blind user
      if (this.config.enableHaptics) {
        Vibration.vibrate([0, 150, 100, 150]);
      }

      // Voice confirmation
      if (this.config.enableVoiceAnnouncement) {
        speakViaBle('Kính BSmart đã kết nối thành công', this.bleService).catch(() => {});
      }
    } else if (state === BleConnectionState.BLUETOOTH_OFF) {
      this.cancelScheduledReconnect();
      this.isAttemptingConnect = false;
      this.log('Bluetooth is turned off. Auto-connect paused.');
    } else if (state === BleConnectionState.DISCONNECTED) {
      this.isAttemptingConnect = false;

      // Haptic confirmation: ONLY vibrate if glasses were previously connected in this session!
      if (this.hasConnectedOnce && this.config.enableHaptics) {
        Vibration.vibrate(400);
        this.hasConnectedOnce = false;
        // Unexpected disconnect after having been connected: trigger reconnect
        this.retryCount = 0;
        this.scheduleReconnect(this.config.reconnectIntervalMs);
      }
    }
  }

  /**
   * Schedule next reconnect attempt
   */
  private scheduleReconnect(delayMs: number): void {
    if (!this.isAutoConnectActive || this.reconnectTimer) return;

    this.reconnectTimer = setTimeout(async () => {
      this.reconnectTimer = null;
      if (!this.isAutoConnectActive) return;

      const currentState = this.bleService.getConnectionState();
      if (
        currentState === BleConnectionState.CONNECTED ||
        currentState === BleConnectionState.BLUETOOTH_OFF
      ) {
        return;
      }

      if (this.isAttemptingConnect) {
        return;
      }

      if (this.retryCount >= this.config.maxRetries) {
        this.log('Không tìm thấy kính thông minh. Chuyển sang chế độ độc lập trên điện thoại.');
        this.stop();
        return;
      }

      this.retryCount++;
      this.isAttemptingConnect = true;
      this.log(`Tìm kiếm kính BSmart (lần ${this.retryCount}/${this.config.maxRetries})...`);

      try {
        await this.bleService.connect();
      } catch (err: any) {
        this.isAttemptingConnect = false;
        if (this.retryCount < this.config.maxRetries) {
          this.log(`Không thấy kính (${err?.message || 'hết thời gian quét'}). Sẽ thử lại lần cuối...`);
          this.scheduleReconnect(this.config.reconnectIntervalMs);
        } else {
          this.log('Không tìm thấy kính BSmart. Đã dừng tìm kiếm để tiết kiệm pin (Chế độ độc lập sẵn sàng).');
          this.stop();
        }
      }
    }, delayMs);
  }

  private cancelScheduledReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private log(message: string): void {
    if (this.logCallback) {
      this.logCallback(`[AutoConnect] ${message}`);
    }
  }

  isActive(): boolean {
    return this.isAutoConnectActive;
  }
}
