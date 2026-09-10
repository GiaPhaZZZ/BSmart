/**
 * BSmart App State Machine Hook
 * The central state machine that coordinates all features.
 * See: SKILL.md §3, docs/requirement.md §3
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, BleConnectionState, LogEntry } from '../types';
import { getBleService, getMockBleService, BleAutoConnectService } from '../services/ble';
import { transcribeAudio, askQA } from '../services/api/ApiService';
import { transcribeAudioOnDevice } from '../services/ai/OnDeviceAsrService';
import { askQAOnDevice } from '../services/ai/OnDeviceVlmService';
import { speakViaBle, speakUrgent, stopSpeaking } from '../services/tts/TtsService';
import {
  playFeatureActivationSound,
  stopFeatureSound,
} from '../services/audio/SoundEffectService';
import {
  processNavigationFrame,
  warningsToVietnamese,
  resetCooldowns,
} from '../services/navigation/NavigationEngine';
import {
  runInference,
  isInferenceAvailable,
} from '../services/navigation/OnDeviceInference';
import { NAVIGATION_FRAME_INTERVAL_MS } from '../constants/navigationRules';
import { saveCapturedImage } from '../services/storage/ImageStorageService';

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function makeLog(message: string, type: LogEntry['type'] = 'info'): LogEntry {
  return {
    id: generateId(),
    timestamp: new Date().toLocaleTimeString('vi-VN'),
    message,
    type,
  };
}

/**
 * Keyword matching for feature selection
 * See: docs/requirement.md §4
 */
function matchFeatureKeyword(
  text: string,
): 'feature1' | 'feature2' | 'feature3' | null {
  const lower = text.toLowerCase().trim();
  if (
    lower.includes('tính năng 1') ||
    lower.includes('tính năng một') ||
    lower.includes('gpt')
  ) {
    return 'feature1';
  }
  if (lower.includes('tính năng 2') || lower.includes('tính năng hai')) {
    return 'feature2';
  }
  if (lower.includes('tính năng 3') || lower.includes('tính năng ba')) {
    return 'feature3';
  }
  return null;
}

export interface AppStateMachineResult {
  appState: AppState;
  connectionState: BleConnectionState;
  logs: LogEntry[];
  currentImage: string | null;
  useMock: boolean;
  // Mock controls
  onMockButtonHold: () => void;
  onMockButtonRelease: () => void;
  onMockShortPress: () => void;
  onMockImage: () => void;
  // BLE controls
  onConnect: () => void;
  onDisconnect: () => void;
}

export function useAppStateMachine(): AppStateMachineResult {
  const [appState, setAppState] = useState<AppState>(AppState.IDLE);
  const [connectionState, setConnectionState] = useState<BleConnectionState>(
    BleConnectionState.DISCONNECTED,
  );
  const [logs, setLogs] = useState<LogEntry[]>([
    makeLog('BSmart App started. Mock BLE active.'),
  ]);
  const [currentImage, setCurrentImage] = useState<string | null>(null);

  const bleService = useRef(getBleService());
  const mockService = useRef(getMockBleService());
  const appStateRef = useRef<AppState>(AppState.IDLE);
  const isProcessingFrame = useRef(false);
  const navIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const recordedAudioRef = useRef<string | null>(null);
  const capturedImageRef = useRef<string | null>(null);

  const addLog = useCallback(
    (message: string, type: LogEntry['type'] = 'info') => {
      setLogs(prev => {
        const next = [makeLog(message, type), ...prev];
        return next.slice(0, 50); // Keep last 50 entries
      });
    },
    [],
  );

  const transitionTo = useCallback(
    (newState: AppState) => {
      appStateRef.current = newState;
      setAppState(newState);
      addLog(`State → ${newState}`);
    },
    [addLog],
  );

  // ─── Return to IDLE (short press handler) ───────────────────────────
  const returnToIdle = useCallback(async () => {
    stopSpeaking();
    stopFeatureSound();
    stopNavigation();
    resetCooldowns();
    transitionTo(AppState.IDLE);
    addLog('Short press → return to IDLE');
    try {
      await speakViaBle('Đã về trang chủ', bleService.current);
    } catch {
      // ignore cancellation
    }
  }, [transitionTo, addLog]);

  // ─── Navigation mode helpers ─────────────────────────────────────────
  function stopNavigation() {
    if (navIntervalRef.current !== null) {
      clearInterval(navIntervalRef.current);
      navIntervalRef.current = null;
    }
    isProcessingFrame.current = false;
    if (mockService.current) {
      mockService.current.stopAutoImage();
    }
  }

  async function processNavigationImage(imageBase64: string) {
    if (isProcessingFrame.current) {
      addLog('Nav: skipping frame (prev still processing)', 'warn');
      return;
    }
    if (appStateRef.current !== AppState.FEATURE_3_NAVIGATION) return;

    isProcessingFrame.current = true;
    try {
      let detectedObjects;

      if (isInferenceAvailable()) {
        // Real on-device inference path (unblocked once models are available)
        const result = await runInference(imageBase64);
        detectedObjects = result.objects;
        addLog(`Nav: inference found ${detectedObjects.length} object(s)`);
      } else {
        // BLOCKED: On-device models not yet available.
        // Using mock data to keep navigation flow testable.
        // Replace this block when OnDeviceInference is unblocked.
        addLog('Nav: inference BLOCKED — using mock data', 'warn');
        detectedObjects = [
          {
            class: 'person',
            confidence: 0.85,
            x: 0.5,
            y: 0.5,
            width: 0.2,
            height: 0.4,
            depthScore: 0.2,
          },
        ];
      }

      const warnings = processNavigationFrame(detectedObjects);
      const text = warningsToVietnamese(warnings);

      if (text) {
        addLog(`Nav warning: ${text}`);
        await speakUrgent(text, bleService.current);
      } else {
        addLog('Nav: no warnings this frame');
      }
    } catch (e) {
      addLog(`Nav frame error: ${e}`, 'error');
    } finally {
      isProcessingFrame.current = false;
    }
  }

  // ─── Button event handler ────────────────────────────────────────────
  const handleButtonHold = useCallback(async () => {
    const state = appStateRef.current;
    addLog('Button HOLD');

    if (state === AppState.IDLE || state === AppState.FEATURE_1_QA) {
      transitionTo(AppState.LISTENING);
      recordedAudioRef.current = null;
      capturedImageRef.current = null;

      // In real implementation: start audio capture from BLE
      try {
        await speakViaBle('Chờ nhận lệnh', bleService.current);
      } catch {
        // ignored
      }
    }
  }, [addLog, transitionTo]);

  const handleButtonRelease = useCallback(async () => {
    const state = appStateRef.current;
    addLog('Button RELEASE');

    if (state !== AppState.LISTENING) return;

    transitionTo(AppState.PROCESSING);

    try {
      // In real implementation: recordedAudioRef.current = actual BLE audio
      const audioData = recordedAudioRef.current ?? 'MOCK_AUDIO_BASE64';
      const imageData = capturedImageRef.current ?? null;

      if (appStateRef.current === AppState.FEATURE_1_QA) {
        // Feature 1: on-demand audio + image for QA (100% Offline, no server dependence)
        addLog('Processing on-demand QA request (100% Offline)...');
        addLog('Transcribing question using on-device PhoWhisper...');
        const onDeviceAsr = await transcribeAudioOnDevice(audioData);
        const question = onDeviceAsr.text || 'Trước mặt tôi có gì?';
        addLog(`Transcribed: "${question}"`);

        const qaImage = imageData ?? 'MOCK_IMAGE_BASE64';
        addLog('Running on-device SmolVLM2-256M Visual QA (On-Demand)...');
        const onDeviceVlm = await askQAOnDevice(qaImage, question);
        const answer = onDeviceVlm.answer;
        addLog(`Answer: "${answer}"`);

        await speakViaBle(answer, bleService.current);
        transitionTo(AppState.FEATURE_1_QA); // Stay in QA
      } else {
        // Feature selection from IDLE→LISTENING (100% Offline)
        addLog('Transcribing command using on-device PhoWhisper...');
        const onDeviceAsr = await transcribeAudioOnDevice(audioData);
        const text = onDeviceAsr.text;
        addLog(`Transcribed: "${text}"`);

        const feature = matchFeatureKeyword(text);
        if (!feature) {
          addLog('No feature keyword matched', 'warn');
          await speakViaBle(
            'Không nhận diện được lệnh, vui lòng thử lại',
            bleService.current,
          );
          transitionTo(AppState.IDLE);
          return;
        }

        if (feature === 'feature1') {
          transitionTo(AppState.FEATURE_1_QA);
          await playFeatureActivationSound('feature1', bleService.current);
        } else if (feature === 'feature2') {
          transitionTo(AppState.FEATURE_2_CAPTURE);
          await playFeatureActivationSound('feature2', bleService.current);
          // Feature 2: capture + save
          await captureAndSave();
        } else if (feature === 'feature3') {
          transitionTo(AppState.FEATURE_3_NAVIGATION);
          await playFeatureActivationSound('feature3', bleService.current);
          startNavigation();
        }
      }
    } catch (e: any) {
      addLog(`Processing error: ${e?.message ?? e}`, 'error');
      await speakViaBle(
        'Không thể xử lý yêu cầu, vui lòng thử lại',
        bleService.current,
      ).catch(() => {});
      transitionTo(AppState.IDLE);
    }
  }, [addLog, transitionTo]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleShortPress = useCallback(async () => {
    addLog('Button SHORT_PRESS → cancel');
    await returnToIdle();
  }, [addLog, returnToIdle]);

  async function captureAndSave() {
    try {
      addLog('Feature 2: capturing image...');
      const savedFilename = await saveCapturedImage(capturedImageRef.current);
      addLog(`Feature 2: image saved to local storage (${savedFilename})`);
      await speakViaBle(
        'Đã hoàn thành, bạn muốn chọn tính năng nào tiếp theo',
        bleService.current,
      );
    } catch (e: any) {
      addLog(`Capture error: ${e?.message}`, 'error');
      await speakViaBle(
        'Không thể chụp ảnh, vui lòng thử lại',
        bleService.current,
      ).catch(() => {});
    } finally {
      transitionTo(AppState.IDLE);
    }
  }

  function startNavigation() {
    stopNavigation();
    resetCooldowns();
    if (mockService.current) {
      mockService.current.startAutoImage(NAVIGATION_FRAME_INTERVAL_MS);
    }
  }

  // ─── BLE Event Setup ─────────────────────────────────────────────────
  useEffect(() => {
    const svc = bleService.current;

    const unsubButton = svc.onButtonEvent(event => {
      if (event.type === 'BUTTON_SHORT_PRESS') {
        handleShortPress();
      } else if (event.type === 'BUTTON_HOLD') {
        handleButtonHold();
      } else if (event.type === 'BUTTON_RELEASE') {
        handleButtonRelease();
      }
    });

    const unsubImage = svc.onImageReceived(imageBase64 => {
      setCurrentImage(imageBase64);
      capturedImageRef.current = imageBase64;

      if (appStateRef.current === AppState.FEATURE_3_NAVIGATION) {
        processNavigationImage(imageBase64);
      }
    });

    const unsubAudio = svc.onAudioReceived(audioBase64 => {
      recordedAudioRef.current = audioBase64;
    });

    const unsubConnection = svc.onConnectionStateChange(state => {
      setConnectionState(state);
      addLog(`BLE: ${state}`);

      if (state === BleConnectionState.DISCONNECTED) {
        stopSpeaking();
        stopNavigation();
        if (appStateRef.current !== AppState.IDLE) {
          transitionTo(AppState.IDLE);
          addLog('BLE disconnected → reset to IDLE', 'warn');
        }
      }
    });

    const autoConnect = new BleAutoConnectService(svc);
    autoConnect.start(msg => addLog(msg));

    return () => {
      autoConnect.stop();
      unsubButton();
      unsubImage();
      unsubAudio();
      unsubConnection();
      stopNavigation();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Mock controls ────────────────────────────────────────────────────
  const onMockButtonHold = useCallback(() => {
    mockService.current?.simulateButtonHold();
  }, []);

  const onMockButtonRelease = useCallback(() => {
    mockService.current?.simulateButtonRelease();
  }, []);

  const onMockShortPress = useCallback(() => {
    mockService.current?.simulateShortPress();
  }, []);

  const onMockImage = useCallback(() => {
    mockService.current?.simulateImage();
  }, []);

  const onConnect = useCallback(async () => {
    try {
      addLog('Connecting to glasses...');
      await bleService.current.connect();
    } catch (e: any) {
      addLog(`Connect failed: ${e?.message ?? e}`, 'error');
    }
  }, [addLog]);

  const onDisconnect = useCallback(async () => {
    try {
      await bleService.current.disconnect();
    } catch (e: any) {
      addLog(`Disconnect error: ${e?.message ?? e}`, 'error');
    }
  }, [addLog]);

  return {
    appState,
    connectionState,
    logs,
    currentImage,
    useMock: mockService.current !== null,
    onMockButtonHold,
    onMockButtonRelease,
    onMockShortPress,
    onMockImage,
    onConnect,
    onDisconnect,
  };
}
