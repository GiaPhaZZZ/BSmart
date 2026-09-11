/**
 * BSmart App State Machine Hook
 * The central state machine that coordinates all features.
 * See: SKILL.md §3, docs/requirement.md §3
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { AccessibilityInfo } from 'react-native';
import { AppState, BleConnectionState, LogEntry } from '../types';
import {
  getBleService,
  getMockBleService,
  setUseMockBle,
  isUsingMockBle,
} from '../services/ble';
import {
  requestBlePermissions,
  requestAllAppPermissions,
} from '../services/ble/BlePermissionService';
import {
  startBackgroundService,
  stopBackgroundService,
} from '../services/background/ForegroundService';
import { transcribeSpeech } from '../services/ai/OnDeviceAsrService';
import { askQAOnDevice } from '../services/ai/OnDeviceVlmService';
import { modelRegistry } from '../services/ai/ModelRegistry';
import { combinePcmChunksToWav } from '../services/audio/audioUtils';
import {
  startPhoneListening,
  stopPhoneListening,
  cancelPhoneListening,
} from '../services/audio/PhoneSpeechService';
import { capturePhonePhoto } from '../services/camera/PhoneCameraService';
import { speakViaBle, speakUrgent, stopSpeaking } from '../services/tts/TtsService';
import {
  playFeatureActivationSound,
  stopFeatureSound,
  playPttStartFeedback,
  playPttEndFeedback,
  playCancelFeedback,
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
import { MapboxService } from '../services/navigation/MapboxService';
import { RouteGuide } from '../services/navigation/RouteGuide';

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
  if (!text) return null;
  const lower = text.toLowerCase().trim();
  const norm = lower
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/g, 'd')
    .trim();

  if (
    norm.includes('tinh nang 1') ||
    norm.includes('tinh nang mot') ||
    norm.includes('gpt') ||
    norm.includes('hoi dap')
  ) {
    return 'feature1';
  }
  if (
    norm.includes('tinh nang 2') ||
    norm.includes('tinh nang hai') ||
    norm.includes('chup anh') ||
    norm.includes('chup')
  ) {
    return 'feature2';
  }
  if (
    norm.includes('tinh nang 3') ||
    norm.includes('tinh nang ba') ||
    norm.includes('dan duong') ||
    norm.includes('vat can') ||
    norm.includes('canh bao')
  ) {
    return 'feature3';
  }
  return null;
}

export interface AppStateMachineResult {
  appState: AppState;
  connectionState: BleConnectionState;
  connectedDeviceName: string | null;
  logs: LogEntry[];
  currentImage: string | null;
  useMock: boolean;
  isMockBle: boolean;
  toggleMockBle: (enabled: boolean) => void;
  // Blind mode & hardware PTT actions
  onButtonHold: () => void;
  onButtonRelease: () => void;
  onShortPress: () => void;
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
  const [connectedDeviceName, setConnectedDeviceName] = useState<string | null>(null);
  const [isMockBle, setIsMockBle] = useState(isUsingMockBle());
  const [logs, setLogs] = useState<LogEntry[]>([
    makeLog(`BSmart App started. ${isUsingMockBle() ? 'Mock BLE' : 'Real BLE'} active.`),
  ]);
  const [currentImage, setCurrentImage] = useState<string | null>(null);

  const bleService = useRef(getBleService());
  const mockService = useRef(getMockBleService());
  const appStateRef = useRef<AppState>(AppState.IDLE);
  const previousStateRef = useRef<AppState>(AppState.IDLE);
  const isProcessingFrame = useRef(false);
  const navIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const recordedAudioRef = useRef<string | null>(null);
  const audioChunksRef = useRef<string[]>([]);
  const capturedImageRef = useRef<string | null>(null);
  const isPhoneRecordingRef = useRef<boolean>(false);
  const phoneListeningPromiseRef = useRef<Promise<boolean> | null>(null);
  const routeGuideRef = useRef<RouteGuide>(new RouteGuide());
  const lastShortPressTimeRef = useRef<number>(0);
  const lastOffRouteAnnouncedRef = useRef<number>(0);

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

  // ─── Navigation mode helpers ─────────────────────────────────────────
  const stopNavigation = useCallback(() => {
    if (navIntervalRef.current !== null) {
      clearInterval(navIntervalRef.current);
      navIntervalRef.current = null;
    }
    isProcessingFrame.current = false;
    if (mockService.current) {
      mockService.current.stopAutoImage();
    }
    // Send NAV_STOP command to ESP32 to stop auto 4s camera capture
    if (bleService.current?.getConnectionState() === BleConnectionState.CONNECTED) {
      bleService.current.sendAudio('NAV_STOP').catch(err => {
        console.warn('[BLE] Failed to send NAV_STOP to glasses:', err);
      });
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Return to IDLE (short press handler) ───────────────────────────
  const returnToIdle = useCallback(async () => {
    phoneListeningPromiseRef.current = null;
    if (isPhoneRecordingRef.current) {
      isPhoneRecordingRef.current = false;
      cancelPhoneListening().catch(() => {});
    }
    playCancelFeedback();
    stopSpeaking();
    stopFeatureSound();
    stopNavigation();
    resetCooldowns();
    transitionTo(AppState.IDLE);
    AccessibilityInfo.announceForAccessibility('Đã về trang chủ');
    addLog('Short press → return to IDLE');
    try {
      await speakViaBle('Đã về trang chủ', bleService.current);
    } catch {
      // ignore cancellation
    }
  }, [transitionTo, addLog, stopNavigation]);

  async function processNavigationImage(imageBase64: string, destinationText?: string) {
    if (isProcessingFrame.current) {
      addLog('Nav: skipping frame (prev still processing)', 'warn');
      return;
    }
    if (appStateRef.current !== AppState.FEATURE_3_NAVIGATION) return;

    isProcessingFrame.current = true;
    try {
      const { LocationService } = require('../services/location/LocationService');
      let lat: number | undefined;
      let lon: number | undefined;
      try {
        const loc = await LocationService.getCurrentLocation();
        lat = loc.lat;
        lon = loc.lon;
      } catch (e) {
        addLog(`GPS Error: ${e}`, 'warn');
      }

      if (destinationText && lat !== undefined && lon !== undefined) {
        addLog(`Đang tìm đường đến: ${destinationText}...`);
        const destCoords = await MapboxService.searchDestination(destinationText, { lat, lon });
        if (destCoords) {
          addLog('Đã tìm thấy địa điểm, đang lấy lộ trình...');
          const route = await MapboxService.getWalkingDirections({ lat, lon }, destCoords);
          if (route) {
            routeGuideRef.current.setRoute(route);
            addLog('Lấy lộ trình thành công!');
          } else {
            addLog('Không tìm được đường đi tới đây.', 'warn');
            await speakUrgent('Không tìm được đường đi tới vị trí này.', bleService.current);
          }
        } else {
          addLog('Không tìm thấy địa điểm này trên bản đồ.', 'warn');
          await speakUrgent('Không tìm thấy địa điểm này trên bản đồ.', bleService.current);
        }
      } else if (destinationText) {
        addLog('Không có GPS, tự động bỏ qua Mapbox và chuyển sang dùng AI On-device cảnh báo.', 'warn');
        // Không dùng speakUrgent để báo lỗi nữa, cứ để cho AI Inference báo cáo vật cản.
      }

      // 1. Chạy AI Nhận diện vật cản On-device
      let warningText = '';
      if (isInferenceAvailable()) {
        const inferenceResult = await runInference(imageBase64);
        if (inferenceResult.isReady && inferenceResult.objects.length > 0) {
          warningText = warningsToVietnamese(processNavigationFrame(inferenceResult.objects));
        }
      }

      // 2. Lấy hướng dẫn chỉ đường từ RouteGuide
      let navText = '';
      if (lat !== undefined && lon !== undefined) {
        const routeInfo = routeGuideRef.current.updateAndGetInstruction({ lat, lon });
        if (routeInfo.isOffRoute) {
          const now = Date.now();
          if (now - lastOffRouteAnnouncedRef.current > 15000) {
            navText = 'Bạn đã đi lệch đường. ';
            lastOffRouteAnnouncedRef.current = now;
          }
        } else if (routeInfo.instruction) {
          navText = routeInfo.instruction;
          lastOffRouteAnnouncedRef.current = 0; // reset cooldown nếu đã quay lại đúng đường
        }
      }

      // 3. Ghép 2 câu thành 1 và phát ra loa
      let combinedText = '';
      if (navText) combinedText += navText + ' ';
      if (warningText) {
         combinedText += warningText;
      }
      combinedText = combinedText.trim();

      if (combinedText) {
        addLog(`Nav voice: ${combinedText}`);
        await speakUrgent(combinedText, bleService.current);
      } else {
        addLog('Nav: no warnings this frame');
      }
    } catch (e) {
      addLog(`Nav frame error: ${e}`, 'error');
    } finally {
      isProcessingFrame.current = false;
    }
  }

  const handleButtonHold = useCallback(async () => {
    const state = appStateRef.current;
    addLog('Button HOLD');

    if (state === AppState.IDLE || state === AppState.FEATURE_1_QA || state === AppState.FEATURE_3_NAVIGATION) {
      const isBleConnected =
        isUsingMockBle() ||
        bleService.current?.getConnectionState() === BleConnectionState.CONNECTED;

      previousStateRef.current = state;
      transitionTo(AppState.LISTENING);
      recordedAudioRef.current = null;
      capturedImageRef.current = null;
      audioChunksRef.current = [];

      if (!isBleConnected) {
        addLog('Đang nghe từ Micro điện thoại (Chế độ độc lập)...');
        AccessibilityInfo.announceForAccessibility(
          'Đang lắng nghe từ micro điện thoại, hãy nói lệnh',
        );
        
        // Start listening via phone microphone
        isPhoneRecordingRef.current = false;
        const startPromise = startPhoneListening();
        phoneListeningPromiseRef.current = startPromise;
        startPromise
          .then(started => {
            isPhoneRecordingRef.current = started;
            if (started) {
              addLog('Đang thu âm qua Micro điện thoại...');
            }
            playPttStartFeedback(); // Play beep AFTER recognizer starts
          })
          .catch(err => {
            console.warn('[PhoneSpeech] start error:', err);
            playPttStartFeedback();
          });
      } else {
        AccessibilityInfo.announceForAccessibility('Đang lắng nghe, hãy nói lệnh');
        playPttStartFeedback(); // Play beep immediately for BLE
        
        // Still start phone listening as fallback but don't wait for it
        isPhoneRecordingRef.current = false;
        const startPromise = startPhoneListening();
        phoneListeningPromiseRef.current = startPromise;
        startPromise
          .then(started => { isPhoneRecordingRef.current = started; })
          .catch(err => console.warn('[PhoneSpeech] start error:', err));
      }
    }
  }, [addLog, transitionTo]);

  const handleButtonRelease = useCallback(async () => {
    const state = appStateRef.current;
    addLog('Button RELEASE');
    playPttEndFeedback();
    AccessibilityInfo.announceForAccessibility('Đã gửi lệnh, đang xử lý');

    if (state !== AppState.LISTENING) return;

    transitionTo(AppState.PROCESSING);

    // Wait for startPhoneListening promise to resolve if it is still initializing
    if (phoneListeningPromiseRef.current) {
      try {
        await phoneListeningPromiseRef.current;
      } catch {}
      phoneListeningPromiseRef.current = null;
    }

    // Stop phone listening if active
    let phoneTranscript = '';
    if (isPhoneRecordingRef.current) {
      isPhoneRecordingRef.current = false;
      try {
        const phoneResult = await stopPhoneListening();
        phoneTranscript = phoneResult.text.trim();
        if (phoneTranscript) {
          addLog(`Micro điện thoại nhận diện: "${phoneTranscript}"`);
        }
      } catch (err) {
        console.warn('[PhoneSpeech] stop error:', err);
      }
    }

    try {
      const hasBleAudio = audioChunksRef.current.length > 0;
      const audioData = hasBleAudio
        ? combinePcmChunksToWav(audioChunksRef.current)
        : recordedAudioRef.current ?? 'MOCK_AUDIO_BASE64';
      const imageData = capturedImageRef.current ?? null;

      if (previousStateRef.current === AppState.FEATURE_1_QA) {
        // Feature 1: on-demand audio + image for QA
        let question = phoneTranscript;
        if (!question && hasBleAudio) {
          addLog(
            `Processing QA: received ${audioChunksRef.current.length} audio chunk(s)...`,
          );
          addLog('Transcribing question using PhoWhisper ASR...');
          const asrResult = await transcribeSpeech(audioData);
          question = asrResult.text;
        }

        if (!question) {
          question = 'Trước mặt tôi có gì?';
        }
        addLog(`Transcribed: "${question}"`);

        let qaImage = imageData;
        const isBleConnected =
          isUsingMockBle() ||
          bleService.current?.getConnectionState() === BleConnectionState.CONNECTED;

        if (!qaImage && !isBleConnected) {
          addLog('Không kết nối kính: Chụp ảnh bằng Camera điện thoại...');
          try {
            const phonePhoto = await capturePhonePhoto();
            if (phonePhoto) {
              qaImage = phonePhoto;
              setCurrentImage(phonePhoto);
              addLog('Đã chụp ảnh thành công từ Camera điện thoại');
            }
          } catch (camErr: any) {
            addLog(`Lỗi camera điện thoại: ${camErr?.message}`, 'warn');
          }
        }

        qaImage = qaImage ?? 'MOCK_IMAGE_BASE64';
        addLog('Running SmolVLM2 Visual QA...');
        const onDeviceVlm = await askQAOnDevice(qaImage, question);
        const answer = onDeviceVlm.answer;
        addLog(`Answer: "${answer}"`);

        await speakViaBle(answer, bleService.current);
        transitionTo(AppState.FEATURE_1_QA); // Stay in QA
      } else if (previousStateRef.current === AppState.FEATURE_3_NAVIGATION) {
        // Feature 3: update destination
        let destText = phoneTranscript;
        if (!destText && hasBleAudio) {
          addLog(`Processing Nav Update: received ${audioChunksRef.current.length} audio chunk(s)...`);
          addLog('Transcribing new destination using PhoWhisper ASR...');
          const asrResult = await transcribeSpeech(audioData);
          destText = asrResult.text;
        }
        
        if (destText) {
          addLog(`New destination: "${destText}"`);
          // restart nav with new destination
          startNavigation(destText);
        } else {
          addLog('Không nhận diện được điểm đến mới, tiếp tục dẫn đường cũ.', 'warn');
          transitionTo(AppState.FEATURE_3_NAVIGATION);
        }
      } else {
        // Feature selection from IDLE→LISTENING
        let text = phoneTranscript;
        if (!text && hasBleAudio) {
          addLog(
            `Processing Command: received ${audioChunksRef.current.length} audio chunk(s)...`,
          );
          addLog('Transcribing command using PhoWhisper ASR...');
          const asrResult = await transcribeSpeech(audioData);
          text = asrResult.text;
        }
        addLog(`Transcribed: "${text}"`);

        const feature = matchFeatureKeyword(text);
        if (!feature) {
          addLog('No feature keyword matched', 'warn');
          transitionTo(AppState.IDLE);
          await speakViaBle(
            'Không nhận diện được lệnh, vui lòng thử lại',
            bleService.current,
          );
          return;
        }

        if (feature === 'feature1') {
          transitionTo(AppState.FEATURE_1_QA);
          await playFeatureActivationSound('feature1', bleService.current);
          await speakViaBle('Đã bật tính năng hỏi đáp. Hãy nhấn giữ nút để đặt câu hỏi.', bleService.current);
        } else if (feature === 'feature2') {
          transitionTo(AppState.FEATURE_2_CAPTURE);
          await playFeatureActivationSound('feature2', bleService.current);
          // Feature 2: capture + save
          await captureAndSave();
        } else if (feature === 'feature3') {
          transitionTo(AppState.FEATURE_3_NAVIGATION);
          await playFeatureActivationSound('feature3', bleService.current);
          startNavigation(text);
        }
      }
    } catch (e: any) {
      addLog(`Processing error: ${e?.message ?? e}`, 'error');
      await speakViaBle(
        'Không thể xử lý yêu cầu, vui lòng thử lại',
        bleService.current,
      ).catch(() => {});
      transitionTo(AppState.IDLE);
    } finally {
      // Guaranteed safety exit: ensure the app NEVER remains stuck in PROCESSING
      if (appStateRef.current === AppState.PROCESSING) {
        transitionTo(AppState.IDLE);
      }
    }
  }, [addLog, transitionTo]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleShortPress = useCallback(async () => {
    const now = Date.now();
    if (now - lastShortPressTimeRef.current < 500) {
      lastShortPressTimeRef.current = 0;
      addLog('Button DOUBLE_PRESS → return to IDLE');
      await returnToIdle();
    } else {
      lastShortPressTimeRef.current = now;
      addLog('Button SINGLE_PRESS → wait for double press');
    }
  }, [addLog, returnToIdle]);

  async function captureAndSave() {
    try {
      addLog('Feature 2: capturing image...');
      const isBleConnected =
        isUsingMockBle() ||
        bleService.current?.getConnectionState() === BleConnectionState.CONNECTED;

      let photoToSave = capturedImageRef.current;

      // Request immediate photo capture from ESP32 camera if connected
      if (bleService.current?.getConnectionState() === BleConnectionState.CONNECTED) {
        await bleService.current.sendAudio('CAPTURE').catch(err => {
          console.warn('[BLE] Failed to send CAPTURE to glasses:', err);
        });
      } else if (!isUsingMockBle()) {
        // Phone camera fallback when not connected to glasses
        addLog('Không kết nối kính: Chụp ảnh bằng Camera điện thoại...');
        try {
          const phonePhoto = await capturePhonePhoto();
          if (phonePhoto) {
            photoToSave = phonePhoto;
            setCurrentImage(phonePhoto);
            addLog('Đã chụp ảnh thành công từ Camera điện thoại');
          }
        } catch (camErr: any) {
          addLog(`Lỗi camera điện thoại: ${camErr?.message}`, 'warn');
        }
      }

      const savedFilename = await saveCapturedImage(photoToSave);
      addLog(`Feature 2: image saved to local storage (${savedFilename})`);
      await speakViaBle(
        'Đã hoàn thành, bạn muốn chọn tính năng nào tiếp theo',
        bleService.current,
      ).catch(() => {});
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

  function startNavigation(destinationText?: string) {
    stopNavigation();
    resetCooldowns();
    if (mockService.current && isUsingMockBle()) {
      mockService.current.startAutoImage(NAVIGATION_FRAME_INTERVAL_MS);
    } else if (bleService.current?.getConnectionState() === BleConnectionState.CONNECTED) {
      // Send NAV_START command to ESP32 to start auto 4s camera capture
      bleService.current.sendAudio('NAV_START').catch(err => {
        console.warn('[BLE] Failed to send NAV_START to glasses:', err);
      });
    } else {
      // Standalone phone navigation using phone camera
      addLog('Bắt đầu dẫn đường độc lập bằng Camera điện thoại (4s/khung hình)...');
      capturePhonePhoto()
        .then(photo => {
          if (photo && appStateRef.current === AppState.FEATURE_3_NAVIGATION) {
            setCurrentImage(photo);
            processNavigationImage(photo, destinationText);
          }
        })
        .catch(err => {
          addLog(`Lỗi camera dẫn đường: ${err?.message}`, 'warn');
        });

      navIntervalRef.current = setInterval(async () => {
        if (appStateRef.current !== AppState.FEATURE_3_NAVIGATION) {
          stopNavigation();
          return;
        }
        try {
          const photo = await capturePhonePhoto();
          if (photo && appStateRef.current === AppState.FEATURE_3_NAVIGATION) {
            setCurrentImage(photo);
            await processNavigationImage(photo); // Subsequent frames don't need destinationText
          }
        } catch (err: any) {
          console.warn('[PhoneCamera Nav] capture error:', err);
        }
      }, NAVIGATION_FRAME_INTERVAL_MS);
    }
  }

  // ─── BLE Event Setup & On-Device Model Initialization ────────────────
  useEffect(() => {
    // Activate on-device models packaged in APK assets
    modelRegistry.setModelStatus('phowhisper', 'READY');
    modelRegistry.setModelStatus('smolvlm2', 'READY');
    modelRegistry.setModelStatus('yolo26s', 'READY');
    modelRegistry.setModelStatus('zipdepth', 'READY');

    // Request necessary runtime permissions (Mic, Camera, Bluetooth) on launch
    requestAllAppPermissions().catch(() => {});
    
    // Announce app is ready
    setTimeout(() => {
       speakUrgent('Ứng dụng BSmart đã sẵn sàng. Bạn có thể nhấn giữ nút để ra lệnh.', bleService.current).catch(() => {});
    }, 1000);

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
      audioChunksRef.current.push(audioBase64);
      recordedAudioRef.current = audioBase64;
    });

    const unsubConnection = svc.onConnectionStateChange(state => {
      setConnectionState(state);
      if (state === BleConnectionState.CONNECTED) {
        const name = svc.getConnectedDeviceName?.() || 'BSmart Glasses';
        setConnectedDeviceName(name);
        addLog(`Đã kết nối thành công '${name}'`, 'info');
        AccessibilityInfo.announceForAccessibility(`Đã kết nối thành công '${name}'`);
      } else {
        addLog(`BLE: ${state}`);
        if (state === BleConnectionState.DISCONNECTED) {
          setConnectedDeviceName(null);
          // Only interrupt navigation if we were actively streaming navigation frames.
          // DO NOT interrupt user speech (LISTENING) or AI reasoning (PROCESSING)!
          if (appStateRef.current === AppState.FEATURE_3_NAVIGATION) {
            stopSpeaking();
            stopNavigation();
            transitionTo(AppState.IDLE);
            addLog('Mất kết nối kính trong khi dẫn đường → về trang chủ', 'warn');
          }
        }
      }
    });

    return () => {
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

  const toggleMockBle = useCallback(
    (enabled: boolean) => {
      setUseMockBle(enabled);
      bleService.current = getBleService();
      mockService.current = getMockBleService();
      setIsMockBle(enabled);
      if (!enabled) {
        setConnectedDeviceName(null);
      }
      addLog(`Chuyển chế độ BLE: ${enabled ? 'Mock BLE' : 'Real BLE'}`);
    },
    [addLog],
  );

  const onConnect = useCallback(async () => {
    try {
      if (!isUsingMockBle()) {
        addLog('Đang yêu cầu cấp quyền Bluetooth...');
        const granted = await requestBlePermissions();
        if (!granted) {
          addLog('Chưa có đủ quyền Bluetooth/Vị trí', 'warn');
          AccessibilityInfo.announceForAccessibility('Chưa có đủ quyền Bluetooth');
          await speakViaBle(
            'Chưa được cấp quyền Bluetooth, vui lòng cho phép trong cài đặt ứng dụng',
            bleService.current,
          ).catch(() => {});
          return;
        }

        // If Bluetooth is currently off, prompt user / enable it
        const isEnabled = await bleService.current.isBluetoothEnabled?.();
        if (!isEnabled) {
          addLog('Yêu cầu bật Bluetooth...');
          AccessibilityInfo.announceForAccessibility('Đang yêu cầu bật Bluetooth');
          await bleService.current.enableBluetooth?.();
          addLog('Vui lòng bật Bluetooth trong cài đặt để kết nối kính', 'warn');
          return;
        }
      }

      addLog('Đang tìm và kết nối kính...');
      AccessibilityInfo.announceForAccessibility('Đang kết nối với kính');
      await bleService.current.connect();
      const name = bleService.current?.getConnectedDeviceName?.() || 'BSmart Glasses';
      setConnectedDeviceName(name);
      addLog(`Đã kết nối thành công '${name}'`, 'info');
      AccessibilityInfo.announceForAccessibility(`Đã kết nối thành công '${name}'`);
      await speakViaBle(`Đã kết nối thành công ${name}`, bleService.current).catch(() => {});
      // Start background foreground service to keep connection alive when screen is locked
      await startBackgroundService();
    } catch (e: any) {
      addLog(`Kết nối thất bại: ${e?.message ?? e}`, 'error');
      AccessibilityInfo.announceForAccessibility('Kết nối kính thất bại');
      await speakViaBle('Kết nối kính thất bại', bleService.current).catch(() => {});
    }
  }, [addLog]);

  const onDisconnect = useCallback(async () => {
    try {
      await bleService.current.disconnect();
      await stopBackgroundService();
      setConnectedDeviceName(null);
      AccessibilityInfo.announceForAccessibility('Đã ngắt kết nối kính');
      addLog('Đã ngắt kết nối kính');
    } catch (e: any) {
      addLog(`Disconnect error: ${e?.message ?? e}`, 'error');
    }
  }, [addLog]);

  return {
    appState,
    connectionState,
    connectedDeviceName,
    logs,
    currentImage,
    useMock: isMockBle,
    isMockBle,
    toggleMockBle,
    onButtonHold: handleButtonHold,
    onButtonRelease: handleButtonRelease,
    onShortPress: handleShortPress,
    onMockButtonHold,
    onMockButtonRelease,
    onMockShortPress,
    onMockImage,
    onConnect,
    onDisconnect,
  };
}
