/**
 * BSmart On-Device AI & Navigation Inference Tests (MOD-01..04)
 */

declare const require: (moduleName: string) => any;
declare const __dirname: string;

const fs = require('fs');
const path = require('path');

import {
  isInferenceAvailable,
  setInferenceAvailable,
  runInference,
} from '../src/services/navigation/OnDeviceInference';
import {
  processNavigationFrame,
  warningsToVietnamese,
  resetCooldowns,
} from '../src/services/navigation/NavigationEngine';
import {
  matchVoiceCommand,
  normalizeVietnameseText,
  transcribeAudioOnDevice,
} from '../src/services/ai/OnDeviceAsrService';
import { askQAOnDevice } from '../src/services/ai/OnDeviceVlmService';
import {
  askQAGgufOnDevice,
  getGgufVlmStatus,
  isGgufVlmModuleAvailable,
} from '../src/services/ai/OnDeviceGgufVlmService';
import { modelRegistry } from '../src/services/ai/ModelRegistry';
import { AppState } from '../src/types';

describe('OnDeviceInference Service (MOD-03 & MOD-04)', () => {
  beforeEach(() => {
    setInferenceAvailable(true);
    modelRegistry.resetToDefaults();
    resetCooldowns();
  });

  afterEach(() => {
    setInferenceAvailable(null);
  });

  test('isInferenceAvailable returns false by default when models are uninitialized', () => {
    setInferenceAvailable(null);
    expect(isInferenceAvailable()).toBe(false);
  });

  test('isInferenceAvailable returns true when models are marked READY', () => {
    setInferenceAvailable(null);
    modelRegistry.setModelStatus('yolo26s', 'READY');
    modelRegistry.setModelStatus('zipdepth', 'READY');
    expect(isInferenceAvailable()).toBe(true);
  });

  test('setInferenceAvailable(false) successfully disables inference', () => {
    setInferenceAvailable(false);
    expect(isInferenceAvailable()).toBe(false);
  });

  test('runInference returns not ready when inference is disabled', async () => {
    setInferenceAvailable(false);
    const result = await runInference('dummyBase64Data');
    expect(result.isReady).toBe(false);
    expect(result.objects).toHaveLength(0);
  });

  test('runInference handles empty or invalid base64 input gracefully', async () => {
    const result = await runInference('   ');
    expect(result.isReady).toBe(true);
    expect(result.objects).toHaveLength(0);
  });

  test('runInference returns detected objects with valid coordinates and depth', async () => {
    const result = await runInference('validMockBase64JPEGData==');
    expect(result.isReady).toBe(true);
    expect(result.objects.length).toBeGreaterThan(0);

    result.objects.forEach(obj => {
      expect(obj.class).toBeDefined();
      expect(obj.confidence).toBeGreaterThan(0.5);
      expect(obj.x).toBeGreaterThanOrEqual(0);
      expect(obj.x).toBeLessThanOrEqual(1);
      expect(obj.depthScore).toBeDefined();
    });
  });

  test('runInference does not assign one global depth value to all native objects', () => {
    const source = fs.readFileSync(
      path.join(__dirname, '../src/services/navigation/OnDeviceInference.ts'),
      'utf8',
    );

    expect(source).not.toContain('relativeDepthMean');
    expect(source).not.toMatch(/depthScore:\s*relativeDepthMean/);
    expect(source).not.toMatch(/runDepthEstimation\(imageBase64\)/);
  });

  test('integration: runInference output feeds directly into NavigationEngine', async () => {
    const inferenceResult = await runInference('validCameraFrameData');
    const warnings = processNavigationFrame(inferenceResult.objects);
    expect(warnings.length).toBeGreaterThan(0);

    const speechText = warningsToVietnamese(warnings);
    expect(speechText).toContain('Chú ý');
    expect(speechText.length).toBeGreaterThan(10);
  });
});

describe('OnDeviceAsrService (MOD-01)', () => {
  test('normalizeVietnameseText strips accents and converts to lowercase', () => {
    expect(normalizeVietnameseText('Tính Năng 1')).toBe('tinh nang 1');
    expect(normalizeVietnameseText('Hỏi Đáp')).toBe('hoi dap');
  });

  test('matchVoiceCommand recognizes Feature 1 QA triggers', () => {
    expect(matchVoiceCommand('tính năng 1')).toBe(AppState.FEATURE_1_QA);
    expect(matchVoiceCommand('Mở tính năng một')).toBe(AppState.FEATURE_1_QA);
    expect(matchVoiceCommand('Bật GPT')).toBe(AppState.FEATURE_1_QA);
  });

  test('matchVoiceCommand recognizes Feature 2 Capture triggers', () => {
    expect(matchVoiceCommand('tính năng 2')).toBe(AppState.FEATURE_2_CAPTURE);
    expect(matchVoiceCommand('chụp ảnh')).toBe(AppState.FEATURE_2_CAPTURE);
  });

  test('matchVoiceCommand recognizes Feature 3 Navigation triggers', () => {
    expect(matchVoiceCommand('tính năng 3')).toBe(AppState.FEATURE_3_NAVIGATION);
    expect(matchVoiceCommand('bật chế độ dẫn đường')).toBe(AppState.FEATURE_3_NAVIGATION);
  });

  test.each([
    ['1', AppState.FEATURE_1_QA],
    ['s\u1ed1 1', AppState.FEATURE_1_QA],
    ['m\u1ed9t', AppState.FEATURE_1_QA],
    ['s\u1ed1 m\u1ed9t', AppState.FEATURE_1_QA],
    ['t\u00ednh n\u0103ng 1', AppState.FEATURE_1_QA],
    ['t\u00ednh n\u0103ng m\u1ed9t', AppState.FEATURE_1_QA],
    ['t\u00ednh n\u0103ng s\u1ed1 m\u1ed9t', AppState.FEATURE_1_QA],
    ['t\u00ednh n\u0103ng s\u1ed1 1', AppState.FEATURE_1_QA],
    ['h\u1ecfi \u0111\u00e1p', AppState.FEATURE_1_QA],
    ['gpt', AppState.FEATURE_1_QA],
    ['2', AppState.FEATURE_2_CAPTURE],
    ['hai', AppState.FEATURE_2_CAPTURE],
    ['s\u1ed1 2', AppState.FEATURE_2_CAPTURE],
    ['s\u1ed1 hai', AppState.FEATURE_2_CAPTURE],
    ['ch\u1ee5p \u1ea3nh', AppState.FEATURE_2_CAPTURE],
    ['t\u00ednh n\u0103ng 2', AppState.FEATURE_2_CAPTURE],
    ['t\u00ednh n\u0103ng s\u1ed1 2', AppState.FEATURE_2_CAPTURE],
    ['t\u00ednh n\u0103ng s\u1ed1 hai', AppState.FEATURE_2_CAPTURE],
    ['t\u00ednh n\u0103ng hay', AppState.FEATURE_2_CAPTURE],
    ['t\u00ednh n\u0103ng s\u1ed1 hay', AppState.FEATURE_2_CAPTURE],
    ['3', AppState.FEATURE_3_NAVIGATION],
    ['ba', AppState.FEATURE_3_NAVIGATION],
    ['s\u1ed1 3', AppState.FEATURE_3_NAVIGATION],
    ['s\u1ed1 ba', AppState.FEATURE_3_NAVIGATION],
    ['t\u00ednh n\u0103ng 3', AppState.FEATURE_3_NAVIGATION],
    ['t\u00ednh n\u0103ng s\u1ed1 3', AppState.FEATURE_3_NAVIGATION],
    ['t\u00ednh n\u0103ng ba', AppState.FEATURE_3_NAVIGATION],
    ['t\u00ednh n\u0103ng s\u1ed1 ba', AppState.FEATURE_3_NAVIGATION],
    ['t\u00ecm \u0111\u01b0\u1eddng', AppState.FEATURE_3_NAVIGATION],
  ])('matchVoiceCommand recognizes recording alias "%s"', (text, expected) => {
    expect(matchVoiceCommand(text)).toBe(expected);
  });

  test('matchVoiceCommand returns null on unknown text', () => {
    expect(matchVoiceCommand('hôm nay trời đẹp quá')).toBeNull();
  });

  test('transcribeAudioOnDevice returns empty transcription when uninitialized or unwired', async () => {
    const res = await transcribeAudioOnDevice('dGVzdGF1ZGlv');
    expect(res.confidence).toBe(0);
    expect(res.matchedState).toBeNull();
    expect(res.text).toBe('');
  });
});

describe('OnDeviceVlmService (MOD-02)', () => {
  test('askQAOnDevice handles empty image input', async () => {
    const res = await askQAOnDevice('', 'Mô tả ảnh');
    expect(res.isSuccess).toBe(false);
    expect(res.answer).toContain('Không thể xử lý');
  });

  test('askQAOnDevice returns honest unavailable status when model is uninitialized', async () => {
    const res = await askQAOnDevice('mockImage', 'Trước mặt tôi là gì?');
    expect(res.isSuccess).toBe(false);
    expect(res.answer).toContain('Mô hình AI chưa sẵn sàng');
  });

  test('askQAOnDevice returns offline unavailable message when model is READY but native module absent', async () => {
    modelRegistry.setModelStatus('smolvlm2', 'READY');
    const res = await askQAOnDevice('mockImage', 'Trước mặt tôi là gì?');
    expect(res.isSuccess).toBe(false);
    expect(res.answer).toContain('chưa sẵn sàng');
  });

  test('model registry includes separate SmolVLM2 GGUF entry', () => {
    const ggufModel = modelRegistry.getModel('smolvlm2_gguf');
    expect(ggufModel?.filename).toContain('SmolVLM2-256M-Video-Instruct-Q8_0.gguf');
    expect(ggufModel?.status).toBe('UNINITIALIZED');
  });

  test('GGUF service reports unavailable honestly when native module is absent', async () => {
    expect(isGgufVlmModuleAvailable()).toBe(false);
    const status = await getGgufVlmStatus();
    expect(status.runtimeAvailable).toBe(false);
    expect(status.modelLoaded).toBe(false);
    expect(status.message).toContain('not available');
  });

  test('GGUF service handles empty image input without fake answer', async () => {
    const res = await askQAGgufOnDevice('', 'What is in front of me?');
    expect(res.isSuccess).toBe(false);
    expect(res.error).toBe('INVALID_IMAGE');
    expect(res.answer).toContain('Không thể xử lý');
  });
});
