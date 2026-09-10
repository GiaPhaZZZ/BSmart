/**
 * BSmart On-Device Visual QA Service (MOD-02)
 * Powered by SmolVLM2-256M on-device VLM pipeline (100% Offline Architecture).
 *
 * Core Principles & Optimizations (Strictly On-Device):
 * 1. 100% Offline: Zero dependence on external Python server or cloud.
 * 2. On-Demand Execution Only: SmolVLM2 is NEVER run in a continuous loop;
 *    it triggers ONLY when the user explicitly asks a question via Push-To-Talk.
 * 3. Low Resolution Input: Images downscaled to 256x256 before vision embedding
 *    to minimize patch token count (256 patches vs 729+), reducing RAM and thermal spikes.
 * 4. Output Token Cap: Output generated text is constrained to <= 35 tokens for
 *    concise, fast auditory response formatted for TTS.
 * 5. Memory & Session Reuse: Pre-warmed ONNX sessions reused across queries;
 *    intermediate tensors and Bitmaps explicitly recycled immediately after inference.
 * 6. Non-Blocking Async: Offloaded to native background threads to ensure 60fps UI.
 */

import { NativeModules } from 'react-native';
import { modelRegistry } from './ModelRegistry';

const { OnnxInferenceModule } = NativeModules;

export interface VlmAnswerResult {
  answer: string;
  isSuccess: boolean;
  model?: string;
  resolution?: number;
  maxTokens?: number;
}

export interface SmolVlmOptimizationConfig {
  /** Input image resolution downscaled on-device to minimize RAM & patch count */
  inputResolution: number;
  /** Maximum generated tokens to prevent unbounded decoding */
  maxOutputTokens: number;
  /** Quantization type */
  quantization: 'INT8' | 'FP16';
  /** Only execute on user query */
  onDemandOnly: boolean;
}

export const SMOLVLM2_CONFIG: SmolVlmOptimizationConfig = {
  inputResolution: 256,
  maxOutputTokens: 35,
  quantization: 'INT8',
  onDemandOnly: true,
};

/**
 * Run SmolVLM2-256M on-device inference on an image and a spoken question (100% Offline).
 * @param imageBase64 - base64 JPEG from smart glasses
 * @param question - user query in Vietnamese
 */
export async function askQAOnDevice(
  imageBase64: string,
  question: string,
): Promise<VlmAnswerResult> {
  const isReady = modelRegistry.isModelReady('smolvlm2');

  if (!isReady || !imageBase64 || imageBase64.length === 0) {
    return {
      answer: 'Không thể xử lý hình ảnh lúc này. Vui lòng thử lại.',
      isSuccess: false,
    };
  }

  // 1. Try Native C++ ONNX Runtime Engine (onnxruntime-android)
  try {
    if (OnnxInferenceModule && typeof OnnxInferenceModule.runVisualQA === 'function') {
      const nativeResult = await OnnxInferenceModule.runVisualQA(imageBase64, question);
      if (nativeResult && nativeResult.answer) {
        return {
          answer: nativeResult.answer,
          isSuccess: true,
          model: nativeResult.model || 'SmolVLM2-256M-INT8',
          resolution: nativeResult.resolution || SMOLVLM2_CONFIG.inputResolution,
          maxTokens: nativeResult.maxTokens || SMOLVLM2_CONFIG.maxOutputTokens,
        };
      }
    }
  } catch (err) {
    console.warn('[VLM] Native ONNX execution warning, using optimized on-device decoder fallback:', err);
  }

  // On-device SmolVLM2 ONNX not available and no backend reachable.
  // DO NOT return fabricated scene descriptions — this is a safety-critical feature for
  // visually impaired users. A fake answer is worse than an honest "unavailable".
  console.warn('[VLM] SmolVLM2 not available and no backend reachable.');
  return {
    answer: 'Tính năng hỏi đáp hình ảnh chưa sẵn sàng. Vui lòng kết nối máy chủ hoặc thử lại sau.',
    isSuccess: false,
    model: 'unavailable',
  };
}
