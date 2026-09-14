import { NativeModules, Platform } from 'react-native';

const { LlamaCppVlmModule } = NativeModules;

export interface GgufVlmTimings {
  image_preprocessing_ms?: number;
  vision_encoding_ms?: number;
  prompt_processing_ms?: number;
  generation_ms?: number;
  total_vlm_ms?: number;
  model_load_ms?: number;
}

export interface GgufVlmOptions {
  modelPath?: string;
  mmprojPath?: string;
  contextSize?: number;
  threads?: number;
  maxTokens?: number;
  temperature?: number;
}

export interface GgufVlmResult {
  answer: string;
  isSuccess: boolean;
  model?: string;
  runtime?: string;
  quantization?: string;
  maxTokens?: number;
  generatedTokens?: number;
  tokensPerSecond?: number;
  timeToFirstTokenMs?: number;
  timings?: GgufVlmTimings;
  error?: string;
  message?: string;
}

export interface GgufVlmStatus {
  nativeLibraryLoaded: boolean;
  runtimeAvailable: boolean;
  compiledWithLlamaCpp: boolean;
  modelLoaded: boolean;
  runtime?: string;
  model?: string;
  modelPath?: string;
  mmprojPath?: string;
  defaultModelPath?: string;
  defaultMmprojPath?: string;
  defaultModelExists?: boolean;
  defaultMmprojExists?: boolean;
  error?: string;
  message?: string;
}

const DEFAULT_OPTIONS: Required<Pick<GgufVlmOptions, 'contextSize' | 'threads' | 'maxTokens' | 'temperature'>> = {
  contextSize: 2048,
  threads: 2,
  maxTokens: 48,
  temperature: 0,
};

let loadPromise: Promise<GgufVlmStatus> | null = null;

export function isGgufVlmModuleAvailable(): boolean {
  return Platform.OS === 'android' && !!LlamaCppVlmModule;
}

export async function getGgufVlmStatus(): Promise<GgufVlmStatus> {
  if (!isGgufVlmModuleAvailable() || typeof LlamaCppVlmModule.getStatus !== 'function') {
    return {
      nativeLibraryLoaded: false,
      runtimeAvailable: false,
      compiledWithLlamaCpp: false,
      modelLoaded: false,
      runtime: 'unavailable',
      message: 'LlamaCppVlmModule is not available on this platform/build.',
    };
  }
  return LlamaCppVlmModule.getStatus();
}

export async function loadGgufVlmModel(options: GgufVlmOptions = {}): Promise<GgufVlmStatus> {
  if (!isGgufVlmModuleAvailable() || typeof LlamaCppVlmModule.loadModel !== 'function') {
    throw new Error('LlamaCppVlmModule.loadModel is not available.');
  }

  if (loadPromise) {
    return loadPromise;
  }

  const merged = { ...DEFAULT_OPTIONS, ...options };
  const nextLoad = LlamaCppVlmModule.loadModel(
    options.modelPath ?? '',
    options.mmprojPath ?? '',
    merged,
  ).then((status: GgufVlmStatus) => {
    if (!status.runtimeAvailable || !status.modelLoaded) {
      throw new Error(status.message || status.error || 'SmolVLM2 GGUF failed to load.');
    }
    return status;
  }).catch((err: Error) => {
    loadPromise = null;
    throw err;
  });
  loadPromise = nextLoad;
  return nextLoad;
}

export async function unloadGgufVlmModel(): Promise<void> {
  loadPromise = null;
  if (isGgufVlmModuleAvailable() && typeof LlamaCppVlmModule.unloadModel === 'function') {
    await LlamaCppVlmModule.unloadModel();
  }
}

export async function askQAGgufOnDevice(
  imageBase64: string,
  question: string,
  options: GgufVlmOptions = {},
): Promise<GgufVlmResult> {
  if (!imageBase64 || imageBase64.trim().length === 0) {
    return {
      answer: 'Không thể xử lý hình ảnh lúc này. Vui lòng thử lại.',
      isSuccess: false,
      error: 'INVALID_IMAGE',
    };
  }
  if (!question || question.trim().length === 0) {
    return {
      answer: 'Tôi chưa nghe rõ câu hỏi. Vui lòng hỏi lại.',
      isSuccess: false,
      error: 'EMPTY_PROMPT',
    };
  }

  try {
    const merged = { ...DEFAULT_OPTIONS, ...options };
    await loadGgufVlmModel(merged);
    const nativeResult: GgufVlmResult = await LlamaCppVlmModule.runVisualQABase64(
      imageBase64,
      question,
      merged,
    );
    if (!nativeResult?.isSuccess || !nativeResult.answer) {
      return {
        answer: nativeResult?.message || 'Mô hình AI chưa sẵn sàng. Vui lòng thử lại.',
        isSuccess: false,
        model: nativeResult?.model,
        runtime: nativeResult?.runtime,
        error: nativeResult?.error || 'VLM_FAILED',
        message: nativeResult?.message,
        timings: nativeResult?.timings,
      };
    }
    return nativeResult;
  } catch (err: any) {
    console.error('[Feature1][GGUFVLM] Native llama.cpp inference failed:', err);
    return {
      answer: 'Mô hình AI chưa sẵn sàng. Vui lòng thử lại.',
      isSuccess: false,
      model: 'SmolVLM2-256M-GGUF',
      runtime: 'llama.cpp/libmtmd',
      error: 'NATIVE_VLM_ERROR',
      message: err?.message,
    };
  }
}

export async function runGgufVlmWarmBenchmark(
  imageBase64: string,
  prompts: string[],
  runs = 10,
  options: GgufVlmOptions = {},
): Promise<{ results: GgufVlmResult[]; summary: Record<string, number> }> {
  if (!isGgufVlmModuleAvailable() || typeof LlamaCppVlmModule.benchmarkVisualQABase64 !== 'function') {
    throw new Error('LlamaCppVlmModule.benchmarkVisualQABase64 is not available.');
  }
  await loadGgufVlmModel(options);
  return LlamaCppVlmModule.benchmarkVisualQABase64(
    imageBase64,
    prompts,
    runs,
    { ...DEFAULT_OPTIONS, ...options },
  );
}
