/**
 * BSmart On-Device Model Registry (MOD-04)
 * Central manager for tracking, verifying, and configuring on-device AI models.
 * Covers:
 *   - PhoWhisper-tiny (ASR - MOD-01)
 *   - SmolVLM2-256M (Visual QA - MOD-02)
 *   - YOLO26s + ZipDepth (Obstacle Awareness & Depth - MOD-03)
 */

export type ModelStatus =
  | 'UNINITIALIZED'
  | 'LOADING'
  | 'READY'
  | 'MOCK_FALLBACK'
  | 'ERROR';

export interface ModelInfo {
  id: string;
  name: string;
  task: 'asr' | 'vlm' | 'detection' | 'depth';
  filename: string;
  inputShape: number[];
  status: ModelStatus;
  sizeBytes?: number;
  description: string;
}

export interface ModelRegistryConfig {
  asrEnabled: boolean;
  vlmEnabled: boolean;
  navigationEnabled: boolean;
}

const DEFAULT_MODELS: Record<string, ModelInfo> = {
  phowhisper: {
    id: 'phowhisper',
    name: 'PhoWhisper-tiny',
    task: 'asr',
    filename: 'phowhisper/phowhisper_encoder.onnx',
    inputShape: [1, 80, 3000],
    status: 'UNINITIALIZED', // Set to READY only after Native module confirms model loaded
    description: 'Vietnamese Speech-to-Text for voice control and queries',
  },
  smolvlm2: {
    id: 'smolvlm2',
    name: 'SmolVLM2-256M',
    task: 'vlm',
    filename: 'smolvlm2/smolvlm2_vision.onnx',
    inputShape: [1, 3, 384, 384],
    status: 'UNINITIALIZED', // Set to READY only after Native module confirms model loaded
    description: 'Multimodal Vision-Language model for Scene QA',
  },
  yolo26s: {
    id: 'yolo26s',
    name: 'YOLO26s',
    task: 'detection',
    filename: 'navigation/yolo26s.onnx',
    inputShape: [1, 3, 640, 640],
    status: 'UNINITIALIZED', // Set to READY only after Native module confirms model loaded
    description: 'Real-time obstacle and person detector',
  },
  zipdepth: {
    id: 'zipdepth',
    name: 'ZipDepth-base-npu',
    task: 'depth',
    filename: 'navigation/zipdepth.onnx',
    inputShape: [1, 3, 384, 384],
    status: 'UNINITIALIZED', // Set to READY only after Native module confirms model loaded
    description: 'Monocular relative inverse depth estimator',
  },
};


export class ModelRegistry {
  private static instance: ModelRegistry;
  private models: Map<string, ModelInfo> = new Map();

  private constructor() {
    this.resetToDefaults();
  }

  public static getInstance(): ModelRegistry {
    if (!ModelRegistry.instance) {
      ModelRegistry.instance = new ModelRegistry();
    }
    return ModelRegistry.instance;
  }

  public resetToDefaults(): void {
    this.models.clear();
    Object.entries(DEFAULT_MODELS).forEach(([key, info]) => {
      this.models.set(key, { ...info });
    });
  }

  public getModel(id: string): ModelInfo | undefined {
    return this.models.get(id);
  }

  public setModelStatus(id: string, status: ModelStatus): void {
    const model = this.models.get(id);
    if (model) {
      model.status = status;
    }
  }

  public isModelReady(id: string): boolean {
    const model = this.models.get(id);
    return model?.status === 'READY';
  }

  public getAllModels(): ModelInfo[] {
    return Array.from(this.models.values());
  }
}

export const modelRegistry = ModelRegistry.getInstance();
