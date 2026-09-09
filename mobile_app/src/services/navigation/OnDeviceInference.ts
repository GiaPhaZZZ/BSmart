/**
 * BSmart On-Device Inference Service
 *
 * INTEGRATION STATUS: BLOCKED
 *
 * The Python AI pipeline (Funtion2_autopilot.py) runs YOLO26s + ZipDepth as a
 * Python process using:
 *   - ultralytics YOLO26s  → object detection (yolo26s.pt, PyTorch format)
 *   - ZipDepth             → monocular depth (zipdepth_base_npu.pth, PyTorch)
 *
 * Neither model has been exported to ONNX or TFLite format. There is no
 * on-device Android ML runtime configured in this project.
 *
 * WHAT IS NEEDED TO UNBLOCK:
 * 1. Export yolo26s.pt to ONNX:
 *      from ultralytics import YOLO
 *      YOLO('yolo26s.pt').export(format='onnx', imgsz=640)
 *
 * 2. Export ZipDepth to ONNX:
 *      python ZipDepth/scripts/export.py --checkpoint zipdepth_base_npu.pth \
 *             --output zipdepth.onnx
 *
 * 3. Add ONNX Runtime Mobile (or TFLite) dependency to Android:
 *      compile 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'
 *      OR npm install onnxruntime-react-native
 *
 * 4. Bundle model files in Android app assets.
 *
 * 5. Implement InferenceSession loading and image preprocessing here.
 *
 * CURRENT BEHAVIOR:
 * Until models are available, this module exports a typed interface so that
 * useAppStateMachine.ts can be updated to call real inference without
 * architectural changes.
 *
 * See: Funtion2_autopilot.py (Python reference implementation)
 * See: docs/requirement.md §7, §2.3
 */

import { DetectedObject } from '../../types';

export interface InferenceResult {
  objects: DetectedObject[];
  isReady: boolean;
}

/**
 * Run YOLO26s + ZipDepth inference on a base64 JPEG image.
 *
 * BLOCKED: Returns empty result until model exports and runtime are available.
 *
 * @param imageBase64 - base64 encoded JPEG from glasses camera
 * @returns InferenceResult with detected objects and depth scores
 */
export async function runInference(
  imageBase64: string,
): Promise<InferenceResult> {
  // BLOCKED: on-device inference not available.
  // Model files (yolo26s.onnx, zipdepth.onnx) are not yet exported.
  // No ONNX Runtime or TFLite dependency is configured.
  //
  // Return isReady=false so the caller can handle gracefully.
  console.log(
    '[Inference] BLOCKED — models not available. Image length:',
    imageBase64.length,
  );
  return { objects: [], isReady: false };
}

/**
 * Check whether on-device inference is available.
 * Returns false until models and runtime are bundled.
 */
export function isInferenceAvailable(): boolean {
  return false;
}
