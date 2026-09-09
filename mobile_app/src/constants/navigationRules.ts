/**
 * BSmart Constants: Navigation Rule Engine Configuration
 * See: docs/requirement.md §7.1
 */

// Horizontal position thresholds (relative to image width)
export const HORIZONTAL_LEFT_THRESHOLD = 0.33;
export const HORIZONTAL_RIGHT_THRESHOLD = 0.66;

// Depth threshold: values below this are considered NEAR
// ZipDepth returns relative depth; lower = closer to camera
// Calibrate this value against real hardware
export const DEPTH_NEAR_THRESHOLD = 0.4;

// YOLO detection confidence threshold
export const YOLO_CONFIDENCE_THRESHOLD = 0.5;

// Max number of warnings to announce per frame
export const MAX_WARNINGS_PER_FRAME = 2;

// Cooldown duration per (class + position + distance) combo, in milliseconds
export const WARNING_COOLDOWN_MS = 8000;

// Navigation frame capture interval (ms)
export const NAVIGATION_FRAME_INTERVAL_MS = 4000;

// High-priority object classes (order = priority)
export const PRIORITY_OBJECT_CLASSES: string[] = [
  'person',
  'car',
  'bicycle',
  'motorcycle',
  'truck',
  'bus',
  'stairs',
  'chair',
  'table',
];

// Vietnamese class name mapping
export const CLASS_NAME_VI: Record<string, string> = {
  person: 'người',
  car: 'xe hơi',
  bicycle: 'xe đạp',
  motorcycle: 'xe máy',
  truck: 'xe tải',
  bus: 'xe buýt',
  stairs: 'bậc thang',
  chair: 'ghế',
  table: 'bàn',
};
