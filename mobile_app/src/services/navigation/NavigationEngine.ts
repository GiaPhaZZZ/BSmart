/**
 * BSmart Navigation Rule Engine
 * Processes YOLO26s + ZipDepth outputs into Vietnamese obstacle warnings.
 * See: docs/requirement.md §7.1, SKILL.md §4.C, src/constants/navigationRules.ts
 */

import {
  DetectedObject,
  NavigationWarning,
} from '../../types';
import {
  HORIZONTAL_LEFT_THRESHOLD,
  HORIZONTAL_RIGHT_THRESHOLD,
  DEPTH_NEAR_THRESHOLD,
  YOLO_CONFIDENCE_THRESHOLD,
  MAX_WARNINGS_PER_FRAME,
  WARNING_COOLDOWN_MS,
  PRIORITY_OBJECT_CLASSES,
  CLASS_NAME_VI,
} from '../../constants/navigationRules';

// Cooldown tracker: cooldownKey → last-announced timestamp
const cooldownMap = new Map<string, number>();

/**
 * Determine horizontal position label from normalized center X
 */
function getPosition(
  centerX: number,
): 'Trái' | 'Giữa' | 'Phải' {
  if (centerX < HORIZONTAL_LEFT_THRESHOLD) return 'Trái';
  if (centerX > HORIZONTAL_RIGHT_THRESHOLD) return 'Phải';
  return 'Giữa';
}

/**
 * Determine distance label from depth score
 * ZipDepth: lower value = closer to camera
 */
function getDistance(depthScore: number): 'Gần' | 'Xa' {
  return depthScore < DEPTH_NEAR_THRESHOLD ? 'Gần' : 'Xa';
}

/**
 * Compute priority score for an object (lower = higher priority)
 * Priority: center > near > confidence > class priority
 */
function getPriority(obj: DetectedObject): number {
  const positionScore = obj.x > HORIZONTAL_LEFT_THRESHOLD && obj.x < HORIZONTAL_RIGHT_THRESHOLD ? 0 : 1;
  const distanceScore = obj.depthScore < DEPTH_NEAR_THRESHOLD ? 0 : 1;
  const classScore = PRIORITY_OBJECT_CLASSES.indexOf(obj.class);
  const classPriority = classScore === -1 ? PRIORITY_OBJECT_CLASSES.length : classScore;
  const confidenceScore = 1 - obj.confidence;

  // Weight: position > distance > class > confidence
  return positionScore * 1000 + distanceScore * 100 + classPriority * 10 + confidenceScore;
}

/**
 * Process detected objects and produce navigation warnings.
 * Applies cooldown to prevent repetitive announcements.
 *
 * @param objects - objects detected by YOLO26s with ZipDepth scores
 * @returns array of NavigationWarning (max 2, after cooldown filtering)
 */
export function processNavigationFrame(
  objects: DetectedObject[],
): NavigationWarning[] {
  const now = Date.now();

  // Filter by confidence threshold
  const confident = objects.filter(
    obj => obj.confidence >= YOLO_CONFIDENCE_THRESHOLD,
  );

  // Sort by priority
  const sorted = [...confident].sort(
    (a, b) => getPriority(a) - getPriority(b),
  );

  const warnings: NavigationWarning[] = [];

  for (const obj of sorted) {
    if (warnings.length >= MAX_WARNINGS_PER_FRAME) break;

    const position = getPosition(obj.x);
    const distance = getDistance(obj.depthScore);
    const cooldownKey = `${obj.class}|${position}|${distance}`;

    // Check cooldown
    const lastAnnounced = cooldownMap.get(cooldownKey);
    if (lastAnnounced && now - lastAnnounced < WARNING_COOLDOWN_MS) {
      continue; // Still in cooldown, skip
    }

    // Update cooldown
    cooldownMap.set(cooldownKey, now);

    warnings.push({
      objectClass: obj.class,
      position,
      distance,
      cooldownKey,
    });
  }

  return warnings;
}

/**
 * Convert NavigationWarning array to Vietnamese speech text
 * Template: "Lưu ý, có {class} ở {vị trí}, {gần/xa}"
 */
export function warningsToVietnamese(warnings: NavigationWarning[]): string {
  if (warnings.length === 0) return '';

  const parts = warnings.map(w => {
    const className = CLASS_NAME_VI[w.objectClass] ?? w.objectClass;
    const positionText = w.position === 'Giữa' ? 'phía trước' : `bên ${w.position.toLowerCase()}`;
    return `có ${className} ở ${positionText}, ${w.distance.toLowerCase()}`;
  });

  return `Lưu ý, ${parts.join('; ')}`;
}

/**
 * Reset cooldown tracker (e.g., when exiting navigation mode)
 */
export function resetCooldowns(): void {
  cooldownMap.clear();
}
