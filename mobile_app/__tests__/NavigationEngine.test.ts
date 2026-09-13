/**
 * NavigationEngine unit tests
 * Tests YOLO+ZipDepth result processing logic.
 */

import {
  processNavigationFrame,
  warningsToVietnamese,
  resetCooldowns,
} from '../src/services/navigation/NavigationEngine';
import { DetectedObject } from '../src/types';

describe('NavigationEngine', () => {
  beforeEach(() => {
    resetCooldowns();
  });

  it('should classify center object correctly', () => {
    const objects: DetectedObject[] = [
      {
        class: 'person',
        confidence: 0.9,
        x: 0.5,
        y: 0.5,
        width: 0.2,
        height: 0.4,
        depthScore: 0.2, // near
      },
    ];
    const warnings = processNavigationFrame(objects);
    expect(warnings).toHaveLength(1);
    expect(warnings[0].position).toBe('Giữa');
    expect(warnings[0].distance).toBe('Gần');
    expect(warnings[0].objectClass).toBe('person');
  });

  it('should classify left object correctly', () => {
    const objects: DetectedObject[] = [
      {
        class: 'car',
        confidence: 0.75,
        x: 0.2, // left
        y: 0.5,
        width: 0.3,
        height: 0.3,
        depthScore: 0.7, // far
      },
    ];
    const warnings = processNavigationFrame(objects);
    expect(warnings[0].position).toBe('Trái');
    expect(warnings[0].distance).toBe('Xa');
  });

  it('should classify right object correctly', () => {
    const objects: DetectedObject[] = [
      {
        class: 'bicycle',
        confidence: 0.8,
        x: 0.8, // right
        y: 0.5,
        width: 0.15,
        height: 0.3,
        depthScore: 0.3, // near
      },
    ];
    const warnings = processNavigationFrame(objects);
    expect(warnings[0].position).toBe('Phải');
  });

  it('should filter low confidence objects', () => {
    const objects: DetectedObject[] = [
      {
        class: 'person',
        confidence: 0.3, // below threshold 0.5
        x: 0.5,
        y: 0.5,
        width: 0.2,
        height: 0.4,
        depthScore: 0.1,
      },
    ];
    const warnings = processNavigationFrame(objects);
    expect(warnings).toHaveLength(0);
  });

  it('should limit to max 2 warnings per frame', () => {
    const objects: DetectedObject[] = [
      { class: 'person', confidence: 0.9, x: 0.1, y: 0.5, width: 0.2, height: 0.4, depthScore: 0.1 },
      { class: 'car', confidence: 0.8, x: 0.5, y: 0.5, width: 0.3, height: 0.3, depthScore: 0.2 },
      { class: 'bicycle', confidence: 0.7, x: 0.9, y: 0.5, width: 0.15, height: 0.3, depthScore: 0.3 },
    ];
    const warnings = processNavigationFrame(objects);
    expect(warnings.length).toBeLessThanOrEqual(2);
  });

  it('should apply cooldown to prevent repetition', () => {
    const objects: DetectedObject[] = [
      {
        class: 'person',
        confidence: 0.9,
        x: 0.5,
        y: 0.5,
        width: 0.2,
        height: 0.4,
        depthScore: 0.2,
      },
    ];

    // First call: should return warning
    const first = processNavigationFrame(objects);
    expect(first).toHaveLength(1);

    // Second call immediately: same object should be in cooldown
    const second = processNavigationFrame(objects);
    expect(second).toHaveLength(0);
  });

  it('should reset cooldowns correctly', () => {
    const objects: DetectedObject[] = [
      {
        class: 'person',
        confidence: 0.9,
        x: 0.5,
        y: 0.5,
        width: 0.2,
        height: 0.4,
        depthScore: 0.2,
      },
    ];

    processNavigationFrame(objects);
    resetCooldowns();

    const after = processNavigationFrame(objects);
    expect(after).toHaveLength(1);
  });

  it('should generate correct Vietnamese text', () => {
    const objects: DetectedObject[] = [
      {
        class: 'person',
        confidence: 0.9,
        x: 0.5,
        y: 0.5,
        width: 0.2,
        height: 0.4,
        depthScore: 0.1,
      },
    ];
    const warnings = processNavigationFrame(objects);
    const text = warningsToVietnamese(warnings);
    expect(text).toContain('người');
    expect(text).toContain('phía trước');
    expect(text).toContain('gần');
    expect(text).toMatch(/^Chú ý,/);
  });

  it('should announce clear path once, then suppress repeated clear-path messages during cooldown', () => {
    const first = warningsToVietnamese([]);
    const second = warningsToVietnamese([]);

    expect(first).toBe('Đường phía trước trống, tiếp tục di chuyển.');
    expect(second).toBe('');
  });

  it('should generate multi-object Vietnamese text', () => {
    const objects: DetectedObject[] = [
      { class: 'person', confidence: 0.9, x: 0.5, y: 0.5, width: 0.2, height: 0.4, depthScore: 0.1 },
      { class: 'car', confidence: 0.85, x: 0.1, y: 0.5, width: 0.3, height: 0.3, depthScore: 0.8 },
    ];
    const warnings = processNavigationFrame(objects);
    const text = warningsToVietnamese(warnings);
    // Should have both objects in one sentence
    expect(text).toContain('người');
    expect(text).toContain('xe hơi');
  });
});
