import { DetectedObject } from '../../types';

export function estimateBase64Bytes(imageBase64: string): number {
  const payload = imageBase64.trim();
  if (!payload) return 0;
  const commaIndex = payload.indexOf(',');
  const base64Payload = commaIndex >= 0 ? payload.slice(commaIndex + 1) : payload;
  const padding = base64Payload.endsWith('==') ? 2 : base64Payload.endsWith('=') ? 1 : 0;
  return Math.max(0, Math.floor((base64Payload.length * 3) / 4) - padding);
}

export function formatFeature3FrameLog(
  event: 'received' | 'dropped',
  source: string,
  imageBase64: string,
  reason?: string,
): string {
  const reasonText = reason ? ` reason=${reason}` : '';
  return `[F3] frame ${event} source=${source} bytes=${estimateBase64Bytes(imageBase64)}${reasonText}`;
}

export function formatFeature3ObjectLog(obj: DetectedObject): string {
  return `[F3] object ${obj.class} conf=${formatNumber(obj.confidence, 3)} ` +
    `bbox=[x=${formatNumber(obj.x, 4)},y=${formatNumber(obj.y, 4)},` +
    `w=${formatNumber(obj.width, 4)},h=${formatNumber(obj.height, 4)}] ` +
    `depth=${formatNumber(obj.depthScore, 6)}`;
}

export function formatFeature3WarningLog(warningText: string): string {
  return `[F3] warning=${warningText || '<none>'}`;
}

function formatNumber(value: number, digits: number): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'invalid';
}
