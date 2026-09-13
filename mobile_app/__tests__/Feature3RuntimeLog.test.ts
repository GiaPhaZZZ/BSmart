import {
  estimateBase64Bytes,
  formatFeature3FrameLog,
  formatFeature3ObjectLog,
  formatFeature3WarningLog,
} from '../src/services/navigation/Feature3RuntimeLog';

describe('Feature3RuntimeLog', () => {
  test('estimates raw bytes from plain and data-uri base64 payloads', () => {
    expect(estimateBase64Bytes('YWJjZA==')).toBe(4);
    expect(estimateBase64Bytes('data:image/jpeg;base64,YWJjZA==')).toBe(4);
  });

  test('formats frame received and dropped logs', () => {
    expect(formatFeature3FrameLog('received', 'phone', 'YWJjZA==')).toBe(
      '[F3] frame received source=phone bytes=4',
    );
    expect(formatFeature3FrameLog('dropped', 'ble', 'YWJjZA==', 'inference_busy')).toBe(
      '[F3] frame dropped source=ble bytes=4 reason=inference_busy',
    );
  });

  test('formats object depth logs without raw tensor data', () => {
    expect(
      formatFeature3ObjectLog({
        class: 'chair',
        confidence: 0.93761,
        x: 0.29091,
        y: 0.87612,
        width: 0.20622,
        height: 0.24674,
        depthScore: 0.080225,
      }),
    ).toBe(
      '[F3] object chair conf=0.938 bbox=[x=0.2909,y=0.8761,w=0.2062,h=0.2467] depth=0.080225',
    );
  });

  test('formats empty and populated warning logs', () => {
    expect(formatFeature3WarningLog('')).toBe('[F3] warning=<none>');
    expect(formatFeature3WarningLog('Chú ý, có cái ghế ở phía trước, gần')).toBe(
      '[F3] warning=Chú ý, có cái ghế ở phía trước, gần',
    );
  });
});
