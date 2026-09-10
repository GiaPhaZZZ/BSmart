import {
  base64ToBytes,
  bytesToBase64,
  createWavHeader,
  combinePcmChunksToWav,
} from '../src/services/audio/audioUtils';

describe('AudioUtils', () => {
  it('correctly encodes and decodes Base64 roundtrip', () => {
    const original = new Uint8Array([0, 1, 2, 255, 128, 64, 32, 16]);
    const b64 = bytesToBase64(original);
    const decoded = base64ToBytes(b64);
    expect(decoded).toEqual(original);
  });

  it('creates valid 44-byte WAV header with 16kHz mono 16-bit specs', () => {
    const dataSize = 1000;
    const header = createWavHeader(dataSize, 16000);
    expect(header.length).toBe(44);

    const view = new DataView(header.buffer);
    // RIFF
    expect(String.fromCharCode(header[0], header[1], header[2], header[3])).toBe('RIFF');
    expect(view.getUint32(4, true)).toBe(36 + dataSize);
    // WAVE
    expect(String.fromCharCode(header[8], header[9], header[10], header[11])).toBe('WAVE');
    // Channels = 1
    expect(view.getUint16(22, true)).toBe(1);
    // Sample rate = 16000
    expect(view.getUint32(24, true)).toBe(16000);
    // Bits per sample = 16
    expect(view.getUint16(34, true)).toBe(16);
    // Data subchunk size
    expect(view.getUint32(40, true)).toBe(dataSize);
  });

  it('combines multiple streaming PCM chunks into a complete WAV base64 file', () => {
    const chunk1 = bytesToBase64(new Uint8Array(120).fill(1));
    const chunk2 = bytesToBase64(new Uint8Array(120).fill(2));
    const chunk3 = bytesToBase64(new Uint8Array(120).fill(3));

    const wavBase64 = combinePcmChunksToWav([chunk1, chunk2, chunk3]);
    const wavBytes = base64ToBytes(wavBase64);

    // 44 bytes header + 360 bytes PCM = 404 bytes
    expect(wavBytes.length).toBe(44 + 360);

    const view = new DataView(wavBytes.buffer);
    expect(view.getUint32(40, true)).toBe(360);
    // First byte of PCM chunk 1
    expect(wavBytes[44]).toBe(1);
    // First byte of PCM chunk 2
    expect(wavBytes[44 + 120]).toBe(2);
    // First byte of PCM chunk 3
    expect(wavBytes[44 + 240]).toBe(3);
  });

  it('handles empty chunks gracefully', () => {
    expect(combinePcmChunksToWav([])).toBe('');
  });
});
