/**
 * BSmart Audio Utilities
 * Accumulates streaming raw PCM chunks from ESP32 BLE into standard WAV format.
 * Format: 16kHz, 16-bit Mono PCM.
 */

const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const LOOKUP = new Uint8Array(256);
for (let i = 0; i < CHARS.length; i++) {
  LOOKUP[CHARS.charCodeAt(i)] = i;
}

/**
 * Decode Base64 string into Uint8Array
 */
export function base64ToBytes(base64: string): Uint8Array {
  const clean = base64.trim().replace(/[\r\n\t ]/g, '');
  const len = clean.length;
  if (len === 0) return new Uint8Array(0);

  let bufferLength = Math.floor(len * 0.75);
  if (clean[len - 1] === '=') {
    bufferLength--;
    if (clean[len - 2] === '=') {
      bufferLength--;
    }
  }

  const bytes = new Uint8Array(bufferLength);
  let p = 0;

  for (let i = 0; i < len; i += 4) {
    const encoded1 = LOOKUP[clean.charCodeAt(i)];
    const encoded2 = LOOKUP[clean.charCodeAt(i + 1)];
    const encoded3 = LOOKUP[clean.charCodeAt(i + 2)];
    const encoded4 = LOOKUP[clean.charCodeAt(i + 3)];

    bytes[p++] = (encoded1 << 2) | (encoded2 >> 4);
    if (p < bufferLength) {
      bytes[p++] = ((encoded2 & 15) << 4) | (encoded3 >> 2);
    }
    if (p < bufferLength) {
      bytes[p++] = ((encoded3 & 3) << 6) | (encoded4 & 63);
    }
  }

  return bytes;
}

/**
 * Encode Uint8Array into Base64 string
 */
export function bytesToBase64(bytes: Uint8Array): string {
  const len = bytes.length;
  let base64 = '';

  for (let i = 0; i < len; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < len ? bytes[i + 1] : 0;
    const b2 = i + 2 < len ? bytes[i + 2] : 0;

    base64 += CHARS[b0 >> 2];
    base64 += CHARS[((b0 & 3) << 4) | (b1 >> 4)];
    base64 += i + 1 < len ? CHARS[((b1 & 15) << 2) | (b2 >> 6)] : '=';
    base64 += i + 2 < len ? CHARS[b2 & 63] : '=';
  }

  return base64;
}

/**
 * Encode string into Base64 string
 */
export function stringToBase64(str: string): string {
  const bytes = new Uint8Array(str.length);
  for (let i = 0; i < str.length; i++) {
    bytes[i] = str.charCodeAt(i) & 0xff;
  }
  return bytesToBase64(bytes);
}

/**
 * Create standard 44-byte WAV header for 16kHz, 16-bit Mono PCM.
 */
export function createWavHeader(dataSize: number, sampleRate = 16000): Uint8Array {
  const header = new Uint8Array(44);
  const view = new DataView(header.buffer);

  // "RIFF"
  header[0] = 0x52;
  header[1] = 0x49;
  header[2] = 0x46;
  header[3] = 0x46;
  view.setUint32(4, 36 + dataSize, true); // ChunkSize
  // "WAVE"
  header[8] = 0x57;
  header[9] = 0x41;
  header[10] = 0x56;
  header[11] = 0x45;
  // "fmt "
  header[12] = 0x66;
  header[13] = 0x6d;
  header[14] = 0x74;
  header[15] = 0x20;
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
  view.setUint16(22, 1, true); // NumChannels (1 for mono)
  view.setUint32(24, sampleRate, true); // SampleRate (16000)
  view.setUint32(28, sampleRate * 1 * 2, true); // ByteRate (16000 * 1 * 2)
  view.setUint16(32, 2, true); // BlockAlign (1 * 2)
  view.setUint16(34, 16, true); // BitsPerSample (16 bits)
  // "data"
  header[36] = 0x64;
  header[37] = 0x61;
  header[38] = 0x74;
  header[39] = 0x61;
  view.setUint32(40, dataSize, true); // Subchunk2Size

  return header;
}

/**
 * Combine streaming raw PCM base64 chunks from ESP32 into a valid Base64-encoded WAV file.
 */
export function combinePcmChunksToWav(chunks: string[], sampleRate = 16000): string {
  if (!chunks || chunks.length === 0) return '';

  const byteChunks = chunks.map(c => base64ToBytes(c));
  const totalPcmBytes = byteChunks.reduce((sum, b) => sum + b.length, 0);
  if (totalPcmBytes === 0) return '';

  const wavBuffer = new Uint8Array(44 + totalPcmBytes);
  const header = createWavHeader(totalPcmBytes, sampleRate);
  wavBuffer.set(header, 0);

  let offset = 44;
  for (const b of byteChunks) {
    wavBuffer.set(b, offset);
    offset += b.length;
  }

  return bytesToBase64(wavBuffer);
}
