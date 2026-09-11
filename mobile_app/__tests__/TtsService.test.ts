import Tts from 'react-native-tts';
import { speakViaBle, stopSpeaking, speakUrgent } from '../src/services/tts/TtsService';

describe('TtsService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('speaks text and safely cleans up subscriptions using remove() without crashing', async () => {
    const removeFn = jest.fn();
    (Tts.addEventListener as jest.Mock).mockReturnValue({ remove: removeFn });

    const speakPromise = speakViaBle('Xin chào BSmart');

    // Allow microtasks (ensureTtsReady) to tick
    await new Promise<void>(res => setImmediate(res));

    // Simulate tts-finish event callback
    const addListenerCalls = (Tts.addEventListener as jest.Mock).mock.calls;
    expect(addListenerCalls.length).toBeGreaterThanOrEqual(3);

    const finishHandler = addListenerCalls.find(call => call[0] === 'tts-finish')?.[1];
    expect(typeof finishHandler).toBe('function');

    // Trigger finish
    finishHandler({ utteranceId: 'mock-utterance-id' });

    await speakPromise;

    // Verify remove() was called on subscriptions, NOT broken removeEventListener
    expect(removeFn).toHaveBeenCalled();
  });

  it('stops speaking immediately and cancels active utterance cleanly', async () => {
    const removeFn = jest.fn();
    (Tts.addEventListener as jest.Mock).mockReturnValue({ remove: removeFn });

    const speakPromise = speakViaBle('Đang đọc nội dung dài');
    await new Promise<void>(res => setImmediate(res));
    stopSpeaking();

    await speakPromise;
    expect(removeFn).toHaveBeenCalled();
  });

  it('speakUrgent cancels existing speech and speaks new urgent prompt', async () => {
    const mockBleService = {
      getConnectionState: jest.fn(() => 'DISCONNECTED'),
      sendAudio: jest.fn(),
    } as any;

    await speakUrgent('Cảnh báo vật cản', mockBleService);
    expect(Tts.speak).toHaveBeenCalledWith('Cảnh báo vật cản');
  });
});
