import {
  FEATURE_SOUND_MAP,
  playFeatureActivationSound,
  stopFeatureSound,
} from '../src/services/audio/SoundEffectService';
import { BleConnectionState, IBleService } from '../src/types';

jest.mock('react-native-tts', () => ({
  getInitStatus: jest.fn().mockResolvedValue(true),
  setDefaultLanguage: jest.fn().mockResolvedValue(true),
  setDefaultRate: jest.fn().mockResolvedValue(true),
  speak: jest.fn().mockReturnValue('mock-utterance-id'),
  stop: jest.fn().mockResolvedValue(true),
  addEventListener: jest.fn((event, cb) => {
    if (event === 'tts-finish') {
      setTimeout(() => cb({ utteranceId: 'mock-utterance-id' }), 10);
    }
  }),
  removeEventListener: jest.fn(),
}));

describe('SoundEffectService', () => {
  it('correctly defines audio clips and fallback mappings for all 3 features', () => {
    expect(FEATURE_SOUND_MAP.feature1.soundName).toBe('open_f1');
    expect(FEATURE_SOUND_MAP.feature2.soundName).toBe('open_f4');
    expect(FEATURE_SOUND_MAP.feature3.soundName).toBe('open_f2');

    expect(FEATURE_SOUND_MAP.feature1.bleCommand).toBe('CMD:PLAY_F1');
    expect(FEATURE_SOUND_MAP.feature2.bleCommand).toBe('CMD:PLAY_F4');
    expect(FEATURE_SOUND_MAP.feature3.bleCommand).toBe('CMD:PLAY_F2');
  });

  it('sends BLE command when connected to glasses', async () => {
    const mockSendAudio = jest.fn().mockResolvedValue(undefined);
    const mockBleService: Partial<IBleService> = {
      getConnectionState: jest.fn().mockReturnValue(BleConnectionState.CONNECTED),
      sendAudio: mockSendAudio,
    };

    await playFeatureActivationSound('feature1', mockBleService as IBleService);
    expect(mockSendAudio).toHaveBeenCalledWith('CMD:PLAY_F1');
  });

  it('safely stops audio playback without throwing', () => {
    expect(() => stopFeatureSound()).not.toThrow();
  });
});
