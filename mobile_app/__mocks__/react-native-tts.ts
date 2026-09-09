/**
 * Jest manual mock for react-native-tts
 * Provides a no-op TTS that resolves immediately for tests.
 */

const mockTts = {
  getInitStatus: jest.fn(() => Promise.resolve('success')),
  requestInstallEngine: jest.fn(() => Promise.resolve('success')),
  setDefaultLanguage: jest.fn(() => Promise.resolve('success')),
  setDefaultRate: jest.fn(() => Promise.resolve('success')),
  speak: jest.fn(() => 'mock-utterance-id'),
  stop: jest.fn(() => Promise.resolve(true)),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
};

export default mockTts;
