module.exports = {
  preset: '@react-native/jest-preset',
  transformIgnorePatterns: [
    'node_modules/(?!(react-native|@react-native|react-native-ble-plx|react-native-safe-area-context|react-native-tts)/)',
  ],
  moduleNameMapper: {
    'react-native-tts': '<rootDir>/__mocks__/react-native-tts.ts',
  },
};
