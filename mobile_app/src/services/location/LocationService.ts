import { NativeModules } from 'react-native';

const { LocationModule } = NativeModules;

export interface LocationResult {
  lat: number;
  lon: number;
}

export const LocationService = {
  getCurrentLocation: async (): Promise<LocationResult> => {
    if (!LocationModule) {
      throw new Error('LocationModule is not available. Ensure native linking is done.');
    }
    return LocationModule.getCurrentLocation();
  },
};
