/**
 * BSmart Mapbox Service
 * Handles direct communication with Mapbox REST APIs from the mobile app.
 * Includes Geocoding (Forward Search) and Directions API (Walking profile).
 */

import { MAPBOX_ACCESS_TOKEN } from '../../constants/apiConfig';

export interface LocationCoordinates {
  lat: number;
  lon: number;
}

export interface MapboxStep {
  maneuver: {
    instruction: string;
    type: string;
    modifier?: string;
    location?: [number, number];
  };
  distance: number;
  duration: number;
  name: string;
}

export interface MapboxRoute {
  distance: number;
  duration: number;
  geometry: any;
  legs: Array<{
    steps: MapboxStep[];
  }>;
}

export const MapboxService = {
  /**
   * Resolve a spoken destination (e.g. 'Vạn Hạnh Mall') to coordinates via Search Box API.
   */
  async searchDestination(query: string, proximity?: LocationCoordinates): Promise<LocationCoordinates | null> {
    try {
      const token = getMapboxAccessToken();
      if (!token) return null;

      let url = `https://api.mapbox.com/search/searchbox/v1/forward?q=${encodeURIComponent(query)}&language=vi&country=VN&limit=1&access_token=${encodeURIComponent(token)}`;
      if (proximity) {
        url += `&proximity=${proximity.lon},${proximity.lat}`;
      }

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Mapbox search failed: ${response.status}`);
      }

      const data = await response.json();
      const features = data.features || [];
      if (features.length === 0) return null;

      const coords = features[0].geometry?.coordinates;
      if (!coords || coords.length < 2) return null;

      return {
        lon: coords[0], // Mapbox uses [longitude, latitude]
        lat: coords[1]
      };
    } catch (err) {
      console.warn('[MapboxService] Error searching destination:', err);
      return null;
    }
  },

  /**
   * Get walking directions from origin to destination.
   */
  async getWalkingDirections(origin: LocationCoordinates, dest: LocationCoordinates): Promise<MapboxRoute | null> {
    try {
      const token = getMapboxAccessToken();
      if (!token) return null;

      const coordsStr = `${origin.lon},${origin.lat};${dest.lon},${dest.lat}`;
      const url = `https://api.mapbox.com/directions/v5/mapbox/walking/${coordsStr}?steps=true&geometries=geojson&overview=full&language=vi&access_token=${encodeURIComponent(token)}`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Mapbox directions failed: ${response.status}`);
      }

      const data = await response.json();
      const routes = data.routes || [];
      if (routes.length === 0) return null;

      return routes[0] as MapboxRoute;
    } catch (err) {
      console.warn('[MapboxService] Error getting directions:', err);
      return null;
    }
  }
};

function getMapboxAccessToken(): string | null {
  const token = MAPBOX_ACCESS_TOKEN.trim();
  if (!token) {
    console.warn('[MapboxService] MAPBOX_ACCESS_TOKEN is not configured; skipping Mapbox request.');
    return null;
  }
  return token;
}
