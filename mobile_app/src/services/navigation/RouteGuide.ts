import { LocationCoordinates, MapboxRoute } from './MapboxService';

// Earth radius in meters
const EARTH_RADIUS_M = 6371000.0;

function haversineDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const dLat = (lat2 - lat1) * Math.PI / 180.0;
  const dLon = (lon2 - lon1) * Math.PI / 180.0;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180.0) * Math.cos(lat2 * Math.PI / 180.0) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return EARTH_RADIUS_M * c;
}

// Point to line segment distance
function distanceToSegment(lat: number, lon: number, latA: number, lonA: number, latB: number, lonB: number): number {
  const dLat = (latB - latA) * Math.PI / 180.0;
  const dLon = (lonB - lonA) * Math.PI / 180.0;
  const latR = lat * Math.PI / 180.0;
  const lonR = lon * Math.PI / 180.0;
  const latAR = latA * Math.PI / 180.0;
  const lonAR = lonA * Math.PI / 180.0;

  const dx = dLon * Math.cos(latAR);
  const dy = dLat;
  const segmentLenSq = dx * dx + dy * dy;

  if (segmentLenSq === 0) {
    return haversineDistance(lat, lon, latA, lonA);
  }

  const px = (lonR - lonAR) * Math.cos(latAR);
  const py = latR - latAR;
  
  let t = (px * dx + py * dy) / segmentLenSq;
  t = Math.max(0, Math.min(1, t));

  const projLat = latAR + t * dy;
  const projLon = lonAR + t * dLon / Math.cos(latAR);

  return haversineDistance(lat, lon, projLat * 180.0 / Math.PI, projLon * 180.0 / Math.PI);
}

const MANEUVER_VI: Record<string, string> = {
  'depart_': 'bắt đầu di chuyển',
  'arrive_': 'bạn đã đến nơi',
  'turn_left': 'rẽ trái',
  'turn_right': 'rẽ phải',
  'turn_slight left': 'rẽ nhẹ sang trái',
  'turn_slight right': 'rẽ nhẹ sang phải',
  'turn_sharp left': 'rẽ gắt sang trái',
  'turn_sharp right': 'rẽ gắt sang phải',
  'turn_straight': 'đi thẳng',
  'turn_uturn': 'quay đầu',
  'continue_': 'tiếp tục đi thẳng',
  'merge_': 'nhập vào đường',
  'fork_left': 'đi theo nhánh bên trái',
  'fork_right': 'đi theo nhánh bên phải',
  'roundabout_': 'đi vào vòng xuyến',
  'end of road_left': 'cuối đường, rẽ trái',
  'end of road_right': 'cuối đường, rẽ phải',
};

const SKIP_AS_NEXT_MANEUVER = new Set(['depart', 'continue']);

export class RouteGuide {
  private route: MapboxRoute | null = null;
  private currentStepIndex: number = 0;
  private polylineCoords: Array<[number, number]> = [];
  
  private OFF_ROUTE_M = 50.0;
  private ARRIVAL_M = 15.0;
  private MIN_ANNOUNCE_DIST_M = 10.0;

  public isFinished = false;

  public setRoute(route: MapboxRoute) {
    this.route = route;
    this.currentStepIndex = 0;
    this.isFinished = false;
    this.polylineCoords = route.geometry?.coordinates || [];
  }

  public getRoute(): MapboxRoute | null {
    return this.route;
  }

  public updateAndGetInstruction(currentLoc: LocationCoordinates): { instruction: string | null, isOffRoute: boolean, hasArrived: boolean } {
    if (!this.route) return { instruction: null, isOffRoute: false, hasArrived: false };

    // 1. Check if off-route using point-to-segment distance
    const distToRoute = this.getDistanceToRoute(currentLoc);
    if (distToRoute > this.OFF_ROUTE_M) {
      return { instruction: null, isOffRoute: true, hasArrived: false };
    }

    const steps = this.route.legs[0].steps;
    if (this.currentStepIndex >= steps.length) {
      this.isFinished = true;
      return { instruction: 'Bạn đã đến nơi.', isOffRoute: false, hasArrived: true };
    }

    // 2. Find closest step
    // Mapbox provides geometry for steps, but for simplicity we assume the user is on current step
    // In a real app we'd map snap to the step. Here we check distance to destination of current step.
    // For now, we will just give the instruction of the next maneuver.
    
    // We need to know distance to the NEXT maneuver. Let's simplify by just reading the current step.
    let currentStep = steps[this.currentStepIndex];
    
    // If the step is "arrive", we check distance to final coord
    if (currentStep.maneuver.type === 'arrive') {
       // We can just say "bạn đã đến nơi" and finish
       this.isFinished = true;
       return { instruction: 'Bạn đã đến nơi.', isOffRoute: false, hasArrived: true };
    }

    // Find the next significant maneuver
    let nextStepIndex = this.currentStepIndex + 1;
    while (nextStepIndex < steps.length && SKIP_AS_NEXT_MANEUVER.has(steps[nextStepIndex].maneuver.type)) {
      nextStepIndex++;
    }

    let nextStep = steps[nextStepIndex];
    if (!nextStep) {
      nextStep = steps[steps.length - 1]; // fallback to arrive
    }

    // Format instruction
    // Since we don't have accurate along-track distance in this simplified RN version,
    // we just use the step's distance. (In Python, it calculated exactly along polyline)
    // To make it simple: "Đi X mét rồi [hành động]"
    const dist = Math.max(1, Math.round(currentStep.distance));
    const actionVi = this.getManeuverPhrase(nextStep.maneuver.type, nextStep.maneuver.modifier);

    let instruction = `Đi ${dist} mét rồi ${actionVi}.`;

    // Advance step if we are close to the end of the current step
    const stepEndLoc = currentStep.maneuver.location;
    // Mapbox maneuver location is [lon, lat]
    if (stepEndLoc && stepEndLoc.length === 2) {
      const distToEnd = haversineDistance(currentLoc.lat, currentLoc.lon, stepEndLoc[1], stepEndLoc[0]);
      if (distToEnd < this.ARRIVAL_M) {
        this.currentStepIndex++;
      }
    }
    
    return { instruction, isOffRoute: false, hasArrived: false };
  }

  private getDistanceToRoute(loc: LocationCoordinates): number {
    if (this.polylineCoords.length < 2) return 0;
    
    let minDist = Infinity;
    for (let i = 0; i < this.polylineCoords.length - 1; i++) {
      const p1 = this.polylineCoords[i];
      const p2 = this.polylineCoords[i+1];
      const d = distanceToSegment(loc.lat, loc.lon, p1[1], p1[0], p2[1], p2[0]);
      if (d < minDist) minDist = d;
    }
    return minDist;
  }

  private getManeuverPhrase(type: string, modifier?: string): string {
    const keyWithMod = `${type}_${modifier}`;
    const keyWithoutMod = `${type}_`;
    return MANEUVER_VI[keyWithMod] || MANEUVER_VI[keyWithoutMod] || 'tiếp tục di chuyển';
  }
}
