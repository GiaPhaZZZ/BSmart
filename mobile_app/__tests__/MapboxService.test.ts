declare const require: (moduleName: string) => any;
declare const __dirname: string;

const fs = require('fs');
const path = require('path');

import { MapboxService } from '../src/services/navigation/MapboxService';

describe('MapboxService configuration', () => {
  const originalFetch = globalThis.fetch;
  let warnSpy: jest.SpyInstance;

  beforeEach(() => {
    globalThis.fetch = jest.fn() as unknown as typeof fetch;
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    warnSpy.mockRestore();
  });

  it('does not keep a hardcoded Mapbox public token in source', () => {
    const source = fs.readFileSync(
      path.join(__dirname, '../src/services/navigation/MapboxService.ts'),
      'utf8',
    );

    expect(source).not.toContain('pk.ey');
  });

  it('skips Mapbox network calls when no access token is configured', async () => {
    await expect(MapboxService.searchDestination('Vạn Hạnh Mall')).resolves.toBeNull();
    await expect(
      MapboxService.getWalkingDirections({ lat: 10.77, lon: 106.69 }, { lat: 10.78, lon: 106.7 }),
    ).resolves.toBeNull();

    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalledWith(
      '[MapboxService] MAPBOX_ACCESS_TOKEN is not configured; skipping Mapbox request.',
    );
  });
});
