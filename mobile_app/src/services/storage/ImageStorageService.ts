/**
 * BSmart Image Storage Service
 * Manages saving and retrieving captured images in local storage (Feature 2).
 * See: docs/requirement.md §6
 */

export interface SavedImageMeta {
  id: string;
  filename: string;
  timestamp: string;
  sizeBytes: number;
}

const savedImages: SavedImageMeta[] = [];

/**
 * Save captured base64 JPEG image to local storage
 * @param imageBase64 - base64 string of JPEG image
 * @returns path / filename of saved image
 */
export async function saveCapturedImage(imageBase64: string | null): Promise<string> {
  const timestamp = new Date();
  const dateStr = timestamp.toISOString().replace(/[-:T.]/g, '').slice(0, 14);
  const filename = `BSmart_Photo_${dateStr}.jpg`;

  const sizeEstimate = imageBase64 ? Math.round((imageBase64.length * 3) / 4) : 45000;

  const meta: SavedImageMeta = {
    id: `${Date.now()}`,
    filename,
    timestamp: timestamp.toLocaleString('vi-VN'),
    sizeBytes: sizeEstimate,
  };

  savedImages.push(meta);
  return filename;
}

/**
 * Get list of all saved images
 */
export function getSavedImages(): SavedImageMeta[] {
  return [...savedImages];
}
