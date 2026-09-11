/**
 * BSmart Image Storage Service
 * Manages saving and retrieving captured images in local storage (Feature 2).
 * See: docs/requirement.md §6
 */

import { NativeModules } from 'react-native';

const { ImageStorageModule } = NativeModules;

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
  if (!imageBase64) return 'Lỗi: Không có ảnh';
  
  try {
    const filename = await ImageStorageModule.saveImageToGallery(imageBase64);
    
    const meta: SavedImageMeta = {
      id: `${Date.now()}`,
      filename,
      timestamp: new Date().toLocaleString('vi-VN'),
      sizeBytes: Math.round((imageBase64.length * 3) / 4),
    };

    savedImages.push(meta);
    return filename;
  } catch (error) {
    console.error('Failed to save image:', error);
    throw error;
  }
}

/**
 * Get list of all saved images
 */
export function getSavedImages(): SavedImageMeta[] {
  return [...savedImages];
}
