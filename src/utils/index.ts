/**
 * 工具函数统一导出
 */

// 设备检测
export {
  isMobileDevice,
  getRecommendedDPR,
  isWebGPUSupported,
} from './device';

// 几何计算
export {
  computeBoundingBox,
  mergeBoundingBoxes,
  createBoundingBoxFromMinMax,
  transformBoundingBox,
} from './geometry';

// 纹理加载
export {
  loadTextureFromURL,
  loadTextureFromBlob,
  loadTextureFromBuffer,
  createTextureFromImageBitmap,
  TextureCache,
} from './texture';
