/**
 * WebGPU 3D 渲染引擎
 * 库入口文件 - 导出所有公共 API
 */

// ============================================
// 统一类型定义
// ============================================
export type {
  Vec3Tuple,
  Vec4Tuple,
  BoundingBox,
  SimpleBoundingBox,
  Transform,
  TransformableObject as ITransformableObject,
  BoundingBoxProvider as IBoundingBoxProvider,
  MaterialData,
  RendererCapabilities,
} from './types';

export { SHMode, DEFAULT_MATERIAL, DEFAULT_OBJ_MATERIAL } from './types';

// ============================================
// 工具函数
// ============================================
export {
  isMobileDevice,
  getRecommendedDPR,
  isWebGPUSupported,
  computeBoundingBox,
  mergeBoundingBoxes,
  createBoundingBoxFromMinMax,
  transformBoundingBox,
  loadTextureFromURL,
  loadTextureFromBlob,
  loadTextureFromBuffer,
  createTextureFromImageBitmap,
  TextureCache,
} from './utils';

// ============================================
// Core
// ============================================
export { Renderer } from './core/Renderer';
export { Camera } from './core/Camera';
export { OrbitControls } from './core/OrbitControls';
export { ViewportGizmo } from './core/ViewportGizmo';
export { BoundingBoxRenderer } from './core/BoundingBoxRenderer';
export type { BoundingBox as SelectionBoundingBox, BoundingBoxProvider } from './core/BoundingBoxRenderer';

// ============================================
// Mesh
// ============================================
export { Mesh } from './mesh/Mesh';
export type { MeshBoundingBox } from './mesh/Mesh';
export { MeshRenderer } from './mesh/MeshRenderer';

// ============================================
// Loaders
// ============================================
export { GLBLoader } from './loaders/GLBLoader';
export type { LoadedMesh } from './loaders/GLBLoader';
export { OBJLoader } from './loaders/OBJLoader';
export { OBJParser } from './loaders/OBJParser';
export type { ParsedOBJData, ParsedObject } from './loaders/OBJParser';
export { MTLParser } from './loaders/MTLParser';
export type { ParsedMaterial } from './loaders/MTLParser';

// ============================================
// 3D Gaussian Splatting - 接口
// ============================================
export type { 
  IGSSplatRenderer, 
  IGSSplatRendererWithCapabilities,
} from './gs/IGSSplatRenderer';

// ============================================
// 3D Gaussian Splatting - 实现
// ============================================
export { loadPLY } from './gs/PLYLoader';
export type { SplatCPU } from './gs/PLYLoader';
export { loadPLYMobile, parsePLYBuffer, compactDataToGPUBuffer } from './gs/PLYLoaderMobile';
export type { MobileLoadOptions, CompactSplatData } from './gs/PLYLoaderMobile';
export { loadSplat, deserializeSplat } from './gs/SplatLoader';
export { GSSplatRenderer, SHMode as GSSHMode } from './gs/GSSplatRenderer';
export type { BoundingBox as GSSplatBoundingBox } from './gs/GSSplatRenderer';
export { GSSplatSorter } from './gs/GSSplatSorter';
export type { SorterOptions, CullingOptions, ScreenInfo } from './gs/GSSplatSorter';

// ============================================
// 3D Gaussian Splatting - 移动端
// ============================================
export { GSSplatRendererMobile } from './gs/GSSplatRendererMobile';
export { GSSplatSorterMobile } from './gs/GSSplatSorterMobile';
export type { 
  SorterOptions as MobileSorterOptions, 
  CullingOptions as MobileCullingOptions 
} from './gs/GSSplatSorterMobile';
export { 
  compressSplatsToTextures, 
  destroyCompressedTextures,
  calculateTextureDimensions 
} from './gs/TextureCompressor';
export type { CompressedSplatTextures } from './gs/TextureCompressor';

// ============================================
// Scene Management
// ============================================
export { SceneManager } from './scene/SceneManager';
export type { SceneObjectType, SceneObjectInfo } from './scene/SceneManager';

// Scene Proxies
export { 
  SplatTransformProxy,
  MeshGroupProxy,
  SplatBoundingBoxProvider,
} from './scene/proxies';

// ============================================
// Interaction
// ============================================
export { 
  GizmoManager,
  SplatTransformProxy as GizmoSplatTransformProxy,
  MeshGroupProxy as GizmoMeshGroupProxy,
  SplatBoundingBoxProvider as GizmoSplatBoundingBoxProvider
} from './interaction/GizmoManager';

// ============================================
// App
// ============================================
export { App } from './App';
export type { ProgressCallback } from './App';

// ============================================
// Gizmo
// ============================================
export type { TransformableObject } from './core/gizmo/TransformGizmoV2';
export { TransformGizmoV2, GizmoMode } from './core/gizmo/TransformGizmoV2';
export type { GizmoTheme, TransformGizmoConfig, GizmoSpace } from './core/gizmo/TransformGizmoV2';
