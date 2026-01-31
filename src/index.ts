/**
 * WebGPU 3D 渲染引擎
 * 库入口文件 - 导出所有公共 API
 */

// Core
export { Renderer } from './core/Renderer';
export { Camera } from './core/Camera';
export { OrbitControls } from './core/OrbitControls';
export { ViewportGizmo } from './core/ViewportGizmo';
export { BoundingBoxRenderer } from './core/BoundingBoxRenderer';
export type { BoundingBox as SelectionBoundingBox, BoundingBoxProvider } from './core/BoundingBoxRenderer';

// Mesh
export { Mesh } from './mesh/Mesh';
export type { MeshBoundingBox } from './mesh/Mesh';
export { MeshRenderer } from './mesh/MeshRenderer';

// Loaders
export { GLBLoader } from './loaders/GLBLoader';
export type { MaterialData, LoadedMesh } from './loaders/GLBLoader';

// 3D Gaussian Splatting
export { loadPLY } from './gs/PLYLoader';
export type { SplatCPU } from './gs/PLYLoader';
export { loadPLYMobile, parsePLYBuffer, compactDataToGPUBuffer } from './gs/PLYLoaderMobile';
export type { MobileLoadOptions, CompactSplatData } from './gs/PLYLoaderMobile';
export { loadSplat, deserializeSplat } from './gs/SplatLoader';
export { GSSplatRenderer, SHMode, PerformanceTier } from './gs/GSSplatRenderer';
export type { MobileOptimizationConfig, BoundingBox } from './gs/GSSplatRenderer';
export { GSSplatSorter } from './gs/GSSplatSorter';
export type { SorterOptions, CullingOptions, ScreenInfo } from './gs/GSSplatSorter';

// App
export { App, SplatTransformProxy, MeshGroupProxy, SplatBoundingBoxProvider } from './App';
export type { ProgressCallback } from './App';

// Gizmo
export type { TransformableObject } from './core/gizmo/TransformGizmoV2';
export { TransformGizmoV2, GizmoMode } from './core/gizmo/TransformGizmoV2';
export type { GizmoTheme, TransformGizmoConfig, GizmoSpace } from './core/gizmo/TransformGizmoV2';

// 3D Gaussian Splatting - 移动端纹理压缩优化
export { GSSplatRendererMobile } from './gs/GSSplatRendererMobile';
export type { BoundingBox as MobileBoundingBox } from './gs/GSSplatRendererMobile';
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
