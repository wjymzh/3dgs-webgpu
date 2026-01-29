/**
 * WebGPU 3D 渲染引擎
 * 库入口文件 - 导出所有公共 API
 */

// Core
export { Renderer } from './core/Renderer';
export { Camera } from './core/Camera';
export { OrbitControls } from './core/OrbitControls';
export { ViewportGizmo } from './core/ViewportGizmo';

// Mesh
export { Mesh } from './mesh/Mesh';
export { MeshRenderer } from './mesh/MeshRenderer';

// Loaders
export { GLBLoader } from './loaders/GLBLoader';

// 3D Gaussian Splatting
export { loadPLY } from './gs/PLYLoader';
export type { SplatCPU } from './gs/PLYLoader';
export { loadPLYMobile, compactDataToGPUBuffer } from './gs/PLYLoaderMobile';
export type { MobileLoadOptions, CompactSplatData } from './gs/PLYLoaderMobile';
export { GSSplatRenderer, SHMode, PerformanceTier } from './gs/GSSplatRenderer';
export type { MobileOptimizationConfig, BoundingBox } from './gs/GSSplatRenderer';
export { GSSplatSorter } from './gs/GSSplatSorter';
export type { SorterOptions, CullingOptions, ScreenInfo } from './gs/GSSplatSorter';

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

// App
export { App, SplatTransformProxy, MeshGroupProxy } from './App';

// Gizmo
export type { TransformableObject } from './core/gizmo/TransformGizmo';
