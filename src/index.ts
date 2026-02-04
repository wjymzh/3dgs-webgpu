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
export { OBJLoader } from './loaders/OBJLoader';
export { OBJParser } from './loaders/OBJParser';
export type { ParsedOBJData, ParsedObject } from './loaders/OBJParser';
export { MTLParser } from './loaders/MTLParser';
export type { ParsedMaterial } from './loaders/MTLParser';

// 3D Gaussian Splatting - 统一接口
export type { 
  IGSSplatRenderer, 
  IGSSplatRendererWithCapabilities,
  RendererCapabilities,
  BoundingBox,
  SHMode 
} from './gs/IGSSplatRenderer';

// 3D Gaussian Splatting - 实现
export { loadPLY } from './gs/PLYLoader';
export type { SplatCPU } from './gs/PLYLoader';
export { loadPLYMobile, parsePLYBuffer, compactDataToGPUBuffer } from './gs/PLYLoaderMobile';
export type { MobileLoadOptions, CompactSplatData } from './gs/PLYLoaderMobile';
export { loadSplat, deserializeSplat } from './gs/SplatLoader';
export { GSSplatRenderer, PerformanceTier } from './gs/GSSplatRenderer';
// 保留旧的导出以保持向后兼容
export { SHMode as GSSHMode } from './gs/GSSplatRenderer';
export type { MobileOptimizationConfig } from './gs/GSSplatRenderer';
export { GSSplatSorter } from './gs/GSSplatSorter';
export type { SorterOptions, CullingOptions, ScreenInfo } from './gs/GSSplatSorter';

// 3D Gaussian Splatting - V2 优化版本 (基于 rfs-gsplat-render)
export { GSSplatRendererV2, SHMode as GSSHModeV2 } from './gs/GSSplatRendererV2';
export type { BoundingBox as BoundingBoxV2 } from './gs/GSSplatRendererV2';
export { RadixSorter } from './gs/RadixSorter';
export type { CullingOptions as RadixCullingOptions, ScreenInfo as RadixScreenInfo } from './gs/RadixSorter';

// 3D Gaussian Splatting - 移动端
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

// Scene Management
export { SceneManager } from './scene/SceneManager';
export type { SceneObjectType, SceneObjectInfo } from './scene/SceneManager';

// Interaction
export { 
  GizmoManager,
  SplatTransformProxy as GizmoSplatTransformProxy,
  MeshGroupProxy as GizmoMeshGroupProxy,
  SplatBoundingBoxProvider as GizmoSplatBoundingBoxProvider
} from './interaction/GizmoManager';

// App
export { App, SplatTransformProxy, MeshGroupProxy, SplatBoundingBoxProvider } from './App';
export type { ProgressCallback } from './App';

// Gizmo
export type { TransformableObject } from './core/gizmo/TransformGizmoV2';
export { TransformGizmoV2, GizmoMode } from './core/gizmo/TransformGizmoV2';
export type { GizmoTheme, TransformGizmoConfig, GizmoSpace } from './core/gizmo/TransformGizmoV2';
