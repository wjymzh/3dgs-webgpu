/**
 * 类型定义统一导出
 */

// 几何类型
export type {
  Vec3Tuple,
  Vec4Tuple,
  BoundingBox,
  SimpleBoundingBox,
  Transform,
  TransformableObject,
  BoundingBoxProvider,
} from './geometry';

// 材质类型
export type { MaterialData } from './material';
export { DEFAULT_MATERIAL, DEFAULT_OBJ_MATERIAL } from './material';

// Splat 类型
export { SHMode } from './splat';
export type { RendererCapabilities } from './splat';
