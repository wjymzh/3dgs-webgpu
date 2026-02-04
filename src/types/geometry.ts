/**
 * 统一的几何类型定义
 */

/**
 * 3D 向量类型（元组形式）
 */
export type Vec3Tuple = [number, number, number];

/**
 * 4D 向量类型（元组形式）
 */
export type Vec4Tuple = [number, number, number, number];

/**
 * 包围盒接口 - 统一定义
 * 用于 Mesh、Splat、场景等所有 3D 对象
 */
export interface BoundingBox {
  /** 最小点坐标 */
  min: Vec3Tuple;
  /** 最大点坐标 */
  max: Vec3Tuple;
  /** 中心点坐标 */
  center: Vec3Tuple;
  /** 包围球半径 */
  radius: number;
}

/**
 * 简化的包围盒（仅 min/max）
 * 用于 BoundingBoxRenderer 等只需要边界的场景
 */
export interface SimpleBoundingBox {
  min: Vec3Tuple;
  max: Vec3Tuple;
}

/**
 * 变换属性接口
 */
export interface Transform {
  position: Vec3Tuple;
  rotation: Vec3Tuple;  // 欧拉角（弧度）
  scale: Vec3Tuple;
}

/**
 * 可变换对象接口
 * 用于 Gizmo 操作的目标对象
 */
export interface TransformableObject {
  position: Vec3Tuple;
  rotation: Vec3Tuple;
  scale: Vec3Tuple;
  setPosition(x: number, y: number, z: number): void;
  setRotation(x: number, y: number, z: number): void;
  setScale(x: number, y: number, z: number): void;
}

/**
 * 包围盒提供者接口
 * 用于动态获取包围盒（考虑变换）
 */
export interface BoundingBoxProvider {
  getBoundingBox(): SimpleBoundingBox | null;
}
