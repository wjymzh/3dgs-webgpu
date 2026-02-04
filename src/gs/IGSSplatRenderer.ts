/**
 * IGSSplatRenderer - 3D Gaussian Splatting 渲染器统一接口
 * 
 * 桌面端和移动端渲染器都实现此接口，消除平台判断代码
 */

import type { CompactSplatData } from "./PLYLoaderMobile";
import type { SplatCPU } from "./PLYLoader";
import type { BoundingBox, Vec3Tuple } from "../types";
import { SHMode, RendererCapabilities } from "../types";

// 重新导出类型，保持向后兼容
export type { BoundingBox, Vec3Tuple, RendererCapabilities };
export { SHMode };

/**
 * 3D Gaussian Splatting 渲染器接口
 */
export interface IGSSplatRenderer {
  // ============================================
  // 数据设置
  // ============================================
  
  /**
   * 设置紧凑格式的 splat 数据（推荐）
   */
  setCompactData(data: CompactSplatData): void;
  
  /**
   * 设置原始 splat 数据（仅桌面端支持）
   * 移动端实现可以抛出错误或转换为紧凑格式
   */
  setData?(splats: SplatCPU[]): void;

  // ============================================
  // 渲染
  // ============================================
  
  /**
   * 渲染 splats
   */
  render(pass: GPURenderPassEncoder): void;

  // ============================================
  // 变换
  // ============================================
  
  /**
   * 设置位置
   */
  setPosition(x: number, y: number, z: number): void;
  
  /**
   * 获取位置
   */
  getPosition(): Vec3Tuple;
  
  /**
   * 设置旋转（欧拉角，弧度）
   */
  setRotation(x: number, y: number, z: number): void;
  
  /**
   * 获取旋转
   */
  getRotation(): Vec3Tuple;
  
  /**
   * 设置缩放
   */
  setScale(x: number, y: number, z: number): void;
  
  /**
   * 获取缩放
   */
  getScale(): Vec3Tuple;
  
  /**
   * 设置旋转/缩放中心点（pivot）
   */
  setPivot(x: number, y: number, z: number): void;
  
  /**
   * 获取旋转/缩放中心点（pivot）
   */
  getPivot(): Vec3Tuple;
  
  /**
   * 获取模型矩阵
   */
  getModelMatrix(): Float32Array;

  // ============================================
  // 查询
  // ============================================
  
  /**
   * 获取 splat 数量
   */
  getSplatCount(): number;
  
  /**
   * 获取 bounding box
   */
  getBoundingBox(): BoundingBox | null;

  // ============================================
  // SH 模式（可选，移动端可能不支持）
  // ============================================
  
  /**
   * 设置 SH 模式
   * 移动端可能只支持 L0
   */
  setSHMode?(mode: SHMode): void;
  
  /**
   * 获取当前 SH 模式
   */
  getSHMode?(): SHMode;
  
  /**
   * 是否支持指定的 SH 模式
   */
  supportsSHMode?(mode: SHMode): boolean;

  // ============================================
  // 生命周期
  // ============================================
  
  /**
   * 销毁资源
   */
  destroy(): void;
}

/**
 * 获取渲染器能力（可选方法）
 */
export interface IGSSplatRendererWithCapabilities extends IGSSplatRenderer {
  getCapabilities(): RendererCapabilities;
}
