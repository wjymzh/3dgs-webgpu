/**
 * SceneManager - 场景管理器
 * 
 * 负责管理场景中的所有对象：
 * - Mesh 网格
 * - Splat 点云
 * - 场景查询（bounding box 等）
 */

import { Mesh } from "../mesh/Mesh";
import { MeshRenderer } from "../mesh/MeshRenderer";
import { IGSSplatRenderer, BoundingBox } from "../gs/IGSSplatRenderer";
import { MeshBoundingBox } from "../mesh/Mesh";

/**
 * 场景对象类型
 */
export type SceneObjectType = 'mesh' | 'splat' | 'meshGroup';

/**
 * 场景对象信息
 */
export interface SceneObjectInfo {
  type: SceneObjectType;
  index: number;
  name?: string;
}

/**
 * SceneManager - 场景管理器
 */
export class SceneManager {
  private meshRenderer: MeshRenderer;
  private gsRenderer: IGSSplatRenderer | null = null;

  constructor(meshRenderer: MeshRenderer) {
    this.meshRenderer = meshRenderer;
  }

  // ============================================
  // Splat 渲染器管理
  // ============================================

  /**
   * 设置 GS Splat 渲染器
   */
  setGSRenderer(renderer: IGSSplatRenderer | null): void {
    this.gsRenderer = renderer;
  }

  /**
   * 获取 GS Splat 渲染器
   */
  getGSRenderer(): IGSSplatRenderer | null {
    return this.gsRenderer;
  }

  /**
   * 是否有 Splat 数据
   */
  hasSplats(): boolean {
    return this.gsRenderer !== null && this.gsRenderer.getSplatCount() > 0;
  }

  // ============================================
  // Mesh 管理（委托给 MeshRenderer）
  // ============================================

  /**
   * 获取 Mesh 数量
   */
  getMeshCount(): number {
    return this.meshRenderer.getMeshCount();
  }

  /**
   * 获取指定索引的 Mesh
   */
  getMeshByIndex(index: number): Mesh | null {
    return this.meshRenderer.getMeshByIndex(index);
  }

  /**
   * 获取指定范围的 Mesh
   */
  getMeshRange(startIndex: number, count: number): Mesh[] {
    const meshes: Mesh[] = [];
    for (let i = 0; i < count; i++) {
      const mesh = this.meshRenderer.getMeshByIndex(startIndex + i);
      if (mesh) {
        meshes.push(mesh);
      }
    }
    return meshes;
  }

  /**
   * 清空所有 Mesh
   */
  clearMeshes(): void {
    this.meshRenderer.clear();
  }

  /**
   * 按索引移除 Mesh
   */
  removeMeshByIndex(index: number): boolean {
    return this.meshRenderer.removeMeshByIndex(index);
  }

  // ============================================
  // Splat 管理
  // ============================================

  /**
   * 获取 Splat 数量
   */
  getSplatCount(): number {
    return this.gsRenderer?.getSplatCount() ?? 0;
  }

  /**
   * 清空 Splats
   */
  clearSplats(): void {
    if (this.gsRenderer) {
      this.gsRenderer.destroy();
      this.gsRenderer = null;
    }
  }

  // ============================================
  // Splat 变换
  // ============================================

  /**
   * 设置 Splat 位置
   */
  setSplatPosition(x: number, y: number, z: number): void {
    this.gsRenderer?.setPosition(x, y, z);
  }

  /**
   * 获取 Splat 位置
   */
  getSplatPosition(): [number, number, number] | null {
    return this.gsRenderer?.getPosition() ?? null;
  }

  /**
   * 设置 Splat 旋转
   */
  setSplatRotation(x: number, y: number, z: number): void {
    this.gsRenderer?.setRotation(x, y, z);
  }

  /**
   * 获取 Splat 旋转
   */
  getSplatRotation(): [number, number, number] | null {
    return this.gsRenderer?.getRotation() ?? null;
  }

  /**
   * 设置 Splat 缩放
   */
  setSplatScale(x: number, y: number, z: number): void {
    this.gsRenderer?.setScale(x, y, z);
  }

  /**
   * 获取 Splat 缩放
   */
  getSplatScale(): [number, number, number] | null {
    return this.gsRenderer?.getScale() ?? null;
  }

  // ============================================
  // SH 模式
  // ============================================

  /**
   * 设置 SH 模式
   */
  setSHMode(mode: 0 | 1 | 2 | 3): void {
    if (this.gsRenderer?.setSHMode) {
      this.gsRenderer.setSHMode(mode);
    }
  }

  /**
   * 获取当前 SH 模式
   */
  getSHMode(): number {
    return this.gsRenderer?.getSHMode?.() ?? 0;
  }

  // ============================================
  // Bounding Box 查询
  // ============================================

  /**
   * 获取 Mesh 的组合 bounding box
   */
  getMeshBoundingBox(): MeshBoundingBox | null {
    return this.meshRenderer.getCombinedBoundingBox();
  }

  /**
   * 获取 Splat 的 bounding box
   */
  getSplatBoundingBox(): BoundingBox | null {
    return this.gsRenderer?.getBoundingBox() ?? null;
  }

  /**
   * 获取指定 Mesh 范围的组合 bounding box
   */
  getMeshRangeBoundingBox(startIndex: number, count: number): BoundingBox | null {
    const meshes = this.getMeshRange(startIndex, count);
    if (meshes.length === 0) return null;

    let combinedMin: [number, number, number] | null = null;
    let combinedMax: [number, number, number] | null = null;

    for (const mesh of meshes) {
      const bbox = mesh.getWorldBoundingBox();
      if (!bbox) continue;

      if (combinedMin === null || combinedMax === null) {
        combinedMin = [...bbox.min];
        combinedMax = [...bbox.max];
      } else {
        combinedMin[0] = Math.min(combinedMin[0], bbox.min[0]);
        combinedMin[1] = Math.min(combinedMin[1], bbox.min[1]);
        combinedMin[2] = Math.min(combinedMin[2], bbox.min[2]);
        combinedMax[0] = Math.max(combinedMax[0], bbox.max[0]);
        combinedMax[1] = Math.max(combinedMax[1], bbox.max[1]);
        combinedMax[2] = Math.max(combinedMax[2], bbox.max[2]);
      }
    }

    if (combinedMin === null || combinedMax === null) return null;

    const center: [number, number, number] = [
      (combinedMin[0] + combinedMax[0]) / 2,
      (combinedMin[1] + combinedMax[1]) / 2,
      (combinedMin[2] + combinedMax[2]) / 2,
    ];
    const dx = combinedMax[0] - combinedMin[0];
    const dy = combinedMax[1] - combinedMin[1];
    const dz = combinedMax[2] - combinedMin[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    return { min: combinedMin, max: combinedMax, center, radius };
  }

  /**
   * 获取整个场景的组合 bounding box
   */
  getSceneBoundingBox(): BoundingBox | null {
    let combinedMin: [number, number, number] | null = null;
    let combinedMax: [number, number, number] | null = null;

    // 1. 获取 Mesh 的 bounding box
    const meshBBox = this.getMeshBoundingBox();
    if (meshBBox) {
      combinedMin = [...meshBBox.min];
      combinedMax = [...meshBBox.max];
    }

    // 2. 获取 Splat 的 bounding box
    const splatBBox = this.getSplatBoundingBox();
    if (splatBBox) {
      if (combinedMin === null || combinedMax === null) {
        combinedMin = [...splatBBox.min];
        combinedMax = [...splatBBox.max];
      } else {
        combinedMin[0] = Math.min(combinedMin[0], splatBBox.min[0]);
        combinedMin[1] = Math.min(combinedMin[1], splatBBox.min[1]);
        combinedMin[2] = Math.min(combinedMin[2], splatBBox.min[2]);
        combinedMax[0] = Math.max(combinedMax[0], splatBBox.max[0]);
        combinedMax[1] = Math.max(combinedMax[1], splatBBox.max[1]);
        combinedMax[2] = Math.max(combinedMax[2], splatBBox.max[2]);
      }
    }

    if (combinedMin === null || combinedMax === null) {
      return null;
    }

    const center: [number, number, number] = [
      (combinedMin[0] + combinedMax[0]) / 2,
      (combinedMin[1] + combinedMax[1]) / 2,
      (combinedMin[2] + combinedMax[2]) / 2,
    ];
    const dx = combinedMax[0] - combinedMin[0];
    const dy = combinedMax[1] - combinedMin[1];
    const dz = combinedMax[2] - combinedMin[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    return { min: combinedMin, max: combinedMax, center, radius };
  }

  // ============================================
  // 材质颜色
  // ============================================

  /**
   * 获取指定索引 Mesh 的材质颜色
   */
  getMeshColor(index: number): [number, number, number, number] | null {
    return this.meshRenderer.getMeshColor(index);
  }

  /**
   * 设置指定索引 Mesh 的材质颜色
   */
  setMeshColor(index: number, r: number, g: number, b: number, a: number = 1): boolean {
    return this.meshRenderer.setMeshColor(index, r, g, b, a);
  }

  /**
   * 设置指定范围内所有 Mesh 的材质颜色
   */
  setMeshRangeColor(startIndex: number, count: number, r: number, g: number, b: number, a: number = 1): number {
    return this.meshRenderer.setMeshRangeColor(startIndex, count, r, g, b, a);
  }

  /**
   * 销毁场景管理器
   */
  destroy(): void {
    this.clearSplats();
    // 注意：meshRenderer 的销毁由 App 负责
  }
}
