/**
 * 统一的材质类型定义
 */

/**
 * 材质数据接口
 * 用于 GLB/OBJ 等模型的材质
 */
export interface MaterialData {
  /** 基础颜色因子 [r, g, b, a] */
  baseColorFactor: [number, number, number, number];
  /** 基础颜色纹理 */
  baseColorTexture: GPUTexture | null;
  /** 金属度因子 */
  metallicFactor: number;
  /** 粗糙度因子 */
  roughnessFactor: number;
  /** 是否双面渲染 */
  doubleSided: boolean;
}

/**
 * 默认材质
 */
export const DEFAULT_MATERIAL: Readonly<MaterialData> = {
  baseColorFactor: [1, 1, 1, 1],
  baseColorTexture: null,
  metallicFactor: 0,
  roughnessFactor: 0.5,
  doubleSided: false,
};

/**
 * OBJ 默认材质（双面渲染）
 */
export const DEFAULT_OBJ_MATERIAL: Readonly<MaterialData> = {
  baseColorFactor: [1, 1, 1, 1],
  baseColorTexture: null,
  metallicFactor: 0,
  roughnessFactor: 0.5,
  doubleSided: true,
};
