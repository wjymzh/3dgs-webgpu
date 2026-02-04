/**
 * 几何计算工具函数
 */

import type { BoundingBox, Vec3Tuple } from '../types';

/**
 * 从顶点位置数组计算包围盒
 * @param positions 顶点位置数组 [x0, y0, z0, x1, y1, z1, ...]
 * @returns 包围盒信息
 */
export function computeBoundingBox(
  positions: ArrayLike<number>
): BoundingBox {
  if (positions.length < 3) {
    return {
      min: [0, 0, 0],
      max: [0, 0, 0],
      center: [0, 0, 0],
      radius: 0,
    };
  }

  // 初始化为第一个点
  const min: Vec3Tuple = [positions[0], positions[1], positions[2]];
  const max: Vec3Tuple = [positions[0], positions[1], positions[2]];

  // 遍历所有顶点
  for (let i = 3; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];

    min[0] = Math.min(min[0], x);
    min[1] = Math.min(min[1], y);
    min[2] = Math.min(min[2], z);
    max[0] = Math.max(max[0], x);
    max[1] = Math.max(max[1], y);
    max[2] = Math.max(max[2], z);
  }

  // 计算中心点
  const center: Vec3Tuple = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];

  // 计算 bounding sphere 半径
  const dx = max[0] - min[0];
  const dy = max[1] - min[1];
  const dz = max[2] - min[2];
  const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

  return { min, max, center, radius };
}

/**
 * 合并多个包围盒
 * @param boxes 包围盒数组
 * @returns 合并后的包围盒，如果输入为空则返回 null
 */
export function mergeBoundingBoxes(boxes: BoundingBox[]): BoundingBox | null {
  if (boxes.length === 0) return null;

  let combinedMin: Vec3Tuple = [...boxes[0].min];
  let combinedMax: Vec3Tuple = [...boxes[0].max];

  for (let i = 1; i < boxes.length; i++) {
    const box = boxes[i];
    combinedMin[0] = Math.min(combinedMin[0], box.min[0]);
    combinedMin[1] = Math.min(combinedMin[1], box.min[1]);
    combinedMin[2] = Math.min(combinedMin[2], box.min[2]);
    combinedMax[0] = Math.max(combinedMax[0], box.max[0]);
    combinedMax[1] = Math.max(combinedMax[1], box.max[1]);
    combinedMax[2] = Math.max(combinedMax[2], box.max[2]);
  }

  const center: Vec3Tuple = [
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
 * 从 min/max 计算完整的包围盒信息
 */
export function createBoundingBoxFromMinMax(
  min: Vec3Tuple,
  max: Vec3Tuple
): BoundingBox {
  const center: Vec3Tuple = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];

  const dx = max[0] - min[0];
  const dy = max[1] - min[1];
  const dz = max[2] - min[2];
  const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

  return { min, max, center, radius };
}

/**
 * 变换包围盒的 8 个角点并计算新的 AABB
 * @param bbox 原始包围盒
 * @param modelMatrix 4x4 变换矩阵（列主序）
 */
export function transformBoundingBox(
  bbox: BoundingBox,
  modelMatrix: Float32Array
): BoundingBox {
  // 获取 8 个角点
  const corners: Vec3Tuple[] = [
    [bbox.min[0], bbox.min[1], bbox.min[2]],
    [bbox.max[0], bbox.min[1], bbox.min[2]],
    [bbox.min[0], bbox.max[1], bbox.min[2]],
    [bbox.max[0], bbox.max[1], bbox.min[2]],
    [bbox.min[0], bbox.min[1], bbox.max[2]],
    [bbox.max[0], bbox.min[1], bbox.max[2]],
    [bbox.min[0], bbox.max[1], bbox.max[2]],
    [bbox.max[0], bbox.max[1], bbox.max[2]],
  ];

  const m = modelMatrix;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const [x, y, z] of corners) {
    // 应用变换（列主序矩阵）
    const tx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const ty = m[1] * x + m[5] * y + m[9] * z + m[13];
    const tz = m[2] * x + m[6] * y + m[10] * z + m[14];

    minX = Math.min(minX, tx);
    minY = Math.min(minY, ty);
    minZ = Math.min(minZ, tz);
    maxX = Math.max(maxX, tx);
    maxY = Math.max(maxY, ty);
    maxZ = Math.max(maxZ, tz);
  }

  return createBoundingBoxFromMinMax(
    [minX, minY, minZ],
    [maxX, maxY, maxZ]
  );
}
