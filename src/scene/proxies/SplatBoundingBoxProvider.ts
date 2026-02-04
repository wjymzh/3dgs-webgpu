/**
 * SplatBoundingBoxProvider - PLY/Splat 包围盒提供者
 * 动态获取 PLY 的包围盒（考虑变换）
 */

import type { BoundingBoxProvider, SimpleBoundingBox, Vec3Tuple } from '../../types';
import type { IGSSplatRenderer } from '../../gs/IGSSplatRenderer';

export class SplatBoundingBoxProvider implements BoundingBoxProvider {
  private renderer: IGSSplatRenderer;

  constructor(renderer: IGSSplatRenderer) {
    this.renderer = renderer;
  }

  getBoundingBox(): SimpleBoundingBox | null {
    const bbox = this.renderer.getBoundingBox();
    if (!bbox) return null;

    // 获取变换参数
    const position = this.renderer.getPosition();
    const rotation = this.renderer.getRotation();
    const scale = this.renderer.getScale();
    const pivot = this.renderer.getPivot();

    // 获取本地包围盒的 8 个角点
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

    // 计算变换矩阵
    const [sx, sy, sz] = scale;
    const [rx, ry, rz] = rotation;
    const [tx, ty, tz] = position;
    const [px, py, pz] = pivot;

    const cx = Math.cos(rx), sx1 = Math.sin(rx);
    const cy = Math.cos(ry), sy1 = Math.sin(ry);
    const cz = Math.cos(rz), sz1 = Math.sin(rz);

    // 组合旋转矩阵 R = Rz * Ry * Rx
    const r00 = cy * cz;
    const r01 = sx1 * sy1 * cz - cx * sz1;
    const r02 = cx * sy1 * cz + sx1 * sz1;
    const r10 = cy * sz1;
    const r11 = sx1 * sy1 * sz1 + cx * cz;
    const r12 = cx * sy1 * sz1 - sx1 * cz;
    const r20 = -sy1;
    const r21 = sx1 * cy;
    const r22 = cx * cy;

    // RS 矩阵 (旋转 * 缩放)
    const rs00 = r00 * sx, rs01 = r01 * sy, rs02 = r02 * sz;
    const rs10 = r10 * sx, rs11 = r11 * sy, rs12 = r12 * sz;
    const rs20 = r20 * sx, rs21 = r21 * sy, rs22 = r22 * sz;

    // 计算 (I - RS) * pivot
    const dpx = px - (rs00 * px + rs01 * py + rs02 * pz);
    const dpy = py - (rs10 * px + rs11 * py + rs12 * pz);
    const dpz = pz - (rs20 * px + rs21 * py + rs22 * pz);

    // 最终平移 = position + (I - RS) * pivot
    const finalTx = tx + dpx;
    const finalTy = ty + dpy;
    const finalTz = tz + dpz;

    // 变换所有角点
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (const [x, y, z] of corners) {
      const wx = rs00 * x + rs01 * y + rs02 * z + finalTx;
      const wy = rs10 * x + rs11 * y + rs12 * z + finalTy;
      const wz = rs20 * x + rs21 * y + rs22 * z + finalTz;

      minX = Math.min(minX, wx);
      minY = Math.min(minY, wy);
      minZ = Math.min(minZ, wz);
      maxX = Math.max(maxX, wx);
      maxY = Math.max(maxY, wy);
      maxZ = Math.max(maxZ, wz);
    }

    return {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    };
  }
}
