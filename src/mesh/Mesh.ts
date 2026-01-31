/**
 * Bounding Box 结构（与 GSSplatRenderer 共享接口）
 */
export interface MeshBoundingBox {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  radius: number; // bounding sphere 半径
}

/**
 * Mesh - 网格数据结构
 * 存储 GPUBuffer + 变换属性
 */
export class Mesh {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer | null;
  vertexCount: number;
  indexCount: number;
  modelMatrix: Float32Array;

  // 顶点格式信息
  hasUV: boolean = false;
  indexFormat: 'uint16' | 'uint32' = 'uint16';

  // 变换属性（分离存储，便于 Gizmo 操作）
  position: Float32Array = new Float32Array([0, 0, 0]);
  rotation: Float32Array = new Float32Array([0, 0, 0]); // 欧拉角 (弧度)
  scale: Float32Array = new Float32Array([1, 1, 1]);

  // 本地空间的 bounding box（加载时计算，不随变换更新）
  private localBoundingBox: MeshBoundingBox | null = null;

  constructor(
    vertexBuffer: GPUBuffer,
    vertexCount: number,
    indexBuffer: GPUBuffer | null = null,
    indexCount: number = 0,
    boundingBox?: MeshBoundingBox
  ) {
    this.vertexBuffer = vertexBuffer;
    this.vertexCount = vertexCount;
    this.indexBuffer = indexBuffer;
    this.indexCount = indexCount;
    this.modelMatrix = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    this.localBoundingBox = boundingBox || null;
  }

  /**
   * 获取顶点 stride（字节数）
   */
  getVertexStride(): number {
    return this.hasUV ? 32 : 24;
  }

  /**
   * 设置本地 bounding box
   */
  setBoundingBox(bbox: MeshBoundingBox): void {
    this.localBoundingBox = bbox;
  }

  /**
   * 获取本地 bounding box
   */
  getLocalBoundingBox(): MeshBoundingBox | null {
    return this.localBoundingBox;
  }

  /**
   * 获取世界空间的 bounding box（考虑完整变换：缩放、旋转、平移）
   */
  getWorldBoundingBox(): MeshBoundingBox | null {
    if (!this.localBoundingBox) return null;

    const local = this.localBoundingBox;
    
    // 获取本地包围盒的 8 个角点
    const corners: [number, number, number][] = [
      [local.min[0], local.min[1], local.min[2]],
      [local.max[0], local.min[1], local.min[2]],
      [local.min[0], local.max[1], local.min[2]],
      [local.max[0], local.max[1], local.min[2]],
      [local.min[0], local.min[1], local.max[2]],
      [local.max[0], local.min[1], local.max[2]],
      [local.min[0], local.max[1], local.max[2]],
      [local.max[0], local.max[1], local.max[2]],
    ];
    
    // 使用 modelMatrix 变换所有角点
    const m = this.modelMatrix;
    const transformedCorners: [number, number, number][] = corners.map(([x, y, z]) => {
      const tx = m[0] * x + m[4] * y + m[8] * z + m[12];
      const ty = m[1] * x + m[5] * y + m[9] * z + m[13];
      const tz = m[2] * x + m[6] * y + m[10] * z + m[14];
      return [tx, ty, tz];
    });
    
    // 计算变换后的 AABB
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    
    for (const [x, y, z] of transformedCorners) {
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    }
    
    const worldMin: [number, number, number] = [minX, minY, minZ];
    const worldMax: [number, number, number] = [maxX, maxY, maxZ];
    const worldCenter: [number, number, number] = [
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    ];
    
    const dx = maxX - minX;
    const dy = maxY - minY;
    const dz = maxZ - minZ;
    const worldRadius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    return { min: worldMin, max: worldMax, center: worldCenter, radius: worldRadius };
  }

  setPosition(x: number, y: number, z: number): void {
    this.position[0] = x;
    this.position[1] = y;
    this.position[2] = z;
    this.updateModelMatrix();
  }

  getPosition(): [number, number, number] {
    return [this.position[0], this.position[1], this.position[2]];
  }

  setRotation(rx: number, ry: number, rz: number): void {
    this.rotation[0] = rx;
    this.rotation[1] = ry;
    this.rotation[2] = rz;
    this.updateModelMatrix();
  }

  getRotation(): [number, number, number] {
    return [this.rotation[0], this.rotation[1], this.rotation[2]];
  }

  setScale(sx: number, sy: number, sz: number): void {
    this.scale[0] = sx;
    this.scale[1] = sy;
    this.scale[2] = sz;
    this.updateModelMatrix();
  }

  getScale(): [number, number, number] {
    return [this.scale[0], this.scale[1], this.scale[2]];
  }

  updateModelMatrix(): void {
    const [sx, sy, sz] = this.scale;
    const [rx, ry, rz] = this.rotation;
    const [tx, ty, tz] = this.position;

    const cx = Math.cos(rx), sx_ = Math.sin(rx);
    const cy = Math.cos(ry), sy_ = Math.sin(ry);
    const cz = Math.cos(rz), sz_ = Math.sin(rz);

    this.modelMatrix[0] = sx * (cy * cz);
    this.modelMatrix[1] = sx * (cy * sz_);
    this.modelMatrix[2] = sx * (-sy_);
    this.modelMatrix[3] = 0;

    this.modelMatrix[4] = sy * (sx_ * sy_ * cz - cx * sz_);
    this.modelMatrix[5] = sy * (sx_ * sy_ * sz_ + cx * cz);
    this.modelMatrix[6] = sy * (sx_ * cy);
    this.modelMatrix[7] = 0;

    this.modelMatrix[8] = sz * (cx * sy_ * cz + sx_ * sz_);
    this.modelMatrix[9] = sz * (cx * sy_ * sz_ - sx_ * cz);
    this.modelMatrix[10] = sz * (cx * cy);
    this.modelMatrix[11] = 0;

    this.modelMatrix[12] = tx;
    this.modelMatrix[13] = ty;
    this.modelMatrix[14] = tz;
    this.modelMatrix[15] = 1;
  }

  resetTransform(): void {
    this.position.set([0, 0, 0]);
    this.rotation.set([0, 0, 0]);
    this.scale.set([1, 1, 1]);
    this.updateModelMatrix();
  }

  destroy(): void {
    this.vertexBuffer.destroy();
    if (this.indexBuffer) {
      this.indexBuffer.destroy();
    }
  }
}
