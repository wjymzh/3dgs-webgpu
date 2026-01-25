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
   * 获取世界空间的 bounding box（考虑变换）
   * 简化计算：仅考虑位置平移，忽略旋转和非均匀缩放
   */
  getWorldBoundingBox(): MeshBoundingBox | null {
    if (!this.localBoundingBox) return null;

    const local = this.localBoundingBox;
    const [sx, sy, sz] = this.scale;
    const [tx, ty, tz] = this.position;

    // 简化：缩放后平移
    const worldMin: [number, number, number] = [
      local.min[0] * sx + tx,
      local.min[1] * sy + ty,
      local.min[2] * sz + tz,
    ];
    const worldMax: [number, number, number] = [
      local.max[0] * sx + tx,
      local.max[1] * sy + ty,
      local.max[2] * sz + tz,
    ];
    const worldCenter: [number, number, number] = [
      local.center[0] * sx + tx,
      local.center[1] * sy + ty,
      local.center[2] * sz + tz,
    ];
    // 半径按最大缩放因子计算
    const maxScale = Math.max(sx, sy, sz);
    const worldRadius = local.radius * maxScale;

    return {
      min: worldMin,
      max: worldMax,
      center: worldCenter,
      radius: worldRadius,
    };
  }

  /**
   * 设置位置
   */
  setPosition(x: number, y: number, z: number): void {
    this.position[0] = x;
    this.position[1] = y;
    this.position[2] = z;
    this.updateModelMatrix();
  }

  /**
   * 获取位置
   */
  getPosition(): [number, number, number] {
    return [this.position[0], this.position[1], this.position[2]];
  }

  /**
   * 设置旋转（欧拉角，弧度）
   */
  setRotation(rx: number, ry: number, rz: number): void {
    this.rotation[0] = rx;
    this.rotation[1] = ry;
    this.rotation[2] = rz;
    this.updateModelMatrix();
  }

  /**
   * 获取旋转
   */
  getRotation(): [number, number, number] {
    return [this.rotation[0], this.rotation[1], this.rotation[2]];
  }

  /**
   * 设置缩放
   */
  setScale(sx: number, sy: number, sz: number): void {
    this.scale[0] = sx;
    this.scale[1] = sy;
    this.scale[2] = sz;
    this.updateModelMatrix();
  }

  /**
   * 获取缩放
   */
  getScale(): [number, number, number] {
    return [this.scale[0], this.scale[1], this.scale[2]];
  }

  /**
   * 更新模型矩阵 (Scale * Rotation * Translation)
   */
  updateModelMatrix(): void {
    const [sx, sy, sz] = this.scale;
    const [rx, ry, rz] = this.rotation;
    const [tx, ty, tz] = this.position;

    // 计算旋转矩阵（ZYX 欧拉角顺序）
    const cx = Math.cos(rx), sx_ = Math.sin(rx);
    const cy = Math.cos(ry), sy_ = Math.sin(ry);
    const cz = Math.cos(rz), sz_ = Math.sin(rz);

    // 组合 Scale * Rotation
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

    // Translation
    this.modelMatrix[12] = tx;
    this.modelMatrix[13] = ty;
    this.modelMatrix[14] = tz;
    this.modelMatrix[15] = 1;
  }

  /**
   * 重置变换
   */
  resetTransform(): void {
    this.position.set([0, 0, 0]);
    this.rotation.set([0, 0, 0]);
    this.scale.set([1, 1, 1]);
    this.updateModelMatrix();
  }

  /**
   * 销毁 GPU 资源
   */
  destroy(): void {
    this.vertexBuffer.destroy();
    if (this.indexBuffer) {
      this.indexBuffer.destroy();
    }
  }
}
