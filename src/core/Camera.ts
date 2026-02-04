/**
 * Camera - 相机矩阵计算
 * 只负责视图矩阵和投影矩阵
 */
export class Camera {
  // 相机参数
  position: Float32Array = new Float32Array([0, 0, 5]);
  target: Float32Array = new Float32Array([0, 0, 0]);
  up: Float32Array = new Float32Array([0, 1, 0]);

  // 投影参数
  fov: number = Math.PI / 4; // 45度
  aspect: number = 1;
  near: number = 0.1;  // 增大近平面以提高深度精度 (参考实现使用 0.1)
  far: number = 1000;  // 减小远平面以提高深度精度

  // 矩阵
  viewMatrix: Float32Array = new Float32Array(16);
  projectionMatrix: Float32Array = new Float32Array(16);
  viewProjectionMatrix: Float32Array = new Float32Array(16);

  constructor() {
    this.updateMatrix();
  }

  /**
   * 设置宽高比
   */
  setAspect(aspect: number): void {
    this.aspect = aspect;
  }

  /**
   * 更新视图和投影矩阵
   */
  updateMatrix(): void {
    this.updateViewMatrix();
    this.updateProjectionMatrix();
    this.multiplyMatrices(
      this.viewProjectionMatrix,
      this.projectionMatrix,
      this.viewMatrix,
    );
  }

  /**
   * 计算视图矩阵 (lookAt)
   */
  private updateViewMatrix(): void {
    const eye = this.position;
    const target = this.target;
    const up = this.up;

    // 计算相机坐标系
    const zAxis = this.normalize(this.subtract(eye, target));
    const xAxis = this.normalize(this.cross(up, zAxis));
    const yAxis = this.cross(zAxis, xAxis);

    // 构建视图矩阵 (列主序)
    this.viewMatrix[0] = xAxis[0];
    this.viewMatrix[1] = yAxis[0];
    this.viewMatrix[2] = zAxis[0];
    this.viewMatrix[3] = 0;

    this.viewMatrix[4] = xAxis[1];
    this.viewMatrix[5] = yAxis[1];
    this.viewMatrix[6] = zAxis[1];
    this.viewMatrix[7] = 0;

    this.viewMatrix[8] = xAxis[2];
    this.viewMatrix[9] = yAxis[2];
    this.viewMatrix[10] = zAxis[2];
    this.viewMatrix[11] = 0;

    this.viewMatrix[12] = -this.dot(xAxis, eye);
    this.viewMatrix[13] = -this.dot(yAxis, eye);
    this.viewMatrix[14] = -this.dot(zAxis, eye);
    this.viewMatrix[15] = 1;
  }

  /**
   * 计算投影矩阵 (透视投影)
   */
  private updateProjectionMatrix(): void {
    const f = 1.0 / Math.tan(this.fov / 2);
    const rangeInv = 1 / (this.near - this.far);

    // 列主序
    this.projectionMatrix[0] = f / this.aspect;
    this.projectionMatrix[1] = 0;
    this.projectionMatrix[2] = 0;
    this.projectionMatrix[3] = 0;

    this.projectionMatrix[4] = 0;
    this.projectionMatrix[5] = f;
    this.projectionMatrix[6] = 0;
    this.projectionMatrix[7] = 0;

    this.projectionMatrix[8] = 0;
    this.projectionMatrix[9] = 0;
    this.projectionMatrix[10] = (this.near + this.far) * rangeInv;
    this.projectionMatrix[11] = -1;

    this.projectionMatrix[12] = 0;
    this.projectionMatrix[13] = 0;
    this.projectionMatrix[14] = this.near * this.far * rangeInv * 2;
    this.projectionMatrix[15] = 0;
  }

  // ========== 向量/矩阵工具函数 ==========

  private subtract(a: Float32Array, b: Float32Array): Float32Array {
    return new Float32Array([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
  }

  private cross(a: Float32Array, b: Float32Array): Float32Array {
    return new Float32Array([
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ]);
  }

  private dot(a: Float32Array, b: Float32Array): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  private normalize(v: Float32Array): Float32Array {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len < 1e-10) return new Float32Array([0, 0, 1]); // 返回默认方向，避免除零
    return new Float32Array([v[0] / len, v[1] / len, v[2] / len]);
  }

  private multiplyMatrices(
    out: Float32Array,
    a: Float32Array,
    b: Float32Array,
  ): void {
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[i * 4 + j] =
          a[j] * b[i * 4] +
          a[j + 4] * b[i * 4 + 1] +
          a[j + 8] * b[i * 4 + 2] +
          a[j + 12] * b[i * 4 + 3];
      }
    }
  }
}
