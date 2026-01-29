import { Vec3 } from "./Vec3";
import { Quat } from "./Quat";

/**
 * Mat4 - 4x4 Matrix utility class
 * Provides matrix operations for 3D transformations
 * Storage is column-major order (OpenGL/WebGPU convention)
 */
export class Mat4 {
  elements: Float32Array; // 16 elements, column-major order

  constructor() {
    this.elements = new Float32Array(16);
  }

  // Factory methods
  static identity(): Mat4 {
    const m = new Mat4();
    m.elements[0] = 1;
    m.elements[5] = 1;
    m.elements[10] = 1;
    m.elements[15] = 1;
    return m;
  }

  static fromTranslation(v: Vec3): Mat4 {
    const m = Mat4.identity();
    m.elements[12] = v.x;
    m.elements[13] = v.y;
    m.elements[14] = v.z;
    return m;
  }

  static fromRotation(q: Quat): Mat4 {
    const m = new Mat4();
    const e = m.elements;

    const x2 = q.x + q.x;
    const y2 = q.y + q.y;
    const z2 = q.z + q.z;
    const xx = q.x * x2;
    const xy = q.x * y2;
    const xz = q.x * z2;
    const yy = q.y * y2;
    const yz = q.y * z2;
    const zz = q.z * z2;
    const wx = q.w * x2;
    const wy = q.w * y2;
    const wz = q.w * z2;

    e[0] = 1 - (yy + zz);
    e[1] = xy + wz;
    e[2] = xz - wy;
    e[3] = 0;

    e[4] = xy - wz;
    e[5] = 1 - (xx + zz);
    e[6] = yz + wx;
    e[7] = 0;

    e[8] = xz + wy;
    e[9] = yz - wx;
    e[10] = 1 - (xx + yy);
    e[11] = 0;

    e[12] = 0;
    e[13] = 0;
    e[14] = 0;
    e[15] = 1;

    return m;
  }

  static fromScale(v: Vec3): Mat4 {
    const m = new Mat4();
    m.elements[0] = v.x;
    m.elements[5] = v.y;
    m.elements[10] = v.z;
    m.elements[15] = 1;
    return m;
  }

  static compose(position: Vec3, rotation: Quat, scale: Vec3): Mat4 {
    const m = Mat4.fromRotation(rotation);
    const e = m.elements;

    // Apply scale
    e[0] *= scale.x;
    e[1] *= scale.x;
    e[2] *= scale.x;
    e[4] *= scale.y;
    e[5] *= scale.y;
    e[6] *= scale.y;
    e[8] *= scale.z;
    e[9] *= scale.z;
    e[10] *= scale.z;

    // Apply translation
    e[12] = position.x;
    e[13] = position.y;
    e[14] = position.z;

    return m;
  }

  // Operations
  multiply(m: Mat4): Mat4 {
    const result = new Mat4();
    const a = this.elements;
    const b = m.elements;
    const r = result.elements;

    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        r[i * 4 + j] =
          a[j] * b[i * 4] +
          a[j + 4] * b[i * 4 + 1] +
          a[j + 8] * b[i * 4 + 2] +
          a[j + 12] * b[i * 4 + 3];
      }
    }

    return result;
  }

  multiplyInPlace(m: Mat4): Mat4 {
    const temp = this.multiply(m);
    this.elements.set(temp.elements);
    return this;
  }

  inverse(): Mat4 | null {
    const e = this.elements;
    const inv = new Float32Array(16);

    inv[0] =
      e[5] * e[10] * e[15] -
      e[5] * e[11] * e[14] -
      e[9] * e[6] * e[15] +
      e[9] * e[7] * e[14] +
      e[13] * e[6] * e[11] -
      e[13] * e[7] * e[10];
    inv[4] =
      -e[4] * e[10] * e[15] +
      e[4] * e[11] * e[14] +
      e[8] * e[6] * e[15] -
      e[8] * e[7] * e[14] -
      e[12] * e[6] * e[11] +
      e[12] * e[7] * e[10];
    inv[8] =
      e[4] * e[9] * e[15] -
      e[4] * e[11] * e[13] -
      e[8] * e[5] * e[15] +
      e[8] * e[7] * e[13] +
      e[12] * e[5] * e[11] -
      e[12] * e[7] * e[9];
    inv[12] =
      -e[4] * e[9] * e[14] +
      e[4] * e[10] * e[13] +
      e[8] * e[5] * e[14] -
      e[8] * e[6] * e[13] -
      e[12] * e[5] * e[10] +
      e[12] * e[6] * e[9];

    inv[1] =
      -e[1] * e[10] * e[15] +
      e[1] * e[11] * e[14] +
      e[9] * e[2] * e[15] -
      e[9] * e[3] * e[14] -
      e[13] * e[2] * e[11] +
      e[13] * e[3] * e[10];
    inv[5] =
      e[0] * e[10] * e[15] -
      e[0] * e[11] * e[14] -
      e[8] * e[2] * e[15] +
      e[8] * e[3] * e[14] +
      e[12] * e[2] * e[11] -
      e[12] * e[3] * e[10];
    inv[9] =
      -e[0] * e[9] * e[15] +
      e[0] * e[11] * e[13] +
      e[8] * e[1] * e[15] -
      e[8] * e[3] * e[13] -
      e[12] * e[1] * e[11] +
      e[12] * e[3] * e[9];
    inv[13] =
      e[0] * e[9] * e[14] -
      e[0] * e[10] * e[13] -
      e[8] * e[1] * e[14] +
      e[8] * e[2] * e[13] +
      e[12] * e[1] * e[10] -
      e[12] * e[2] * e[9];

    inv[2] =
      e[1] * e[6] * e[15] -
      e[1] * e[7] * e[14] -
      e[5] * e[2] * e[15] +
      e[5] * e[3] * e[14] +
      e[13] * e[2] * e[7] -
      e[13] * e[3] * e[6];
    inv[6] =
      -e[0] * e[6] * e[15] +
      e[0] * e[7] * e[14] +
      e[4] * e[2] * e[15] -
      e[4] * e[3] * e[14] -
      e[12] * e[2] * e[7] +
      e[12] * e[3] * e[6];
    inv[10] =
      e[0] * e[5] * e[15] -
      e[0] * e[7] * e[13] -
      e[4] * e[1] * e[15] +
      e[4] * e[3] * e[13] +
      e[12] * e[1] * e[7] -
      e[12] * e[3] * e[5];
    inv[14] =
      -e[0] * e[5] * e[14] +
      e[0] * e[6] * e[13] +
      e[4] * e[1] * e[14] -
      e[4] * e[2] * e[13] -
      e[12] * e[1] * e[6] +
      e[12] * e[2] * e[5];

    inv[3] =
      -e[1] * e[6] * e[11] +
      e[1] * e[7] * e[10] +
      e[5] * e[2] * e[11] -
      e[5] * e[3] * e[10] -
      e[9] * e[2] * e[7] +
      e[9] * e[3] * e[6];
    inv[7] =
      e[0] * e[6] * e[11] -
      e[0] * e[7] * e[10] -
      e[4] * e[2] * e[11] +
      e[4] * e[3] * e[10] +
      e[8] * e[2] * e[7] -
      e[8] * e[3] * e[6];
    inv[11] =
      -e[0] * e[5] * e[11] +
      e[0] * e[7] * e[9] +
      e[4] * e[1] * e[11] -
      e[4] * e[3] * e[9] -
      e[8] * e[1] * e[7] +
      e[8] * e[3] * e[5];
    inv[15] =
      e[0] * e[5] * e[10] -
      e[0] * e[6] * e[9] -
      e[4] * e[1] * e[10] +
      e[4] * e[2] * e[9] +
      e[8] * e[1] * e[6] -
      e[8] * e[2] * e[5];

    const det = e[0] * inv[0] + e[1] * inv[4] + e[2] * inv[8] + e[3] * inv[12];

    if (Math.abs(det) < 1e-10) {
      return null; // Matrix is singular
    }

    const invDet = 1.0 / det;
    const result = new Mat4();
    for (let i = 0; i < 16; i++) {
      result.elements[i] = inv[i] * invDet;
    }

    return result;
  }

  transpose(): Mat4 {
    const m = new Mat4();
    const e = this.elements;
    const r = m.elements;

    r[0] = e[0];
    r[1] = e[4];
    r[2] = e[8];
    r[3] = e[12];
    r[4] = e[1];
    r[5] = e[5];
    r[6] = e[9];
    r[7] = e[13];
    r[8] = e[2];
    r[9] = e[6];
    r[10] = e[10];
    r[11] = e[14];
    r[12] = e[3];
    r[13] = e[7];
    r[14] = e[11];
    r[15] = e[15];

    return m;
  }

  // Decomposition
  decompose(): { position: Vec3; rotation: Quat; scale: Vec3 } | null {
    const e = this.elements;

    // Extract translation
    const position = new Vec3(e[12], e[13], e[14]);

    // Extract scale
    const sx = Math.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    const sy = Math.sqrt(e[4] * e[4] + e[5] * e[5] + e[6] * e[6]);
    const sz = Math.sqrt(e[8] * e[8] + e[9] * e[9] + e[10] * e[10]);

    const scale = new Vec3(sx, sy, sz);

    // Check for zero scale
    if (sx < 1e-10 || sy < 1e-10 || sz < 1e-10) {
      return null;
    }

    // Extract rotation by removing scale
    const m11 = e[0] / sx;
    const m12 = e[1] / sx;
    const m13 = e[2] / sx;
    const m21 = e[4] / sy;
    const m22 = e[5] / sy;
    const m23 = e[6] / sy;
    const m31 = e[8] / sz;
    const m32 = e[9] / sz;
    const m33 = e[10] / sz;

    // Convert rotation matrix to quaternion
    const trace = m11 + m22 + m33;
    let rotation: Quat;

    if (trace > 0) {
      const s = 0.5 / Math.sqrt(trace + 1.0);
      rotation = new Quat(
        (m23 - m32) * s,
        (m31 - m13) * s,
        (m12 - m21) * s,
        0.25 / s,
      );
    } else if (m11 > m22 && m11 > m33) {
      const s = 2.0 * Math.sqrt(1.0 + m11 - m22 - m33);
      rotation = new Quat(
        0.25 * s,
        (m21 + m12) / s,
        (m31 + m13) / s,
        (m23 - m32) / s,
      );
    } else if (m22 > m33) {
      const s = 2.0 * Math.sqrt(1.0 + m22 - m11 - m33);
      rotation = new Quat(
        (m21 + m12) / s,
        0.25 * s,
        (m32 + m23) / s,
        (m31 - m13) / s,
      );
    } else {
      const s = 2.0 * Math.sqrt(1.0 + m33 - m11 - m22);
      rotation = new Quat(
        (m31 + m13) / s,
        (m32 + m23) / s,
        0.25 * s,
        (m12 - m21) / s,
      );
    }

    return { position, rotation, scale };
  }

  // Transformation
  transformPoint(v: Vec3): Vec3 {
    const e = this.elements;
    const x = v.x;
    const y = v.y;
    const z = v.z;
    const w = e[3] * x + e[7] * y + e[11] * z + e[15];

    return new Vec3(
      (e[0] * x + e[4] * y + e[8] * z + e[12]) / w,
      (e[1] * x + e[5] * y + e[9] * z + e[13]) / w,
      (e[2] * x + e[6] * y + e[10] * z + e[14]) / w,
    );
  }

  transformDirection(v: Vec3): Vec3 {
    const e = this.elements;
    return new Vec3(
      e[0] * v.x + e[4] * v.y + e[8] * v.z,
      e[1] * v.x + e[5] * v.y + e[9] * v.z,
      e[2] * v.x + e[6] * v.y + e[10] * v.z,
    );
  }

  // Utility
  clone(): Mat4 {
    const m = new Mat4();
    m.elements.set(this.elements);
    return m;
  }
}
