import { Vec3 } from "./Vec3";

/**
 * Quat - Quaternion utility class
 * Provides quaternion operations for 3D rotations
 */
export class Quat {
  x: number;
  y: number;
  z: number;
  w: number;

  constructor(x: number = 0, y: number = 0, z: number = 0, w: number = 1) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
  }

  // Factory methods
  static identity(): Quat {
    return new Quat(0, 0, 0, 1);
  }

  /**
   * Create quaternion from Euler angles (ZYX order)
   * @param x - Rotation around X axis in radians
   * @param y - Rotation around Y axis in radians
   * @param z - Rotation around Z axis in radians
   */
  static fromEuler(x: number, y: number, z: number): Quat {
    const cx = Math.cos(x * 0.5);
    const cy = Math.cos(y * 0.5);
    const cz = Math.cos(z * 0.5);
    const sx = Math.sin(x * 0.5);
    const sy = Math.sin(y * 0.5);
    const sz = Math.sin(z * 0.5);

    // ZYX order
    return new Quat(
      sx * cy * cz - cx * sy * sz,
      cx * sy * cz + sx * cy * sz,
      cx * cy * sz - sx * sy * cz,
      cx * cy * cz + sx * sy * sz,
    );
  }

  /**
   * Create quaternion from axis-angle representation
   * @param axis - Rotation axis (should be normalized)
   * @param angle - Rotation angle in radians
   */
  static fromAxisAngle(axis: Vec3, angle: number): Quat {
    const halfAngle = angle * 0.5;
    const s = Math.sin(halfAngle);
    return new Quat(axis.x * s, axis.y * s, axis.z * s, Math.cos(halfAngle));
  }

  // Operations
  /**
   * Multiply this quaternion by another (this * q)
   */
  multiply(q: Quat): Quat {
    return new Quat(
      this.w * q.x + this.x * q.w + this.y * q.z - this.z * q.y,
      this.w * q.y - this.x * q.z + this.y * q.w + this.z * q.x,
      this.w * q.z + this.x * q.y - this.y * q.x + this.z * q.w,
      this.w * q.w - this.x * q.x - this.y * q.y - this.z * q.z,
    );
  }

  /**
   * Convert quaternion to Euler angles (ZYX order)
   */
  toEuler(): Vec3 {
    // Roll (x-axis rotation)
    const sinr_cosp = 2 * (this.w * this.x + this.y * this.z);
    const cosr_cosp = 1 - 2 * (this.x * this.x + this.y * this.y);
    const roll = Math.atan2(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    const sinp = 2 * (this.w * this.y - this.z * this.x);
    let pitch: number;
    if (Math.abs(sinp) >= 1) {
      pitch = (Math.sign(sinp) * Math.PI) / 2; // Use 90 degrees if out of range
    } else {
      pitch = Math.asin(sinp);
    }

    // Yaw (z-axis rotation)
    const siny_cosp = 2 * (this.w * this.z + this.x * this.y);
    const cosy_cosp = 1 - 2 * (this.y * this.y + this.z * this.z);
    const yaw = Math.atan2(siny_cosp, cosy_cosp);

    return new Vec3(roll, pitch, yaw);
  }

  // Normalization
  normalize(): Quat {
    const len = Math.sqrt(
      this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w,
    );
    if (len < 1e-10) {
      return Quat.identity();
    }
    return new Quat(this.x / len, this.y / len, this.z / len, this.w / len);
  }

  normalizeInPlace(): Quat {
    const len = Math.sqrt(
      this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w,
    );
    if (len < 1e-10) {
      this.x = 0;
      this.y = 0;
      this.z = 0;
      this.w = 1;
      return this;
    }
    this.x /= len;
    this.y /= len;
    this.z /= len;
    this.w /= len;
    return this;
  }

  // Interpolation
  /**
   * Spherical linear interpolation between this quaternion and another
   * @param q - Target quaternion
   * @param t - Interpolation factor (0 to 1)
   */
  slerp(q: Quat, t: number): Quat {
    let dot = this.x * q.x + this.y * q.y + this.z * q.z + this.w * q.w;

    // If the dot product is negative, slerp won't take the shorter path
    // Fix by reversing one quaternion
    let q2 = q;
    if (dot < 0) {
      q2 = new Quat(-q.x, -q.y, -q.z, -q.w);
      dot = -dot;
    }

    // If quaternions are very close, use linear interpolation
    if (dot > 0.9995) {
      return new Quat(
        this.x + t * (q2.x - this.x),
        this.y + t * (q2.y - this.y),
        this.z + t * (q2.z - this.z),
        this.w + t * (q2.w - this.w),
      ).normalize();
    }

    // Calculate coefficients
    const theta = Math.acos(dot);
    const sinTheta = Math.sin(theta);
    const w1 = Math.sin((1 - t) * theta) / sinTheta;
    const w2 = Math.sin(t * theta) / sinTheta;

    return new Quat(
      this.x * w1 + q2.x * w2,
      this.y * w1 + q2.y * w2,
      this.z * w1 + q2.z * w2,
      this.w * w1 + q2.w * w2,
    );
  }

  // Utility
  clone(): Quat {
    return new Quat(this.x, this.y, this.z, this.w);
  }
}
