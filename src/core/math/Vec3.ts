/**
 * Vec3 - 3D Vector utility class
 * Provides vector operations for 3D mathematics
 */
export class Vec3 {
  x: number;
  y: number;
  z: number;

  constructor(x: number = 0, y: number = 0, z: number = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  // Factory methods
  static fromArray(arr: Float32Array | number[], offset: number = 0): Vec3 {
    return new Vec3(arr[offset], arr[offset + 1], arr[offset + 2]);
  }

  static zero(): Vec3 {
    return new Vec3(0, 0, 0);
  }

  static one(): Vec3 {
    return new Vec3(1, 1, 1);
  }

  // Basic operations (return new Vec3)
  add(v: Vec3): Vec3 {
    return new Vec3(this.x + v.x, this.y + v.y, this.z + v.z);
  }

  subtract(v: Vec3): Vec3 {
    return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z);
  }

  multiply(scalar: number): Vec3 {
    return new Vec3(this.x * scalar, this.y * scalar, this.z * scalar);
  }

  divide(scalar: number): Vec3 {
    return new Vec3(this.x / scalar, this.y / scalar, this.z / scalar);
  }

  // In-place operations (modify this)
  addInPlace(v: Vec3): Vec3 {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }

  subtractInPlace(v: Vec3): Vec3 {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }

  multiplyInPlace(scalar: number): Vec3 {
    this.x *= scalar;
    this.y *= scalar;
    this.z *= scalar;
    return this;
  }

  // Vector operations
  dot(v: Vec3): number {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }

  cross(v: Vec3): Vec3 {
    return new Vec3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x,
    );
  }

  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }

  lengthSquared(): number {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }

  distance(v: Vec3): number {
    const dx = this.x - v.x;
    const dy = this.y - v.y;
    const dz = this.z - v.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  distanceSquared(v: Vec3): number {
    const dx = this.x - v.x;
    const dy = this.y - v.y;
    const dz = this.z - v.z;
    return dx * dx + dy * dy + dz * dz;
  }

  // Normalization
  normalize(): Vec3 {
    const len = this.length();
    if (len < 1e-10) {
      // Handle zero-length vector - return default direction
      return new Vec3(0, 0, 1);
    }
    return new Vec3(this.x / len, this.y / len, this.z / len);
  }

  normalizeInPlace(): Vec3 {
    const len = this.length();
    if (len < 1e-10) {
      // Handle zero-length vector - set to default direction
      this.x = 0;
      this.y = 0;
      this.z = 1;
      return this;
    }
    this.x /= len;
    this.y /= len;
    this.z /= len;
    return this;
  }

  // Utility
  clone(): Vec3 {
    return new Vec3(this.x, this.y, this.z);
  }

  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  set(x: number, y: number, z: number): Vec3 {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }

  /**
   * Check if this vector equals another vector
   */
  equals(v: Vec3): boolean {
    return this.x === v.x && this.y === v.y && this.z === v.z;
  }

  /**
   * Check if this vector approximately equals another vector
   */
  equalsApprox(v: Vec3, epsilon: number = 1e-6): boolean {
    return (
      Math.abs(this.x - v.x) < epsilon &&
      Math.abs(this.y - v.y) < epsilon &&
      Math.abs(this.z - v.z) < epsilon
    );
  }
}
