import { Vec3 } from "./Vec3";
import { Camera } from "../Camera";

/**
 * Ray - Ray utility class
 * Provides ray operations for 3D picking and intersection tests
 */
export class Ray {
  origin: Vec3;
  direction: Vec3; // Should be normalized

  constructor(origin: Vec3, direction: Vec3) {
    this.origin = origin;
    this.direction = direction;
  }

  /**
   * Create a ray from screen coordinates
   * @param screenX - Screen X coordinate (0 to canvasWidth)
   * @param screenY - Screen Y coordinate (0 to canvasHeight)
   * @param canvasWidth - Canvas width in pixels
   * @param canvasHeight - Canvas height in pixels
   * @param camera - Camera instance
   */
  static fromScreenPoint(
    screenX: number,
    screenY: number,
    canvasWidth: number,
    canvasHeight: number,
    camera: Camera,
  ): Ray {
    // Convert screen coordinates to normalized device coordinates (-1 to 1)
    const ndcX = (screenX / canvasWidth) * 2 - 1;
    const ndcY = -(screenY / canvasHeight) * 2 + 1; // Flip Y axis

    // Create inverse view-projection matrix
    const invViewProj = Ray.invertMatrix(camera.viewProjectionMatrix);

    // Transform NDC point to world space (near plane)
    const nearPoint = Ray.transformPoint(invViewProj, ndcX, ndcY, -1);
    const farPoint = Ray.transformPoint(invViewProj, ndcX, ndcY, 1);

    // Ray origin is camera position
    const origin = new Vec3(
      camera.position[0],
      camera.position[1],
      camera.position[2],
    );

    // Ray direction is from near to far point
    const direction = new Vec3(
      farPoint.x - nearPoint.x,
      farPoint.y - nearPoint.y,
      farPoint.z - nearPoint.z,
    ).normalize();

    return new Ray(origin, direction);
  }

  /**
   * Get point at distance along ray
   * @param distance - Distance along ray
   */
  at(distance: number): Vec3 {
    return new Vec3(
      this.origin.x + this.direction.x * distance,
      this.origin.y + this.direction.y * distance,
      this.origin.z + this.direction.z * distance,
    );
  }

  /**
   * Intersect ray with plane
   * @param planeOrigin - Point on the plane
   * @param planeNormal - Normal vector of the plane (should be normalized)
   * @returns Distance along ray to intersection, or null if parallel/no intersection
   */
  intersectPlane(planeOrigin: Vec3, planeNormal: Vec3): number | null {
    const denom = this.direction.dot(planeNormal);

    // Check if ray is parallel to plane
    if (Math.abs(denom) < 1e-6) {
      return null;
    }

    const t = planeOrigin.subtract(this.origin).dot(planeNormal) / denom;

    // Return null if intersection is behind ray origin
    if (t < 0) {
      return null;
    }

    return t;
  }

  /**
   * Compute distance from ray to a point
   * @param point - Point in 3D space
   */
  distanceToPoint(point: Vec3): number {
    const v = point.subtract(this.origin);
    const t = v.dot(this.direction);

    // If t < 0, closest point is the ray origin
    if (t < 0) {
      return this.origin.distance(point);
    }

    // Closest point on ray
    const closestPoint = this.at(t);
    return closestPoint.distance(point);
  }

  /**
   * Compute distance from ray to a line segment (for capsule hit testing)
   * @param segmentStart - Start point of line segment
   * @param segmentEnd - End point of line segment
   */
  distanceToSegment(segmentStart: Vec3, segmentEnd: Vec3): number {
    const segmentDir = segmentEnd.subtract(segmentStart);
    const segmentLength = segmentDir.length();

    // Handle degenerate segment (point)
    if (segmentLength < 1e-10) {
      return this.distanceToPoint(segmentStart);
    }

    const segmentDirNorm = segmentDir.divide(segmentLength);

    // Compute closest points between ray and line segment
    const w0 = this.origin.subtract(segmentStart);
    const a = this.direction.dot(this.direction);
    const b = this.direction.dot(segmentDirNorm);
    const c = segmentDirNorm.dot(segmentDirNorm);
    const d = this.direction.dot(w0);
    const e = segmentDirNorm.dot(w0);

    const denom = a * c - b * b;
    let t = 0; // Parameter on ray
    let s = 0; // Parameter on segment

    if (Math.abs(denom) < 1e-6) {
      // Ray and segment are parallel
      t = 0;
      s = e / c;
    } else {
      t = (b * e - c * d) / denom;
      s = (a * e - b * d) / denom;
    }

    // Clamp t to non-negative (ray starts at origin)
    t = Math.max(0, t);

    // Clamp s to segment bounds [0, segmentLength]
    s = Math.max(0, Math.min(segmentLength, s));

    const pointOnRay = this.at(t);
    const pointOnSegment = segmentStart.add(segmentDirNorm.multiply(s));

    return pointOnRay.distance(pointOnSegment);
  }

  /**
   * Clone this ray
   */
  clone(): Ray {
    return new Ray(this.origin.clone(), this.direction.clone());
  }

  /**
   * Transform ray by a matrix
   * @param matrix - Transformation matrix
   */
  transform(matrix: { transformPoint(v: Vec3): Vec3; transformVector(v: Vec3): Vec3 }): Ray {
    const newOrigin = matrix.transformPoint(this.origin);
    const newDirection = matrix.transformVector(this.direction).normalize();
    return new Ray(newOrigin, newDirection);
  }

  /**
   * Intersect ray with triangle using Möller–Trumbore algorithm
   * @param v0 - First vertex of triangle
   * @param v1 - Second vertex of triangle
   * @param v2 - Third vertex of triangle
   * @returns Distance to intersection, or null if no intersection
   */
  intersectTriangle(v0: Vec3, v1: Vec3, v2: Vec3): number | null {
    const EPSILON = 1e-6;
    
    const edge1 = v1.subtract(v0);
    const edge2 = v2.subtract(v0);
    
    const h = this.direction.cross(edge2);
    const a = edge1.dot(h);
    
    // Ray is parallel to triangle
    if (Math.abs(a) < EPSILON) {
      return null;
    }
    
    const f = 1.0 / a;
    const s = this.origin.subtract(v0);
    const u = f * s.dot(h);
    
    // Intersection is outside triangle
    if (u < 0.0 || u > 1.0) {
      return null;
    }
    
    const q = s.cross(edge1);
    const v = f * this.direction.dot(q);
    
    // Intersection is outside triangle
    if (v < 0.0 || u + v > 1.0) {
      return null;
    }
    
    const t = f * edge2.dot(q);
    
    // Intersection is behind ray origin
    if (t < EPSILON) {
      return null;
    }
    
    return t;
  }

  // Helper methods for matrix operations
  private static invertMatrix(m: Float32Array): Float32Array {
    const inv = new Float32Array(16);
    const e = m;

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
      // Return identity if singular
      const identity = new Float32Array(16);
      identity[0] = identity[5] = identity[10] = identity[15] = 1;
      return identity;
    }

    const invDet = 1.0 / det;
    const result = new Float32Array(16);
    for (let i = 0; i < 16; i++) {
      result[i] = inv[i] * invDet;
    }

    return result;
  }

  private static transformPoint(
    m: Float32Array,
    x: number,
    y: number,
    z: number,
  ): Vec3 {
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    return new Vec3(
      (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
      (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
      (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
    );
  }
}
