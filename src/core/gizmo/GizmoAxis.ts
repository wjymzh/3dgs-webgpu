import { Vec3 } from "../math/Vec3";
import { Ray } from "../math/Ray";

/**
 * AxisType - Enum for axis types (X, Y, Z)
 */
export enum AxisType {
  X = 0,
  Y = 1,
  Z = 2,
}

/**
 * GizmoMode - Enum for gizmo manipulation modes
 */
export enum GizmoMode {
  Translate = 0,
  Rotate = 1,
  Scale = 2,
}

/**
 * GizmoAxisConfig - Configuration for a gizmo axis
 */
export interface GizmoAxisConfig {
  type: AxisType;
  mode: GizmoMode;
  color: Vec3; // Base color (e.g., red for X)
  direction: Vec3; // Axis direction in local space
}

/**
 * GizmoAxis - Represents a single axis or ring component of the gizmo
 * Handles geometry creation, hit testing, and visual state for one axis
 */
export class GizmoAxis {
  config: GizmoAxisConfig;

  // Visual state
  isHovered: boolean = false;
  isActive: boolean = false;

  // Geometry (GPU buffers created lazily)
  private vertexBuffer: GPUBuffer | null = null;
  private indexBuffer: GPUBuffer | null = null;
  private vertexCount: number = 0;
  private indexCount: number = 0;

  constructor(config: GizmoAxisConfig) {
    this.config = config;
  }

  /**
   * Create GPU buffers for this axis geometry
   * @param device - WebGPU device
   */
  createGeometry(device: GPUDevice): void {
    // Generate geometry based on mode
    let geometryData: { vertices: Float32Array; indices: Uint16Array };

    switch (this.config.mode) {
      case GizmoMode.Translate:
        geometryData = this.createTranslateGeometry();
        break;
      case GizmoMode.Rotate:
        geometryData = this.createRotateGeometry();
        break;
      case GizmoMode.Scale:
        geometryData = this.createScaleGeometry();
        break;
      default:
        throw new Error(`Unknown gizmo mode: ${this.config.mode}`);
    }

    // Store counts
    this.vertexCount = geometryData.vertices.length / 6; // 6 floats per vertex (pos + color)
    this.indexCount = geometryData.indices.length;

    // Create vertex buffer
    this.vertexBuffer = device.createBuffer({
      size: geometryData.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(
      geometryData.vertices,
    );
    this.vertexBuffer.unmap();

    // Create index buffer
    this.indexBuffer = device.createBuffer({
      size: geometryData.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(
      geometryData.indices,
    );
    this.indexBuffer.unmap();
  }

  /**
   * Get vertex buffer
   */
  getVertexBuffer(): GPUBuffer | null {
    return this.vertexBuffer;
  }

  /**
   * Get index buffer
   */
  getIndexBuffer(): GPUBuffer | null {
    return this.indexBuffer;
  }

  /**
   * Get vertex count
   */
  getVertexCount(): number {
    return this.vertexCount;
  }

  /**
   * Get index count
   */
  getIndexCount(): number {
    return this.indexCount;
  }

  /**
   * Create geometry for translate mode (arrow: cylinder + cone)
   * @private
   */
  private createTranslateGeometry(): {
    vertices: Float32Array;
    indices: Uint16Array;
  } {
    const cylinderLength = 0.8;
    const cylinderRadius = 0.04;
    const coneLength = 0.25;
    const coneRadius = 0.1;
    const segments = 12;

    const vertices: number[] = [];
    const indices: number[] = [];

    const color = this.config.color;
    const dir = this.config.direction;

    // Helper to add vertex
    const addVertex = (pos: Vec3) => {
      vertices.push(pos.x, pos.y, pos.z, color.x, color.y, color.z);
    };

    // Compute perpendicular vectors for cylinder/cone cross-section
    let perpX: Vec3, perpY: Vec3;
    if (Math.abs(dir.x) < 0.9) {
      perpX = new Vec3(1, 0, 0).cross(dir).normalize();
    } else {
      perpX = new Vec3(0, 1, 0).cross(dir).normalize();
    }
    perpY = dir.cross(perpX).normalize();

    // Generate cylinder
    const cylinderStart = new Vec3(0, 0, 0);
    const cylinderEnd = dir.multiply(cylinderLength);

    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      const offset = perpX
        .multiply(cos * cylinderRadius)
        .add(perpY.multiply(sin * cylinderRadius));

      // Bottom vertex
      addVertex(cylinderStart.add(offset));
      // Top vertex
      addVertex(cylinderEnd.add(offset));
    }

    // Cylinder indices
    for (let i = 0; i < segments; i++) {
      const base = i * 2;
      indices.push(base, base + 1, base + 2);
      indices.push(base + 1, base + 3, base + 2);
    }

    // Generate cone
    const coneBase = dir.multiply(cylinderLength);
    const coneTip = dir.multiply(cylinderLength + coneLength);

    const coneBaseIndex = vertices.length / 6;

    // Cone base vertices
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      const offset = perpX
        .multiply(cos * coneRadius)
        .add(perpY.multiply(sin * coneRadius));
      addVertex(coneBase.add(offset));
    }

    // Cone tip
    const coneTipIndex = vertices.length / 6;
    addVertex(coneTip);

    // Cone indices
    for (let i = 0; i < segments; i++) {
      indices.push(coneBaseIndex + i, coneTipIndex, coneBaseIndex + i + 1);
    }

    return {
      vertices: new Float32Array(vertices),
      indices: new Uint16Array(indices),
    };
  }

  /**
   * Create geometry for rotate mode (torus ring)
   * @private
   */
  private createRotateGeometry(): {
    vertices: Float32Array;
    indices: Uint16Array;
  } {
    const majorRadius = 1.0;
    const minorRadius = 0.05; // 增大管道半径，提升可见性和可交互性
    const segments = 64;
    const tubeSegments = 12; // 增加管道分段数，使圆环更平滑

    const vertices: number[] = [];
    const indices: number[] = [];

    const color = this.config.color;
    const dir = this.config.direction;

    // Helper to add vertex
    const addVertex = (pos: Vec3) => {
      vertices.push(pos.x, pos.y, pos.z, color.x, color.y, color.z);
    };

    // Compute perpendicular vectors for torus plane
    let perpX: Vec3, perpY: Vec3;
    if (Math.abs(dir.x) < 0.9) {
      perpX = new Vec3(1, 0, 0).cross(dir).normalize();
    } else {
      perpX = new Vec3(0, 1, 0).cross(dir).normalize();
    }
    perpY = dir.cross(perpX).normalize();

    // Generate torus vertices
    for (let i = 0; i <= segments; i++) {
      const u = (i / segments) * Math.PI * 2;
      const cosU = Math.cos(u);
      const sinU = Math.sin(u);

      // Center of tube at this angle
      const tubeCenter = perpX
        .multiply(cosU * majorRadius)
        .add(perpY.multiply(sinU * majorRadius));

      // Tangent direction for tube
      const tubeTangent = perpX
        .multiply(-sinU)
        .add(perpY.multiply(cosU))
        .normalize();
      const tubeNormal = dir;
      const tubeBinormal = tubeTangent.cross(tubeNormal).normalize();

      for (let j = 0; j <= tubeSegments; j++) {
        const v = (j / tubeSegments) * Math.PI * 2;
        const cosV = Math.cos(v);
        const sinV = Math.sin(v);

        // Offset from tube center
        const offset = tubeNormal
          .multiply(cosV * minorRadius)
          .add(tubeBinormal.multiply(sinV * minorRadius));

        addVertex(tubeCenter.add(offset));
      }
    }

    // Generate torus indices
    for (let i = 0; i < segments; i++) {
      for (let j = 0; j < tubeSegments; j++) {
        const a = i * (tubeSegments + 1) + j;
        const b = a + tubeSegments + 1;
        const c = a + 1;
        const d = b + 1;

        indices.push(a, b, c);
        indices.push(b, d, c);
      }
    }

    return {
      vertices: new Float32Array(vertices),
      indices: new Uint16Array(indices),
    };
  }

  /**
   * Create geometry for scale mode (cylinder + cube)
   * @private
   */
  private createScaleGeometry(): {
    vertices: Float32Array;
    indices: Uint16Array;
  } {
    const cylinderLength = 0.8;
    const cylinderRadius = 0.04;
    const cubeSize = 0.15;
    const segments = 12;

    const vertices: number[] = [];
    const indices: number[] = [];

    const color = this.config.color;
    const dir = this.config.direction;

    // Helper to add vertex
    const addVertex = (pos: Vec3) => {
      vertices.push(pos.x, pos.y, pos.z, color.x, color.y, color.z);
    };

    // Compute perpendicular vectors for cylinder cross-section
    let perpX: Vec3, perpY: Vec3;
    if (Math.abs(dir.x) < 0.9) {
      perpX = new Vec3(1, 0, 0).cross(dir).normalize();
    } else {
      perpX = new Vec3(0, 1, 0).cross(dir).normalize();
    }
    perpY = dir.cross(perpX).normalize();

    // Generate cylinder
    const cylinderStart = new Vec3(0, 0, 0);
    const cylinderEnd = dir.multiply(cylinderLength);

    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      const offset = perpX
        .multiply(cos * cylinderRadius)
        .add(perpY.multiply(sin * cylinderRadius));

      // Bottom vertex
      addVertex(cylinderStart.add(offset));
      // Top vertex
      addVertex(cylinderEnd.add(offset));
    }

    // Cylinder indices
    for (let i = 0; i < segments; i++) {
      const base = i * 2;
      indices.push(base, base + 1, base + 2);
      indices.push(base + 1, base + 3, base + 2);
    }

    // Generate cube at end
    const cubeCenter = dir.multiply(cylinderLength + cubeSize / 2);
    const halfSize = cubeSize / 2;

    const cubeBaseIndex = vertices.length / 6;

    // Cube vertices (8 corners)
    const cubeOffsets = [
      perpX
        .multiply(-halfSize)
        .add(perpY.multiply(-halfSize))
        .add(dir.multiply(-halfSize)),
      perpX
        .multiply(halfSize)
        .add(perpY.multiply(-halfSize))
        .add(dir.multiply(-halfSize)),
      perpX
        .multiply(halfSize)
        .add(perpY.multiply(halfSize))
        .add(dir.multiply(-halfSize)),
      perpX
        .multiply(-halfSize)
        .add(perpY.multiply(halfSize))
        .add(dir.multiply(-halfSize)),
      perpX
        .multiply(-halfSize)
        .add(perpY.multiply(-halfSize))
        .add(dir.multiply(halfSize)),
      perpX
        .multiply(halfSize)
        .add(perpY.multiply(-halfSize))
        .add(dir.multiply(halfSize)),
      perpX
        .multiply(halfSize)
        .add(perpY.multiply(halfSize))
        .add(dir.multiply(halfSize)),
      perpX
        .multiply(-halfSize)
        .add(perpY.multiply(halfSize))
        .add(dir.multiply(halfSize)),
    ];

    for (const offset of cubeOffsets) {
      addVertex(cubeCenter.add(offset));
    }

    // Cube indices (12 triangles, 6 faces)
    const cubeFaces = [
      [0, 1, 2, 0, 2, 3], // Front
      [4, 6, 5, 4, 7, 6], // Back
      [0, 4, 5, 0, 5, 1], // Bottom
      [2, 6, 7, 2, 7, 3], // Top
      [0, 3, 7, 0, 7, 4], // Left
      [1, 5, 6, 1, 6, 2], // Right
    ];

    for (const face of cubeFaces) {
      for (const idx of face) {
        indices.push(cubeBaseIndex + idx);
      }
    }

    return {
      vertices: new Float32Array(vertices),
      indices: new Uint16Array(indices),
    };
  }

  /**
   * Test if ray hits this axis
   * @param ray - Ray to test
   * @param gizmoPosition - Position of gizmo in world space
   * @param gizmoScale - Screen-space scale factor
   */
  testHit(ray: Ray, gizmoPosition: Vec3, gizmoScale: number): boolean {
    if (this.config.mode === GizmoMode.Rotate) {
      // Use ring hit test for rotation mode
      const ringRadius = 1.0 * gizmoScale;
      const ringWidth = 0.2 * gizmoScale; // 增大碰撞检测宽度，便于点击
      return this.testRingHit(
        ray,
        gizmoPosition,
        this.config.direction,
        ringRadius,
        ringWidth,
      );
    } else {
      // Use axis hit test for translate and scale modes
      const axisLength = 1.05 * gizmoScale; // Slightly longer than visual (0.8 + 0.25)
      const axisStart = gizmoPosition;
      const axisEnd = gizmoPosition.add(
        this.config.direction.multiply(axisLength),
      );
      const radius = 0.15 * gizmoScale; // Hit threshold
      return this.testAxisHit(ray, axisStart, axisEnd, radius);
    }
  }

  /**
   * Test if ray hits an axis (capsule hit test)
   * @param ray - Ray to test
   * @param axisStart - Start point of axis
   * @param axisEnd - End point of axis
   * @param radius - Capsule radius
   * @private
   */
  private testAxisHit(
    ray: Ray,
    axisStart: Vec3,
    axisEnd: Vec3,
    radius: number,
  ): boolean {
    const distance = ray.distanceToSegment(axisStart, axisEnd);
    return distance < radius;
  }

  /**
   * Test if ray hits a rotation ring (torus hit test)
   * @param ray - Ray to test
   * @param ringCenter - Center of ring
   * @param ringNormal - Normal of ring plane
   * @param ringRadius - Major radius of torus
   * @param ringWidth - Minor radius (thickness) of torus
   * @private
   */
  private testRingHit(
    ray: Ray,
    ringCenter: Vec3,
    ringNormal: Vec3,
    ringRadius: number,
    ringWidth: number,
  ): boolean {
    // Intersect ray with plane perpendicular to axis
    const distance = ray.intersectPlane(ringCenter, ringNormal);
    if (distance === null || distance < 0) {
      return false;
    }

    const hitPoint = ray.at(distance);

    // Compute distance from ring center
    const offset = hitPoint.subtract(ringCenter);
    const distFromCenter = offset.length();

    // Check if within ring width tolerance
    const minDist = ringRadius - ringWidth;
    const maxDist = ringRadius + ringWidth;

    return distFromCenter >= minDist && distFromCenter <= maxDist;
  }

  /**
   * Get color based on hover/active state
   */
  getColor(): Vec3 {
    const baseColor = this.config.color;

    if (this.isActive) {
      // Active: 150% brightness
      return baseColor.multiply(1.5);
    } else if (this.isHovered) {
      // Hovered: 120% brightness
      return baseColor.multiply(1.2);
    } else {
      // Inactive: 60% brightness (semi-transparent)
      return baseColor.multiply(0.6);
    }
  }

  /**
   * Release GPU resources
   */
  destroy(): void {
    if (this.vertexBuffer) {
      this.vertexBuffer.destroy();
      this.vertexBuffer = null;
    }

    if (this.indexBuffer) {
      this.indexBuffer.destroy();
      this.indexBuffer = null;
    }

    this.vertexCount = 0;
    this.indexCount = 0;
  }
}
