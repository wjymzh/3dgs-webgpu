# Design Document: Transform Gizmo System

## Overview

The Transform Gizmo system provides interactive 3D manipulation controls for a WebGPU-based editor. It consists of a main `TransformGizmo` controller class that manages visual handles (axes and rings) for translating, rotating, and scaling 3D objects. The system integrates with the existing Camera and Mesh infrastructure, using ray-based hit testing for precise interaction and real-time transformation feedback.

### Key Design Principles

1. **Separation of Concerns**: The `TransformGizmo` class handles coordination and state management, while `GizmoAxis` handles individual axis rendering and hit testing
2. **Mathematical Robustness**: All geometric operations handle edge cases (parallel rays, zero vectors, etc.) gracefully
3. **Screen-Space Consistency**: The gizmo maintains constant apparent size regardless of camera distance
4. **Real-Time Feedback**: Transformations are applied immediately during drag operations for responsive interaction
5. **Extensibility**: The architecture supports future enhancements like planar translation, snapping, and coordinate space switching

## Architecture

### Component Hierarchy

```
TransformGizmo (Main Controller)
├── GizmoAxis[] (X/Y/Z axes for current mode)
├── Math Utilities (Vec3, Quat, Mat4, Ray)
├── Rendering Pipeline (WebGPU)
└── Event Handlers (pointer events)
```

### Data Flow

1. **Initialization**: TransformGizmo creates GizmoAxis instances for each mode, sets up WebGPU pipeline
2. **Frame Update**: TransformGizmo updates gizmo position from target object, computes screen-space scale
3. **Pointer Events**:
   - Move → Hit testing → Update hover states
   - Down → Capture pointer → Enter drag mode
   - Move (dragging) → Compute transformation delta → Apply to target
   - Up → Release pointer → Exit drag mode
4. **Rendering**: TransformGizmo renders active GizmoAxis instances with appropriate visual states

### Integration Points

- **Camera**: Provides view/projection matrices for rendering and ray generation
- **Mesh/Target Objects**: Provides position/rotation/scale properties for manipulation
- **Renderer**: Provides WebGPU device and render pass for drawing
- **Canvas**: Provides pointer events and screen coordinates

## Components and Interfaces

### Math Utility Classes

These utility classes provide reusable mathematical operations for the gizmo and other parts of the codebase.

#### Vec3 Class

```typescript
class Vec3 {
  x: number;
  y: number;
  z: number;

  constructor(x: number = 0, y: number = 0, z: number = 0);

  // Factory methods
  static fromArray(arr: Float32Array | number[], offset?: number): Vec3;
  static zero(): Vec3;
  static one(): Vec3;

  // Basic operations (return new Vec3)
  add(v: Vec3): Vec3;
  subtract(v: Vec3): Vec3;
  multiply(scalar: number): Vec3;
  divide(scalar: number): Vec3;

  // In-place operations (modify this)
  addInPlace(v: Vec3): Vec3;
  subtractInPlace(v: Vec3): Vec3;
  multiplyInPlace(scalar: number): Vec3;

  // Vector operations
  dot(v: Vec3): number;
  cross(v: Vec3): Vec3;
  length(): number;
  lengthSquared(): number;
  distance(v: Vec3): number;
  distanceSquared(v: Vec3): number;

  // Normalization
  normalize(): Vec3; // Returns new normalized vector
  normalizeInPlace(): Vec3; // Normalizes this vector (handles zero-length)

  // Utility
  clone(): Vec3;
  toArray(): [number, number, number];
  set(x: number, y: number, z: number): Vec3;
}
```

#### Quat Class

```typescript
class Quat {
  x: number;
  y: number;
  z: number;
  w: number;

  constructor(x: number = 0, y: number = 0, z: number = 0, w: number = 1);

  // Factory methods
  static identity(): Quat;
  static fromEuler(x: number, y: number, z: number): Quat; // Euler angles in radians (ZYX order)
  static fromAxisAngle(axis: Vec3, angle: number): Quat;

  // Operations
  multiply(q: Quat): Quat; // Quaternion multiplication
  toEuler(): Vec3; // Returns Euler angles (ZYX order)

  // Normalization
  normalize(): Quat;
  normalizeInPlace(): Quat;

  // Interpolation
  slerp(q: Quat, t: number): Quat; // Spherical linear interpolation

  // Utility
  clone(): Quat;
}
```

#### Mat4 Class

```typescript
class Mat4 {
  elements: Float32Array; // 16 elements, column-major order

  constructor();

  // Factory methods
  static identity(): Mat4;
  static fromTranslation(v: Vec3): Mat4;
  static fromRotation(q: Quat): Mat4;
  static fromScale(v: Vec3): Mat4;
  static compose(position: Vec3, rotation: Quat, scale: Vec3): Mat4;

  // Operations
  multiply(m: Mat4): Mat4;
  multiplyInPlace(m: Mat4): Mat4;
  inverse(): Mat4 | null; // Returns null if not invertible
  transpose(): Mat4;

  // Decomposition
  decompose(): { position: Vec3; rotation: Quat; scale: Vec3 } | null;

  // Transformation
  transformPoint(v: Vec3): Vec3; // Apply to point (w=1)
  transformDirection(v: Vec3): Vec3; // Apply to direction (w=0)

  // Utility
  clone(): Mat4;
}
```

#### Ray Class

```typescript
class Ray {
  origin: Vec3;
  direction: Vec3; // Should be normalized

  constructor(origin: Vec3, direction: Vec3);

  // Factory method
  static fromScreenPoint(
    screenX: number,
    screenY: number,
    canvasWidth: number,
    canvasHeight: number,
    camera: Camera,
  ): Ray;

  // Operations
  at(distance: number): Vec3; // Returns point at distance along ray

  // Intersection tests
  intersectPlane(planeOrigin: Vec3, planeNormal: Vec3): number | null; // Returns distance or null
  distanceToPoint(point: Vec3): number;
  distanceToSegment(segmentStart: Vec3, segmentEnd: Vec3): number;

  // Utility
  clone(): Ray;
}
```

### GizmoAxis Class

Represents a single axis or ring component of the gizmo.

```typescript
enum AxisType {
  X = 0,
  Y = 1,
  Z = 2,
}

enum GizmoMode {
  Translate = 0,
  Rotate = 1,
  Scale = 2,
}

interface GizmoAxisConfig {
  type: AxisType;
  mode: GizmoMode;
  color: Vec3; // Base color (e.g., red for X)
  direction: Vec3; // Axis direction in local space
}

class GizmoAxis {
  config: GizmoAxisConfig;

  // Visual state
  isHovered: boolean = false;
  isActive: boolean = false;

  // Geometry (GPU buffers created lazily)
  private vertexBuffer: GPUBuffer | null = null;
  private indexBuffer: GPUBuffer | null = null;
  private vertexCount: number = 0;
  private indexCount: number = 0;

  constructor(config: GizmoAxisConfig);

  // Geometry creation
  createGeometry(device: GPUDevice): void;
  private createTranslateGeometry(): {
    vertices: Float32Array;
    indices: Uint16Array;
  };
  private createRotateGeometry(): {
    vertices: Float32Array;
    indices: Uint16Array;
  };
  private createScaleGeometry(): {
    vertices: Float32Array;
    indices: Uint16Array;
  };

  // Hit testing
  testHit(ray: Ray, gizmoPosition: Vec3, gizmoScale: number): boolean;
  private testAxisHit(
    ray: Ray,
    axisStart: Vec3,
    axisEnd: Vec3,
    radius: number,
  ): boolean;
  private testRingHit(
    ray: Ray,
    ringCenter: Vec3,
    ringNormal: Vec3,
    ringRadius: number,
    ringWidth: number,
  ): boolean;

  // Rendering
  getColor(): Vec3; // Returns color based on hover/active state

  // Cleanup
  destroy(): void;
}
```

### TransformGizmo Class

Main controller class that manages the gizmo system.

```typescript
interface TransformGizmoConfig {
  renderer: Renderer;
  camera: Camera;
  canvas: HTMLCanvasElement;
  size?: number; // Base size in world units (default: 1.0)
  hitThreshold?: number; // Hit detection threshold (default: 0.15)
}

class TransformGizmo {
  private renderer: Renderer;
  private camera: Camera;
  private canvas: HTMLCanvasElement;

  // Configuration
  private baseSize: number = 1.0;
  private hitThreshold: number = 0.15;

  // State
  private mode: GizmoMode = GizmoMode.Translate;
  private targetObject: Mesh | null = null;
  private axes: GizmoAxis[] = [];

  // Interaction state
  private activeAxis: GizmoAxis | null = null;
  private hoveredAxis: GizmoAxis | null = null;
  private isDragging: boolean = false;
  private dragStartPoint: Vec3 | null = null;
  private dragStartTransform: {
    position: Vec3;
    rotation: Vec3;
    scale: Vec3;
  } | null = null;

  // Rendering resources
  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;

  // Screen-space scaling
  private currentScale: number = 1.0;

  constructor(config: TransformGizmoConfig);

  // Initialization
  init(): void;
  private createPipeline(): void;
  private createAxes(): void;

  // Mode management
  setMode(mode: GizmoMode): void;
  getMode(): GizmoMode;

  // Target management
  setTarget(object: Mesh | null): void;
  getTarget(): Mesh | null;

  // Event handlers
  onPointerMove(event: PointerEvent): void;
  onPointerDown(event: PointerEvent): void;
  onPointerUp(event: PointerEvent): void;

  // Hit testing
  private performHitTest(ray: Ray): GizmoAxis | null;

  // Transformation computation
  private computeTranslationDelta(ray: Ray): Vec3 | null;
  private computeRotationDelta(ray: Ray): number | null;
  private computeScaleDelta(ray: Ray): Vec3 | null;

  // Transformation application
  private applyTranslation(delta: Vec3): void;
  private applyRotation(angle: number): void;
  private applyScale(delta: Vec3): void;

  // Rendering
  render(pass: GPURenderPassEncoder): void;
  private updateScreenSpaceScale(): void;
  private updateUniforms(): void;

  // Utility
  private screenToRay(screenX: number, screenY: number): Ray;

  // Cleanup
  destroy(): void;
}
```

## Data Models

### Gizmo Visual State

The gizmo's visual appearance is determined by:

1. **Mode**: Determines which geometry to display (arrows, rings, or scale handles)
2. **Hover State**: Brightens the color of the axis under the pointer
3. **Active State**: Further brightens the color of the axis being dragged
4. **Inactive State**: Semi-transparent rendering for non-interacted axes

Color computation:

```typescript
function getAxisColor(
  baseColor: Vec3,
  isHovered: boolean,
  isActive: boolean,
): Vec3 {
  if (isActive) {
    return baseColor.multiply(1.5); // 150% brightness
  } else if (isHovered) {
    return baseColor.multiply(1.2); // 120% brightness
  } else {
    return baseColor.multiply(0.6); // 60% brightness (semi-transparent)
  }
}
```

### Transformation State

During a drag operation, the gizmo maintains:

```typescript
interface DragState {
  axis: GizmoAxis; // The axis being dragged
  startPoint: Vec3; // Initial 3D point where drag started
  startTransform: {
    position: Vec3;
    rotation: Vec3; // Euler angles
    scale: Vec3;
  }; // Object's transform at drag start
  currentRay: Ray; // Current pointer ray
}
```

### Geometry Data

Each GizmoAxis stores vertex data in the format:

```
Vertex: [position.x, position.y, position.z, color.r, color.g, color.b]
Stride: 24 bytes (6 floats)
```

Geometry specifications:

- **Translate Arrow**: Cylinder (0.8 units) + Cone (0.25 units), radius 0.04
- **Rotate Ring**: Torus with major radius 1.0, minor radius 0.02, 64 segments
- **Scale Handle**: Cylinder (0.8 units) + Cube (0.15 units), radius 0.04

### Screen-Space Scaling

To maintain constant apparent size, the gizmo scales based on camera distance:

```typescript
function computeScreenSpaceScale(
  gizmoPosition: Vec3,
  cameraPosition: Vec3,
  fov: number,
): number {
  const distance = gizmoPosition.distance(cameraPosition);
  const scale = distance * Math.tan(fov / 2) * 0.15; // 0.15 is tuning factor
  return scale;
}
```

## Mathematical Operations

### Translation Along Axis

Given:

- `ray`: Current pointer ray
- `axis`: Axis direction (normalized)
- `gizmoPos`: Gizmo position
- `startPoint`: 3D point where drag started

Algorithm:

1. Project ray onto axis to find closest point on axis
2. Compute displacement from start point to current point
3. Project displacement onto axis direction to get scalar
4. Apply: `newPosition = startPosition + axis * scalar`

```typescript
function computeTranslationDelta(
  ray: Ray,
  axis: Vec3,
  gizmoPos: Vec3,
  startPoint: Vec3,
): Vec3 | null {
  // Find closest point on axis to ray
  const axisPoint = closestPointOnRayToLine(ray, gizmoPos, axis);
  if (axisPoint === null) return null;

  // Compute displacement
  const displacement = axisPoint.subtract(startPoint);

  // Project onto axis
  const scalar = displacement.dot(axis);

  return axis.multiply(scalar);
}

function closestPointOnRayToLine(
  ray: Ray,
  linePoint: Vec3,
  lineDir: Vec3,
): Vec3 | null {
  // Using formula for closest point between two lines
  const w0 = ray.origin.subtract(linePoint);
  const a = ray.direction.dot(ray.direction);
  const b = ray.direction.dot(lineDir);
  const c = lineDir.dot(lineDir);
  const d = ray.direction.dot(w0);
  const e = lineDir.dot(w0);

  const denom = a * c - b * b;
  if (Math.abs(denom) < 1e-6) return null; // Parallel

  const t = (b * e - c * d) / denom;
  const s = (a * e - b * d) / denom;

  return linePoint.add(lineDir.multiply(s));
}
```

### Rotation Around Axis

Given:

- `ray`: Current pointer ray
- `axis`: Rotation axis (normalized)
- `gizmoPos`: Gizmo position
- `startPoint`: 3D point where drag started (on plane perpendicular to axis)

Algorithm:

1. Intersect ray with plane perpendicular to axis
2. Compute vector from gizmo center to intersection point
3. Compute angle between start vector and current vector using atan2
4. Apply: `newRotation = startRotation + angle * axis`

```typescript
function computeRotationDelta(
  ray: Ray,
  axis: Vec3,
  gizmoPos: Vec3,
  startPoint: Vec3,
): number | null {
  // Intersect ray with plane perpendicular to axis
  const distance = ray.intersectPlane(gizmoPos, axis);
  if (distance === null || distance < 0) return null;

  const currentPoint = ray.at(distance);

  // Compute vectors from center to points (projected onto plane)
  const startVec = startPoint.subtract(gizmoPos);
  const currentVec = currentPoint.subtract(gizmoPos);

  // Project onto plane (remove component along axis)
  const startProj = startVec.subtract(axis.multiply(startVec.dot(axis)));
  const currentProj = currentVec.subtract(axis.multiply(currentVec.dot(axis)));

  // Normalize
  const startNorm = startProj.normalize();
  const currentNorm = currentProj.normalize();

  // Compute angle using atan2 for correct sign
  const cosAngle = startNorm.dot(currentNorm);
  const sinAngle = startNorm.cross(currentNorm).dot(axis);

  return Math.atan2(sinAngle, cosAngle);
}
```

### Scale Along Axis

Given:

- `ray`: Current pointer ray
- `axis`: Scale axis (normalized)
- `gizmoPos`: Gizmo position
- `startPoint`: 3D point where drag started

Algorithm:

1. Similar to translation, project ray onto axis
2. Compute displacement scalar
3. Map displacement to scale factor (e.g., 0.1 units = 1.1x scale)
4. Clamp to minimum scale (0.001)
5. Apply: `newScale = startScale * scaleFactor`

```typescript
function computeScaleDelta(
  ray: Ray,
  axis: Vec3,
  gizmoPos: Vec3,
  startPoint: Vec3,
): Vec3 | null {
  // Find closest point on axis to ray (same as translation)
  const axisPoint = closestPointOnRayToLine(ray, gizmoPos, axis);
  if (axisPoint === null) return null;

  // Compute displacement
  const displacement = axisPoint.subtract(startPoint);
  const scalar = displacement.dot(axis);

  // Map to scale factor (1 unit displacement = 2x scale)
  const scaleFactor = 1.0 + scalar;

  // Clamp to minimum
  const clampedFactor = Math.max(0.001, scaleFactor);

  // Return scale delta for this axis only
  const delta = new Vec3();
  if (axis.x !== 0) delta.x = clampedFactor;
  if (axis.y !== 0) delta.y = clampedFactor;
  if (axis.z !== 0) delta.z = clampedFactor;

  return delta;
}
```

### Ray-Capsule Distance (Axis Hit Testing)

Given:

- `ray`: Pointer ray
- `segmentStart`: Axis start point
- `segmentEnd`: Axis end point
- `radius`: Capsule radius

Algorithm:

1. Compute closest points between ray and line segment
2. Calculate distance between closest points
3. Hit if distance < radius

```typescript
function raySegmentDistance(
  ray: Ray,
  segmentStart: Vec3,
  segmentEnd: Vec3,
): number {
  const segmentDir = segmentEnd.subtract(segmentStart);
  const segmentLength = segmentDir.length();
  const segmentDirNorm = segmentDir.divide(segmentLength);

  // Closest point on ray to segment
  const w0 = ray.origin.subtract(segmentStart);
  const a = ray.direction.dot(ray.direction);
  const b = ray.direction.dot(segmentDirNorm);
  const c = segmentDirNorm.dot(segmentDirNorm);
  const d = ray.direction.dot(w0);
  const e = segmentDirNorm.dot(w0);

  const denom = a * c - b * b;
  let t = 0;
  let s = 0;

  if (Math.abs(denom) < 1e-6) {
    // Parallel
    t = 0;
    s = e / c;
  } else {
    t = (b * e - c * d) / denom;
    s = (a * e - b * d) / denom;
  }

  // Clamp s to segment
  s = Math.max(0, Math.min(segmentLength, s));

  const pointOnRay = ray.at(t);
  const pointOnSegment = segmentStart.add(segmentDirNorm.multiply(s));

  return pointOnRay.distance(pointOnSegment);
}
```

### Ray-Torus Intersection (Ring Hit Testing)

Given:

- `ray`: Pointer ray
- `ringCenter`: Center of rotation ring
- `ringNormal`: Normal of ring plane (axis direction)
- `ringRadius`: Major radius of torus
- `ringWidth`: Minor radius (thickness) of torus

Algorithm:

1. Intersect ray with ring plane
2. Compute distance from intersection to ring center
3. Hit if distance is within [ringRadius - ringWidth, ringRadius + ringWidth]

```typescript
function testRingHit(
  ray: Ray,
  ringCenter: Vec3,
  ringNormal: Vec3,
  ringRadius: number,
  ringWidth: number,
): boolean {
  // Intersect with plane
  const distance = ray.intersectPlane(ringCenter, ringNormal);
  if (distance === null || distance < 0) return false;

  const hitPoint = ray.at(distance);

  // Distance from ring center
  const offset = hitPoint.subtract(ringCenter);
  const distFromCenter = offset.length();

  // Check if within ring width
  const minDist = ringRadius - ringWidth;
  const maxDist = ringRadius + ringWidth;

  return distFromCenter >= minDist && distFromCenter <= maxDist;
}
```

## Error Handling

### Mathematical Edge Cases

1. **Zero-Length Vectors**: All normalization operations check for zero length and return a default vector (e.g., [0, 0, 1]) instead of NaN
2. **Parallel Rays**: Ray-line and ray-plane operations return null when rays are parallel to avoid division by zero
3. **Degenerate Transformations**: Scale operations clamp to minimum value (0.001) to prevent zero or negative scales
4. **Matrix Inversion**: Matrix inverse operations return null for singular matrices

### User Interaction Edge Cases

1. **No Target Object**: Gizmo does not render when no target is set
2. **Camera Behind Gizmo**: Hit testing rejects negative ray distances
3. **Rapid Mode Switching**: Mode changes clear all interaction state to prevent stale references
4. **Pointer Capture Loss**: Pointer up event always clears drag state, even if capture was lost

### WebGPU Resource Management

1. **Device Loss**: Gizmo gracefully handles device loss by checking device state before rendering
2. **Buffer Creation Failure**: Geometry creation catches buffer allocation errors and logs warnings
3. **Pipeline Creation Failure**: Pipeline creation errors are caught and reported, gizmo disables rendering

## Testing Strategy

The Transform Gizmo system will be tested using a dual approach: property-based tests for universal correctness properties and unit tests for specific examples and edge cases.

### Property-Based Testing

Property-based tests will validate universal mathematical properties across randomized inputs. Each test will run a minimum of 100 iterations with randomly generated test data.

**Testing Library**: We will use `fast-check` for TypeScript property-based testing.

**Test Configuration**:

- Minimum 100 iterations per property test
- Each test tagged with: `Feature: transform-gizmo, Property {N}: {property description}`
- Tests organized by component (Vec3, Quat, Mat4, Ray, GizmoAxis, TransformGizmo)

### Unit Testing

Unit tests will validate specific examples, edge cases, and integration points:

1. **Math Utilities**: Test specific known values (e.g., Vec3(1,0,0).cross(Vec3(0,1,0)) === Vec3(0,0,1))
2. **Edge Cases**: Test zero vectors, parallel rays, degenerate matrices
3. **Integration**: Test gizmo initialization, mode switching, target setting
4. **Rendering**: Test that geometry buffers are created correctly

**Testing Library**: Standard Jest/Vitest for unit tests.

### Test Organization

```
tests/
├── math/
│   ├── Vec3.property.test.ts
│   ├── Vec3.unit.test.ts
│   ├── Quat.property.test.ts
│   ├── Quat.unit.test.ts
│   ├── Mat4.property.test.ts
│   ├── Mat4.unit.test.ts
│   ├── Ray.property.test.ts
│   └── Ray.unit.test.ts
├── gizmo/
│   ├── GizmoAxis.property.test.ts
│   ├── GizmoAxis.unit.test.ts
│   ├── TransformGizmo.property.test.ts
│   └── TransformGizmo.unit.test.ts
└── integration/
    └── TransformGizmo.integration.test.ts
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Math Utility Properties

#### Vec3 Properties

Property 1: Vector addition is commutative
_For any_ two Vec3 vectors `a` and `b`, `a.add(b)` should equal `b.add(a)`
**Validates: Requirements 14.5**

Property 2: Cross product is perpendicular
_For any_ two non-parallel Vec3 vectors `a` and `b`, `a.cross(b)` should be perpendicular to both `a` and `b` (dot product equals zero)
**Validates: Requirements 14.5**

Property 3: Normalization produces unit vectors
_For any_ non-zero Vec3 vector `v`, `v.normalize().length()` should equal 1.0 (within floating point tolerance)
**Validates: Requirements 14.5**

Property 4: Distance is symmetric
_For any_ two Vec3 vectors `a` and `b`, `a.distance(b)` should equal `b.distance(a)`
**Validates: Requirements 14.5**

Property 5: Zero vector normalization is safe
_For any_ zero-length Vec3 vector, calling `normalize()` should return a valid vector without NaN or Infinity values
**Validates: Requirements 10.3**

#### Quat Properties

Property 6: Euler round trip preserves rotation
_For any_ valid Euler angles (x, y, z), converting to quaternion via `Quat.fromEuler(x, y, z)` then back via `toEuler()` should produce equivalent angles (accounting for gimbal lock cases)
**Validates: Requirements 14.6**

Property 7: Quaternion multiplication is associative
_For any_ three quaternions `a`, `b`, and `c`, `a.multiply(b.multiply(c))` should equal `a.multiply(b).multiply(c)`
**Validates: Requirements 14.6**

Property 8: Identity quaternion is neutral
_For any_ quaternion `q`, `q.multiply(Quat.identity())` should equal `q`
**Validates: Requirements 14.6**

#### Mat4 Properties

Property 9: Matrix multiplication by inverse yields identity
_For any_ invertible Mat4 matrix `m`, `m.multiply(m.inverse())` should equal the identity matrix (within floating point tolerance)
**Validates: Requirements 14.7**

Property 10: Compose then decompose is round trip
_For any_ position, rotation, and scale vectors, composing them into a matrix via `Mat4.compose()` then decomposing via `decompose()` should return equivalent values
**Validates: Requirements 14.7**

Property 11: Matrix transpose is involutive
_For any_ Mat4 matrix `m`, `m.transpose().transpose()` should equal `m`
**Validates: Requirements 14.7**

#### Ray Properties

Property 12: Ray at zero distance returns origin
_For any_ Ray, calling `at(0)` should return the ray's origin point
**Validates: Requirements 14.8**

Property 13: Ray-plane intersection is on plane
_For any_ Ray and plane (origin, normal), if `intersectPlane()` returns a distance `d`, then `ray.at(d)` should lie on the plane (dot product with normal equals zero)
**Validates: Requirements 14.8**

Property 14: Ray distance to point is non-negative
_For any_ Ray and point, `distanceToPoint()` should return a value >= 0
**Validates: Requirements 14.8**

Property 15: Parallel ray and plane return null
_For any_ Ray and plane where the ray direction is perpendicular to the plane normal (dot product near zero), `intersectPlane()` should return null
**Validates: Requirements 10.2**

### Gizmo State Management Properties

Property 16: Mode change clears interaction state
_For any_ TransformGizmo with active or hovered axes, calling `setMode()` should result in all axes having `isHovered = false` and `isActive = false`
**Validates: Requirements 1.3**

Property 17: Mode change preserves target
_For any_ TransformGizmo with a target object, calling `setMode()` should not change the target object reference
**Validates: Requirements 1.4**

Property 18: Gizmo position follows target position
_For any_ target object with position `p`, setting it as the gizmo target should result in the gizmo rendering at position `p`
**Validates: Requirements 2.1**

Property 19: Target position changes update gizmo
_For any_ target object, if its position changes from `p1` to `p2`, the gizmo position should update to `p2` on the next render
**Validates: Requirements 2.2**

### Visual State Properties

Property 20: Screen-space scale is proportional to distance
_For any_ camera distance `d1` and `d2` where `d2 = 2 * d1`, the computed screen-space scale at `d2` should be approximately `2 * scale(d1)`
**Validates: Requirements 3.4, 12.3**

Property 21: Hovered axis is brighter than inactive
_For any_ GizmoAxis with base color `c`, when `isHovered = true` and `isActive = false`, `getColor()` should return a color brighter than `c * 0.6`
**Validates: Requirements 3.5**

Property 22: Active axis is brightest
_For any_ GizmoAxis with base color `c`, when `isActive = true`, `getColor()` should return a color brighter than when `isHovered = true`
**Validates: Requirements 3.6**

Property 23: Inactive axis is dimmed
_For any_ GizmoAxis with base color `c`, when `isHovered = false` and `isActive = false`, `getColor()` should return `c * 0.6`
**Validates: Requirements 3.7**

### Translation Properties

Property 24: Translation displacement is along axis
_For any_ translation computation with axis direction `a`, the resulting displacement vector `d` should satisfy `d.cross(a).length() ≈ 0` (displacement is parallel to axis)
**Validates: Requirements 4.2, 4.5**

Property 25: Translation updates target position
_For any_ target object with initial position `p0` and computed displacement `d`, after applying translation, the target position should be `p0 + d`
**Validates: Requirements 4.3**

Property 26: Pointer down on axis enters drag mode
_For any_ GizmoAxis that registers a hit, after processing a pointer down event, `isDragging` should be `true` and `activeAxis` should reference that axis
**Validates: Requirements 4.1**

Property 27: Pointer up exits drag mode
_For any_ TransformGizmo in drag mode, after processing a pointer up event, `isDragging` should be `false` and `activeAxis` should be `null`
**Validates: Requirements 4.4**

### Rotation Properties

Property 28: Rotation angle computation is correct
_For any_ two points on a rotation plane at angles `θ1` and `θ2` from a reference direction, the computed rotation angle should equal `θ2 - θ1`
**Validates: Requirements 5.2, 5.5**

Property 29: Rotation updates target rotation
_For any_ target object with initial rotation `r0` and computed angle `θ` around axis `a`, after applying rotation, the rotation should reflect an additional `θ` radians around `a`
**Validates: Requirements 5.3**

### Scale Properties

Property 30: Scale factor is clamped to minimum
_For any_ computed scale factor, the resulting scale applied to the target should never be less than 0.001 on any axis
**Validates: Requirements 6.5, 10.5**

Property 31: Scale computation is along axis
_For any_ scale computation with axis direction `a`, the resulting scale delta should only affect the component(s) corresponding to non-zero elements of `a`
**Validates: Requirements 6.2, 6.6**

Property 32: Scale updates target scale
_For any_ target object with initial scale `s0` and computed scale factor `f` on axis `a`, after applying scale, the target scale should be `s0` multiplied by `f` on the axis corresponding to `a`
**Validates: Requirements 6.3**

### Hit Testing Properties

Property 33: Axis hit detection uses distance threshold
_For any_ ray and axis (modeled as line segment with radius `r`), if the ray-segment distance is less than `r`, hit testing should return `true`; if greater than `r`, it should return `false`
**Validates: Requirements 7.2, 7.3**

Property 34: Ring hit detection uses radius tolerance
_For any_ ray and rotation ring with radius `R` and width `w`, if the ray-plane intersection distance from center is in `[R-w, R+w]`, hit testing should return `true`
**Validates: Requirements 8.2, 8.3**

Property 35: Ray-segment distance is symmetric
_For any_ ray and line segment, the computed distance should be the same regardless of segment direction
**Validates: Requirements 7.2**

### Pointer Event Properties

Property 36: Pointer move updates hover state
_For any_ pointer move event, after processing, exactly zero or one axis should have `isHovered = true` (the one closest to the ray, if any are within threshold)
**Validates: Requirements 9.1**

Property 37: Pointer capture enables drag mode
_For any_ pointer down event on a hit axis, the gizmo should enter drag mode with that axis as active
**Validates: Requirements 9.2**

Property 38: Continuous drag applies cumulative transformations
_For any_ sequence of pointer move events during drag, each event should compute and apply a transformation delta, resulting in cumulative changes to the target
**Validates: Requirements 9.3**

Property 39: Pointer release clears drag state
_For any_ pointer up event, after processing, `isDragging` should be `false`, `activeAxis` should be `null`, and pointer capture should be released
**Validates: Requirements 9.4**

### Edge Case Properties

Property 40: Parallel ray and axis handled gracefully
_For any_ ray and axis that are parallel (dot product of directions ≈ 1), translation computation should return `null` or a safe default without NaN or Infinity
**Validates: Requirements 10.1**

Property 41: Zero-length vector operations are safe
_For any_ operation that normalizes a vector, if the input is zero-length, the result should be a valid vector (not NaN or Infinity)
**Validates: Requirements 10.3**

Property 42: Degenerate angle computation is safe
_For any_ angle computation between parallel or anti-parallel vectors, the result should be a valid number (0 or π) without NaN
**Validates: Requirements 10.4**

### Camera Integration Properties

Property 43: Screen-to-ray conversion is consistent
_For any_ screen point `(x, y)`, converting to a ray and then projecting back to screen space should yield approximately `(x, y)`
**Validates: Requirements 12.2**

Property 44: Camera movement updates gizmo scale
_For any_ camera position change that alters the distance to the gizmo, the screen-space scale factor should be recalculated on the next frame
**Validates: Requirements 12.4**
