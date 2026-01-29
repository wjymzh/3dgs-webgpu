import { Renderer } from "../Renderer";
import { Camera } from "../Camera";
import { GizmoAxis, GizmoMode, AxisType } from "./GizmoAxis";
import { Vec3 } from "../math/Vec3";
import { Ray } from "../math/Ray";
import { Mat4 } from "../math/Mat4";
import { Quat } from "../math/Quat";

/**
 * TransformableObject - Gizmo 可以操作的对象接口
 * Mesh 和 SplatTransformProxy 都实现这个接口
 * 注意：position/rotation/scale 可以是数组或 Float32Array（兼容 Mesh）
 */
export interface TransformableObject {
  position: [number, number, number] | Float32Array;
  rotation: [number, number, number] | Float32Array;
  scale: [number, number, number] | Float32Array;
  setPosition(x: number, y: number, z: number): void;
  setRotation(x: number, y: number, z: number): void;
  setScale(x: number, y: number, z: number): void;
}

/**
 * TransformGizmoConfig - Configuration for the transform gizmo
 */
export interface TransformGizmoConfig {
  renderer: Renderer;
  camera: Camera;
  canvas: HTMLCanvasElement;
  size?: number; // Base size in world units (default: 1.0)
  hitThreshold?: number; // Hit detection threshold (default: 0.15)
}

/**
 * TransformGizmo - Main controller class for interactive 3D manipulation controls
 * Provides visual handles for translating, rotating, and scaling 3D objects
 */
export class TransformGizmo {
  private renderer: Renderer;
  private camera: Camera;
  private canvas: HTMLCanvasElement;

  // Configuration
  private baseSize: number = 1.0;
  private hitThreshold: number = 0.2;

  // State
  private mode: GizmoMode = GizmoMode.Translate;
  private targetObject: TransformableObject | null = null;
  private axes: GizmoAxis[] = [];

  // 仅显示模式（用于 PLY 等不支持变换的对象）
  private displayOnlyPosition: Vec3 | null = null;
  private isDisplayOnly: boolean = false;

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

  // Drag state callback
  private onDragStateChange: ((isDragging: boolean) => void) | null = null;

  // Rendering resources
  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;

  // Screen-space scaling
  private currentScale: number = 1.0;

  constructor(config: TransformGizmoConfig) {
    this.renderer = config.renderer;
    this.camera = config.camera;
    this.canvas = config.canvas;

    if (config.size !== undefined) {
      this.baseSize = config.size;
    }

    if (config.hitThreshold !== undefined) {
      this.hitThreshold = config.hitThreshold;
    }
  }

  /**
   * Initialize the gizmo - create pipeline and initial axes
   */
  init(): void {
    this.createPipeline();
    this.createAxes();
  }

  /**
   * Create the WebGPU render pipeline for gizmo rendering
   * @private
   */
  private createPipeline(): void {
    const device = this.renderer.device;

    // WGSL shader code
    const shaderCode = `
      struct Uniforms {
        viewProjection: mat4x4<f32>,
        model: mat4x4<f32>,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;

      struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) color: vec3<f32>,
      }

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec3<f32>,
      }

      @vertex
      fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        let worldPos = uniforms.model * vec4<f32>(input.position, 1.0);
        output.position = uniforms.viewProjection * worldPos;
        output.color = input.color;
        return output;
      }

      @fragment
      fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        return vec4<f32>(input.color, 1.0);
      }
    `;

    // Create shader module
    const shaderModule = device.createShaderModule({
      code: shaderCode,
    });

    // Create uniform buffer (2 mat4x4 = 128 bytes)
    this.uniformBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create bind group layout
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
      ],
    });

    // Create bind group
    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: this.uniformBuffer },
        },
      ],
    });

    // Create pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    // Create render pipeline
    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vertexMain",
        buffers: [
          {
            arrayStride: 24, // 6 floats * 4 bytes = 24 bytes
            attributes: [
              {
                // position
                shaderLocation: 0,
                offset: 0,
                format: "float32x3",
              },
              {
                // color
                shaderLocation: 1,
                offset: 12,
                format: "float32x3",
              },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [
          {
            format: this.renderer.format,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "back",
      },
      depthStencil: {
        format: this.renderer.depthFormat,
        depthWriteEnabled: false,  // 不写入深度，避免影响其他物体
        depthCompare: "always",    // 始终通过深度测试，确保 Gizmo 永远可见
      },
    });
  }

  /**
   * Create axes for the current mode
   * @private
   */
  private createAxes(): void {
    const device = this.renderer.device;

    // Clear existing axes
    for (const axis of this.axes) {
      axis.destroy();
    }
    this.axes = [];

    // Create three axes (X, Y, Z) for current mode
    const axisConfigs = [
      {
        type: AxisType.X,
        color: new Vec3(0.9, 0.2, 0.2), // Red
        direction: new Vec3(1, 0, 0),
      },
      {
        type: AxisType.Y,
        color: new Vec3(0.2, 0.9, 0.2), // Green
        direction: new Vec3(0, 1, 0),
      },
      {
        type: AxisType.Z,
        color: new Vec3(0.2, 0.4, 0.9), // Blue
        direction: new Vec3(0, 0, 1),
      },
    ];

    for (const config of axisConfigs) {
      const axis = new GizmoAxis({
        type: config.type,
        mode: this.mode,
        color: config.color,
        direction: config.direction,
      });

      // Create geometry for this axis
      axis.createGeometry(device);

      this.axes.push(axis);
    }
  }

  /**
   * Set the manipulation mode (translate, rotate, or scale)
   * @param mode - The new mode
   */
  setMode(mode: GizmoMode): void {
    // Clear all axes' hover and active states
    for (const axis of this.axes) {
      axis.isHovered = false;
      axis.isActive = false;
    }

    // Clear interaction state
    this.activeAxis = null;
    this.hoveredAxis = null;
    this.isDragging = false;
    this.dragStartPoint = null;
    this.dragStartTransform = null;

    // Update mode
    this.mode = mode;

    // Recreate axes for new mode
    this.createAxes();
  }

  /**
   * Get the current manipulation mode
   * @returns The current mode
   */
  getMode(): GizmoMode {
    return this.mode;
  }

  /**
   * Set the target object to manipulate
   * @param object - The target object (Mesh/SplatTransformProxy) or null to clear
   */
  setTarget(object: TransformableObject | null): void {
    this.targetObject = object;

    // Clear display-only mode when setting a real target
    this.displayOnlyPosition = null;
    this.isDisplayOnly = false;

    // Clear interaction state when target changes
    this.activeAxis = null;
    this.hoveredAxis = null;
    this.isDragging = false;
    this.dragStartPoint = null;
    this.dragStartTransform = null;

    // Clear all axes' hover and active states
    for (const axis of this.axes) {
      axis.isHovered = false;
      axis.isActive = false;
    }
  }

  /**
   * Set a display-only position for the gizmo (no interaction)
   * Used for objects that don't support transformation (e.g., PLY point clouds)
   * @param x - X coordinate
   * @param y - Y coordinate
   * @param z - Z coordinate
   */
  setDisplayPosition(x: number, y: number, z: number): void {
    this.displayOnlyPosition = new Vec3(x, y, z);
    this.isDisplayOnly = true;

    // Clear any mesh target
    this.targetObject = null;

    // Clear interaction state
    this.activeAxis = null;
    this.hoveredAxis = null;
    this.isDragging = false;
    this.dragStartPoint = null;
    this.dragStartTransform = null;

    // Clear all axes' hover and active states
    for (const axis of this.axes) {
      axis.isHovered = false;
      axis.isActive = false;
    }
  }

  /**
   * Clear the display-only position
   */
  clearDisplayPosition(): void {
    this.displayOnlyPosition = null;
    this.isDisplayOnly = false;
  }

  /**
   * Check if gizmo is in display-only mode
   */
  isInDisplayOnlyMode(): boolean {
    return this.isDisplayOnly;
  }

  /**
   * Get the current target object
   * @returns The target object or null
   */
  getTarget(): TransformableObject | null {
    return this.targetObject;
  }

  /**
   * Set callback for drag state changes
   * @param callback - Function called when drag state changes
   */
  setOnDragStateChange(callback: ((isDragging: boolean) => void) | null): void {
    this.onDragStateChange = callback;
  }

  /**
   * Check if gizmo is currently being dragged
   * @returns True if dragging
   */
  getIsDragging(): boolean {
    return this.isDragging;
  }

  /**
   * Get the current gizmo position
   * Returns position from target object or display-only position
   * @private
   */
  private getGizmoPosition(): Vec3 | null {
    if (this.isDisplayOnly && this.displayOnlyPosition) {
      return this.displayOnlyPosition;
    }
    if (this.targetObject) {
      return new Vec3(
        this.targetObject.position[0],
        this.targetObject.position[1],
        this.targetObject.position[2],
      );
    }
    return null;
  }

  /**
   * Check if gizmo should be visible
   * @private
   */
  private isVisible(): boolean {
    return this.targetObject !== null || (this.isDisplayOnly && this.displayOnlyPosition !== null);
  }

  /**
   * Update the screen-space scale factor based on camera distance
   * Maintains constant apparent size regardless of camera distance
   * @private
   */
  private updateScreenSpaceScale(): void {
    const gizmoPosition = this.getGizmoPosition();
    if (!gizmoPosition) {
      return;
    }

    // Get camera position
    const cameraPosition = new Vec3(
      this.camera.position[0],
      this.camera.position[1],
      this.camera.position[2],
    );

    // Compute distance from camera to gizmo
    const distance = gizmoPosition.distance(cameraPosition);

    // Calculate scale factor: distance * tan(fov/2) * 0.25 (增大系数使 Gizmo 更大)
    const scale = distance * Math.tan(this.camera.fov / 2) * 0.25;

    // Store in currentScale property
    this.currentScale = scale * this.baseSize;
  }

  /**
   * Convert screen coordinates to a ray in world space
   * @param screenX - Screen X coordinate (clientX, relative to viewport)
   * @param screenY - Screen Y coordinate (clientY, relative to viewport)
   * @returns Ray in world space
   * @private
   */
  private screenToRay(screenX: number, screenY: number): Ray {
    // Convert client coordinates to canvas-relative coordinates
    const rect = this.canvas.getBoundingClientRect();
    const canvasX = screenX - rect.left;
    const canvasY = screenY - rect.top;
    
    // Account for canvas scaling (CSS size vs actual pixel size)
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    
    const pixelX = canvasX * scaleX;
    const pixelY = canvasY * scaleY;
    
    return Ray.fromScreenPoint(
      pixelX,
      pixelY,
      this.canvas.width,
      this.canvas.height,
      this.camera,
    );
  }

  /**
   * Perform hit testing to find which axis (if any) the ray intersects
   * @param ray - Ray to test
   * @returns The first axis that registers a hit, or null if no hit
   * @private
   */
  private performHitTest(ray: Ray): GizmoAxis | null {
    // No hit test in display-only mode (no interaction allowed)
    if (this.isDisplayOnly) {
      return null;
    }

    const gizmoPosition = this.getGizmoPosition();
    if (!gizmoPosition) {
      return null;
    }

    // Iterate through all axes and test for hits
    for (const axis of this.axes) {
      if (axis.testHit(ray, gizmoPosition, this.currentScale)) {
        return axis;
      }
    }

    return null;
  }

  /**
   * Handle pointer move events
   * @param event - Pointer event
   */
  onPointerMove(event: PointerEvent): void {
    // No interaction in display-only mode
    if (this.isDisplayOnly || !this.targetObject) {
      return;
    }

    // Convert pointer position to ray
    const ray = this.screenToRay(event.clientX, event.clientY);

    // Perform hit testing
    const hitAxis = this.performHitTest(ray);

    // Update hover state if not dragging
    if (!this.isDragging) {
      // Clear previous hover state
      if (this.hoveredAxis) {
        this.hoveredAxis.isHovered = false;
      }

      // Set new hover state
      this.hoveredAxis = hitAxis;
      if (this.hoveredAxis) {
        this.hoveredAxis.isHovered = true;
      }
    } else {
      // If dragging, compute and apply transformation delta
      if (this.mode === GizmoMode.Translate) {
        const delta = this.computeTranslationDelta(ray);
        if (delta !== null) {
          this.applyTranslation(delta);
        }
      } else if (this.mode === GizmoMode.Rotate) {
        const angle = this.computeRotationDelta(ray);
        if (angle !== null) {
          this.applyRotation(angle);
        }
      } else if (this.mode === GizmoMode.Scale) {
        const delta = this.computeScaleDelta(ray);
        if (delta !== null) {
          this.applyScale(delta);
        }
      }
    }
  }

  /**
   * Handle pointer down events
   * @param event - Pointer event
   */
  onPointerDown(event: PointerEvent): void {
    // No interaction in display-only mode
    if (this.isDisplayOnly || !this.targetObject) {
      return;
    }

    // Convert pointer position to ray
    const ray = this.screenToRay(event.clientX, event.clientY);

    // Perform hit testing
    const hitAxis = this.performHitTest(ray);

    // If hit, enter drag mode
    if (hitAxis) {
      this.activeAxis = hitAxis;
      this.activeAxis.isActive = true;
      this.isDragging = true;

      // Notify drag state change
      if (this.onDragStateChange) {
        this.onDragStateChange(true);
      }

      // Capture pointer on canvas
      this.canvas.setPointerCapture(event.pointerId);

      // Get gizmo position
      const gizmoPosition = new Vec3(
        this.targetObject.position[0],
        this.targetObject.position[1],
        this.targetObject.position[2],
      );

      // Store drag start point (intersection with appropriate geometry)
      // For now, we'll compute this based on mode
      if (this.mode === GizmoMode.Rotate) {
        // For rotation, intersect with plane perpendicular to axis
        const distance = ray.intersectPlane(
          gizmoPosition,
          this.activeAxis.config.direction,
        );
        if (distance !== null) {
          this.dragStartPoint = ray.at(distance);
        }
      } else {
        // For translate/scale, find closest point on axis to ray
        const axisLength = 1.05 * this.currentScale;
        const axisStart = gizmoPosition;
        const axisEnd = gizmoPosition.add(
          this.activeAxis.config.direction.multiply(axisLength),
        );

        // Use a simplified closest point calculation
        // Project ray origin onto axis
        const toOrigin = ray.origin.subtract(axisStart);
        const axisDir = axisEnd.subtract(axisStart).normalize();
        const projection = toOrigin.dot(axisDir);
        const clampedProjection = Math.max(0, Math.min(axisLength, projection));
        this.dragStartPoint = axisStart.add(
          axisDir.multiply(clampedProjection),
        );
      }

      // Store drag start transform
      this.dragStartTransform = {
        position: new Vec3(
          this.targetObject.position[0],
          this.targetObject.position[1],
          this.targetObject.position[2],
        ),
        rotation: new Vec3(
          this.targetObject.rotation[0],
          this.targetObject.rotation[1],
          this.targetObject.rotation[2],
        ),
        scale: new Vec3(
          this.targetObject.scale[0],
          this.targetObject.scale[1],
          this.targetObject.scale[2],
        ),
      };
    }
  }

  /**
   * Handle pointer up events
   * @param event - Pointer event
   */
  onPointerUp(event: PointerEvent): void {
    // Check if was dragging before clearing state
    const wasDragging = this.isDragging;

    // Exit drag mode
    this.isDragging = false;

    // Clear active axis
    if (this.activeAxis) {
      this.activeAxis.isActive = false;
      this.activeAxis = null;
    }

    // Release pointer capture
    if (this.canvas.hasPointerCapture(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }

    // Clear drag state
    this.dragStartPoint = null;
    this.dragStartTransform = null;

    // Notify drag state change
    if (wasDragging && this.onDragStateChange) {
      this.onDragStateChange(false);
    }
  }

  /**
   * Compute translation delta along active axis
   * @param ray - Current pointer ray
   * @returns Displacement vector or null if computation fails
   * @private
   */
  private computeTranslationDelta(ray: Ray): Vec3 | null {
    if (!this.activeAxis || !this.targetObject || !this.dragStartPoint || !this.dragStartTransform) {
      return null;
    }

    // 使用拖拽开始时的位置，而不是当前位置，确保计算稳定
    const gizmoPosition = this.dragStartTransform.position;

    // Get active axis direction (normalized)
    const axis = this.activeAxis.config.direction;

    // Find closest point on axis to current ray
    const axisPoint = this.closestPointOnRayToLine(ray, gizmoPosition, axis);
    if (axisPoint === null) {
      return null; // Parallel ray/axis case
    }

    // Compute displacement from drag start point
    const displacement = axisPoint.subtract(this.dragStartPoint);

    // Project displacement onto axis direction to get scalar
    const scalar = displacement.dot(axis);

    // Return displacement vector along axis
    return axis.multiply(scalar);
  }

  /**
   * Helper method to find closest point on a line to a ray
   * @param ray - The ray
   * @param linePoint - A point on the line
   * @param lineDir - Direction of the line (normalized)
   * @returns Closest point on line or null if parallel
   * @private
   */
  private closestPointOnRayToLine(
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
    if (Math.abs(denom) < 1e-6) {
      return null; // Parallel
    }

    const t = (b * e - c * d) / denom;
    const s = (a * e - b * d) / denom;

    // Return closest point on line
    return linePoint.add(lineDir.multiply(s));
  }

  /**
   * Compute rotation delta around active axis
   * @param ray - Current pointer ray
   * @returns Rotation angle in radians or null if computation fails
   * @private
   */
  private computeRotationDelta(ray: Ray): number | null {
    if (!this.activeAxis || !this.targetObject || !this.dragStartPoint || !this.dragStartTransform) {
      return null;
    }

    // 使用拖拽开始时的位置，而不是当前位置，确保计算稳定
    const gizmoPosition = this.dragStartTransform.position;

    // Get active axis direction (normalized) - this is the plane normal
    const axis = this.activeAxis.config.direction;

    // Intersect current ray with plane perpendicular to axis
    const distance = ray.intersectPlane(gizmoPosition, axis);
    if (distance === null || distance < 0) {
      return null; // Parallel ray/plane case or behind camera
    }

    const currentPoint = ray.at(distance);

    // Compute vectors from gizmo center to points
    const startVec = this.dragStartPoint.subtract(gizmoPosition);
    const currentVec = currentPoint.subtract(gizmoPosition);

    // Project onto plane (remove component along axis)
    const startProj = startVec.subtract(axis.multiply(startVec.dot(axis)));
    const currentProj = currentVec.subtract(
      axis.multiply(currentVec.dot(axis)),
    );

    // Check for degenerate cases (vectors too close to axis)
    if (
      startProj.lengthSquared() < 1e-6 ||
      currentProj.lengthSquared() < 1e-6
    ) {
      return null;
    }

    // Normalize
    const startNorm = startProj.normalize();
    const currentNorm = currentProj.normalize();

    // Compute angle using atan2 for correct sign
    const cosAngle = startNorm.dot(currentNorm);
    const sinAngle = startNorm.cross(currentNorm).dot(axis);

    return Math.atan2(sinAngle, cosAngle);
  }

  /**
   * Compute scale delta along active axis
   * @param ray - Current pointer ray
   * @returns Scale delta vector or null if computation fails
   * @private
   */
  private computeScaleDelta(ray: Ray): Vec3 | null {
    if (!this.activeAxis || !this.targetObject || !this.dragStartPoint || !this.dragStartTransform) {
      return null;
    }

    // 使用拖拽开始时的位置，而不是当前位置，确保计算稳定
    const gizmoPosition = this.dragStartTransform.position;

    // Get active axis direction (normalized)
    const axis = this.activeAxis.config.direction;

    // Find closest point on axis to current ray (same as translation)
    const axisPoint = this.closestPointOnRayToLine(ray, gizmoPosition, axis);
    if (axisPoint === null) {
      return null; // Parallel ray/axis case
    }

    // Compute displacement from drag start point
    const displacement = axisPoint.subtract(this.dragStartPoint);

    // Project displacement onto axis direction to get scalar
    const scalar = displacement.dot(axis);

    // Map to scale factor (1.0 + scalar)
    const scaleFactor = 1.0 + scalar;

    // Clamp to minimum 0.001
    const clampedFactor = Math.max(0.001, scaleFactor);

    // Return scale delta for active axis only
    const delta = new Vec3(1.0, 1.0, 1.0);
    if (Math.abs(axis.x) > 0.5) {
      delta.x = clampedFactor;
    }
    if (Math.abs(axis.y) > 0.5) {
      delta.y = clampedFactor;
    }
    if (Math.abs(axis.z) > 0.5) {
      delta.z = clampedFactor;
    }

    return delta;
  }

  /**
   * Apply translation delta to target object
   * @param delta - Displacement vector to add to target position
   * @private
   */
  private applyTranslation(delta: Vec3): void {
    if (!this.targetObject || !this.dragStartTransform) {
      return;
    }

    // Add displacement delta to target object's position
    const newPosition = this.dragStartTransform.position.add(delta);

    // Call target.setPosition() with new position
    this.targetObject.setPosition(newPosition.x, newPosition.y, newPosition.z);
  }

  /**
   * Apply rotation delta to target object
   * @param angle - Rotation angle in radians to add to target rotation
   * @private
   */
  private applyRotation(angle: number): void {
    if (!this.targetObject || !this.dragStartTransform || !this.activeAxis) {
      return;
    }

    // Get the active axis direction to determine which rotation component to modify
    const axis = this.activeAxis.config.direction;

    // Start with the original rotation from drag start
    const newRotation = this.dragStartTransform.rotation.clone();

    // Add angle delta to target object's rotation on active axis
    if (Math.abs(axis.x) > 0.5) {
      newRotation.x += angle;
    } else if (Math.abs(axis.y) > 0.5) {
      newRotation.y += angle;
    } else if (Math.abs(axis.z) > 0.5) {
      newRotation.z += angle;
    }

    // Call target.setRotation() with new rotation
    this.targetObject.setRotation(newRotation.x, newRotation.y, newRotation.z);
  }

  /**
   * Apply scale delta to target object
   * @param delta - Scale factor vector to multiply with target scale
   * @private
   */
  private applyScale(delta: Vec3): void {
    if (!this.targetObject || !this.dragStartTransform) {
      return;
    }

    // Multiply target object's scale by scale factor on active axis
    const newScale = new Vec3(
      this.dragStartTransform.scale.x * delta.x,
      this.dragStartTransform.scale.y * delta.y,
      this.dragStartTransform.scale.z * delta.z,
    );

    // Call target.setScale() with new scale
    this.targetObject.setScale(newScale.x, newScale.y, newScale.z);
  }

  /**
   * Update uniform buffer with current matrices
   * Computes model matrix from gizmo position and scale, combines with camera view-projection
   * @private
   */
  private updateUniforms(): void {
    const gizmoPosition = this.getGizmoPosition();
    if (!gizmoPosition || !this.uniformBuffer) {
      return;
    }

    const device = this.renderer.device;

    // Create model matrix from gizmo position and currentScale
    const scaleVec = new Vec3(
      this.currentScale,
      this.currentScale,
      this.currentScale,
    );
    const modelMatrix = Mat4.compose(gizmoPosition, Quat.identity(), scaleVec);

    // Get camera view-projection matrix
    const viewProjectionMatrix = new Mat4();
    viewProjectionMatrix.elements.set(this.camera.viewProjectionMatrix);

    // Create uniform data buffer (2 mat4x4 = 128 bytes)
    const uniformData = new Float32Array(32); // 32 floats = 128 bytes

    // Copy view-projection matrix (first 16 floats)
    uniformData.set(viewProjectionMatrix.elements, 0);

    // Copy model matrix (next 16 floats)
    uniformData.set(modelMatrix.elements, 16);

    // Write to uniform buffer
    device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
  }

  /**
   * Render the gizmo
   * @param pass - WebGPU render pass encoder
   */
  render(pass: GPURenderPassEncoder): void {
    // Return early if not visible
    if (!this.isVisible()) {
      return;
    }

    // Return early if pipeline or bind group not initialized
    if (!this.pipeline || !this.bindGroup) {
      return;
    }

    // Update screen-space scale
    this.updateScreenSpaceScale();

    // Update uniforms
    this.updateUniforms();

    // Set pipeline and bind group
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);

    // For each axis, set vertex/index buffers and draw
    for (const axis of this.axes) {
      // Get the color based on hover/active state
      const color = axis.getColor();

      // Get vertex and index buffers from axis
      const vertexBuffer = axis.getVertexBuffer();
      const indexBuffer = axis.getIndexBuffer();
      const indexCount = axis.getIndexCount();

      if (!vertexBuffer || !indexBuffer || indexCount === 0) {
        continue;
      }

      // Update vertex colors based on state
      // Note: For simplicity, we're using the geometry's base colors
      // In a more advanced implementation, we could update colors dynamically

      // Set vertex buffer
      pass.setVertexBuffer(0, vertexBuffer);

      // Set index buffer
      pass.setIndexBuffer(indexBuffer, "uint16");

      // Draw
      pass.drawIndexed(indexCount);
    }
  }

  /**
   * Destroy the gizmo and release GPU resources
   */
  destroy(): void {
    // Destroy all axes
    for (const axis of this.axes) {
      axis.destroy();
    }
    this.axes = [];

    // Destroy GPU buffers
    if (this.uniformBuffer) {
      this.uniformBuffer.destroy();
      this.uniformBuffer = null;
    }

    // Note: pipeline and bindGroup don't have explicit destroy methods in WebGPU
    // They will be garbage collected when no longer referenced
    this.pipeline = null;
    this.bindGroup = null;
  }
}
