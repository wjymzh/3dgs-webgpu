import { Renderer } from "../Renderer";
import { Camera } from "../Camera";
import { Vec3 } from "../math/Vec3";
import { Ray } from "../math/Ray";
import { Mat4 } from "../math/Mat4";
import { Quat } from "../math/Quat";
import { Shape, GizmoAxisType } from "./Shape";
import { ArrowShape } from "./ArrowShape";
import { PlaneShape } from "./PlaneShape";
import { SphereShape } from "./SphereShape";
import { ArcShape, ArcDisplayMode } from "./ArcShape";
import { BoxLineShape } from "./BoxLineShape";

/**
 * GizmoMode - Gizmo 操作模式
 */
export enum GizmoMode {
  Translate = 'translate',
  Rotate = 'rotate',
  Scale = 'scale',
}

/**
 * GizmoSpace - 坐标空间
 */
export type GizmoSpace = 'world' | 'local';

/**
 * GizmoDragMode - 拖拽时的显示模式
 */
export type GizmoDragMode = 'show' | 'hide' | 'selected';

/**
 * TransformableObject - 可变换对象接口
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
 * GizmoTheme - Gizmo 主题颜色
 */
export interface GizmoTheme {
  shapeBase: {
    x: Vec3;
    y: Vec3;
    z: Vec3;
    xyz: Vec3;
    f: Vec3;  // 面向相机的轴
  };
  shapeHover: {
    x: Vec3;
    y: Vec3;
    z: Vec3;
    xyz: Vec3;
    f: Vec3;
  };
  guideBase: {
    x: Vec3;
    y: Vec3;
    z: Vec3;
  };
  disabled: Vec3;
}

// 默认主题
const DEFAULT_THEME: GizmoTheme = {
  shapeBase: {
    x: new Vec3(0.9, 0.2, 0.2),
    y: new Vec3(0.2, 0.9, 0.2),
    z: new Vec3(0.2, 0.4, 0.9),
    xyz: new Vec3(0.8, 0.8, 0.8),
    f: new Vec3(0.8, 0.8, 0.8),
  },
  shapeHover: {
    x: new Vec3(1.0, 0.6, 0.6),
    y: new Vec3(0.6, 1.0, 0.6),
    z: new Vec3(0.6, 0.6, 1.0),
    xyz: new Vec3(1.0, 1.0, 1.0),
    f: new Vec3(1.0, 1.0, 1.0),
  },
  guideBase: {
    x: new Vec3(0.9, 0.2, 0.2),
    y: new Vec3(0.2, 0.9, 0.2),
    z: new Vec3(0.2, 0.4, 0.9),
  },
  disabled: new Vec3(0.5, 0.5, 0.5),
};

// 常量
const GLANCE_EPSILON = 0.01;
const RING_FACING_EPSILON = 1e-4;
const PERS_SCALE_RATIO = 0.3;
const MIN_SCALE = 1e-4;
const RAD_TO_DEG = 180 / Math.PI;

/**
 * TransformGizmoConfig - Gizmo 配置
 */
export interface TransformGizmoConfig {
  renderer: Renderer;
  camera: Camera;
  canvas: HTMLCanvasElement;
  size?: number;
  snap?: boolean;
  snapIncrement?: number;
}

/**
 * TransformGizmoV2 - 重构后的变换 Gizmo
 * 参考 PlayCanvas 引擎的 TransformGizmo 实现
 */
export class TransformGizmoV2 {
  private renderer: Renderer;
  private camera: Camera;
  private canvas: HTMLCanvasElement;
  
  // 配置
  private _size: number = 1.0;
  private _scale: number = 1.0;
  private _mode: GizmoMode = GizmoMode.Translate;
  private _coordSpace: GizmoSpace = 'world';
  private _theme: GizmoTheme = DEFAULT_THEME;
  
  // Snap 功能
  snap: boolean = false;
  snapIncrement: number = 1;
  
  // 拖拽模式
  dragMode: GizmoDragMode = 'selected';
  
  // 平面翻转
  flipPlanes: boolean = true;
  
  // 目标对象
  private _target: TransformableObject | null = null;
  
  // 形状
  private _shapes: Map<GizmoAxisType | 'f', Shape> = new Map();
  
  // 交互状态
  private _hoverAxis: GizmoAxisType | 'f' | '' = '';
  private _selectedAxis: GizmoAxisType | 'f' | '' = '';
  private _hoverIsPlane: boolean = false;
  private _selectedIsPlane: boolean = false;
  private _dragging: boolean = false;

  // 拖拽起始状态
  private _rootStartPos: Vec3 = new Vec3();
  private _rootStartRot: Quat = Quat.identity();
  private _selectionStartPoint: Vec3 = new Vec3();
  private _dragStartTransform: {
    position: Vec3;
    rotation: Vec3;
    scale: Vec3;
  } | null = null;
  
  // 面向相机的方向缓存
  private _facingDir: Vec3 = new Vec3();
  
  // 回调
  private _onDragStateChange: ((isDragging: boolean) => void) | null = null;
  
  // GPU 资源
  private pipeline: GPURenderPipeline | null = null;
  private linePipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  
  // 每个 Shape 的 uniform buffer 和 bind group
  private shapeUniformBuffers: Map<GizmoAxisType | 'f', GPUBuffer> = new Map();
  private shapeBindGroups: Map<GizmoAxisType | 'f', GPUBindGroup> = new Map();
  
  // 辅助线资源
  private guideLineBuffer: GPUBuffer | null = null;
  private guideLineBindGroup: GPUBindGroup | null = null;
  
  constructor(config: TransformGizmoConfig) {
    this.renderer = config.renderer;
    this.camera = config.camera;
    this.canvas = config.canvas;
    
    if (config.size !== undefined) this._size = config.size;
    if (config.snap !== undefined) this.snap = config.snap;
    if (config.snapIncrement !== undefined) this.snapIncrement = config.snapIncrement;
  }

  /**
   * 初始化 Gizmo
   */
  init(): void {
    this.createPipeline();
    this.createLinePipeline();
    this.createShapes();
  }

  /**
   * 创建 WebGPU 渲染管线
   */
  private createPipeline(): void {
    const device = this.renderer.device;
    
    // Shader 支持从 uniform 读取颜色（包含透明度）
    const shaderCode = `
      struct Uniforms {
        viewProjection: mat4x4<f32>,
        model: mat4x4<f32>,
        color: vec4<f32>,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;

      struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) vertexColor: vec3<f32>,
      }

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
      }

      @vertex
      fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        let worldPos = uniforms.model * vec4<f32>(input.position, 1.0);
        output.position = uniforms.viewProjection * worldPos;
        // 使用 uniform 中的颜色，忽略顶点颜色
        output.color = uniforms.color;
        return output;
      }

      @fragment
      fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        // 如果透明度为 0，丢弃片元
        if (input.color.a < 0.01) {
          discard;
        }
        return input.color;
      }
    `;
    
    const shaderModule = device.createShaderModule({ code: shaderCode });
    
    this.uniformBuffer = device.createBuffer({
      size: 144,  // 64 + 64 + 16 = 144 bytes (viewProj + model + color)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      }],
    });

    this.bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });
    
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vertexMain",
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ],
        }],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [{
          format: this.renderer.format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: this.renderer.depthFormat,
        depthWriteEnabled: false,
        depthCompare: "always",
      },
    });
  }

  /**
   * 创建辅助线渲染管线
   */
  private createLinePipeline(): void {
    const device = this.renderer.device;
    
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
        return vec4<f32>(input.color, 0.6);
      }
    `;
    
    const shaderModule = device.createShaderModule({ code: shaderCode });
    
    if (!this.bindGroupLayout) return;
    
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.linePipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vertexMain",
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ],
        }],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [{
          format: this.renderer.format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        }],
      },
      primitive: { topology: "line-list", cullMode: "none" },
      depthStencil: {
        format: this.renderer.depthFormat,
        depthWriteEnabled: false,
        depthCompare: "always",
      },
    });
    
    // 创建辅助线缓冲区
    this.guideLineBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    this.guideLineBindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.guideLineBuffer } }],
    });
  }

  /**
   * 创建形状
   */
  private createShapes(): void {
    const device = this.renderer.device;
    
    // 清除现有形状和资源
    for (const shape of this._shapes.values()) {
      shape.destroy();
    }
    this._shapes.clear();
    
    for (const buffer of this.shapeUniformBuffers.values()) {
      buffer.destroy();
    }
    this.shapeUniformBuffers.clear();
    this.shapeBindGroups.clear();
    
    if (this._mode === GizmoMode.Translate) {
      this.createTranslateShapes(device);
    } else if (this._mode === GizmoMode.Rotate) {
      this.createRotateShapes(device);
    } else if (this._mode === GizmoMode.Scale) {
      this.createScaleShapes(device);
    }
    
    this.createShapeUniformBuffers(device);
  }
  
  /**
   * 为每个 Shape 创建独立的 uniform buffer 和 bind group
   */
  private createShapeUniformBuffers(device: GPUDevice): void {
    if (!this.bindGroupLayout) return;
    
    for (const [axis] of this._shapes) {
      const uniformBuffer = device.createBuffer({
        size: 144,  // 64 + 64 + 16 = 144 bytes (viewProj + model + color)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      
      const bindGroup = device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
      });
      
      this.shapeUniformBuffers.set(axis, uniformBuffer);
      this.shapeBindGroups.set(axis, bindGroup);
    }
  }

  /**
   * 创建平移模式的形状
   */
  private createTranslateShapes(device: GPUDevice): void {
    const theme = this._theme;
    
    // 中心球
    const center = new SphereShape({
      axis: 'xyz',
      defaultColor: theme.shapeBase.xyz,
      hoverColor: theme.shapeHover.xyz,
      disabledColor: theme.disabled,
      radius: 0.1,
    });
    center.createGeometry(device);
    this._shapes.set('xyz', center);
    
    // X 轴箭头
    const xArrow = new ArrowShape({
      axis: 'x',
      defaultColor: theme.shapeBase.x,
      hoverColor: theme.shapeHover.x,
      disabledColor: theme.disabled,
      rotation: new Vec3(0, 0, -90),
    });
    xArrow.createGeometry(device);
    this._shapes.set('x', xArrow);
    
    // Y 轴箭头
    const yArrow = new ArrowShape({
      axis: 'y',
      defaultColor: theme.shapeBase.y,
      hoverColor: theme.shapeHover.y,
      disabledColor: theme.disabled,
      rotation: new Vec3(0, 0, 0),
    });
    yArrow.createGeometry(device);
    this._shapes.set('y', yArrow);
    
    // Z 轴箭头
    const zArrow = new ArrowShape({
      axis: 'z',
      defaultColor: theme.shapeBase.z,
      hoverColor: theme.shapeHover.z,
      disabledColor: theme.disabled,
      rotation: new Vec3(90, 0, 0),
    });
    zArrow.createGeometry(device);
    this._shapes.set('z', zArrow);
    
    // YZ 平面 (X 轴方向)
    const yzPlane = new PlaneShape({
      axis: 'yz',
      defaultColor: theme.shapeBase.x,
      hoverColor: theme.shapeHover.x,
      disabledColor: theme.disabled,
      rotation: new Vec3(0, 0, -90),
    });
    yzPlane.createGeometry(device);
    this._shapes.set('yz', yzPlane);
    
    // XZ 平面 (Y 轴方向)
    const xzPlane = new PlaneShape({
      axis: 'xz',
      defaultColor: theme.shapeBase.y,
      hoverColor: theme.shapeHover.y,
      disabledColor: theme.disabled,
    });
    xzPlane.createGeometry(device);
    this._shapes.set('xz', xzPlane);
    
    // XY 平面 (Z 轴方向)
    const xyPlane = new PlaneShape({
      axis: 'xy',
      defaultColor: theme.shapeBase.z,
      hoverColor: theme.shapeHover.z,
      disabledColor: theme.disabled,
      rotation: new Vec3(90, 0, 0),
    });
    xyPlane.createGeometry(device);
    this._shapes.set('xy', xyPlane);
  }

  /**
   * 创建旋转模式的形状
   */
  private createRotateShapes(device: GPUDevice): void {
    const theme = this._theme;
    
    // X 轴旋转环
    const xArc = new ArcShape({
      axis: 'x',
      defaultColor: theme.shapeBase.x,
      hoverColor: theme.shapeHover.x,
      disabledColor: theme.disabled,
      rotation: new Vec3(0, 0, -90),
      ringRadius: 0.5,
      tubeRadius: 0.015,
      sectorAngle: 180,
      tolerance: 0.05,
    });
    xArc.createGeometry(device);
    this._shapes.set('x', xArc);
    
    // Y 轴旋转环
    const yArc = new ArcShape({
      axis: 'y',
      defaultColor: theme.shapeBase.y,
      hoverColor: theme.shapeHover.y,
      disabledColor: theme.disabled,
      rotation: new Vec3(0, 0, 0),
      ringRadius: 0.5,
      tubeRadius: 0.015,
      sectorAngle: 180,
      tolerance: 0.05,
    });
    yArc.createGeometry(device);
    this._shapes.set('y', yArc);
    
    // Z 轴旋转环
    const zArc = new ArcShape({
      axis: 'z',
      defaultColor: theme.shapeBase.z,
      hoverColor: theme.shapeHover.z,
      disabledColor: theme.disabled,
      rotation: new Vec3(90, 0, 90),
      ringRadius: 0.5,
      tubeRadius: 0.015,
      sectorAngle: 180,
      tolerance: 0.05,
    });
    zArc.createGeometry(device);
    this._shapes.set('z', zArc);
    
    // 面向相机的旋转环
    const fArc = new ArcShape({
      axis: 'xyz',  // 使用 xyz 作为内部 axis
      defaultColor: theme.shapeBase.f,
      hoverColor: theme.shapeHover.f,
      disabledColor: theme.disabled,
      ringRadius: 0.55,
      tubeRadius: 0.015,
      sectorAngle: 360,
      tolerance: 0.05,
    });
    fArc.createGeometry(device);
    this._shapes.set('f', fArc);
    
    // 中心球（用于自由旋转）- 默认半透明，hover 时更明显
    const center = new SphereShape({
      axis: 'xyz',
      defaultColor: theme.shapeBase.xyz,
      hoverColor: theme.shapeHover.xyz,
      disabledColor: theme.disabled,
      defaultAlpha: 0.0,   // 默认完全透明
      hoverAlpha: 0.3,     // hover 时半透明
      radius: 0.45,
    });
    center.createGeometry(device);
    this._shapes.set('xyz', center);
  }

  /**
   * 创建缩放模式的形状
   */
  private createScaleShapes(device: GPUDevice): void {
    const theme = this._theme;
    
    // 中心球（统一缩放）
    const center = new SphereShape({
      axis: 'xyz',
      defaultColor: theme.shapeBase.xyz,
      hoverColor: theme.shapeHover.xyz,
      disabledColor: theme.disabled,
      radius: 0.1,
    });
    center.createGeometry(device);
    this._shapes.set('xyz', center);
    
    // X 轴缩放
    const xBox = new BoxLineShape({
      axis: 'x',
      defaultColor: theme.shapeBase.x,
      hoverColor: theme.shapeHover.x,
      disabledColor: theme.disabled,
      rotation: new Vec3(0, 0, -90),
    });
    xBox.createGeometry(device);
    this._shapes.set('x', xBox);
    
    // Y 轴缩放
    const yBox = new BoxLineShape({
      axis: 'y',
      defaultColor: theme.shapeBase.y,
      hoverColor: theme.shapeHover.y,
      disabledColor: theme.disabled,
    });
    yBox.createGeometry(device);
    this._shapes.set('y', yBox);
    
    // Z 轴缩放
    const zBox = new BoxLineShape({
      axis: 'z',
      defaultColor: theme.shapeBase.z,
      hoverColor: theme.shapeHover.z,
      disabledColor: theme.disabled,
      rotation: new Vec3(90, 0, 0),
    });
    zBox.createGeometry(device);
    this._shapes.set('z', zBox);
  }

  // ==================== 属性访问器 ====================
  
  get mode(): GizmoMode { return this._mode; }
  set mode(value: GizmoMode) {
    if (this._mode !== value) {
      this._mode = value;
      this.createShapes();
      this._clearInteractionState();
    }
  }
  
  get coordSpace(): GizmoSpace { return this._coordSpace; }
  set coordSpace(value: GizmoSpace) { this._coordSpace = value; }
  
  get size(): number { return this._size; }
  set size(value: number) { this._size = value; }
  
  get target(): TransformableObject | null { return this._target; }
  
  get isDragging(): boolean { return this._dragging; }

  // ==================== 目标管理 ====================
  
  setTarget(object: TransformableObject | null): void {
    this._target = object;
    this._clearInteractionState();
  }
  
  setOnDragStateChange(callback: ((isDragging: boolean) => void) | null): void {
    this._onDragStateChange = callback;
  }
  
  private _clearInteractionState(): void {
    this._hoverAxis = '';
    this._selectedAxis = '';
    this._hoverIsPlane = false;
    this._selectedIsPlane = false;
    this._dragging = false;
    this._dragStartTransform = null;
    
    for (const shape of this._shapes.values()) {
      shape.hover(false);
    }
  }

  // ==================== 缩放计算 ====================
  
  private getGizmoPosition(): Vec3 | null {
    if (!this._target) return null;
    return new Vec3(
      this._target.position[0],
      this._target.position[1],
      this._target.position[2]
    );
  }
  
  /**
   * 获取 Gizmo 的视觉旋转（用于渲染形状）
   * - 平移模式：根据 coordSpace 设置
   * - 旋转模式：始终使用世界坐标（三个轴保持正交）
   * - 缩放模式：始终使用本地坐标（跟随物体旋转）
   */
  private getGizmoRotation(): Quat {
    if (this._mode === GizmoMode.Rotate) {
      // 旋转模式始终使用世界坐标，保持三个轴正交
      return Quat.identity();
    }
    if (this._mode === GizmoMode.Scale && this._target) {
      // 缩放模式始终使用本地坐标
      return Quat.fromEuler(
        this._target.rotation[0],
        this._target.rotation[1],
        this._target.rotation[2]
      );
    }
    // 平移模式根据 coordSpace 设置
    if (this._coordSpace === 'local' && this._target) {
      return Quat.fromEuler(
        this._target.rotation[0],
        this._target.rotation[1],
        this._target.rotation[2]
      );
    }
    return Quat.identity();
  }
  
  private updateScale(): void {
    const gizmoPos = this.getGizmoPosition();
    if (!gizmoPos) return;
    
    const cameraPos = new Vec3(
      this.camera.position[0],
      this.camera.position[1],
      this.camera.position[2]
    );
    
    const dist = gizmoPos.distance(cameraPos);
    this._scale = Math.tan(0.5 * this.camera.fov) * dist * PERS_SCALE_RATIO;
    this._scale = Math.max(this._scale * this._size, MIN_SCALE);
  }
  
  /**
   * 获取面向相机的方向（从 Gizmo 指向相机）
   */
  private getFacingDir(): Vec3 {
    const gizmoPos = this.getGizmoPosition();
    if (!gizmoPos) return new Vec3(0, 0, 1);
    
    const cameraPos = new Vec3(
      this.camera.position[0],
      this.camera.position[1],
      this.camera.position[2]
    );
    
    return cameraPos.subtract(gizmoPos).normalize();
  }

  // ==================== 动态形状调整 ====================
  
  /**
   * 更新形状以面向相机
   * 参考 PlayCanvas 的 _shapesLookAtCamera
   */
  private _shapesLookAtCamera(): void {
    if (this._mode === GizmoMode.Translate) {
      this._updateTranslateShapesForCamera();
    } else if (this._mode === GizmoMode.Rotate) {
      this._updateRotateShapesForCamera();
    }
  }
  
  /**
   * 更新平移形状
   */
  private _updateTranslateShapesForCamera(): void {
    const facingDir = this.getFacingDir();
    const gizmoRot = this.getGizmoRotation();
    
    // 计算 Gizmo 的三个轴方向
    const right = gizmoRot.transformVector(new Vec3(1, 0, 0));
    const up = gizmoRot.transformVector(new Vec3(0, 1, 0));
    const forward = gizmoRot.transformVector(new Vec3(0, 0, 1));
    
    // 轴可见性
    const xShape = this._shapes.get('x');
    if (xShape) {
      const dot = Math.abs(facingDir.dot(right));
      xShape.visible = (1 - dot) > GLANCE_EPSILON;
    }
    
    const yShape = this._shapes.get('y');
    if (yShape) {
      const dot = Math.abs(facingDir.dot(up));
      yShape.visible = (1 - dot) > GLANCE_EPSILON;
    }
    
    const zShape = this._shapes.get('z');
    if (zShape) {
      const dot = Math.abs(facingDir.dot(forward));
      zShape.visible = (1 - dot) > GLANCE_EPSILON;
    }
    
    // 平面可见性和翻转
    const yzPlane = this._shapes.get('yz') as PlaneShape | undefined;
    if (yzPlane) {
      const cross = facingDir.cross(right);
      yzPlane.visible = (1 - cross.length()) > GLANCE_EPSILON;
      if (this.flipPlanes) {
        const flipped = new Vec3(
          0,
          cross.dot(forward) < 0 ? 1 : 0,
          cross.dot(up) < 0 ? 1 : 0
        );
        yzPlane.setFlipped(flipped);
      }
    }
    
    const xzPlane = this._shapes.get('xz') as PlaneShape | undefined;
    if (xzPlane) {
      const cross = facingDir.cross(up);
      xzPlane.visible = (1 - cross.length()) > GLANCE_EPSILON;
      if (this.flipPlanes) {
        const flipped = new Vec3(
          cross.dot(forward) > 0 ? 1 : 0,
          0,
          cross.dot(right) > 0 ? 1 : 0
        );
        xzPlane.setFlipped(flipped);
      }
    }
    
    const xyPlane = this._shapes.get('xy') as PlaneShape | undefined;
    if (xyPlane) {
      const cross = facingDir.cross(forward);
      xyPlane.visible = (1 - cross.length()) > GLANCE_EPSILON;
      if (this.flipPlanes) {
        const flipped = new Vec3(
          cross.dot(up) < 0 ? 1 : 0,
          cross.dot(right) > 0 ? 1 : 0,
          0
        );
        xyPlane.setFlipped(flipped);
      }
    }
  }

  /**
   * 更新旋转形状
   */
  private _updateRotateShapesForCamera(): void {
    const facingDir = this.getFacingDir();
    const gizmoRot = this.getGizmoRotation();
    
    // 将面向方向转换到 Gizmo 局部空间
    const invRot = gizmoRot.inverse();
    const localFacingDir = invRot.transformVector(facingDir.clone());
    
    // 计算 Gizmo 的三个轴方向
    const right = gizmoRot.transformVector(new Vec3(1, 0, 0));
    const up = gizmoRot.transformVector(new Vec3(0, 1, 0));
    const forward = gizmoRot.transformVector(new Vec3(0, 0, 1));
    
    // X 轴旋转环
    const xArc = this._shapes.get('x') as ArcShape | undefined;
    if (xArc) {
      const angle = Math.atan2(localFacingDir.z, localFacingDir.y) * RAD_TO_DEG;
      xArc.setDynamicRotation(new Vec3(0, angle - 90, 0));
      
      const dot = facingDir.dot(right);
      if (!this._dragging) {
        const showSector = 1 - Math.abs(dot) > RING_FACING_EPSILON;
        xArc.show(showSector ? 'sector' : 'ring');
      }
    }
    
    // Y 轴旋转环
    const yArc = this._shapes.get('y') as ArcShape | undefined;
    if (yArc) {
      const angle = Math.atan2(localFacingDir.x, localFacingDir.z) * RAD_TO_DEG;
      yArc.setDynamicRotation(new Vec3(0, angle, 0));
      
      const dot = facingDir.dot(up);
      if (!this._dragging) {
        const showSector = 1 - Math.abs(dot) > RING_FACING_EPSILON;
        yArc.show(showSector ? 'sector' : 'ring');
      }
    }
    
    // Z 轴旋转环
    const zArc = this._shapes.get('z') as ArcShape | undefined;
    if (zArc) {
      const angle = Math.atan2(localFacingDir.y, localFacingDir.x) * RAD_TO_DEG;
      zArc.setDynamicRotation(new Vec3(0, 0, angle));
      
      const dot = facingDir.dot(forward);
      if (!this._dragging) {
        const showSector = 1 - Math.abs(dot) > RING_FACING_EPSILON;
        zArc.show(showSector ? 'sector' : 'ring');
      }
    }
    
    // 面向相机的旋转环
    const fArc = this._shapes.get('f') as ArcShape | undefined;
    if (fArc) {
      // 计算面向相机的旋转
      const cameraPos = new Vec3(
        this.camera.position[0],
        this.camera.position[1],
        this.camera.position[2]
      );
      const gizmoPos = this.getGizmoPosition()!;
      const dir = cameraPos.subtract(gizmoPos).normalize();
      
      const elev = Math.atan2(-dir.y, Math.sqrt(dir.x * dir.x + dir.z * dir.z)) * RAD_TO_DEG;
      const azim = Math.atan2(-dir.x, -dir.z) * RAD_TO_DEG;
      
      // 设置面向相机的旋转（覆盖基础旋转）
      fArc.setDynamicRotation(new Vec3(-elev + 90, azim, 0));
    }
    
    this._facingDir = facingDir;
  }

  /**
   * 拖拽时更新形状显示
   */
  private _updateDragVisibility(isDragging: boolean): void {
    if (this._mode === GizmoMode.Rotate) {
      this._updateRotateDragVisibility(isDragging);
    } else if (this._mode === GizmoMode.Translate) {
      this._updateTranslateDragVisibility(isDragging);
    } else if (this._mode === GizmoMode.Scale) {
      this._updateScaleDragVisibility(isDragging);
    }
  }
  
  private _updateRotateDragVisibility(isDragging: boolean): void {
    for (const [axis, shape] of this._shapes) {
      if (!(shape instanceof ArcShape)) continue;
      
      switch (this.dragMode) {
        case 'show':
          break;
        case 'hide':
          if (isDragging) {
            (shape as ArcShape).show(axis === this._selectedAxis ? 'ring' : 'none');
          } else {
            (shape as ArcShape).show('sector');
          }
          break;
        case 'selected':
          if (isDragging) {
            (shape as ArcShape).show(axis === this._selectedAxis ? 'ring' : 'sector');
          }
          break;
      }
    }
  }
  
  private _updateTranslateDragVisibility(isDragging: boolean): void {
    for (const [axis, shape] of this._shapes) {
      switch (this.dragMode) {
        case 'show':
          break;
        case 'hide':
          shape.visible = !isDragging;
          break;
        case 'selected':
          if (this._selectedAxis === 'xyz') {
            shape.visible = isDragging ? axis.length === 1 : true;
          } else if (this._selectedIsPlane) {
            shape.visible = isDragging ? axis.length === 1 && !axis.includes(this._selectedAxis) : true;
          } else {
            shape.visible = isDragging ? axis === this._selectedAxis : true;
          }
          break;
      }
    }
  }
  
  private _updateScaleDragVisibility(isDragging: boolean): void {
    for (const [axis, shape] of this._shapes) {
      switch (this.dragMode) {
        case 'show':
          break;
        case 'hide':
          shape.visible = !isDragging;
          break;
        case 'selected':
          if (this._selectedAxis === 'xyz') {
            shape.visible = isDragging ? axis.length === 1 : true;
          } else {
            shape.visible = isDragging ? axis === this._selectedAxis : true;
          }
          break;
      }
    }
  }

  // ==================== 射线转换 ====================
  
  private screenToRay(screenX: number, screenY: number): Ray {
    const rect = this.canvas.getBoundingClientRect();
    const canvasX = screenX - rect.left;
    const canvasY = screenY - rect.top;
    
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    
    const pixelX = canvasX * scaleX;
    const pixelY = canvasY * scaleY;
    
    return Ray.fromScreenPoint(
      pixelX,
      pixelY,
      this.canvas.width,
      this.canvas.height,
      this.camera
    );
  }
  
  private performHitTest(ray: Ray): { axis: GizmoAxisType | 'f' | ''; isPlane: boolean } {
    if (!this._target) {
      return { axis: '', isPlane: false };
    }
    
    const gizmoPos = this.getGizmoPosition()!;
    const gizmoRot = this.getGizmoRotation();
    
    const scale = new Vec3(this._scale, this._scale, this._scale);
    const worldTransform = Mat4.compose(gizmoPos, gizmoRot, scale);
    
    let closestDist: number | null = null;
    let closestAxis: GizmoAxisType | 'f' | '' = '';
    let closestIsPlane = false;
    
    for (const [axis, shape] of this._shapes) {
      // 不检查 visible，让 Shape.intersect() 通过 interactable 属性控制
      // 这样透明但可交互的形状（如旋转模式的中心球）仍然可以被选中
      if (shape.disabled) continue;
      
      const dist = shape.intersect(ray, worldTransform);
      if (dist !== null && (closestDist === null || dist < closestDist)) {
        closestDist = dist;
        closestAxis = axis;
        closestIsPlane = axis.length === 2;
      }
    }
    
    return { axis: closestAxis, isPlane: closestIsPlane };
  }

  // ==================== 事件处理 ====================
  
  onPointerMove(event: PointerEvent): void {
    if (!this._target) return;
    
    const ray = this.screenToRay(event.clientX, event.clientY);
    
    if (!this._dragging) {
      const hit = this.performHitTest(ray);
      this._updateHover(hit.axis, hit.isPlane);
    } else {
      const point = this._screenToPoint(event.clientX, event.clientY);
      this._applyTransform(point);
    }
  }
  
  onPointerDown(event: PointerEvent): void {
    if (!this._target) return;
    
    const ray = this.screenToRay(event.clientX, event.clientY);
    const hit = this.performHitTest(ray);
    
    if (hit.axis) {
      this._selectedAxis = hit.axis;
      this._selectedIsPlane = hit.isPlane;
      this._dragging = true;
      
      this._rootStartPos = this.getGizmoPosition()!.clone();
      this._rootStartRot = this.getGizmoRotation();
      
      const point = this._screenToPoint(event.clientX, event.clientY);
      this._selectionStartPoint = point.clone();
      
      this._dragStartTransform = {
        position: new Vec3(
          this._target.position[0],
          this._target.position[1],
          this._target.position[2]
        ),
        rotation: new Vec3(
          this._target.rotation[0],
          this._target.rotation[1],
          this._target.rotation[2]
        ),
        scale: new Vec3(
          this._target.scale[0],
          this._target.scale[1],
          this._target.scale[2]
        ),
      };
      
      this.canvas.setPointerCapture(event.pointerId);
      
      // 更新拖拽时的形状显示
      this._updateDragVisibility(true);
      
      if (this._onDragStateChange) {
        this._onDragStateChange(true);
      }
    }
  }
  
  onPointerUp(event: PointerEvent): void {
    const wasDragging = this._dragging;
    
    this._dragging = false;
    this._selectedAxis = '';
    this._selectedIsPlane = false;
    this._dragStartTransform = null;
    
    if (this.canvas.hasPointerCapture(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
    
    // 恢复形状显示
    this._updateDragVisibility(false);
    
    if (wasDragging && this._onDragStateChange) {
      this._onDragStateChange(false);
    }
  }

  private _updateHover(axis: GizmoAxisType | 'f' | '', isPlane: boolean): void {
    if (this._dragging) return;
    
    if (this._hoverAxis !== axis) {
      // 清除之前的 hover 状态
      for (const shape of this._shapes.values()) {
        shape.hover(false);
      }
      
      this._hoverAxis = axis;
      this._hoverIsPlane = isPlane;
      
      if (axis) {
        if (axis === 'xyz') {
          this._shapes.get('x')?.hover(true);
          this._shapes.get('y')?.hover(true);
          this._shapes.get('z')?.hover(true);
          this._shapes.get('xyz')?.hover(true);
        } else if (axis === 'f') {
          this._shapes.get('f')?.hover(true);
        } else if (isPlane) {
          const shape = this._shapes.get(axis);
          shape?.hover(true);
          for (const char of axis) {
            this._shapes.get(char as GizmoAxisType)?.hover(true);
          }
        } else {
          this._shapes.get(axis)?.hover(true);
        }
      }
    }
  }

  // ==================== 变换计算 ====================
  
  private _screenToPoint(x: number, y: number): Vec3 {
    const ray = this.screenToRay(x, y);
    const axis = this._selectedAxis;
    const isPlane = this._selectedIsPlane;
    
    // 旋转模式使用不同的平面
    if (this._mode === GizmoMode.Rotate && axis && axis !== 'xyz' && axis.length === 1) {
      const plane = this._createRotationPlane(axis);
      const dist = ray.intersectPlane(plane.point, plane.normal);
      if (dist === null) {
        return this._rootStartPos.clone();
      }
      return ray.at(dist);
    }
    
    // 面向相机的旋转
    if (this._mode === GizmoMode.Rotate && axis === 'f') {
      const plane = this._createFacingPlane();
      const dist = ray.intersectPlane(plane.point, plane.normal);
      if (dist === null) {
        return this._rootStartPos.clone();
      }
      return ray.at(dist);
    }
    
    const plane = this._createInteractionPlane(axis as GizmoAxisType, isPlane);
    
    const dist = ray.intersectPlane(plane.point, plane.normal);
    if (dist === null) {
      return new Vec3(0, 0, 0);
    }
    
    const point = ray.at(dist);
    
    const localPoint = point.subtract(this._rootStartPos);
    const invRot = this._rootStartRot.inverse();
    const rotatedPoint = invRot.transformVector(localPoint);
    
    if (!isPlane && axis !== 'xyz' && axis !== 'f' && axis.length === 1) {
      this._projectToAxis(rotatedPoint, axis);
    }
    
    return rotatedPoint;
  }

  private _createRotationPlane(axis: string): { point: Vec3; normal: Vec3 } {
    const point = this._rootStartPos.clone();
    let normal = new Vec3(0, 1, 0);
    
    if (this._coordSpace === 'local') {
      if (axis === 'x') normal = this._rootStartRot.transformVector(new Vec3(1, 0, 0));
      else if (axis === 'y') normal = this._rootStartRot.transformVector(new Vec3(0, 1, 0));
      else if (axis === 'z') normal = this._rootStartRot.transformVector(new Vec3(0, 0, 1));
    } else {
      if (axis === 'x') normal = new Vec3(1, 0, 0);
      else if (axis === 'y') normal = new Vec3(0, 1, 0);
      else if (axis === 'z') normal = new Vec3(0, 0, 1);
    }
    
    return { point, normal };
  }
  
  private _createFacingPlane(): { point: Vec3; normal: Vec3 } {
    const point = this._rootStartPos.clone();
    const normal = this.getFacingDir().multiply(-1);
    return { point, normal };
  }
  
  private _createInteractionPlane(
    axis: GizmoAxisType | '', 
    isPlane: boolean
  ): { point: Vec3; normal: Vec3 } {
    const point = this._rootStartPos.clone();
    let normal = new Vec3(0, 1, 0);
    
    if (axis === 'xyz' || isPlane) {
      const facingDir = this.getFacingDir();
      normal = facingDir.multiply(-1);
    } else if (axis === 'x') {
      const axisDir = this._rootStartRot.transformVector(new Vec3(1, 0, 0));
      const cameraDir = this.getFacingDir();
      const cross = axisDir.cross(cameraDir);
      if (cross.lengthSquared() > 1e-6) {
        normal = cross.cross(axisDir).normalize();
      } else {
        normal = this._rootStartRot.transformVector(new Vec3(0, 1, 0));
      }
    } else if (axis === 'y') {
      const axisDir = this._rootStartRot.transformVector(new Vec3(0, 1, 0));
      const cameraDir = this.getFacingDir();
      const cross = axisDir.cross(cameraDir);
      if (cross.lengthSquared() > 1e-6) {
        normal = cross.cross(axisDir).normalize();
      } else {
        normal = this._rootStartRot.transformVector(new Vec3(1, 0, 0));
      }
    } else if (axis === 'z') {
      const axisDir = this._rootStartRot.transformVector(new Vec3(0, 0, 1));
      const cameraDir = this.getFacingDir();
      const cross = axisDir.cross(cameraDir);
      if (cross.lengthSquared() > 1e-6) {
        normal = cross.cross(axisDir).normalize();
      } else {
        normal = this._rootStartRot.transformVector(new Vec3(0, 1, 0));
      }
    }
    
    return { point, normal };
  }
  
  private _projectToAxis(point: Vec3, axis: string): void {
    if (axis === 'x') {
      point.y = 0;
      point.z = 0;
    } else if (axis === 'y') {
      point.x = 0;
      point.z = 0;
    } else if (axis === 'z') {
      point.x = 0;
      point.y = 0;
    }
  }

  private _applyTransform(point: Vec3): void {
    if (!this._target || !this._dragStartTransform) return;
    
    const delta = point.subtract(this._selectionStartPoint);
    
    if (this._mode === GizmoMode.Translate) {
      this._applyTranslation(delta);
    } else if (this._mode === GizmoMode.Rotate) {
      this._applyRotation(point);
    } else if (this._mode === GizmoMode.Scale) {
      this._applyScale(delta);
    }
  }
  
  private _applyTranslation(delta: Vec3): void {
    if (!this._target || !this._dragStartTransform) return;
    
    if (this.snap) {
      delta.x = Math.round(delta.x / this.snapIncrement) * this.snapIncrement;
      delta.y = Math.round(delta.y / this.snapIncrement) * this.snapIncrement;
      delta.z = Math.round(delta.z / this.snapIncrement) * this.snapIncrement;
    }
    
    const worldDelta = this._rootStartRot.transformVector(delta);
    
    const newPos = this._dragStartTransform.position.add(worldDelta);
    this._target.setPosition(newPos.x, newPos.y, newPos.z);
  }
  
  private _applyRotation(point: Vec3): void {
    if (!this._target || !this._dragStartTransform) return;
    
    const axis = this._selectedAxis;
    if (!axis) return;
    
    // 面向相机的旋转
    if (axis === 'f') {
      this._applyFacingRotation(point);
      return;
    }
    
    if (axis === 'xyz') return;
    
    const gizmoPos = this._rootStartPos;
    
    const startVec = this._selectionStartPoint.subtract(gizmoPos);
    const currentVec = point.subtract(gizmoPos);
    
    let rotAxis = new Vec3(0, 1, 0);
    if (this._coordSpace === 'local') {
      if (axis === 'x') rotAxis = this._rootStartRot.transformVector(new Vec3(1, 0, 0));
      else if (axis === 'y') rotAxis = this._rootStartRot.transformVector(new Vec3(0, 1, 0));
      else if (axis === 'z') rotAxis = this._rootStartRot.transformVector(new Vec3(0, 0, 1));
    } else {
      if (axis === 'x') rotAxis = new Vec3(1, 0, 0);
      else if (axis === 'y') rotAxis = new Vec3(0, 1, 0);
      else if (axis === 'z') rotAxis = new Vec3(0, 0, 1);
    }
    
    if (startVec.lengthSquared() < 1e-6 || currentVec.lengthSquared() < 1e-6) {
      return;
    }
    
    const startNorm = startVec.normalize();
    const currentNorm = currentVec.normalize();
    
    const cosAngle = Math.max(-1, Math.min(1, startNorm.dot(currentNorm)));
    const crossVec = startNorm.cross(currentNorm);
    const sinAngle = crossVec.dot(rotAxis);
    let angleDelta = Math.atan2(sinAngle, cosAngle);
    
    if (this.snap) {
      const snapAngle = this.snapIncrement * Math.PI / 180;
      angleDelta = Math.round(angleDelta / snapAngle) * snapAngle;
    }
    
    const deltaQuat = Quat.fromAxisAngle(rotAxis, angleDelta);
    
    const startRot = Quat.fromEuler(
      this._dragStartTransform.rotation.x,
      this._dragStartTransform.rotation.y,
      this._dragStartTransform.rotation.z
    );
    
    const newRot = deltaQuat.multiply(startRot);
    
    const euler = newRot.toEuler();
    this._target.setRotation(euler.x, euler.y, euler.z);
  }

  private _applyFacingRotation(point: Vec3): void {
    if (!this._target || !this._dragStartTransform) return;
    
    const gizmoPos = this._rootStartPos;
    
    const startVec = this._selectionStartPoint.subtract(gizmoPos);
    const currentVec = point.subtract(gizmoPos);
    
    // 旋转轴是面向相机的方向
    const rotAxis = this.getFacingDir();
    
    if (startVec.lengthSquared() < 1e-6 || currentVec.lengthSquared() < 1e-6) {
      return;
    }
    
    const startNorm = startVec.normalize();
    const currentNorm = currentVec.normalize();
    
    const cosAngle = Math.max(-1, Math.min(1, startNorm.dot(currentNorm)));
    const crossVec = startNorm.cross(currentNorm);
    const sinAngle = crossVec.dot(rotAxis);
    let angleDelta = Math.atan2(sinAngle, cosAngle);
    
    if (this.snap) {
      const snapAngle = this.snapIncrement * Math.PI / 180;
      angleDelta = Math.round(angleDelta / snapAngle) * snapAngle;
    }
    
    const deltaQuat = Quat.fromAxisAngle(rotAxis, angleDelta);
    
    const startRot = Quat.fromEuler(
      this._dragStartTransform.rotation.x,
      this._dragStartTransform.rotation.y,
      this._dragStartTransform.rotation.z
    );
    
    const newRot = deltaQuat.multiply(startRot);
    
    const euler = newRot.toEuler();
    this._target.setRotation(euler.x, euler.y, euler.z);
  }
  
  private _applyScale(delta: Vec3): void {
    if (!this._target || !this._dragStartTransform) return;
    
    const axis = this._selectedAxis;
    
    let scaleFactor = 1.0;
    if (axis === 'x') scaleFactor = 1.0 + delta.x;
    else if (axis === 'y') scaleFactor = 1.0 + delta.y;
    else if (axis === 'z') scaleFactor = 1.0 + delta.z;
    else if (axis === 'xyz') scaleFactor = 1.0 + (delta.x + delta.y + delta.z) / 3;
    
    scaleFactor = Math.max(0.001, scaleFactor);
    
    if (this.snap) {
      scaleFactor = Math.round(scaleFactor / this.snapIncrement) * this.snapIncrement;
      scaleFactor = Math.max(0.001, scaleFactor);
    }
    
    const newScale = this._dragStartTransform.scale.clone();
    if (axis === 'x') newScale.x *= scaleFactor;
    else if (axis === 'y') newScale.y *= scaleFactor;
    else if (axis === 'z') newScale.z *= scaleFactor;
    else if (axis === 'xyz') {
      newScale.x *= scaleFactor;
      newScale.y *= scaleFactor;
      newScale.z *= scaleFactor;
    }
    
    this._target.setScale(newScale.x, newScale.y, newScale.z);
  }

  // ==================== 渲染 ====================
  
  private getShapeModelMatrix(shape: Shape): Mat4 {
    const gizmoPos = this.getGizmoPosition()!;
    const gizmoRot = this.getGizmoRotation();
    const scale = new Vec3(this._scale, this._scale, this._scale);
    
    const gizmoMatrix = Mat4.compose(gizmoPos, gizmoRot, scale);
    const shapeLocalMatrix = shape.getLocalTransform();
    
    return gizmoMatrix.multiply(shapeLocalMatrix);
  }
  
  private updateUniformsForShape(axis: GizmoAxisType | 'f', shape: Shape): void {
    const uniformBuffer = this.shapeUniformBuffers.get(axis);
    if (!uniformBuffer) return;
    
    const device = this.renderer.device;
    
    const modelMatrix = this.getShapeModelMatrix(shape);
    
    const viewProjectionMatrix = new Mat4();
    viewProjectionMatrix.elements.set(this.camera.viewProjectionMatrix);
    
    // 获取当前颜色（包含透明度）
    const color = shape.getColor();
    
    // uniform 数据：viewProj(64) + model(64) + color(16) = 144 bytes
    // 但需要 16 字节对齐，所以用 36 个 float
    const uniformData = new Float32Array(36);
    uniformData.set(viewProjectionMatrix.elements, 0);
    uniformData.set(modelMatrix.elements, 16);
    uniformData[32] = color.r;
    uniformData[33] = color.g;
    uniformData[34] = color.b;
    uniformData[35] = color.a;
    
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
  }
  
  render(pass: GPURenderPassEncoder): void {
    if (!this._target) return;
    if (!this.pipeline) return;
    
    this.updateScale();
    this._shapesLookAtCamera();
    
    pass.setPipeline(this.pipeline);
    
    for (const [axis, shape] of this._shapes) {
      if (!shape.visible) continue;
      this.updateUniformsForShape(axis, shape);
    }
    
    for (const [axis, shape] of this._shapes) {
      if (!shape.visible) continue;
      
      const vertexBuffer = shape.getVertexBuffer();
      const indexBuffer = shape.getIndexBuffer();
      const indexCount = shape.getIndexCount();
      const bindGroup = this.shapeBindGroups.get(axis);
      
      if (!vertexBuffer || !indexBuffer || indexCount === 0 || !bindGroup) {
        continue;
      }
      
      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.setIndexBuffer(indexBuffer, "uint16");
      pass.drawIndexed(indexCount);
    }
  }

  destroy(): void {
    for (const shape of this._shapes.values()) {
      shape.destroy();
    }
    this._shapes.clear();
    
    for (const buffer of this.shapeUniformBuffers.values()) {
      buffer.destroy();
    }
    this.shapeUniformBuffers.clear();
    this.shapeBindGroups.clear();
    
    if (this.uniformBuffer) {
      this.uniformBuffer.destroy();
      this.uniformBuffer = null;
    }
    
    if (this.guideLineBuffer) {
      this.guideLineBuffer.destroy();
      this.guideLineBuffer = null;
    }
    
    this.pipeline = null;
    this.linePipeline = null;
    this.bindGroup = null;
    this.guideLineBindGroup = null;
    this.bindGroupLayout = null;
  }
}
