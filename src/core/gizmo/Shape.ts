import { Vec3 } from "../math/Vec3";
import { Ray } from "../math/Ray";
import { Mat4 } from "../math/Mat4";
import { Quat } from "../math/Quat";

/**
 * GizmoAxis - 轴类型枚举
 */
export type GizmoAxisType = 'x' | 'y' | 'z' | 'xy' | 'xz' | 'yz' | 'xyz';

/**
 * Vec4Color - 带透明度的颜色
 */
export interface Vec4Color {
  r: number;
  g: number;
  b: number;
  a: number;
}

/**
 * ShapeConfig - Shape 配置
 */
export interface ShapeConfig {
  axis: GizmoAxisType;
  defaultColor: Vec3;
  hoverColor: Vec3;
  disabledColor: Vec3;
  defaultAlpha?: number;  // 默认透明度
  hoverAlpha?: number;    // hover 透明度
  rotation?: Vec3;        // 欧拉角旋转
}

/**
 * TriData - 三角形碰撞数据
 */
export interface TriData {
  vertices: Float32Array;  // 三角形顶点
  indices: Uint16Array;    // 索引
  transform: Mat4;         // 局部变换矩阵
  priority: number;        // 碰撞优先级（越高越优先）
}

/**
 * Shape - Gizmo 形状基类
 * 参考 PlayCanvas 引擎的 Shape 实现
 */
export abstract class Shape {
  axis: GizmoAxisType;
  
  // 状态
  protected _disabled: boolean = false;
  protected _visible: boolean = true;
  protected _hovered: boolean = false;
  protected _interactable: boolean = true;  // 是否可交互（碰撞检测）
  
  // 颜色
  protected _defaultColor: Vec3;
  protected _hoverColor: Vec3;
  protected _disabledColor: Vec3;
  protected _defaultAlpha: number = 1.0;
  protected _hoverAlpha: number = 1.0;
  
  // 变换
  protected _position: Vec3 = new Vec3(0, 0, 0);
  protected _rotation: Vec3 = new Vec3(0, 0, 0);  // 欧拉角
  protected _scale: Vec3 = new Vec3(1, 1, 1);
  
  // 碰撞数据
  triData: TriData[] = [];
  
  // GPU 资源
  protected device: GPUDevice | null = null;
  protected vertexBuffer: GPUBuffer | null = null;
  protected indexBuffer: GPUBuffer | null = null;
  protected vertexCount: number = 0;
  protected indexCount: number = 0;
  
  // 翻转状态（用于平面）
  flipped: Vec3 = new Vec3(0, 0, 0);
  
  constructor(config: ShapeConfig) {
    this.axis = config.axis;
    this._defaultColor = config.defaultColor.clone();
    this._hoverColor = config.hoverColor.clone();
    this._disabledColor = config.disabledColor.clone();
    this._defaultAlpha = config.defaultAlpha ?? 1.0;
    this._hoverAlpha = config.hoverAlpha ?? 1.0;
    
    if (config.rotation) {
      this._rotation = config.rotation.clone();
    }
  }
  
  get disabled(): boolean {
    return this._disabled;
  }
  
  set disabled(value: boolean) {
    this._disabled = value;
    if (value) {
      this._hovered = false;
    }
  }
  
  get visible(): boolean {
    return this._visible;
  }
  
  set visible(value: boolean) {
    this._visible = value;
    // 默认情况下，visible 也控制 interactable
    // 但可以单独设置 interactable 来覆盖
  }
  
  get interactable(): boolean {
    return this._interactable;
  }
  
  set interactable(value: boolean) {
    this._interactable = value;
  }
  
  /**
   * 设置 hover 状态
   */
  hover(state: boolean): void {
    if (this._disabled) {
      this._hovered = false;
      return;
    }
    this._hovered = state;
  }
  
  /**
   * 获取当前颜色（包含透明度）
   */
  getColor(): Vec4Color {
    if (this._disabled) {
      return { r: this._disabledColor.x, g: this._disabledColor.y, b: this._disabledColor.z, a: 1.0 };
    }
    if (this._hovered) {
      return { r: this._hoverColor.x, g: this._hoverColor.y, b: this._hoverColor.z, a: this._hoverAlpha };
    }
    return { r: this._defaultColor.x, g: this._defaultColor.y, b: this._defaultColor.z, a: this._defaultAlpha };
  }
  
  /**
   * 获取局部变换矩阵
   */
  getLocalTransform(): Mat4 {
    // 将欧拉角（度）转换为弧度
    const rotX = this._rotation.x * Math.PI / 180;
    const rotY = this._rotation.y * Math.PI / 180;
    const rotZ = this._rotation.z * Math.PI / 180;
    
    const rotation = Quat.fromEuler(rotX, rotY, rotZ);
    
    // Debug: 输出旋转信息
    // console.log(`Shape ${this.axis} rotation (deg):`, this._rotation);
    // console.log(`Shape ${this.axis} rotation (rad):`, rotX, rotY, rotZ);
    // console.log(`Shape ${this.axis} quaternion:`, rotation);
    
    return Mat4.compose(this._position, rotation, this._scale);
  }
  
  /**
   * 射线碰撞检测
   * @param ray - 世界空间射线
   * @param parentTransform - 父级变换矩阵（gizmo 的世界变换）
   * @returns 碰撞距离，null 表示未碰撞
   */
  intersect(ray: Ray, parentTransform: Mat4): number | null {
    // disabled 或 不可交互时跳过碰撞检测
    if (this._disabled || !this._interactable) {
      return null;
    }
    
    let closestDist: number | null = null;
    let highestPriority = -Infinity;
    
    for (const tri of this.triData) {
      // 计算完整变换：parent * local * triTransform
      const localTransform = this.getLocalTransform();
      const worldTransform = parentTransform.multiply(localTransform).multiply(tri.transform);
      const invTransform = worldTransform.invert();
      
      // 将射线变换到局部空间
      const localRay = ray.transform(invTransform);
      
      // 与三角形进行碰撞检测
      const dist = this.intersectTriangles(localRay, tri.vertices, tri.indices);
      
      if (dist !== null) {
        // 考虑优先级
        if (tri.priority > highestPriority || 
            (tri.priority === highestPriority && (closestDist === null || dist < closestDist))) {
          closestDist = dist;
          highestPriority = tri.priority;
        }
      }
    }
    
    return closestDist;
  }
  
  /**
   * 射线与三角形列表碰撞检测
   */
  protected intersectTriangles(
    ray: Ray, 
    vertices: Float32Array, 
    indices: Uint16Array
  ): number | null {
    let closestDist: number | null = null;
    
    for (let i = 0; i < indices.length; i += 3) {
      const i0 = indices[i] * 3;
      const i1 = indices[i + 1] * 3;
      const i2 = indices[i + 2] * 3;
      
      const v0 = new Vec3(vertices[i0], vertices[i0 + 1], vertices[i0 + 2]);
      const v1 = new Vec3(vertices[i1], vertices[i1 + 1], vertices[i1 + 2]);
      const v2 = new Vec3(vertices[i2], vertices[i2 + 1], vertices[i2 + 2]);
      
      const dist = ray.intersectTriangle(v0, v1, v2);
      
      if (dist !== null && dist > 0) {
        if (closestDist === null || dist < closestDist) {
          closestDist = dist;
        }
      }
    }
    
    return closestDist;
  }
  
  /**
   * 创建 GPU 资源
   */
  abstract createGeometry(device: GPUDevice): void;
  
  /**
   * 获取顶点缓冲区
   */
  getVertexBuffer(): GPUBuffer | null {
    return this.vertexBuffer;
  }
  
  /**
   * 获取索引缓冲区
   */
  getIndexBuffer(): GPUBuffer | null {
    return this.indexBuffer;
  }
  
  /**
   * 获取索引数量
   */
  getIndexCount(): number {
    return this.indexCount;
  }
  
  /**
   * 销毁 GPU 资源
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
  }
  
  /**
   * 创建 GPU 缓冲区
   */
  protected createBuffers(
    device: GPUDevice,
    vertices: Float32Array,
    indices: Uint16Array
  ): void {
    this.device = device;
    this.vertexCount = vertices.length / 6;
    this.indexCount = indices.length;
    
    // 创建顶点缓冲区
    this.vertexBuffer = device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();
    
    // 创建索引缓冲区
    this.indexBuffer = device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();
  }
}
