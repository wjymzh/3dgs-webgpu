import { Vec3 } from "../math/Vec3";
import { Mat4 } from "../math/Mat4";
import { Quat } from "../math/Quat";
import { Shape, ShapeConfig } from "./Shape";

/**
 * ArcShapeConfig - 圆弧形状配置
 */
export interface ArcShapeConfig extends ShapeConfig {
  tubeRadius?: number;   // 管道半径（粗细）
  ringRadius?: number;   // 圆环半径
  sectorAngle?: number;  // 扇形角度（度）
  tolerance?: number;    // 碰撞检测容差
}

/**
 * ArcDisplayMode - 圆弧显示模式
 */
export type ArcDisplayMode = 'sector' | 'ring' | 'none';

/**
 * ArcShape - 圆弧形状（用于旋转 Gizmo）
 * 参考 PlayCanvas 引擎的 ArcShape 实现
 * 
 * 圆环默认在 XZ 平面上（绕 Y 轴），通过 rotation 旋转到正确的轴
 */
export class ArcShape extends Shape {
  private _tubeRadius: number = 0.02;
  private _ringRadius: number = 0.5;
  private _sectorAngle: number = 180;  // 默认半圆弧
  private _tolerance: number = 0.05;
  
  // 显示模式
  private _displayMode: ArcDisplayMode = 'sector';
  
  // 双缓冲：sector 和 ring 的几何数据
  private _sectorVertexBuffer: GPUBuffer | null = null;
  private _sectorIndexBuffer: GPUBuffer | null = null;
  private _sectorIndexCount: number = 0;
  private _ringVertexBuffer: GPUBuffer | null = null;
  private _ringIndexBuffer: GPUBuffer | null = null;
  private _ringIndexCount: number = 0;
  
  // 动态旋转角度（用于面向相机）
  private _dynamicRotation: Vec3 = new Vec3(0, 0, 0);
  
  constructor(config: ArcShapeConfig) {
    super(config);
    
    this._tubeRadius = config.tubeRadius ?? this._tubeRadius;
    this._ringRadius = config.ringRadius ?? this._ringRadius;
    this._sectorAngle = config.sectorAngle ?? this._sectorAngle;
    this._tolerance = config.tolerance ?? this._tolerance;
    
    this._updateTriData();
  }
  
  get tubeRadius(): number { return this._tubeRadius; }
  set tubeRadius(value: number) { this._tubeRadius = value; this._updateTriData(); }
  
  get ringRadius(): number { return this._ringRadius; }
  set ringRadius(value: number) { this._ringRadius = value; this._updateTriData(); }
  
  get sectorAngle(): number { return this._sectorAngle; }
  set sectorAngle(value: number) { this._sectorAngle = value; this._updateTriData(); }
  
  get tolerance(): number { return this._tolerance; }
  set tolerance(value: number) { this._tolerance = value; this._updateTriData(); }
  
  get displayMode(): ArcDisplayMode { return this._displayMode; }
  
  /**
   * 设置动态旋转（用于面向相机）
   */
  setDynamicRotation(rotation: Vec3): void {
    this._dynamicRotation = rotation.clone();
  }
  
  /**
   * 获取局部变换矩阵（包含动态旋转）
   */
  override getLocalTransform(): Mat4 {
    // 基础旋转
    const baseRotX = this._rotation.x * Math.PI / 180;
    const baseRotY = this._rotation.y * Math.PI / 180;
    const baseRotZ = this._rotation.z * Math.PI / 180;
    const baseQuat = Quat.fromEuler(baseRotX, baseRotY, baseRotZ);
    
    // 动态旋转
    const dynRotX = this._dynamicRotation.x * Math.PI / 180;
    const dynRotY = this._dynamicRotation.y * Math.PI / 180;
    const dynRotZ = this._dynamicRotation.z * Math.PI / 180;
    const dynQuat = Quat.fromEuler(dynRotX, dynRotY, dynRotZ);
    
    // 组合旋转：先应用基础旋转，再应用动态旋转
    const finalQuat = baseQuat.multiply(dynQuat);
    
    return Mat4.compose(this._position, finalQuat, this._scale);
  }
  
  /**
   * 更新碰撞数据
   */
  private _updateTriData(): void {
    // 根据显示模式创建碰撞几何体
    const angle = this._displayMode === 'ring' ? 360 : this._sectorAngle;
    const torusGeo = this.createTorusGeometry(20, 8, this._tubeRadius + this._tolerance, angle);
    
    const transform = Mat4.identity();
    
    this.triData = [{
      vertices: torusGeo.vertices,
      indices: torusGeo.indices,
      transform: transform,
      priority: 0
    }];
  }
  
  /**
   * 切换显示模式
   */
  show(mode: ArcDisplayMode): void {
    if (this._displayMode === mode) return;
    
    this._displayMode = mode;
    this._visible = mode !== 'none';
    
    // 更新碰撞数据
    this._updateTriData();
  }
  
  /**
   * 创建圆环几何体
   */
  private createTorusGeometry(
    segments: number,
    tubeSegments: number,
    tubeRadius?: number,
    sectorAngle?: number
  ): { vertices: Float32Array; indices: Uint16Array } {
    const vertices: number[] = [];
    const indices: number[] = [];
    
    const tube = tubeRadius ?? this._tubeRadius;
    const ring = this._ringRadius;
    const angle = sectorAngle ?? this._sectorAngle;
    const sectorRad = (angle * Math.PI) / 180;
    
    for (let i = 0; i <= segments; i++) {
      const u = (i / segments) * sectorRad;
      const cosU = Math.cos(u);
      const sinU = Math.sin(u);
      
      for (let j = 0; j <= tubeSegments; j++) {
        const v = (j / tubeSegments) * Math.PI * 2;
        const cosV = Math.cos(v);
        const sinV = Math.sin(v);
        
        const x = (ring + tube * cosV) * cosU;
        const y = tube * sinV;
        const z = (ring + tube * cosV) * sinU;
        
        vertices.push(x, y, z);
      }
    }
    
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
      indices: new Uint16Array(indices)
    };
  }
  
  /**
   * 创建渲染用的 GPU 几何体（同时创建 sector 和 ring 两种）
   */
  createGeometry(device: GPUDevice): void {
    this.device = device;
    
    // 创建 sector 几何体
    const sectorData = this._createTorusMeshData(this._sectorAngle);
    this._sectorVertexBuffer = device.createBuffer({
      size: sectorData.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this._sectorVertexBuffer.getMappedRange()).set(sectorData.vertices);
    this._sectorVertexBuffer.unmap();
    
    this._sectorIndexBuffer = device.createBuffer({
      size: sectorData.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this._sectorIndexBuffer.getMappedRange()).set(sectorData.indices);
    this._sectorIndexBuffer.unmap();
    this._sectorIndexCount = sectorData.indices.length;
    
    // 创建 ring 几何体（完整圆环）
    const ringData = this._createTorusMeshData(360);
    this._ringVertexBuffer = device.createBuffer({
      size: ringData.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this._ringVertexBuffer.getMappedRange()).set(ringData.vertices);
    this._ringVertexBuffer.unmap();
    
    this._ringIndexBuffer = device.createBuffer({
      size: ringData.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this._ringIndexBuffer.getMappedRange()).set(ringData.indices);
    this._ringIndexBuffer.unmap();
    this._ringIndexCount = ringData.indices.length;
    
    // 默认使用 sector
    this.vertexBuffer = this._sectorVertexBuffer;
    this.indexBuffer = this._sectorIndexBuffer;
    this.indexCount = this._sectorIndexCount;
  }
  
  /**
   * 创建圆环网格数据
   */
  private _createTorusMeshData(sectorAngle: number): { vertices: Float32Array; indices: Uint16Array } {
    const segments = 64;
    const tubeSegments = 12;
    const vertices: number[] = [];
    const indices: number[] = [];
    
    const color = this._defaultColor;
    const tube = this._tubeRadius;
    const ring = this._ringRadius;
    const sectorRad = (sectorAngle * Math.PI) / 180;
    
    for (let i = 0; i <= segments; i++) {
      const u = (i / segments) * sectorRad;
      const cosU = Math.cos(u);
      const sinU = Math.sin(u);
      
      for (let j = 0; j <= tubeSegments; j++) {
        const v = (j / tubeSegments) * Math.PI * 2;
        const cosV = Math.cos(v);
        const sinV = Math.sin(v);
        
        const x = (ring + tube * cosV) * cosU;
        const y = tube * sinV;
        const z = (ring + tube * cosV) * sinU;
        
        vertices.push(x, y, z, color.x, color.y, color.z);
      }
    }
    
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
      indices: new Uint16Array(indices)
    };
  }
  
  /**
   * 获取顶点缓冲区（根据显示模式）
   */
  override getVertexBuffer(): GPUBuffer | null {
    if (this._displayMode === 'ring') {
      return this._ringVertexBuffer;
    }
    return this._sectorVertexBuffer;
  }
  
  /**
   * 获取索引缓冲区（根据显示模式）
   */
  override getIndexBuffer(): GPUBuffer | null {
    if (this._displayMode === 'ring') {
      return this._ringIndexBuffer;
    }
    return this._sectorIndexBuffer;
  }
  
  /**
   * 获取索引数量（根据显示模式）
   */
  override getIndexCount(): number {
    if (this._displayMode === 'ring') {
      return this._ringIndexCount;
    }
    return this._sectorIndexCount;
  }
  
  /**
   * 销毁 GPU 资源
   */
  override destroy(): void {
    if (this._sectorVertexBuffer) {
      this._sectorVertexBuffer.destroy();
      this._sectorVertexBuffer = null;
    }
    if (this._sectorIndexBuffer) {
      this._sectorIndexBuffer.destroy();
      this._sectorIndexBuffer = null;
    }
    if (this._ringVertexBuffer) {
      this._ringVertexBuffer.destroy();
      this._ringVertexBuffer = null;
    }
    if (this._ringIndexBuffer) {
      this._ringIndexBuffer.destroy();
      this._ringIndexBuffer = null;
    }
    super.destroy();
  }
}
