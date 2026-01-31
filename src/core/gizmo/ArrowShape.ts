import { Vec3 } from "../math/Vec3";
import { Mat4 } from "../math/Mat4";
import { Quat } from "../math/Quat";
import { Shape, ShapeConfig, TriData } from "./Shape";

/**
 * ArrowShapeConfig - 箭头形状配置
 */
export interface ArrowShapeConfig extends ShapeConfig {
  gap?: number;           // 箭头起点与中心的间距
  lineThickness?: number; // 线条粗细
  lineLength?: number;    // 线条长度
  arrowThickness?: number;// 箭头粗细
  arrowLength?: number;   // 箭头长度
  tolerance?: number;     // 碰撞检测容差
}

/**
 * ArrowShape - 箭头形状（用于平移 Gizmo）
 * 参考 PlayCanvas 引擎的 ArrowShape 实现
 */
export class ArrowShape extends Shape {
  private _gap: number = 0;
  private _lineThickness: number = 0.02;
  private _lineLength: number = 0.5;
  private _arrowThickness: number = 0.12;
  private _arrowLength: number = 0.18;
  private _tolerance: number = 0.1;
  
  constructor(config: ArrowShapeConfig) {
    super(config);
    
    this._gap = config.gap ?? this._gap;
    this._lineThickness = config.lineThickness ?? this._lineThickness;
    this._lineLength = config.lineLength ?? this._lineLength;
    this._arrowThickness = config.arrowThickness ?? this._arrowThickness;
    this._arrowLength = config.arrowLength ?? this._arrowLength;
    this._tolerance = config.tolerance ?? this._tolerance;
    
    this._updateTriData();
  }
  
  get gap(): number { return this._gap; }
  set gap(value: number) { this._gap = value; this._updateTriData(); }
  
  get lineThickness(): number { return this._lineThickness; }
  set lineThickness(value: number) { this._lineThickness = value; this._updateTriData(); }
  
  get lineLength(): number { return this._lineLength; }
  set lineLength(value: number) { this._lineLength = value; this._updateTriData(); }
  
  get arrowThickness(): number { return this._arrowThickness; }
  set arrowThickness(value: number) { this._arrowThickness = value; this._updateTriData(); }
  
  get arrowLength(): number { return this._arrowLength; }
  set arrowLength(value: number) { this._arrowLength = value; this._updateTriData(); }
  
  /**
   * 更新碰撞数据
   */
  private _updateTriData(): void {
    // 箭头锥体碰撞数据
    const conePos = new Vec3(0, this._gap + this._arrowLength * 0.5 + this._lineLength, 0);
    const coneScale = new Vec3(this._arrowThickness, this._arrowLength, this._arrowThickness);
    const coneTransform = Mat4.compose(conePos, Quat.identity(), coneScale);
    
    // 线条圆柱碰撞数据（加上容差）
    const linePos = new Vec3(0, this._gap + this._lineLength * 0.5, 0);
    const lineScale = new Vec3(
      this._lineThickness + this._tolerance,
      this._lineLength,
      this._lineThickness + this._tolerance
    );
    const lineTransform = Mat4.compose(linePos, Quat.identity(), lineScale);
    
    // 生成单位锥体和圆柱的三角形数据
    const coneGeo = this.createConeGeometry();
    const cylinderGeo = this.createCylinderGeometry();
    
    this.triData = [
      {
        vertices: coneGeo.vertices,
        indices: coneGeo.indices,
        transform: coneTransform,
        priority: 0
      },
      {
        vertices: cylinderGeo.vertices,
        indices: cylinderGeo.indices,
        transform: lineTransform,
        priority: 1  // 线条优先级更高，更容易选中
      }
    ];
  }
  
  /**
   * 创建单位锥体几何体（底面半径1，高度1，底面在y=0，顶点在y=1）
   */
  private createConeGeometry(): { vertices: Float32Array; indices: Uint16Array } {
    const segments = 12;
    const vertices: number[] = [];
    const indices: number[] = [];
    
    // 底面中心
    vertices.push(0, -0.5, 0);
    
    // 底面顶点
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      vertices.push(Math.cos(angle) * 0.5, -0.5, Math.sin(angle) * 0.5);
    }
    
    // 顶点
    const tipIndex = vertices.length / 3;
    vertices.push(0, 0.5, 0);
    
    // 底面三角形
    for (let i = 1; i <= segments; i++) {
      indices.push(0, i + 1, i);
    }
    
    // 侧面三角形
    for (let i = 1; i <= segments; i++) {
      indices.push(i, i + 1, tipIndex);
    }
    
    return {
      vertices: new Float32Array(vertices),
      indices: new Uint16Array(indices)
    };
  }
  
  /**
   * 创建单位圆柱几何体（半径1，高度1，中心在原点）
   */
  private createCylinderGeometry(): { vertices: Float32Array; indices: Uint16Array } {
    const segments = 12;
    const vertices: number[] = [];
    const indices: number[] = [];
    
    // 底面和顶面顶点
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const x = Math.cos(angle) * 0.5;
      const z = Math.sin(angle) * 0.5;
      
      // 底面
      vertices.push(x, -0.5, z);
      // 顶面
      vertices.push(x, 0.5, z);
    }
    
    // 侧面三角形
    for (let i = 0; i < segments; i++) {
      const base = i * 2;
      indices.push(base, base + 1, base + 2);
      indices.push(base + 1, base + 3, base + 2);
    }
    
    return {
      vertices: new Float32Array(vertices),
      indices: new Uint16Array(indices)
    };
  }
  
  /**
   * 创建渲染用的 GPU 几何体
   */
  createGeometry(device: GPUDevice): void {
    const segments = 12;
    const vertices: number[] = [];
    const indices: number[] = [];
    
    const color = this._defaultColor;
    
    // 辅助函数：添加顶点
    const addVertex = (x: number, y: number, z: number) => {
      vertices.push(x, y, z, color.x, color.y, color.z);
    };
    
    // 计算垂直向量
    let perpX: Vec3, perpY: Vec3;
    const dir = new Vec3(0, 1, 0);  // 默认沿 Y 轴
    perpX = new Vec3(1, 0, 0);
    perpY = new Vec3(0, 0, 1);
    
    // 生成圆柱（线条部分）
    const cylinderStart = this._gap;
    const cylinderEnd = this._gap + this._lineLength;
    
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      
      const offsetX = cos * this._lineThickness;
      const offsetZ = sin * this._lineThickness;
      
      // 底部顶点
      addVertex(offsetX, cylinderStart, offsetZ);
      // 顶部顶点
      addVertex(offsetX, cylinderEnd, offsetZ);
    }
    
    // 圆柱索引
    for (let i = 0; i < segments; i++) {
      const base = i * 2;
      indices.push(base, base + 1, base + 2);
      indices.push(base + 1, base + 3, base + 2);
    }
    
    // 生成锥体（箭头部分）
    const coneBase = this._gap + this._lineLength;
    const coneTip = coneBase + this._arrowLength;
    
    const coneBaseIndex = vertices.length / 6;
    
    // 锥体底面顶点
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      
      const offsetX = cos * this._arrowThickness;
      const offsetZ = sin * this._arrowThickness;
      
      addVertex(offsetX, coneBase, offsetZ);
    }
    
    // 锥体顶点
    const coneTipIndex = vertices.length / 6;
    addVertex(0, coneTip, 0);
    
    // 锥体索引
    for (let i = 0; i < segments; i++) {
      indices.push(coneBaseIndex + i, coneTipIndex, coneBaseIndex + i + 1);
    }
    
    this.createBuffers(device, new Float32Array(vertices), new Uint16Array(indices));
  }
}
