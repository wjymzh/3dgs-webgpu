import { Vec3 } from "../math/Vec3";
import { Mat4 } from "../math/Mat4";
import { Quat } from "../math/Quat";
import { Shape, ShapeConfig, TriData } from "./Shape";

/**
 * BoxLineShapeConfig - 方块线形状配置（用于缩放 Gizmo）
 */
export interface BoxLineShapeConfig extends ShapeConfig {
  gap?: number;           // 起点与中心的间距
  lineThickness?: number; // 线条粗细
  lineLength?: number;    // 线条长度
  boxSize?: number;       // 末端方块大小
  tolerance?: number;     // 碰撞检测容差
}

/**
 * BoxLineShape - 方块线形状（用于缩放 Gizmo）
 * 参考 PlayCanvas 引擎的 BoxLineShape 实现
 */
export class BoxLineShape extends Shape {
  private _gap: number = 0;
  private _lineThickness: number = 0.02;
  private _lineLength: number = 0.5;
  private _boxSize: number = 0.1;
  private _tolerance: number = 0.1;
  
  constructor(config: BoxLineShapeConfig) {
    super(config);
    
    this._gap = config.gap ?? this._gap;
    this._lineThickness = config.lineThickness ?? this._lineThickness;
    this._lineLength = config.lineLength ?? this._lineLength;
    this._boxSize = config.boxSize ?? this._boxSize;
    this._tolerance = config.tolerance ?? this._tolerance;
    
    this._updateTriData();
  }
  
  get gap(): number { return this._gap; }
  set gap(value: number) { this._gap = value; this._updateTriData(); }
  
  get lineThickness(): number { return this._lineThickness; }
  set lineThickness(value: number) { this._lineThickness = value; this._updateTriData(); }
  
  get lineLength(): number { return this._lineLength; }
  set lineLength(value: number) { this._lineLength = value; this._updateTriData(); }
  
  get boxSize(): number { return this._boxSize; }
  set boxSize(value: number) { this._boxSize = value; this._updateTriData(); }
  
  /**
   * 更新碰撞数据
   */
  private _updateTriData(): void {
    // 方块碰撞数据
    const boxPos = new Vec3(0, this._gap + this._lineLength + this._boxSize * 0.5, 0);
    const boxScale = new Vec3(this._boxSize, this._boxSize, this._boxSize);
    const boxTransform = Mat4.compose(boxPos, Quat.identity(), boxScale);
    
    // 线条圆柱碰撞数据（加上容差）
    const linePos = new Vec3(0, this._gap + this._lineLength * 0.5, 0);
    const lineScale = new Vec3(
      this._lineThickness + this._tolerance,
      this._lineLength,
      this._lineThickness + this._tolerance
    );
    const lineTransform = Mat4.compose(linePos, Quat.identity(), lineScale);
    
    // 生成单位方块和圆柱的三角形数据
    const boxGeo = this.createBoxGeometry();
    const cylinderGeo = this.createCylinderGeometry();
    
    this.triData = [
      {
        vertices: boxGeo.vertices,
        indices: boxGeo.indices,
        transform: boxTransform,
        priority: 0
      },
      {
        vertices: cylinderGeo.vertices,
        indices: cylinderGeo.indices,
        transform: lineTransform,
        priority: 1
      }
    ];
  }
  
  /**
   * 创建单位方块几何体（边长1，中心在原点）
   */
  private createBoxGeometry(): { vertices: Float32Array; indices: Uint16Array } {
    const h = 0.5;
    const vertices = new Float32Array([
      // 前面
      -h, -h,  h,
       h, -h,  h,
       h,  h,  h,
      -h,  h,  h,
      // 后面
      -h, -h, -h,
      -h,  h, -h,
       h,  h, -h,
       h, -h, -h,
    ]);
    
    const indices = new Uint16Array([
      // 前
      0, 1, 2, 0, 2, 3,
      // 后
      4, 5, 6, 4, 6, 7,
      // 上
      3, 2, 6, 3, 6, 5,
      // 下
      4, 7, 1, 4, 1, 0,
      // 右
      1, 7, 6, 1, 6, 2,
      // 左
      4, 0, 3, 4, 3, 5
    ]);
    
    return { vertices, indices };
  }
  
  /**
   * 创建单位圆柱几何体
   */
  private createCylinderGeometry(): { vertices: Float32Array; indices: Uint16Array } {
    const segments = 12;
    const vertices: number[] = [];
    const indices: number[] = [];
    
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const x = Math.cos(angle) * 0.5;
      const z = Math.sin(angle) * 0.5;
      
      vertices.push(x, -0.5, z);
      vertices.push(x, 0.5, z);
    }
    
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
    
    // 生成圆柱（线条部分）
    const cylinderStart = this._gap;
    const cylinderEnd = this._gap + this._lineLength;
    
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      
      const offsetX = cos * this._lineThickness;
      const offsetZ = sin * this._lineThickness;
      
      addVertex(offsetX, cylinderStart, offsetZ);
      addVertex(offsetX, cylinderEnd, offsetZ);
    }
    
    // 圆柱索引
    for (let i = 0; i < segments; i++) {
      const base = i * 2;
      indices.push(base, base + 1, base + 2);
      indices.push(base + 1, base + 3, base + 2);
    }
    
    // 生成方块
    const boxCenter = this._gap + this._lineLength + this._boxSize * 0.5;
    const h = this._boxSize * 0.5;
    
    const boxBaseIndex = vertices.length / 6;
    
    // 方块8个顶点
    addVertex(-h, boxCenter - h,  h);
    addVertex( h, boxCenter - h,  h);
    addVertex( h, boxCenter + h,  h);
    addVertex(-h, boxCenter + h,  h);
    addVertex(-h, boxCenter - h, -h);
    addVertex(-h, boxCenter + h, -h);
    addVertex( h, boxCenter + h, -h);
    addVertex( h, boxCenter - h, -h);
    
    // 方块索引
    const boxIndices = [
      0, 1, 2, 0, 2, 3,  // 前
      4, 5, 6, 4, 6, 7,  // 后
      3, 2, 6, 3, 6, 5,  // 上
      4, 7, 1, 4, 1, 0,  // 下
      1, 7, 6, 1, 6, 2,  // 右
      4, 0, 3, 4, 3, 5   // 左
    ];
    
    for (const idx of boxIndices) {
      indices.push(boxBaseIndex + idx);
    }
    
    this.createBuffers(device, new Float32Array(vertices), new Uint16Array(indices));
  }
}
