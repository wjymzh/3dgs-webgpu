import { Vec3 } from "../math/Vec3";
import { Mat4 } from "../math/Mat4";
import { Quat } from "../math/Quat";
import { Shape, ShapeConfig, TriData } from "./Shape";

/**
 * SphereShapeConfig - 球体形状配置
 */
export interface SphereShapeConfig extends ShapeConfig {
  radius?: number;  // 球体半径
}

/**
 * SphereShape - 球体形状（用于统一缩放/中心选择）
 * 参考 PlayCanvas 引擎的 SphereShape 实现
 */
export class SphereShape extends Shape {
  private _radius: number = 0.1;
  
  constructor(config: SphereShapeConfig) {
    super(config);
    
    this._radius = config.radius ?? this._radius;
    
    this._updateTriData();
  }
  
  get radius(): number { return this._radius; }
  set radius(value: number) { this._radius = value; this._updateTriData(); }
  
  /**
   * 更新碰撞数据
   */
  private _updateTriData(): void {
    const sphereGeo = this.createSphereGeometry(8, 6);
    
    const scale = new Vec3(this._radius * 2, this._radius * 2, this._radius * 2);
    const transform = Mat4.compose(new Vec3(0, 0, 0), Quat.identity(), scale);
    
    this.triData = [{
      vertices: sphereGeo.vertices,
      indices: sphereGeo.indices,
      transform: transform,
      priority: 3  // 中心球优先级最高
    }];
  }
  
  /**
   * 创建单位球体几何体（半径0.5，中心在原点）
   */
  private createSphereGeometry(
    segments: number, 
    rings: number
  ): { vertices: Float32Array; indices: Uint16Array } {
    const vertices: number[] = [];
    const indices: number[] = [];
    
    // 生成球体顶点
    for (let ring = 0; ring <= rings; ring++) {
      const phi = (ring / rings) * Math.PI;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);
      
      for (let seg = 0; seg <= segments; seg++) {
        const theta = (seg / segments) * Math.PI * 2;
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);
        
        const x = 0.5 * sinPhi * cosTheta;
        const y = 0.5 * cosPhi;
        const z = 0.5 * sinPhi * sinTheta;
        
        vertices.push(x, y, z);
      }
    }
    
    // 生成球体索引
    for (let ring = 0; ring < rings; ring++) {
      for (let seg = 0; seg < segments; seg++) {
        const a = ring * (segments + 1) + seg;
        const b = a + segments + 1;
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
   * 创建渲染用的 GPU 几何体
   */
  createGeometry(device: GPUDevice): void {
    const segments = 16;
    const rings = 12;
    const vertices: number[] = [];
    const indices: number[] = [];
    
    const color = this._defaultColor;
    
    // 生成球体顶点
    for (let ring = 0; ring <= rings; ring++) {
      const phi = (ring / rings) * Math.PI;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);
      
      for (let seg = 0; seg <= segments; seg++) {
        const theta = (seg / segments) * Math.PI * 2;
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);
        
        const x = this._radius * sinPhi * cosTheta;
        const y = this._radius * cosPhi;
        const z = this._radius * sinPhi * sinTheta;
        
        vertices.push(x, y, z, color.x, color.y, color.z);
      }
    }
    
    // 生成球体索引
    for (let ring = 0; ring < rings; ring++) {
      for (let seg = 0; seg < segments; seg++) {
        const a = ring * (segments + 1) + seg;
        const b = a + segments + 1;
        const c = a + 1;
        const d = b + 1;
        
        indices.push(a, b, c);
        indices.push(b, d, c);
      }
    }
    
    this.createBuffers(device, new Float32Array(vertices), new Uint16Array(indices));
  }
}
