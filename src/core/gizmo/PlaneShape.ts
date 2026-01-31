import { Vec3 } from "../math/Vec3";
import { Mat4 } from "../math/Mat4";
import { Quat } from "../math/Quat";
import { Shape, ShapeConfig } from "./Shape";

/**
 * PlaneShapeConfig - 平面形状配置
 */
export interface PlaneShapeConfig extends ShapeConfig {
  size?: number;  // 平面大小
  gap?: number;   // 与中心的间距
}

/**
 * PlaneShape - 平面形状（用于双轴平移）
 * 参考 PlayCanvas 引擎的 PlaneShape 实现
 */
export class PlaneShape extends Shape {
  private _size: number = 0.2;
  private _gap: number = 0.1;
  
  // 缓存不同翻转状态的几何数据
  private _geometryCache: Map<string, { vertexBuffer: GPUBuffer; indexBuffer: GPUBuffer }> = new Map();
  private _currentFlipKey: string = '0,0,0';
  
  constructor(config: PlaneShapeConfig) {
    super(config);
    
    this._size = config.size ?? this._size;
    this._gap = config.gap ?? this._gap;
    
    this._updateTriData();
  }
  
  get size(): number { return this._size; }
  set size(value: number) { 
    this._size = value; 
    this._updateTriData(); 
    this._clearGeometryCache();
  }
  
  get gap(): number { return this._gap; }
  set gap(value: number) { 
    this._gap = value; 
    this._updateTriData(); 
    this._clearGeometryCache();
  }
  
  /**
   * 更新碰撞数据
   */
  private _updateTriData(): void {
    const planeGeo = this.createPlaneGeometry();
    
    // 平面位置（考虑翻转）
    const pos = new Vec3(
      (this._gap + this._size * 0.5) * (1 - this.flipped.x * 2),
      0,
      (this._gap + this._size * 0.5) * (1 - this.flipped.z * 2)
    );
    
    const scale = new Vec3(this._size, 1, this._size);
    const transform = Mat4.compose(pos, Quat.identity(), scale);
    
    this.triData = [{
      vertices: planeGeo.vertices,
      indices: planeGeo.indices,
      transform: transform,
      priority: 2
    }];
  }
  
  /**
   * 创建单位平面几何体
   */
  private createPlaneGeometry(): { vertices: Float32Array; indices: Uint16Array } {
    const vertices = new Float32Array([
      -0.5, 0, -0.5,
       0.5, 0, -0.5,
       0.5, 0,  0.5,
      -0.5, 0,  0.5
    ]);
    
    const indices = new Uint16Array([
      0, 1, 2,
      0, 2, 3,
      0, 2, 1,
      0, 3, 2
    ]);
    
    return { vertices, indices };
  }
  
  /**
   * 创建渲染用的 GPU 几何体
   */
  createGeometry(device: GPUDevice): void {
    this.device = device;
    this._currentFlipKey = `${this.flipped.x},${this.flipped.y},${this.flipped.z}`;
    
    // 检查缓存
    if (this._geometryCache.has(this._currentFlipKey)) {
      const cached = this._geometryCache.get(this._currentFlipKey)!;
      this.vertexBuffer = cached.vertexBuffer;
      this.indexBuffer = cached.indexBuffer;
      this.indexCount = 12;
      return;
    }
    
    const color = this._defaultColor;
    
    // 计算平面位置（考虑翻转）
    const offsetX = (this._gap + this._size * 0.5) * (1 - this.flipped.x * 2);
    const offsetZ = (this._gap + this._size * 0.5) * (1 - this.flipped.z * 2);
    const halfSize = this._size * 0.5;
    
    const vertices = new Float32Array([
      offsetX - halfSize, 0, offsetZ - halfSize, color.x, color.y, color.z,
      offsetX + halfSize, 0, offsetZ - halfSize, color.x, color.y, color.z,
      offsetX + halfSize, 0, offsetZ + halfSize, color.x, color.y, color.z,
      offsetX - halfSize, 0, offsetZ + halfSize, color.x, color.y, color.z,
    ]);
    
    const indices = new Uint16Array([
      0, 1, 2,
      0, 2, 3,
      0, 2, 1,
      0, 3, 2
    ]);
    
    this.createBuffers(device, vertices, indices);
    
    // 缓存
    if (this.vertexBuffer && this.indexBuffer) {
      this._geometryCache.set(this._currentFlipKey, {
        vertexBuffer: this.vertexBuffer,
        indexBuffer: this.indexBuffer
      });
    }
  }
  
  /**
   * 清除几何缓存
   */
  private _clearGeometryCache(): void {
    for (const cached of this._geometryCache.values()) {
      cached.vertexBuffer.destroy();
      cached.indexBuffer.destroy();
    }
    this._geometryCache.clear();
    this.vertexBuffer = null;
    this.indexBuffer = null;
  }
  
  /**
   * 更新翻转状态
   */
  setFlipped(newFlipped: Vec3): void {
    const newKey = `${newFlipped.x},${newFlipped.y},${newFlipped.z}`;
    if (this._currentFlipKey === newKey) return;
    
    this.flipped = newFlipped.clone();
    this._updateTriData();
    
    if (this.device) {
      this._currentFlipKey = newKey;
      
      // 检查缓存
      if (this._geometryCache.has(newKey)) {
        const cached = this._geometryCache.get(newKey)!;
        this.vertexBuffer = cached.vertexBuffer;
        this.indexBuffer = cached.indexBuffer;
      } else {
        // 创建新的几何体
        this.createGeometry(this.device);
      }
    }
  }
  
  /**
   * 销毁 GPU 资源
   */
  override destroy(): void {
    this._clearGeometryCache();
    super.destroy();
  }
}
