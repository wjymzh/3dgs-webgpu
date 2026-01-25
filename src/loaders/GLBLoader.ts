import { Mesh, MeshBoundingBox } from '../mesh/Mesh';

/**
 * GLB 文件格式常量
 */
const GLB_MAGIC = 0x46546C67; // 'glTF'
const GLB_VERSION = 2;
const CHUNK_TYPE_JSON = 0x4E4F534A; // 'JSON'
const CHUNK_TYPE_BIN = 0x004E4942;  // 'BIN\0'

/**
 * glTF 访问器组件类型
 */
const COMPONENT_TYPES: Record<number, { size: number; type: 'float' | 'uint16' | 'uint32' }> = {
  5120: { size: 1, type: 'uint16' },  // BYTE
  5121: { size: 1, type: 'uint16' },  // UNSIGNED_BYTE
  5122: { size: 2, type: 'uint16' },  // SHORT
  5123: { size: 2, type: 'uint16' },  // UNSIGNED_SHORT
  5125: { size: 4, type: 'uint32' },  // UNSIGNED_INT
  5126: { size: 4, type: 'float' },   // FLOAT
};

/**
 * glTF 类型元素数量
 */
const TYPE_SIZES: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};

/**
 * GLBLoader - GLB 文件加载器
 * 解析 GLB 文件并生成 Mesh[]
 */
export class GLBLoader {
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * 加载 GLB 文件
   */
  async load(url: string): Promise<Mesh[]> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`无法加载 GLB 文件: ${url}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    return this.parse(arrayBuffer);
  }

  /**
   * 解析 GLB 二进制数据
   */
  private parse(buffer: ArrayBuffer): Mesh[] {
    const dataView = new DataView(buffer);
    let offset = 0;

    // 读取 GLB 头部
    const magic = dataView.getUint32(offset, true);
    offset += 4;
    if (magic !== GLB_MAGIC) {
      throw new Error('无效的 GLB 文件');
    }

    const version = dataView.getUint32(offset, true);
    offset += 4;
    if (version !== GLB_VERSION) {
      throw new Error(`不支持的 GLB 版本: ${version}`);
    }

    const _length = dataView.getUint32(offset, true);
    offset += 4;

    // 读取 JSON chunk
    const jsonChunkLength = dataView.getUint32(offset, true);
    offset += 4;
    const jsonChunkType = dataView.getUint32(offset, true);
    offset += 4;

    if (jsonChunkType !== CHUNK_TYPE_JSON) {
      throw new Error('第一个 chunk 必须是 JSON');
    }

    const jsonData = new Uint8Array(buffer, offset, jsonChunkLength);
    const jsonString = new TextDecoder().decode(jsonData);
    const gltf = JSON.parse(jsonString);
    offset += jsonChunkLength;

    // 读取 BIN chunk（可选）
    let binData: ArrayBuffer | null = null;
    if (offset < buffer.byteLength) {
      const binChunkLength = dataView.getUint32(offset, true);
      offset += 4;
      const binChunkType = dataView.getUint32(offset, true);
      offset += 4;

      if (binChunkType === CHUNK_TYPE_BIN) {
        binData = buffer.slice(offset, offset + binChunkLength);
      }
    }

    // 解析网格
    return this.parseMeshes(gltf, binData);
  }

  /**
   * 解析所有网格
   */
  private parseMeshes(gltf: any, binData: ArrayBuffer | null): Mesh[] {
    const meshes: Mesh[] = [];

    if (!gltf.meshes || !binData) {
      console.warn('GLB 文件中没有网格数据');
      return meshes;
    }

    for (const gltfMesh of gltf.meshes) {
      for (const primitive of gltfMesh.primitives) {
        const mesh = this.parsePrimitive(gltf, primitive, binData);
        if (mesh) {
          meshes.push(mesh);
        }
      }
    }

    return meshes;
  }

  /**
   * 解析单个图元
   */
  private parsePrimitive(gltf: any, primitive: any, binData: ArrayBuffer): Mesh | null {
    const attributes = primitive.attributes;
    
    // 获取位置数据
    if (attributes.POSITION === undefined) {
      console.warn('图元缺少 POSITION 属性');
      return null;
    }

    const positionAccessor = gltf.accessors[attributes.POSITION];
    const positions = this.getAccessorData(gltf, positionAccessor, binData);

    // 获取法线数据（可选，如果没有则生成）
    let normals: Float32Array;
    if (attributes.NORMAL !== undefined) {
      const normalAccessor = gltf.accessors[attributes.NORMAL];
      normals = this.getAccessorData(gltf, normalAccessor, binData) as Float32Array;
    } else {
      // 生成默认法线（指向 +Y）
      normals = new Float32Array(positions.length);
      for (let i = 0; i < positions.length; i += 3) {
        normals[i] = 0;
        normals[i + 1] = 1;
        normals[i + 2] = 0;
      }
    }

    // 创建交错顶点数据: position(3) + normal(3)
    const vertexCount = positionAccessor.count;
    const vertexData = new Float32Array(vertexCount * 6);
    for (let i = 0; i < vertexCount; i++) {
      vertexData[i * 6 + 0] = positions[i * 3 + 0];
      vertexData[i * 6 + 1] = positions[i * 3 + 1];
      vertexData[i * 6 + 2] = positions[i * 3 + 2];
      vertexData[i * 6 + 3] = normals[i * 3 + 0];
      vertexData[i * 6 + 4] = normals[i * 3 + 1];
      vertexData[i * 6 + 5] = normals[i * 3 + 2];
    }

    // 创建顶点缓冲区
    const vertexBuffer = this.device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertexBuffer, 0, vertexData);

    // 获取索引数据（可选）
    let indexBuffer: GPUBuffer | null = null;
    let indexCount = 0;

    if (primitive.indices !== undefined) {
      const indexAccessor = gltf.accessors[primitive.indices];
      const indices = this.getAccessorData(gltf, indexAccessor, binData);
      indexCount = indexAccessor.count;

      // 转换为 Uint16Array
      const indexData = new Uint16Array(indexCount);
      for (let i = 0; i < indexCount; i++) {
        indexData[i] = indices[i];
      }

      indexBuffer = this.device.createBuffer({
        size: indexData.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(indexBuffer, 0, indexData);
    }

    // 计算 bounding box
    const boundingBox = this.computeBoundingBox(positions);

    return new Mesh(vertexBuffer, vertexCount, indexBuffer, indexCount, boundingBox);
  }

  /**
   * 计算顶点数据的 bounding box
   */
  private computeBoundingBox(positions: Float32Array | Uint16Array | Uint32Array): MeshBoundingBox {
    if (positions.length < 3) {
      return {
        min: [0, 0, 0],
        max: [0, 0, 0],
        center: [0, 0, 0],
        radius: 0,
      };
    }

    // 初始化为第一个点
    const min: [number, number, number] = [positions[0], positions[1], positions[2]];
    const max: [number, number, number] = [positions[0], positions[1], positions[2]];

    // 遍历所有顶点
    for (let i = 3; i < positions.length; i += 3) {
      const x = positions[i];
      const y = positions[i + 1];
      const z = positions[i + 2];

      min[0] = Math.min(min[0], x);
      min[1] = Math.min(min[1], y);
      min[2] = Math.min(min[2], z);
      max[0] = Math.max(max[0], x);
      max[1] = Math.max(max[1], y);
      max[2] = Math.max(max[2], z);
    }

    // 计算中心点
    const center: [number, number, number] = [
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    ];

    // 计算 bounding sphere 半径
    const dx = max[0] - min[0];
    const dy = max[1] - min[1];
    const dz = max[2] - min[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    return { min, max, center, radius };
  }

  /**
   * 获取访问器数据
   */
  private getAccessorData(gltf: any, accessor: any, binData: ArrayBuffer): Float32Array | Uint16Array | Uint32Array {
    const bufferView = gltf.bufferViews[accessor.bufferView];
    const componentType = COMPONENT_TYPES[accessor.componentType];
    const typeSize = TYPE_SIZES[accessor.type];
    const count = accessor.count * typeSize;

    const byteOffset = (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);

    switch (componentType.type) {
      case 'float':
        return new Float32Array(binData, byteOffset, count);
      case 'uint16':
        return new Uint16Array(binData, byteOffset, count);
      case 'uint32':
        return new Uint32Array(binData, byteOffset, count);
      default:
        throw new Error(`不支持的组件类型: ${accessor.componentType}`);
    }
  }

  /**
   * 创建测试立方体（用于调试）
   */
  createTestCube(): Mesh {
    // 立方体顶点数据: position(3) + normal(3)
    const vertices = new Float32Array([
      // 前面 (z = 0.5)
      -0.5, -0.5,  0.5,  0, 0, 1,
       0.5, -0.5,  0.5,  0, 0, 1,
       0.5,  0.5,  0.5,  0, 0, 1,
      -0.5,  0.5,  0.5,  0, 0, 1,
      // 后面 (z = -0.5)
       0.5, -0.5, -0.5,  0, 0, -1,
      -0.5, -0.5, -0.5,  0, 0, -1,
      -0.5,  0.5, -0.5,  0, 0, -1,
       0.5,  0.5, -0.5,  0, 0, -1,
      // 上面 (y = 0.5)
      -0.5,  0.5,  0.5,  0, 1, 0,
       0.5,  0.5,  0.5,  0, 1, 0,
       0.5,  0.5, -0.5,  0, 1, 0,
      -0.5,  0.5, -0.5,  0, 1, 0,
      // 下面 (y = -0.5)
      -0.5, -0.5, -0.5,  0, -1, 0,
       0.5, -0.5, -0.5,  0, -1, 0,
       0.5, -0.5,  0.5,  0, -1, 0,
      -0.5, -0.5,  0.5,  0, -1, 0,
      // 右面 (x = 0.5)
       0.5, -0.5,  0.5,  1, 0, 0,
       0.5, -0.5, -0.5,  1, 0, 0,
       0.5,  0.5, -0.5,  1, 0, 0,
       0.5,  0.5,  0.5,  1, 0, 0,
      // 左面 (x = -0.5)
      -0.5, -0.5, -0.5,  -1, 0, 0,
      -0.5, -0.5,  0.5,  -1, 0, 0,
      -0.5,  0.5,  0.5,  -1, 0, 0,
      -0.5,  0.5, -0.5,  -1, 0, 0,
    ]);

    const indices = new Uint16Array([
      0, 1, 2, 0, 2, 3,       // 前
      4, 5, 6, 4, 6, 7,       // 后
      8, 9, 10, 8, 10, 11,    // 上
      12, 13, 14, 12, 14, 15, // 下
      16, 17, 18, 16, 18, 19, // 右
      20, 21, 22, 20, 22, 23, // 左
    ]);

    const vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertexBuffer, 0, vertices);

    const indexBuffer = this.device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(indexBuffer, 0, indices);

    // 立方体 bounding box: -0.5 到 0.5
    const cubeBbox: MeshBoundingBox = {
      min: [-0.5, -0.5, -0.5],
      max: [0.5, 0.5, 0.5],
      center: [0, 0, 0],
      radius: Math.sqrt(0.75), // sqrt(0.5^2 * 3) / 2 * 2 = sqrt(0.75)
    };

    return new Mesh(vertexBuffer, 24, indexBuffer, 36, cubeBbox);
  }

  /**
   * 创建测试球体
   */
  createTestSphere(radius: number = 0.5, segments: number = 32, rings: number = 16): Mesh {
    const vertices: number[] = [];
    const indices: number[] = [];

    // 生成顶点
    for (let ring = 0; ring <= rings; ring++) {
      const phi = (ring / rings) * Math.PI;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);

      for (let seg = 0; seg <= segments; seg++) {
        const theta = (seg / segments) * Math.PI * 2;
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);

        // 位置
        const x = radius * sinPhi * cosTheta;
        const y = radius * cosPhi;
        const z = radius * sinPhi * sinTheta;

        // 法线（球体法线就是归一化的位置）
        const nx = sinPhi * cosTheta;
        const ny = cosPhi;
        const nz = sinPhi * sinTheta;

        vertices.push(x, y, z, nx, ny, nz);
      }
    }

    // 生成索引
    for (let ring = 0; ring < rings; ring++) {
      for (let seg = 0; seg < segments; seg++) {
        const current = ring * (segments + 1) + seg;
        const next = current + segments + 1;

        indices.push(current, next, current + 1);
        indices.push(current + 1, next, next + 1);
      }
    }

    const vertexData = new Float32Array(vertices);
    const indexData = new Uint16Array(indices);

    const vertexBuffer = this.device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertexBuffer, 0, vertexData);

    const indexBuffer = this.device.createBuffer({
      size: indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(indexBuffer, 0, indexData);

    // 球体 bounding box
    const sphereBbox: MeshBoundingBox = {
      min: [-radius, -radius, -radius],
      max: [radius, radius, radius],
      center: [0, 0, 0],
      radius: radius,
    };

    return new Mesh(vertexBuffer, vertexData.length / 6, indexBuffer, indexData.length, sphereBbox);
  }
}
