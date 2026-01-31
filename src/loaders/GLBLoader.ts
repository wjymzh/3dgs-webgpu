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
const COMPONENT_TYPES: Record<number, { size: number; type: 'float' | 'uint8' | 'uint16' | 'uint32' | 'int8' | 'int16' }> = {
  5120: { size: 1, type: 'int8' },     // BYTE
  5121: { size: 1, type: 'uint8' },    // UNSIGNED_BYTE
  5122: { size: 2, type: 'int16' },    // SHORT
  5123: { size: 2, type: 'uint16' },   // UNSIGNED_SHORT
  5125: { size: 4, type: 'uint32' },   // UNSIGNED_INT
  5126: { size: 4, type: 'float' },    // FLOAT
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
 * 材质数据
 */
export interface MaterialData {
  baseColorFactor: [number, number, number, number];
  baseColorTexture: GPUTexture | null;
  metallicFactor: number;
  roughnessFactor: number;
  doubleSided: boolean;
}

/**
 * 加载后的 Mesh 数据（包含材质）
 */
export interface LoadedMesh {
  mesh: Mesh;
  material: MaterialData;
}

/**
 * GLBLoader - GLB 文件加载器
 * 解析 GLB 文件并生成 Mesh[]，支持贴图
 */
export class GLBLoader {
  private device: GPUDevice;
  private textureCache: Map<number, GPUTexture> = new Map();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * 加载 GLB 文件
   */
  async load(url: string): Promise<LoadedMesh[]> {
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
  private async parse(buffer: ArrayBuffer): Promise<LoadedMesh[]> {
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

    // 清空纹理缓存
    this.textureCache.clear();

    // 解析网格
    return this.parseMeshes(gltf, binData);
  }

  /**
   * 解析所有网格
   */
  private async parseMeshes(gltf: any, binData: ArrayBuffer | null): Promise<LoadedMesh[]> {
    const meshes: LoadedMesh[] = [];

    if (!gltf.meshes || !binData) {
      console.warn('GLB 文件中没有网格数据');
      return meshes;
    }

    for (const gltfMesh of gltf.meshes) {
      for (const primitive of gltfMesh.primitives) {
        const loadedMesh = await this.parsePrimitive(gltf, primitive, binData);
        if (loadedMesh) {
          meshes.push(loadedMesh);
        }
      }
    }

    return meshes;
  }

  /**
   * 解析单个图元
   */
  private async parsePrimitive(gltf: any, primitive: any, binData: ArrayBuffer): Promise<LoadedMesh | null> {
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
      const normalData = this.getAccessorData(gltf, normalAccessor, binData);
      normals = new Float32Array(normalData);
    } else {
      // 生成默认法线（指向 +Y）
      normals = new Float32Array(positions.length);
      for (let i = 0; i < positions.length; i += 3) {
        normals[i] = 0;
        normals[i + 1] = 1;
        normals[i + 2] = 0;
      }
    }

    // 获取 UV 坐标（可选）
    let uvs: Float32Array | null = null;
    if (attributes.TEXCOORD_0 !== undefined) {
      const uvAccessor = gltf.accessors[attributes.TEXCOORD_0];
      const uvData = this.getAccessorData(gltf, uvAccessor, binData);
      uvs = new Float32Array(uvData);
    }

    // 创建交错顶点数据: position(3) + normal(3) + uv(2)
    const vertexCount = positionAccessor.count;
    const hasUV = uvs !== null;
    const stride = hasUV ? 8 : 6; // 有 UV 时 8 floats，否则 6 floats
    const vertexData = new Float32Array(vertexCount * stride);
    
    for (let i = 0; i < vertexCount; i++) {
      const baseIdx = i * stride;
      vertexData[baseIdx + 0] = positions[i * 3 + 0];
      vertexData[baseIdx + 1] = positions[i * 3 + 1];
      vertexData[baseIdx + 2] = positions[i * 3 + 2];
      vertexData[baseIdx + 3] = normals[i * 3 + 0];
      vertexData[baseIdx + 4] = normals[i * 3 + 1];
      vertexData[baseIdx + 5] = normals[i * 3 + 2];
      if (hasUV && uvs) {
        vertexData[baseIdx + 6] = uvs[i * 2 + 0];
        vertexData[baseIdx + 7] = uvs[i * 2 + 1];
      }
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
    let indexFormat: 'uint16' | 'uint32' = 'uint16';

    if (primitive.indices !== undefined) {
      const indexAccessor = gltf.accessors[primitive.indices];
      const indices = this.getAccessorData(gltf, indexAccessor, binData);
      indexCount = indexAccessor.count;

      // 根据顶点数量决定索引格式
      if (vertexCount > 65535) {
        indexFormat = 'uint32';
        const indexData = new Uint32Array(indexCount);
        for (let i = 0; i < indexCount; i++) {
          indexData[i] = indices[i];
        }
        indexBuffer = this.device.createBuffer({
          size: indexData.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(indexBuffer, 0, indexData);
      } else {
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
    }

    // 计算 bounding box
    const boundingBox = this.computeBoundingBox(positions);

    // 解析材质
    const material = await this.parseMaterial(gltf, primitive.material, binData);

    // 创建 Mesh
    const mesh = new Mesh(vertexBuffer, vertexCount, indexBuffer, indexCount, boundingBox);
    mesh.hasUV = hasUV;
    mesh.indexFormat = indexFormat;

    return { mesh, material };
  }

  /**
   * 解析材质
   */
  private async parseMaterial(gltf: any, materialIndex: number | undefined, binData: ArrayBuffer): Promise<MaterialData> {
    // 默认材质
    const defaultMaterial: MaterialData = {
      baseColorFactor: [1, 1, 1, 1],
      baseColorTexture: null,
      metallicFactor: 1,
      roughnessFactor: 1,
      doubleSided: false,
    };

    if (materialIndex === undefined || !gltf.materials) {
      return defaultMaterial;
    }

    const gltfMaterial = gltf.materials[materialIndex];
    if (!gltfMaterial) {
      return defaultMaterial;
    }

    const material: MaterialData = { ...defaultMaterial };

    // 解析 doubleSided
    if (gltfMaterial.doubleSided !== undefined) {
      material.doubleSided = gltfMaterial.doubleSided;
    }

    // 解析 PBR 材质
    const pbr = gltfMaterial.pbrMetallicRoughness;
    if (pbr) {
      // baseColorFactor
      if (pbr.baseColorFactor) {
        material.baseColorFactor = pbr.baseColorFactor;
      }

      // metallicFactor
      if (pbr.metallicFactor !== undefined) {
        material.metallicFactor = pbr.metallicFactor;
      }

      // roughnessFactor
      if (pbr.roughnessFactor !== undefined) {
        material.roughnessFactor = pbr.roughnessFactor;
      }

      // baseColorTexture
      if (pbr.baseColorTexture) {
        const textureIndex = pbr.baseColorTexture.index;
        material.baseColorTexture = await this.loadTexture(gltf, textureIndex, binData);
      }
    }

    return material;
  }

  /**
   * 加载纹理
   */
  private async loadTexture(gltf: any, textureIndex: number, binData: ArrayBuffer): Promise<GPUTexture | null> {
    // 检查缓存
    if (this.textureCache.has(textureIndex)) {
      return this.textureCache.get(textureIndex)!;
    }

    if (!gltf.textures || !gltf.images) {
      return null;
    }

    const texture = gltf.textures[textureIndex];
    if (!texture || texture.source === undefined) {
      return null;
    }

    const image = gltf.images[texture.source];
    if (!image) {
      return null;
    }

    try {
      let imageBitmap: ImageBitmap;

      if (image.bufferView !== undefined) {
        // 从 buffer 加载图片
        const bufferView = gltf.bufferViews[image.bufferView];
        const byteOffset = bufferView.byteOffset || 0;
        const byteLength = bufferView.byteLength;
        const imageData = new Uint8Array(binData, byteOffset, byteLength);
        const blob = new Blob([imageData], { type: image.mimeType || 'image/png' });
        imageBitmap = await createImageBitmap(blob);
      } else if (image.uri) {
        // 从 URI 加载（data URI 或外部 URL）
        if (image.uri.startsWith('data:')) {
          const response = await fetch(image.uri);
          const blob = await response.blob();
          imageBitmap = await createImageBitmap(blob);
        } else {
          // 外部 URL - 这里简化处理，实际可能需要相对路径解析
          const response = await fetch(image.uri);
          const blob = await response.blob();
          imageBitmap = await createImageBitmap(blob);
        }
      } else {
        return null;
      }

      // 创建 GPU 纹理
      const gpuTexture = this.device.createTexture({
        size: [imageBitmap.width, imageBitmap.height, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      this.device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture: gpuTexture },
        [imageBitmap.width, imageBitmap.height]
      );

      // 缓存纹理
      this.textureCache.set(textureIndex, gpuTexture);

      return gpuTexture;
    } catch (error) {
      console.error('加载纹理失败:', error);
      return null;
    }
  }

  /**
   * 计算顶点数据的 bounding box
   */
  private computeBoundingBox(positions: Float32Array | Uint16Array | Uint32Array | Int8Array | Int16Array | Uint8Array): MeshBoundingBox {
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
   * 获取访问器数据 - 修复字节对齐问题
   */
  private getAccessorData(gltf: any, accessor: any, binData: ArrayBuffer): Float32Array | Uint16Array | Uint32Array | Int8Array | Int16Array | Uint8Array {
    const bufferView = gltf.bufferViews[accessor.bufferView];
    const componentType = COMPONENT_TYPES[accessor.componentType];
    const typeSize = TYPE_SIZES[accessor.type];
    const count = accessor.count * typeSize;

    const byteOffset = (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);
    const byteLength = count * componentType.size;

    // 复制数据到新的 ArrayBuffer 以避免对齐问题
    const alignedBuffer = new ArrayBuffer(byteLength);
    const srcView = new Uint8Array(binData, byteOffset, byteLength);
    const dstView = new Uint8Array(alignedBuffer);
    dstView.set(srcView);

    switch (componentType.type) {
      case 'float':
        return new Float32Array(alignedBuffer);
      case 'uint8':
        return new Uint8Array(alignedBuffer);
      case 'uint16':
        return new Uint16Array(alignedBuffer);
      case 'uint32':
        return new Uint32Array(alignedBuffer);
      case 'int8':
        return new Int8Array(alignedBuffer);
      case 'int16':
        return new Int16Array(alignedBuffer);
      default:
        throw new Error(`不支持的组件类型: ${accessor.componentType}`);
    }
  }

  /**
   * 创建测试立方体（用于调试）
   */
  createTestCube(): LoadedMesh {
    // 立方体顶点数据: position(3) + normal(3) + uv(2)
    const vertices = new Float32Array([
      // 前面 (z = 0.5)
      -0.5, -0.5,  0.5,  0, 0, 1,  0, 1,
       0.5, -0.5,  0.5,  0, 0, 1,  1, 1,
       0.5,  0.5,  0.5,  0, 0, 1,  1, 0,
      -0.5,  0.5,  0.5,  0, 0, 1,  0, 0,
      // 后面 (z = -0.5)
       0.5, -0.5, -0.5,  0, 0, -1,  0, 1,
      -0.5, -0.5, -0.5,  0, 0, -1,  1, 1,
      -0.5,  0.5, -0.5,  0, 0, -1,  1, 0,
       0.5,  0.5, -0.5,  0, 0, -1,  0, 0,
      // 上面 (y = 0.5)
      -0.5,  0.5,  0.5,  0, 1, 0,  0, 1,
       0.5,  0.5,  0.5,  0, 1, 0,  1, 1,
       0.5,  0.5, -0.5,  0, 1, 0,  1, 0,
      -0.5,  0.5, -0.5,  0, 1, 0,  0, 0,
      // 下面 (y = -0.5)
      -0.5, -0.5, -0.5,  0, -1, 0,  0, 1,
       0.5, -0.5, -0.5,  0, -1, 0,  1, 1,
       0.5, -0.5,  0.5,  0, -1, 0,  1, 0,
      -0.5, -0.5,  0.5,  0, -1, 0,  0, 0,
      // 右面 (x = 0.5)
       0.5, -0.5,  0.5,  1, 0, 0,  0, 1,
       0.5, -0.5, -0.5,  1, 0, 0,  1, 1,
       0.5,  0.5, -0.5,  1, 0, 0,  1, 0,
       0.5,  0.5,  0.5,  1, 0, 0,  0, 0,
      // 左面 (x = -0.5)
      -0.5, -0.5, -0.5,  -1, 0, 0,  0, 1,
      -0.5, -0.5,  0.5,  -1, 0, 0,  1, 1,
      -0.5,  0.5,  0.5,  -1, 0, 0,  1, 0,
      -0.5,  0.5, -0.5,  -1, 0, 0,  0, 0,
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
      radius: Math.sqrt(0.75),
    };

    const mesh = new Mesh(vertexBuffer, 24, indexBuffer, 36, cubeBbox);
    mesh.hasUV = true;
    mesh.indexFormat = 'uint16';

    return {
      mesh,
      material: {
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: null,
        metallicFactor: 0,
        roughnessFactor: 0.5,
        doubleSided: false,
      },
    };
  }

  /**
   * 创建测试球体
   */
  createTestSphere(radius: number = 0.5, segments: number = 32, rings: number = 16): LoadedMesh {
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

        // UV
        const u = seg / segments;
        const v = ring / rings;

        vertices.push(x, y, z, nx, ny, nz, u, v);
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

    const mesh = new Mesh(vertexBuffer, vertexData.length / 8, indexBuffer, indexData.length, sphereBbox);
    mesh.hasUV = true;
    mesh.indexFormat = 'uint16';

    return {
      mesh,
      material: {
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: null,
        metallicFactor: 0,
        roughnessFactor: 0.5,
        doubleSided: false,
      },
    };
  }
}
