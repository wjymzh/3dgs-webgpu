/**
 * OBJLoader - OBJ 文件加载器
 * 解析 OBJ 文件并生成 Mesh[]，支持 MTL 材质
 */

import { Mesh, MeshBoundingBox } from '../mesh/Mesh';
import type { MaterialData } from '../types';
import { DEFAULT_OBJ_MATERIAL } from '../types';
import { computeBoundingBox } from '../utils';
import { LoadedMesh } from './GLBLoader';
import { OBJParser, ParsedOBJData, ParsedObject } from './OBJParser';
import { MTLParser, ParsedMaterial } from './MTLParser';

/**
 * OBJLoader - OBJ 文件加载器
 * 解析 OBJ 文件并生成 LoadedMesh[]
 */
export class OBJLoader {
  private device: GPUDevice;
  private textureCache: Map<string, GPUTexture> = new Map();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * 加载 OBJ 文件
   * @param url OBJ 文件 URL
   * @returns 加载后的网格数组
   * 
   * Requirements: 2.1, 2.2, 2.3
   */
  async load(url: string): Promise<LoadedMesh[]> {
    // Requirement 2.1: 获取文件内容
    const response = await fetch(url);
    
    // Requirement 2.2: 处理获取失败
    if (!response.ok) {
      throw new Error(`无法加载 OBJ 文件: ${url} (HTTP ${response.status})`);
    }

    const text = await response.text();
    
    // 计算基础 URL（用于加载 MTL 和纹理）
    const baseUrl = url.substring(0, url.lastIndexOf('/') + 1);
    
    // Requirement 2.3: 解析并返回 LoadedMesh 数组
    return this.parseFromText(text, baseUrl);
  }

  /**
   * 从文本解析 OBJ（用于直接传入内容）
   * @param text OBJ 文本内容
   * @param baseUrl 基础 URL（用于加载 MTL 和纹理）
   * @returns 加载后的网格数组
   * 
   * Requirements: 2.1, 2.3
   */
  async parseFromText(text: string, baseUrl?: string): Promise<LoadedMesh[]> {
    // 解析 OBJ 文本
    const parser = new OBJParser();
    const parsedData = parser.parse(text);

    // 提取 MTL 文件引用
    const mtlFile = this.extractMTLReference(text);
    
    // 加载材质（如果有）
    let materials: Map<string, ParsedMaterial> = new Map();
    if (mtlFile && baseUrl) {
      materials = await this.loadMTL(baseUrl + mtlFile);
    }

    // 转换为 LoadedMesh 数组
    return this.createMeshes(parsedData, materials, baseUrl);
  }

  /**
   * 提取 MTL 文件引用
   */
  private extractMTLReference(text: string): string | null {
    const lines = text.split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('mtllib ')) {
        return trimmed.substring(7).trim();
      }
    }
    return null;
  }

  /**
   * 加载 MTL 材质文件
   * Requirement 4.1, 4.4: 加载 MTL 文件，失败时使用默认材质
   */
  private async loadMTL(url: string): Promise<Map<string, ParsedMaterial>> {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        return new Map();
      }
      const text = await response.text();
      const mtlParser = new MTLParser();
      return mtlParser.parse(text);
    } catch (e) {
      return new Map();
    }
  }

  /**
   * 创建 LoadedMesh 数组
   */
  private async createMeshes(
    parsedData: ParsedOBJData,
    materials: Map<string, ParsedMaterial>,
    baseUrl?: string
  ): Promise<LoadedMesh[]> {
    const loadedMeshes: LoadedMesh[] = [];

    for (const obj of parsedData.objects) {
      // 跳过空对象
      if (obj.indices.length === 0) {
        continue;
      }

      const loadedMesh = await this.createMesh(obj, materials, baseUrl);
      if (loadedMesh) {
        loadedMeshes.push(loadedMesh);
      }
    }

    return loadedMeshes;
  }

  /**
   * 创建单个 Mesh
   */
  private async createMesh(
    obj: ParsedObject,
    materials: Map<string, ParsedMaterial>,
    baseUrl?: string
  ): Promise<LoadedMesh | null> {
    const hasNormals = obj.normals.length > 0;
    const hasUVs = obj.uvs.length > 0;
    const vertexCount = obj.positions.length / 3;

    // 如果没有法线，生成平面法线
    let normals = obj.normals;
    if (!hasNormals) {
      normals = this.generateFlatNormals(obj.positions, obj.indices);
    }

    // 创建交错顶点数据: position(3) + normal(3) + uv(2)
    const stride = hasUVs ? 8 : 6; // floats per vertex
    const vertexData = new Float32Array(vertexCount * stride);

    for (let i = 0; i < vertexCount; i++) {
      const baseIdx = i * stride;
      // Position
      vertexData[baseIdx + 0] = obj.positions[i * 3 + 0];
      vertexData[baseIdx + 1] = obj.positions[i * 3 + 1];
      vertexData[baseIdx + 2] = obj.positions[i * 3 + 2];
      // Normal
      vertexData[baseIdx + 3] = normals[i * 3 + 0];
      vertexData[baseIdx + 4] = normals[i * 3 + 1];
      vertexData[baseIdx + 5] = normals[i * 3 + 2];
      // UV (if present)
      if (hasUVs) {
        vertexData[baseIdx + 6] = obj.uvs[i * 2 + 0];
        vertexData[baseIdx + 7] = obj.uvs[i * 2 + 1];
      }
    }

    // 创建顶点缓冲区
    const vertexBuffer = this.device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertexBuffer, 0, vertexData);

    // 创建索引缓冲区
    // Requirement 3.4: 根据顶点数选择索引格式
    const indexCount = obj.indices.length;
    let indexBuffer: GPUBuffer;
    let indexFormat: 'uint16' | 'uint32' = 'uint16';

    if (vertexCount > 65535) {
      indexFormat = 'uint32';
      const indexData = new Uint32Array(obj.indices);
      indexBuffer = this.device.createBuffer({
        size: indexData.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(indexBuffer, 0, indexData);
    } else {
      const indexData = new Uint16Array(obj.indices);
      indexBuffer = this.device.createBuffer({
        size: indexData.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(indexBuffer, 0, indexData);
    }

    // Requirement 3.5: 计算 bounding box
    const boundingBox = this.computeBoundingBoxFromPositions(obj.positions);

    // 创建 Mesh
    const mesh = new Mesh(vertexBuffer, vertexCount, indexBuffer, indexCount, boundingBox);
    mesh.hasUV = hasUVs;
    mesh.indexFormat = indexFormat;

    // 创建材质
    const material = await this.createMaterial(obj.materialName, materials, baseUrl);

    return { mesh, material };
  }

  /**
   * 生成平面法线（当 OBJ 缺少法线数据时）
   * Requirement 3.2: 从面几何计算平面法线
   */
  private generateFlatNormals(positions: number[], indices: number[]): number[] {
    const normals = new Array(positions.length).fill(0);

    // 遍历每个三角形
    for (let i = 0; i < indices.length; i += 3) {
      const i0 = indices[i];
      const i1 = indices[i + 1];
      const i2 = indices[i + 2];

      // 获取三角形顶点
      const v0 = [positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]];
      const v1 = [positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]];
      const v2 = [positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]];

      // 计算边向量
      const edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
      const edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

      // 计算叉积（面法线）
      const nx = edge1[1] * edge2[2] - edge1[2] * edge2[1];
      const ny = edge1[2] * edge2[0] - edge1[0] * edge2[2];
      const nz = edge1[0] * edge2[1] - edge1[1] * edge2[0];

      // 归一化
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      const normalizedNx = len > 0 ? nx / len : 0;
      const normalizedNy = len > 0 ? ny / len : 1;
      const normalizedNz = len > 0 ? nz / len : 0;

      // 为三角形的每个顶点设置相同的法线（平面着色）
      for (const idx of [i0, i1, i2]) {
        normals[idx * 3 + 0] = normalizedNx;
        normals[idx * 3 + 1] = normalizedNy;
        normals[idx * 3 + 2] = normalizedNz;
      }
    }

    return normals;
  }

  /**
   * 计算顶点数据的 bounding box
   * Requirement 3.5: 计算并存储 bounding box 信息
   */
  private computeBoundingBoxFromPositions(positions: number[]): MeshBoundingBox {
    return computeBoundingBox(positions);
  }

  /**
   * 创建材质数据
   * Requirement 4.3: 将材质关联到网格
   */
  private async createMaterial(
    materialName: string | null,
    materials: Map<string, ParsedMaterial>,
    baseUrl?: string
  ): Promise<MaterialData> {
    if (!materialName || !materials.has(materialName)) {
      return { ...DEFAULT_OBJ_MATERIAL };
    }

    const parsedMaterial = materials.get(materialName)!;
    
    // 转换为 MaterialData - OBJ 模型默认双面渲染
    const material: MaterialData = {
      baseColorFactor: [
        parsedMaterial.diffuseColor[0],
        parsedMaterial.diffuseColor[1],
        parsedMaterial.diffuseColor[2],
        parsedMaterial.opacity,
      ],
      baseColorTexture: null,
      metallicFactor: 0,
      roughnessFactor: 0.5,
      doubleSided: true,
    };

    // 加载漫反射纹理
    if (parsedMaterial.diffuseTexture && baseUrl) {
      material.baseColorTexture = await this.loadTexture(baseUrl + parsedMaterial.diffuseTexture);
    }

    return material;
  }

  /**
   * 加载纹理
   * Requirement 4.5, 6.3: 加载纹理，失败时使用 null
   */
  private async loadTexture(url: string): Promise<GPUTexture | null> {
    // 检查缓存
    if (this.textureCache.has(url)) {
      return this.textureCache.get(url)!;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        return null;
      }

      const blob = await response.blob();
      const imageBitmap = await createImageBitmap(blob);

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
      this.textureCache.set(url, gpuTexture);

      return gpuTexture;
    } catch (error) {
      return null;
    }
  }
}
