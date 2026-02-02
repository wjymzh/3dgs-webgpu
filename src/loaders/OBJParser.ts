/**
 * OBJParser - OBJ 文件文本解析器
 * 解析 OBJ 文件文本内容，提取几何数据
 */

/**
 * 解析后的 OBJ 数据结构
 */
export interface ParsedOBJData {
  objects: ParsedObject[];
}

/**
 * 单个对象/组的数据
 */
export interface ParsedObject {
  name: string;
  positions: number[];      // 展开后的顶点位置 [x,y,z, x,y,z, ...]
  normals: number[];        // 展开后的法线 [nx,ny,nz, ...]
  uvs: number[];            // 展开后的 UV [u,v, u,v, ...]
  indices: number[];        // 三角形索引
  materialName: string | null;
}

/**
 * 面顶点索引结构
 */
interface FaceVertex {
  positionIndex: number;
  uvIndex: number | null;
  normalIndex: number | null;
}

/**
 * OBJ 文本解析器
 */
export class OBJParser {
  // 全局顶点池
  private positions: number[] = [];   // v 指令收集的顶点位置
  private uvs: number[] = [];         // vt 指令收集的纹理坐标
  private normals: number[] = [];     // vn 指令收集的法线

  // 当前对象数据
  private currentObject: ParsedObject | null = null;
  private objects: ParsedObject[] = [];

  // 当前材质
  private currentMaterial: string | null = null;

  // 顶点去重映射 (用于索引缓冲区)
  private vertexMap: Map<string, number> = new Map();
  private vertexCount: number = 0;

  // 已警告的不支持指令集合（用于只警告首次）
  private warnedDirectives: Set<string> = new Set();

  /**
   * 解析 OBJ 文本内容
   * @param text OBJ 文件文本
   * @returns 解析后的数据结构
   */
  parse(text: string): ParsedOBJData {
    // 重置状态
    this.reset();

    // 按行分割文本
    const lines = text.split(/\r?\n/);

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // 跳过空行和注释
      if (line === '' || line.startsWith('#')) {
        continue;
      }

      this.parseLine(line, i + 1);
    }

    // 确保最后一个对象被添加
    this.finalizeCurrentObject();

    // 如果没有任何对象，但有面数据，创建默认对象
    if (this.objects.length === 0 && this.currentObject && this.currentObject.indices.length > 0) {
      this.objects.push(this.currentObject);
    }

    return { objects: this.objects };
  }

  /**
   * 重置解析器状态
   */
  private reset(): void {
    this.positions = [];
    this.uvs = [];
    this.normals = [];
    this.currentObject = null;
    this.objects = [];
    this.currentMaterial = null;
    this.vertexMap = new Map();
    this.vertexCount = 0;
    this.warnedDirectives = new Set();
  }

  /**
   * 警告不支持的指令（仅首次出现时警告）
   * @param directive 指令名称
   * @param lineNum 行号
   */
  private warnUnsupportedDirective(directive: string, lineNum: number): void {
    if (!this.warnedDirectives.has(directive)) {
      this.warnedDirectives.add(directive);
    }
  }

  /**
   * 解析单行
   */
  private parseLine(line: string, lineNum: number): void {
    // 分割指令和参数
    const parts = line.split(/\s+/);
    const directive = parts[0].toLowerCase();

    try {
      switch (directive) {
        case 'v':
          this.parseVertex(parts);
          break;
        case 'vt':
          this.parseTextureCoord(parts);
          break;
        case 'vn':
          this.parseNormal(parts);
          break;
        case 'f':
          this.parseFace(parts, lineNum);
          break;
        case 'o':
        case 'g':
          this.parseObjectOrGroup(parts);
          break;
        case 'usemtl':
          this.parseUseMaterial(parts);
          break;
        case 'mtllib':
          // MTL 文件引用，在 OBJLoader 中处理
          break;
        case 's':
          // 平滑组，暂不支持，警告首次
          this.warnUnsupportedDirective(directive, lineNum);
          break;
        default:
          // 不支持的指令，警告首次
          this.warnUnsupportedDirective(directive, lineNum);
          break;
      }
    } catch (e) {
      // 解析错误，跳过该行
    }
  }

  /**
   * 解析顶点位置 (v x y z [w])
   */
  private parseVertex(parts: string[]): void {
    if (parts.length < 4) {
      throw new Error('顶点数据不完整');
    }

    const x = parseFloat(parts[1]);
    const y = parseFloat(parts[2]);
    const z = parseFloat(parts[3]);

    if (isNaN(x) || isNaN(y) || isNaN(z)) {
      throw new Error('无效的顶点坐标');
    }

    this.positions.push(x, y, z);
  }

  /**
   * 解析纹理坐标 (vt u [v] [w])
   */
  private parseTextureCoord(parts: string[]): void {
    if (parts.length < 2) {
      throw new Error('纹理坐标数据不完整');
    }

    const u = parseFloat(parts[1]);
    const v = parts.length > 2 ? parseFloat(parts[2]) : 0;

    if (isNaN(u) || isNaN(v)) {
      throw new Error('无效的纹理坐标');
    }

    this.uvs.push(u, v);
  }

  /**
   * 解析法线 (vn x y z)
   */
  private parseNormal(parts: string[]): void {
    if (parts.length < 4) {
      throw new Error('法线数据不完整');
    }

    const x = parseFloat(parts[1]);
    const y = parseFloat(parts[2]);
    const z = parseFloat(parts[3]);

    if (isNaN(x) || isNaN(y) || isNaN(z)) {
      throw new Error('无效的法线');
    }

    this.normals.push(x, y, z);
  }

  /**
   * 解析面 (f v1 v2 v3 ...)
   * 支持四种格式:
   * - f v v v (仅顶点)
   * - f v/vt v/vt v/vt (顶点/纹理)
   * - f v/vt/vn v/vt/vn v/vt/vn (顶点/纹理/法线)
   * - f v//vn v//vn v//vn (顶点//法线)
   */
  private parseFace(parts: string[], _lineNum: number): void {
    if (parts.length < 4) {
      throw new Error('面数据不完整，至少需要3个顶点');
    }

    // 确保有当前对象
    this.ensureCurrentObject();

    // 解析面顶点
    const faceVertices: FaceVertex[] = [];
    for (let i = 1; i < parts.length; i++) {
      const vertex = this.parseFaceVertex(parts[i]);
      if (vertex) {
        faceVertices.push(vertex);
      }
    }

    if (faceVertices.length < 3) {
      throw new Error('面顶点数量不足');
    }

    // 三角化多边形（扇形三角化）
    // 对于 n 边形 [v0, v1, v2, ..., vn-1]:
    // 生成三角形: (v0, v1, v2), (v0, v2, v3), ..., (v0, vn-2, vn-1)
    for (let i = 1; i < faceVertices.length - 1; i++) {
      this.addTriangle(faceVertices[0], faceVertices[i], faceVertices[i + 1]);
    }
  }

  /**
   * 解析面顶点索引
   * 支持格式: v, v/vt, v/vt/vn, v//vn
   */
  private parseFaceVertex(vertexStr: string): FaceVertex | null {
    const parts = vertexStr.split('/');
    
    // 解析位置索引（必需）
    const positionIndex = this.resolveIndex(parseInt(parts[0]), this.positions.length / 3);
    if (isNaN(positionIndex)) {
      return null;
    }

    // 解析纹理坐标索引（可选）
    let uvIndex: number | null = null;
    if (parts.length > 1 && parts[1] !== '') {
      uvIndex = this.resolveIndex(parseInt(parts[1]), this.uvs.length / 2);
      if (isNaN(uvIndex)) {
        uvIndex = null;
      }
    }

    // 解析法线索引（可选）
    let normalIndex: number | null = null;
    if (parts.length > 2 && parts[2] !== '') {
      normalIndex = this.resolveIndex(parseInt(parts[2]), this.normals.length / 3);
      if (isNaN(normalIndex)) {
        normalIndex = null;
      }
    }

    return { positionIndex, uvIndex, normalIndex };
  }

  /**
   * 解析索引，支持负索引
   * 负索引表示相对于当前顶点数的偏移（-1 表示最后一个顶点）
   */
  private resolveIndex(index: number, count: number): number {
    if (isNaN(index)) {
      return NaN;
    }

    if (index < 0) {
      // 负索引：-1 表示最后一个，-2 表示倒数第二个
      return count + index;
    } else {
      // 正索引：OBJ 索引从 1 开始
      return index - 1;
    }
  }

  /**
   * 添加三角形到当前对象
   */
  private addTriangle(v0: FaceVertex, v1: FaceVertex, v2: FaceVertex): void {
    const idx0 = this.addVertex(v0);
    const idx1 = this.addVertex(v1);
    const idx2 = this.addVertex(v2);

    this.currentObject!.indices.push(idx0, idx1, idx2);
  }

  /**
   * 添加顶点到当前对象，返回索引
   * 使用顶点去重，相同的顶点组合只添加一次
   */
  private addVertex(vertex: FaceVertex): number {
    // 创建顶点键用于去重
    const key = `${vertex.positionIndex}/${vertex.uvIndex ?? ''}/${vertex.normalIndex ?? ''}`;

    // 检查是否已存在
    if (this.vertexMap.has(key)) {
      return this.vertexMap.get(key)!;
    }

    // 添加新顶点
    const index = this.vertexCount++;
    this.vertexMap.set(key, index);

    // 添加位置数据
    const posIdx = vertex.positionIndex * 3;
    if (posIdx >= 0 && posIdx + 2 < this.positions.length) {
      this.currentObject!.positions.push(
        this.positions[posIdx],
        this.positions[posIdx + 1],
        this.positions[posIdx + 2]
      );
    } else {
      // 无效索引，使用默认值
      this.currentObject!.positions.push(0, 0, 0);
    }

    // 添加法线数据
    if (vertex.normalIndex !== null) {
      const normIdx = vertex.normalIndex * 3;
      if (normIdx >= 0 && normIdx + 2 < this.normals.length) {
        this.currentObject!.normals.push(
          this.normals[normIdx],
          this.normals[normIdx + 1],
          this.normals[normIdx + 2]
        );
      } else {
        this.currentObject!.normals.push(0, 1, 0);
      }
    }

    // 添加 UV 数据
    if (vertex.uvIndex !== null) {
      const uvIdx = vertex.uvIndex * 2;
      if (uvIdx >= 0 && uvIdx + 1 < this.uvs.length) {
        this.currentObject!.uvs.push(
          this.uvs[uvIdx],
          this.uvs[uvIdx + 1]
        );
      } else {
        this.currentObject!.uvs.push(0, 0);
      }
    }

    return index;
  }

  /**
   * 解析对象或组 (o name / g name)
   */
  private parseObjectOrGroup(parts: string[]): void {
    // 完成当前对象
    this.finalizeCurrentObject();

    // 创建新对象
    const name = parts.length > 1 ? parts.slice(1).join(' ') : 'default';
    this.createNewObject(name);
  }

  /**
   * 解析材质引用 (usemtl name)
   */
  private parseUseMaterial(parts: string[]): void {
    if (parts.length > 1) {
      this.currentMaterial = parts.slice(1).join(' ');
      
      // 如果当前对象已有面数据，需要创建新对象
      if (this.currentObject && this.currentObject.indices.length > 0) {
        this.finalizeCurrentObject();
        this.createNewObject(this.currentObject?.name || 'default');
      }
      
      if (this.currentObject) {
        this.currentObject.materialName = this.currentMaterial;
      }
    }
  }

  /**
   * 确保存在当前对象
   */
  private ensureCurrentObject(): void {
    if (!this.currentObject) {
      this.createNewObject('default');
    }
  }

  /**
   * 创建新对象
   */
  private createNewObject(name: string): void {
    this.currentObject = {
      name,
      positions: [],
      normals: [],
      uvs: [],
      indices: [],
      materialName: this.currentMaterial,
    };
    this.vertexMap = new Map();
    this.vertexCount = 0;
  }

  /**
   * 完成当前对象并添加到列表
   */
  private finalizeCurrentObject(): void {
    if (this.currentObject && this.currentObject.indices.length > 0) {
      this.objects.push(this.currentObject);
    }
    this.currentObject = null;
  }
}
