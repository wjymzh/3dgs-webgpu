/**
 * MTLParser - MTL 材质文件解析器
 * 解析 MTL 文件文本内容，提取材质数据
 */

/**
 * 解析后的材质数据
 */
export interface ParsedMaterial {
  name: string;
  diffuseColor: [number, number, number];  // Kd
  diffuseTexture: string | null;           // map_Kd
  opacity: number;                          // d 或 1-Tr
}

/**
 * MTL 文本解析器
 */
export class MTLParser {
  // 当前材质
  private currentMaterial: ParsedMaterial | null = null;
  private materials: Map<string, ParsedMaterial> = new Map();

  /**
   * 解析 MTL 文本内容
   * @param text MTL 文件文本
   * @returns 材质名称到材质数据的映射
   */
  parse(text: string): Map<string, ParsedMaterial> {
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

    // 确保最后一个材质被添加
    this.finalizeCurrentMaterial();

    return this.materials;
  }

  /**
   * 重置解析器状态
   */
  private reset(): void {
    this.currentMaterial = null;
    this.materials = new Map();
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
        case 'newmtl':
          this.parseNewMaterial(parts);
          break;
        case 'kd':
          this.parseDiffuseColor(parts);
          break;
        case 'map_kd':
          this.parseDiffuseTexture(parts);
          break;
        case 'd':
          this.parseOpacity(parts);
          break;
        case 'tr':
          this.parseTransparency(parts);
          break;
        default:
          // 忽略不支持的指令 (Ka, Ks, Ns, illum, etc.)
          break;
      }
    } catch (e) {
      // 解析错误，跳过该行
    }
  }

  /**
   * 解析新材质定义 (newmtl name)
   */
  private parseNewMaterial(parts: string[]): void {
    // 完成当前材质
    this.finalizeCurrentMaterial();

    // 创建新材质
    const name = parts.length > 1 ? parts.slice(1).join(' ') : 'default';
    this.currentMaterial = this.createDefaultMaterial(name);
  }

  /**
   * 解析漫反射颜色 (Kd r g b)
   */
  private parseDiffuseColor(parts: string[]): void {
    if (!this.currentMaterial) {
      return;
    }

    if (parts.length < 4) {
      throw new Error('漫反射颜色数据不完整');
    }

    const r = parseFloat(parts[1]);
    const g = parseFloat(parts[2]);
    const b = parseFloat(parts[3]);

    if (isNaN(r) || isNaN(g) || isNaN(b)) {
      throw new Error('无效的漫反射颜色值');
    }

    this.currentMaterial.diffuseColor = [r, g, b];
  }

  /**
   * 解析漫反射纹理路径 (map_Kd path)
   */
  private parseDiffuseTexture(parts: string[]): void {
    if (!this.currentMaterial) {
      return;
    }

    if (parts.length < 2) {
      throw new Error('漫反射纹理路径不完整');
    }

    // 纹理路径可能包含空格，所以需要合并剩余部分
    const texturePath = parts.slice(1).join(' ');
    this.currentMaterial.diffuseTexture = texturePath;
  }

  /**
   * 解析透明度 (d factor)
   * d = 1.0 表示完全不透明，d = 0.0 表示完全透明
   */
  private parseOpacity(parts: string[]): void {
    if (!this.currentMaterial) {
      return;
    }

    if (parts.length < 2) {
      throw new Error('透明度数据不完整');
    }

    const opacity = parseFloat(parts[1]);

    if (isNaN(opacity)) {
      throw new Error('无效的透明度值');
    }

    this.currentMaterial.opacity = opacity;
  }

  /**
   * 解析透明度 (Tr factor)
   * Tr = 0.0 表示完全不透明，Tr = 1.0 表示完全透明
   * 与 d 相反，所以 opacity = 1 - Tr
   */
  private parseTransparency(parts: string[]): void {
    if (!this.currentMaterial) {
      return;
    }

    if (parts.length < 2) {
      throw new Error('透明度数据不完整');
    }

    const transparency = parseFloat(parts[1]);

    if (isNaN(transparency)) {
      throw new Error('无效的透明度值');
    }

    // Tr 是透明度，需要转换为不透明度
    this.currentMaterial.opacity = 1 - transparency;
  }

  /**
   * 创建默认材质
   */
  private createDefaultMaterial(name: string): ParsedMaterial {
    return {
      name,
      diffuseColor: [1, 1, 1],  // 默认白色
      diffuseTexture: null,
      opacity: 1,               // 默认完全不透明
    };
  }

  /**
   * 完成当前材质并添加到映射
   */
  private finalizeCurrentMaterial(): void {
    if (this.currentMaterial) {
      this.materials.set(this.currentMaterial.name, this.currentMaterial);
    }
    this.currentMaterial = null;
  }
}
