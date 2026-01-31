/**
 * PLYLoaderMobile - 移动端优化的 PLY 加载器
 * 
 * 优化点：
 * 1. 直接输出 GPU buffer 格式，避免中间对象
 * 2. 流式降采样，减少内存峰值
 * 3. 可选跳过 SH 系数（移动端 L0 模式）
 * 4. 使用 TypedArray 而非对象数组
 * 5. 支持多种 PLY 数据类型
 * 6. 确定性采样（基于文件内容的种子）
 */

/**
 * PLY 数据类型到字节大小的映射
 */
const TYPE_SIZES: Record<string, number> = {
  char: 1,
  uchar: 1,
  int8: 1,
  uint8: 1,
  short: 2,
  ushort: 2,
  int16: 2,
  uint16: 2,
  int: 4,
  uint: 4,
  int32: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
};

/**
 * PLY 文件格式类型
 */
type PLYFormat = "binary_little_endian" | "binary_big_endian" | "ascii";

/**
 * 移动端加载配置
 */
export interface MobileLoadOptions {
  /** 最大 splat 数量，超过则降采样 */
  maxSplats?: number;
  /** 是否加载 SH 系数（false 时只加载 DC 颜色） */
  loadSH?: boolean;
  /** 进度回调 */
  onProgress?: (loaded: number, total: number) => void;
  /** 随机种子（用于确定性采样，默认使用文件大小作为种子） */
  seed?: number;
}

/**
 * 紧凑 Splat 数据（用于移动端）
 * 直接返回 Float32Array，而非对象数组
 */
export interface CompactSplatData {
  /** splat 数量 */
  count: number;
  /** 位置数据 Float32Array [x,y,z, x,y,z, ...] */
  positions: Float32Array;
  /** 缩放数据 Float32Array [sx,sy,sz, sx,sy,sz, ...] */
  scales: Float32Array;
  /** 旋转四元数 Float32Array [w,x,y,z, w,x,y,z, ...] */
  rotations: Float32Array;
  /** DC 颜色 Float32Array [r,g,b, r,g,b, ...] */
  colors: Float32Array;
  /** 不透明度 Float32Array [a, a, ...] */
  opacities: Float32Array;
  /** SH 系数（可选）Float32Array，每个 splat 45 个系数 */
  shCoeffs?: Float32Array;
}

/**
 * PLY 属性信息
 */
interface PropertyInfo {
  name: string;
  type: string;
  byteOffset: number;
  byteSize: number;
}

/**
 * 验证 PLY 文件魔数
 */
function validatePLYMagic(buffer: ArrayBuffer): void {
  const bytes = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 10));
  const decoder = new TextDecoder("ascii");
  const header = decoder.decode(bytes);

  if (!header.startsWith("ply")) {
    throw new Error("无效的 PLY 文件: 缺少 'ply' 魔数标识");
  }
}

/**
 * 解析 PLY header
 */
function parseHeader(headerText: string): {
  vertexCount: number;
  properties: PropertyInfo[];
  stride: number;
  format: PLYFormat;
} {
  const lines = headerText.split("\n");
  let vertexCount = 0;
  let format: PLYFormat = "binary_little_endian";
  const properties: PropertyInfo[] = [];
  let currentOffset = 0;
  let inVertexElement = false;

  for (const line of lines) {
    const trimmed = line.trim();

    // 解析格式
    if (trimmed.startsWith("format ")) {
      const parts = trimmed.split(/\s+/);
      const formatStr = parts[1];
      if (formatStr === "ascii") {
        format = "ascii";
      } else if (formatStr === "binary_big_endian") {
        format = "binary_big_endian";
      } else if (formatStr === "binary_little_endian") {
        format = "binary_little_endian";
      } else {
        throw new Error(`不支持的 PLY 格式: ${formatStr}`);
      }
    }

    // 解析 element vertex N
    if (trimmed.startsWith("element vertex")) {
      const parts = trimmed.split(/\s+/);
      vertexCount = parseInt(parts[2], 10);
      inVertexElement = true;
    } else if (trimmed.startsWith("element ")) {
      inVertexElement = false;
    }

    // 解析 property（只收集 vertex element 的属性）
    if (inVertexElement && trimmed.startsWith("property")) {
      const parts = trimmed.split(/\s+/);
      if (parts[1] === "list") {
        console.warn(`PLY: 跳过 list 类型属性: ${trimmed}`);
        continue;
      }
      const type = parts[1];
      const name = parts[2];
      const byteSize = TYPE_SIZES[type];

      if (byteSize === undefined) {
        throw new Error(`不支持的 PLY 属性类型: ${type}`);
      }

      properties.push({
        name,
        type,
        byteOffset: currentOffset,
        byteSize,
      });

      currentOffset += byteSize;
    }
  }

  return { vertexCount, properties, stride: currentOffset, format };
}

/**
 * 提取 header
 */
function extractHeader(buffer: ArrayBuffer): {
  headerText: string;
  dataOffset: number;
} {
  // 首先验证魔数
  validatePLYMagic(buffer);

  const bytes = new Uint8Array(buffer);
  const decoder = new TextDecoder("ascii");

  const maxHeaderSize = Math.min(bytes.length, 10000);
  const headerBytes = bytes.slice(0, maxHeaderSize);
  const headerText = decoder.decode(headerBytes);

  // 支持 Unix (\n) 和 Windows (\r\n) 换行符
  let endIndex = headerText.indexOf("end_header\n");
  let markerLength = "end_header\n".length;

  if (endIndex === -1) {
    endIndex = headerText.indexOf("end_header\r\n");
    markerLength = "end_header\r\n".length;
  }

  if (endIndex === -1) {
    throw new Error("无法找到 PLY header 结束标记 'end_header'");
  }

  const dataOffset = endIndex + markerLength;
  return {
    headerText: headerText.substring(0, endIndex),
    dataOffset,
  };
}

/**
 * Sigmoid 函数
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * 从 DataView 读取属性值（支持多种数据类型）
 */
function readProperty(
  dataView: DataView,
  offset: number,
  type: string,
  littleEndian: boolean
): number {
  switch (type) {
    case "float":
    case "float32":
      return dataView.getFloat32(offset, littleEndian);
    case "double":
    case "float64":
      return dataView.getFloat64(offset, littleEndian);
    case "int":
    case "int32":
      return dataView.getInt32(offset, littleEndian);
    case "uint":
    case "uint32":
      return dataView.getUint32(offset, littleEndian);
    case "short":
    case "int16":
      return dataView.getInt16(offset, littleEndian);
    case "ushort":
    case "uint16":
      return dataView.getUint16(offset, littleEndian);
    case "char":
    case "int8":
      return dataView.getInt8(offset);
    case "uchar":
    case "uint8":
      return dataView.getUint8(offset);
    default:
      return dataView.getFloat32(offset, littleEndian);
  }
}

/**
 * 简单的确定性伪随机数生成器 (Mulberry32)
 * 给定相同的种子，总是产生相同的序列
 */
function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SH_C0 = 0.28209479177387814;

/**
 * 智能采样：基于重要性（opacity * scale）采样
 * 使用 Floyd 采样算法 + 确定性种子，保证相同文件产生相同结果
 * 时间复杂度 O(n)，空间复杂度 O(k)，其中 k 是采样数量
 */
function computeImportanceSampling(
  buffer: ArrayBuffer,
  dataOffset: number,
  stride: number,
  totalCount: number,
  sampleCount: number,
  opacityOffset: number,
  opacityType: string,
  scale0Offset: number,
  scale0Type: string,
  scale1Offset: number,
  scale1Type: string,
  scale2Offset: number,
  scale2Type: string,
  littleEndian: boolean,
  seed: number
): Uint32Array {
  const dataView = new DataView(buffer, dataOffset);
  const random = createSeededRandom(seed);

  // 计算每个 splat 的重要性分数
  const importance = new Float32Array(totalCount);
  let totalImportance = 0;

  for (let i = 0; i < totalCount; i++) {
    const base = i * stride;

    // 获取 opacity（需要 sigmoid 转换）
    const rawOpacity = opacityOffset >= 0
      ? readProperty(dataView, base + opacityOffset, opacityType, littleEndian)
      : 0;
    const opacity = sigmoid(rawOpacity);

    // 获取 scale（需要 exp 转换）
    const s0 = scale0Offset >= 0
      ? Math.exp(readProperty(dataView, base + scale0Offset, scale0Type, littleEndian))
      : 1;
    const s1 = scale1Offset >= 0
      ? Math.exp(readProperty(dataView, base + scale1Offset, scale1Type, littleEndian))
      : 1;
    const s2 = scale2Offset >= 0
      ? Math.exp(readProperty(dataView, base + scale2Offset, scale2Type, littleEndian))
      : 1;
    const maxScale = Math.max(s0, s1, s2);

    // 重要性分数：opacity * scale
    importance[i] = opacity * maxScale;
    totalImportance += importance[i];
  }

  // 使用加权随机采样（Reservoir Sampling with Weights）
  // 这是一种 O(n) 的算法，比完整排序更高效
  const result = new Uint32Array(sampleCount);
  const weights = new Float32Array(sampleCount);

  // 初始化：填充前 sampleCount 个元素
  for (let i = 0; i < Math.min(sampleCount, totalCount); i++) {
    result[i] = i;
    // 使用 -log(random) / weight 作为键值（Efraimidis-Spirakis 算法）
    weights[i] = importance[i] > 0 ? -Math.log(random()) / importance[i] : Infinity;
  }

  // 构建最小堆（用于维护 top-k）
  // 简化实现：直接找最大权重的位置
  let maxWeightIdx = 0;
  let maxWeight = weights[0];
  for (let i = 1; i < sampleCount; i++) {
    if (weights[i] > maxWeight) {
      maxWeight = weights[i];
      maxWeightIdx = i;
    }
  }

  // 处理剩余元素
  for (let i = sampleCount; i < totalCount; i++) {
    const key = importance[i] > 0 ? -Math.log(random()) / importance[i] : Infinity;

    // 如果当前元素的键值小于堆中最大的键值，替换它
    if (key < maxWeight) {
      result[maxWeightIdx] = i;
      weights[maxWeightIdx] = key;

      // 重新找最大权重
      maxWeight = weights[0];
      maxWeightIdx = 0;
      for (let j = 1; j < sampleCount; j++) {
        if (weights[j] > maxWeight) {
          maxWeight = weights[j];
          maxWeightIdx = j;
        }
      }
    }
  }

  // 按原始顺序排列（保持空间局部性，有利于缓存）
  result.sort((a, b) => a - b);

  return result;
}

/**
 * 移动端优化的 PLY 加载器
 * 直接输出紧凑格式，避免创建大量中间对象
 */
export async function loadPLYMobile(
  url: string,
  options: MobileLoadOptions = {}
): Promise<CompactSplatData> {
  // 获取文件
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`无法加载 PLY 文件: ${url}`);
  }
  const buffer = await response.arrayBuffer();

  return parsePLYBuffer(buffer, options);
}

/**
 * 解析 PLY ArrayBuffer
 * 可以直接传入 ArrayBuffer 进行解析，用于本地文件加载
 */
export async function parsePLYBuffer(
  buffer: ArrayBuffer,
  options: MobileLoadOptions = {}
): Promise<CompactSplatData> {
  const {
    maxSplats = 200000,
    loadSH = false,
    onProgress,
  } = options;

  // 使用文件大小作为默认种子（确保相同文件产生相同结果）
  const seed = options.seed ?? buffer.byteLength;

  // 解析 header
  const { headerText, dataOffset } = extractHeader(buffer);
  const { vertexCount, properties, stride, format } = parseHeader(headerText);

  // 验证格式
  if (format === "ascii") {
    throw new Error("不支持 ASCII 格式的 PLY 文件，请使用 binary_little_endian 或 binary_big_endian 格式");
  }

  const littleEndian = format === "binary_little_endian";
  console.log(`PLYLoaderMobile: ${vertexCount} vertices, stride=${stride} bytes, format=${format}`);

  // 构建属性偏移映射
  const propMap = new Map<string, PropertyInfo>();
  for (const prop of properties) {
    propMap.set(prop.name, prop);
  }

  // 获取基本属性信息
  const getProp = (name: string) => propMap.get(name);
  const getOffset = (name: string) => propMap.get(name)?.byteOffset ?? -1;
  const getType = (name: string) => propMap.get(name)?.type ?? "float";

  const offsets = {
    x: getOffset("x"),
    y: getOffset("y"),
    z: getOffset("z"),
    scale_0: getOffset("scale_0"),
    scale_1: getOffset("scale_1"),
    scale_2: getOffset("scale_2"),
    rot_0: getOffset("rot_0"),
    rot_1: getOffset("rot_1"),
    rot_2: getOffset("rot_2"),
    rot_3: getOffset("rot_3"),
    f_dc_0: getOffset("f_dc_0"),
    f_dc_1: getOffset("f_dc_1"),
    f_dc_2: getOffset("f_dc_2"),
    opacity: getOffset("opacity"),
  };

  const types = {
    x: getType("x"),
    y: getType("y"),
    z: getType("z"),
    scale_0: getType("scale_0"),
    scale_1: getType("scale_1"),
    scale_2: getType("scale_2"),
    rot_0: getType("rot_0"),
    rot_1: getType("rot_1"),
    rot_2: getType("rot_2"),
    rot_3: getType("rot_3"),
    f_dc_0: getType("f_dc_0"),
    f_dc_1: getType("f_dc_1"),
    f_dc_2: getType("f_dc_2"),
    opacity: getType("opacity"),
  };

  // SH 系数属性（可选）
  let shProps: PropertyInfo[] = [];
  if (loadSH) {
    shProps = properties
      .filter((p) => p.name.startsWith("f_rest_"))
      .sort((a, b) => {
        const idxA = parseInt(a.name.replace("f_rest_", ""), 10);
        const idxB = parseInt(b.name.replace("f_rest_", ""), 10);
        return idxA - idxB;
      });
    console.log(`PLYLoaderMobile: 加载 ${shProps.length} 个 SH 系数`);
  }

  // 计算实际加载数量
  const needSample = vertexCount > maxSplats;
  const actualCount = Math.min(vertexCount, maxSplats);
  
  // 估算内存使用（纹理压缩模式约 52 bytes/splat）
  const estimatedMemoryMB = (actualCount * 52) / (1024 * 1024);
  console.log(`PLYLoaderMobile: 预估 GPU 内存 = ${estimatedMemoryMB.toFixed(1)} MB (${actualCount.toLocaleString()} splats)`);
  
  if (estimatedMemoryMB > 300) {
    console.warn(`⚠️ 警告: GPU 内存占用较高 (${estimatedMemoryMB.toFixed(0)} MB)，移动端可能崩溃！`);
  }
  
  // 如果需要降采样，使用智能采样（基于重要性，确定性种子）
  let sampleIndices: Uint32Array | null = null;
  if (needSample) {
    console.log(`PLYLoaderMobile: 开始智能采样 ${vertexCount} -> ${actualCount} (${(actualCount/vertexCount*100).toFixed(1)}%), seed=${seed}`);
    sampleIndices = computeImportanceSampling(
      buffer, dataOffset, stride, vertexCount, actualCount,
      offsets.opacity, types.opacity,
      offsets.scale_0, types.scale_0,
      offsets.scale_1, types.scale_1,
      offsets.scale_2, types.scale_2,
      littleEndian, seed
    );
    console.log(`PLYLoaderMobile: 智能采样完成`);
  }

  // 预分配输出数组（一次性分配，避免多次扩容）
  const positions = new Float32Array(actualCount * 3);
  const scales = new Float32Array(actualCount * 3);
  const rotations = new Float32Array(actualCount * 4);
  const colors = new Float32Array(actualCount * 3);
  const opacities = new Float32Array(actualCount);
  const shCoeffs = loadSH ? new Float32Array(actualCount * 45) : undefined;

  // 创建 DataView
  const dataView = new DataView(buffer, dataOffset);

  // 流式解析
  let outputIdx = 0;
  let lastProgress = 0;

  for (let i = 0; i < actualCount; i++) {
    // 计算源索引：使用智能采样或直接索引
    const srcIdx = sampleIndices ? sampleIndices[i] : i;
    const base = srcIdx * stride;

    // 位置
    positions[outputIdx * 3 + 0] = offsets.x >= 0 ? readProperty(dataView, base + offsets.x, types.x, littleEndian) : 0;
    positions[outputIdx * 3 + 1] = offsets.y >= 0 ? readProperty(dataView, base + offsets.y, types.y, littleEndian) : 0;
    positions[outputIdx * 3 + 2] = offsets.z >= 0 ? readProperty(dataView, base + offsets.z, types.z, littleEndian) : 0;

    // 缩放（exp 转换）
    scales[outputIdx * 3 + 0] = offsets.scale_0 >= 0 ? Math.exp(readProperty(dataView, base + offsets.scale_0, types.scale_0, littleEndian)) : 1;
    scales[outputIdx * 3 + 1] = offsets.scale_1 >= 0 ? Math.exp(readProperty(dataView, base + offsets.scale_1, types.scale_1, littleEndian)) : 1;
    scales[outputIdx * 3 + 2] = offsets.scale_2 >= 0 ? Math.exp(readProperty(dataView, base + offsets.scale_2, types.scale_2, littleEndian)) : 1;

    // 旋转四元数（归一化）
    const rot_0 = offsets.rot_0 >= 0 ? readProperty(dataView, base + offsets.rot_0, types.rot_0, littleEndian) : 1;
    const rot_1 = offsets.rot_1 >= 0 ? readProperty(dataView, base + offsets.rot_1, types.rot_1, littleEndian) : 0;
    const rot_2 = offsets.rot_2 >= 0 ? readProperty(dataView, base + offsets.rot_2, types.rot_2, littleEndian) : 0;
    const rot_3 = offsets.rot_3 >= 0 ? readProperty(dataView, base + offsets.rot_3, types.rot_3, littleEndian) : 0;
    const qlen = Math.sqrt(rot_0 * rot_0 + rot_1 * rot_1 + rot_2 * rot_2 + rot_3 * rot_3);
    const qnorm = qlen > 0 ? 1 / qlen : 1;
    rotations[outputIdx * 4 + 0] = rot_0 * qnorm;
    rotations[outputIdx * 4 + 1] = rot_1 * qnorm;
    rotations[outputIdx * 4 + 2] = rot_2 * qnorm;
    rotations[outputIdx * 4 + 3] = rot_3 * qnorm;

    // DC 颜色（SH0 -> RGB）
    // 注意：不要在这里 clamp，因为 SH 贡献可能是负数
    // 最终颜色会在 shader 中 clamp
    const f_dc_0 = offsets.f_dc_0 >= 0 ? readProperty(dataView, base + offsets.f_dc_0, types.f_dc_0, littleEndian) : 0;
    const f_dc_1 = offsets.f_dc_1 >= 0 ? readProperty(dataView, base + offsets.f_dc_1, types.f_dc_1, littleEndian) : 0;
    const f_dc_2 = offsets.f_dc_2 >= 0 ? readProperty(dataView, base + offsets.f_dc_2, types.f_dc_2, littleEndian) : 0;
    colors[outputIdx * 3 + 0] = 0.5 + SH_C0 * f_dc_0;
    colors[outputIdx * 3 + 1] = 0.5 + SH_C0 * f_dc_1;
    colors[outputIdx * 3 + 2] = 0.5 + SH_C0 * f_dc_2;

    // 不透明度（sigmoid）
    const rawOpacity = offsets.opacity >= 0 ? readProperty(dataView, base + offsets.opacity, types.opacity, littleEndian) : 0;
    opacities[outputIdx] = sigmoid(rawOpacity);

    // SH 系数（可选）
    // PLY 文件中 f_rest_* 的顺序是 channel-first:
    // [R0..R14, G0..G14, B0..B14] - 每通道 15 个系数
    // 我们转换为 interleaved 格式: [R0,G0,B0, R1,G1,B1, ...]
    if (shCoeffs && shProps.length > 0) {
      const shBase = outputIdx * 45;
      const perChannel = Math.floor(shProps.length / 3); // 每通道的系数数量
      
      for (let coefIdx = 0; coefIdx < perChannel && coefIdx < 15; coefIdx++) {
        // PLY 中: R 在 [0..perChannel-1], G 在 [perChannel..2*perChannel-1], B 在 [2*perChannel..3*perChannel-1]
        const srcR = coefIdx;
        const srcG = perChannel + coefIdx;
        const srcB = 2 * perChannel + coefIdx;
        
        // 目标: interleaved [R0,G0,B0, R1,G1,B1, ...]
        const dstBase = coefIdx * 3;
        
        if (srcR < shProps.length) {
          const prop = shProps[srcR];
          shCoeffs[shBase + dstBase + 0] = readProperty(dataView, base + prop.byteOffset, prop.type, littleEndian);
        }
        if (srcG < shProps.length) {
          const prop = shProps[srcG];
          shCoeffs[shBase + dstBase + 1] = readProperty(dataView, base + prop.byteOffset, prop.type, littleEndian);
        }
        if (srcB < shProps.length) {
          const prop = shProps[srcB];
          shCoeffs[shBase + dstBase + 2] = readProperty(dataView, base + prop.byteOffset, prop.type, littleEndian);
        }
      }
    }

    outputIdx++;

    // 进度回调
    if (onProgress) {
      const progress = Math.floor((i / actualCount) * 100);
      if (progress > lastProgress) {
        lastProgress = progress;
        onProgress(i, actualCount);
      }
    }
  }

  console.log(`PLYLoaderMobile: 加载完成，共 ${outputIdx} 个 splats`);

  return {
    count: outputIdx,
    positions,
    scales,
    rotations,
    colors,
    opacities,
    shCoeffs,
  };
}

/**
 * 将 CompactSplatData 转换为 GPU buffer 格式
 * 直接输出可以上传到 GPU 的 Float32Array
 * 
 * @param data 紧凑 splat 数据
 * @param includeFullSH 是否包含完整 SH 系数（256 字节/splat），否则只包含基本数据（64 字节/splat）
 */
export function compactDataToGPUBuffer(
  data: CompactSplatData,
  includeFullSH: boolean = false
): Float32Array {
  const count = data.count;

  if (includeFullSH) {
    // 完整格式：256 字节/splat = 64 floats
    const buffer = new Float32Array(count * 64);

    for (let i = 0; i < count; i++) {
      const offset = i * 64;

      // mean (vec3) + padding
      buffer[offset + 0] = data.positions[i * 3 + 0];
      buffer[offset + 1] = data.positions[i * 3 + 1];
      buffer[offset + 2] = data.positions[i * 3 + 2];
      buffer[offset + 3] = 0;

      // scale (vec3) + padding
      buffer[offset + 4] = data.scales[i * 3 + 0];
      buffer[offset + 5] = data.scales[i * 3 + 1];
      buffer[offset + 6] = data.scales[i * 3 + 2];
      buffer[offset + 7] = 0;

      // rotation (vec4)
      buffer[offset + 8] = data.rotations[i * 4 + 0];
      buffer[offset + 9] = data.rotations[i * 4 + 1];
      buffer[offset + 10] = data.rotations[i * 4 + 2];
      buffer[offset + 11] = data.rotations[i * 4 + 3];

      // colorDC (vec3) + opacity
      buffer[offset + 12] = data.colors[i * 3 + 0];
      buffer[offset + 13] = data.colors[i * 3 + 1];
      buffer[offset + 14] = data.colors[i * 3 + 2];
      buffer[offset + 15] = data.opacities[i];

      // SH 系数
      if (data.shCoeffs) {
        const shBase = i * 45;
        // sh1 (9 floats)
        for (let j = 0; j < 9; j++) {
          buffer[offset + 16 + j] = data.shCoeffs[shBase + j];
        }
        // sh2 (15 floats)
        for (let j = 0; j < 15; j++) {
          buffer[offset + 25 + j] = data.shCoeffs[shBase + 9 + j];
        }
        // sh3 (21 floats)
        for (let j = 0; j < 21; j++) {
          buffer[offset + 40 + j] = data.shCoeffs[shBase + 24 + j];
        }
      }
      // padding 已经是 0（Float32Array 默认初始化为 0）
    }

    return buffer;
  } else {
    // 紧凑格式：64 字节/splat = 16 floats（只包含基本渲染数据）
    const buffer = new Float32Array(count * 64); // 保持 256 字节对齐，但只填充基本数据
    
    for (let i = 0; i < count; i++) {
      const offset = i * 64;
      
      // mean (vec3) + padding
      buffer[offset + 0] = data.positions[i * 3 + 0];
      buffer[offset + 1] = data.positions[i * 3 + 1];
      buffer[offset + 2] = data.positions[i * 3 + 2];
      buffer[offset + 3] = 0;
      
      // scale (vec3) + padding
      buffer[offset + 4] = data.scales[i * 3 + 0];
      buffer[offset + 5] = data.scales[i * 3 + 1];
      buffer[offset + 6] = data.scales[i * 3 + 2];
      buffer[offset + 7] = 0;
      
      // rotation (vec4)
      buffer[offset + 8] = data.rotations[i * 4 + 0];
      buffer[offset + 9] = data.rotations[i * 4 + 1];
      buffer[offset + 10] = data.rotations[i * 4 + 2];
      buffer[offset + 11] = data.rotations[i * 4 + 3];
      
      // colorDC (vec3) + opacity
      buffer[offset + 12] = data.colors[i * 3 + 0];
      buffer[offset + 13] = data.colors[i * 3 + 1];
      buffer[offset + 14] = data.colors[i * 3 + 2];
      buffer[offset + 15] = data.opacities[i];
      
      // 其余保持为 0（SH 系数为空）
    }
    
    return buffer;
  }
}
