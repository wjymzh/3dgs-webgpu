/**
 * PLYLoader - 加载 3D Gaussian Splatting 的 PLY 文件
 * 支持 binary_little_endian 和 binary_big_endian 格式
 * 支持多种数据类型: float, double, int, uint, char, uchar, short, ushort
 */

/**
 * CPU 端 Splat 数据结构
 */
export type SplatCPU = {
  mean: [number, number, number];
  scale: [number, number, number];
  rotation: [number, number, number, number]; // 四元数顺序: [w, x, y, z]
  colorDC: [number, number, number];
  opacity: number;
  shRest?: Float32Array; // 完整 SH 系数: L1(9) + L2(15) + L3(21) = 45
};

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
 * PLY 属性信息
 */
type PropertyInfo = {
  name: string;
  type: string;
  byteOffset: number;
  byteSize: number;
};

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
      // 其他 element 类型，停止收集属性
      inVertexElement = false;
    }

    // 解析 property <type> <name>（只收集 vertex element 的属性）
    if (inVertexElement && trimmed.startsWith("property")) {
      const parts = trimmed.split(/\s+/);
      // 跳过 list 类型属性
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

  return {
    vertexCount,
    properties,
    stride: currentOffset,
    format,
  };
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
 * 从 ArrayBuffer 中提取 header 文本和数据起始位置
 */
function extractHeader(buffer: ArrayBuffer): {
  headerText: string;
  dataOffset: number;
} {
  // 首先验证魔数
  validatePLYMagic(buffer);

  const bytes = new Uint8Array(buffer);
  const decoder = new TextDecoder("ascii");

  // 查找 "end_header\n" 或 "end_header\r\n" 标记
  const maxHeaderSize = Math.min(bytes.length, 10000); // header 不应超过 10KB
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
 * 构建属性名到属性信息的映射
 */
function buildPropertyMap(
  properties: PropertyInfo[]
): Map<string, PropertyInfo> {
  const map = new Map<string, PropertyInfo>();
  for (const prop of properties) {
    map.set(prop.name, prop);
  }
  return map;
}

/**
 * 从 DataView 读取属性值（支持多种数据类型）
 */
function readProperty(
  dataView: DataView,
  baseOffset: number,
  prop: PropertyInfo | undefined,
  littleEndian: boolean
): number {
  if (prop === undefined) {
    return 0;
  }

  const offset = baseOffset + prop.byteOffset;

  switch (prop.type) {
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
      console.warn(`未知属性类型: ${prop.type}，使用 float32`);
      return dataView.getFloat32(offset, littleEndian);
  }
}

/**
 * Sigmoid 函数，用于将 opacity 从原始值转换为 [0, 1]
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * 加载并解析 PLY 文件
 * @param url PLY 文件的 URL
 * @returns SplatCPU 数组
 */
export async function loadPLY(url: string): Promise<SplatCPU[]> {
  // 获取文件
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`无法加载 PLY 文件: ${url}`);
  }
  const buffer = await response.arrayBuffer();

  // 解析 header
  const { headerText, dataOffset } = extractHeader(buffer);
  const { vertexCount, properties, stride, format } = parseHeader(headerText);

  // 验证格式
  if (format === "ascii") {
    throw new Error("不支持 ASCII 格式的 PLY 文件，请使用 binary_little_endian 或 binary_big_endian 格式");
  }

  const littleEndian = format === "binary_little_endian";
  console.log(`PLY: ${vertexCount} vertices, stride=${stride} bytes, format=${format}`);

  // 构建属性映射
  const propMap = buildPropertyMap(properties);

  // 获取各属性的信息
  const props = {
    x: propMap.get("x"),
    y: propMap.get("y"),
    z: propMap.get("z"),
    scale_0: propMap.get("scale_0"),
    scale_1: propMap.get("scale_1"),
    scale_2: propMap.get("scale_2"),
    rot_0: propMap.get("rot_0"),
    rot_1: propMap.get("rot_1"),
    rot_2: propMap.get("rot_2"),
    rot_3: propMap.get("rot_3"),
    f_dc_0: propMap.get("f_dc_0"),
    f_dc_1: propMap.get("f_dc_1"),
    f_dc_2: propMap.get("f_dc_2"),
    opacity: propMap.get("opacity"),
  };

  // 收集 f_rest_* 字段并按索引排序（用于 SH 系数）
  const shRestProps = properties
    .filter((p) => p.name.startsWith("f_rest_"))
    .sort((a, b) => {
      const idxA = parseInt(a.name.replace("f_rest_", ""), 10);
      const idxB = parseInt(b.name.replace("f_rest_", ""), 10);
      return idxA - idxB;
    });

  console.log(`PLY: 找到 ${shRestProps.length} 个 f_rest_* SH 系数`);
  
  // 打印前几个 f_rest_* 属性的名称，验证排序是否正确
  if (shRestProps.length > 0) {
    console.log(`PLY: f_rest_* 属性顺序: ${shRestProps.slice(0, 5).map(p => p.name).join(', ')}...`);
  }

  // 创建 DataView 用于读取二进制数据
  const dataView = new DataView(buffer, dataOffset);

  // 预分配共享的 SH 系数数组（优化内存）
  const shRestBuffer = new Float32Array(vertexCount * 45);

  // 解析每个 vertex
  const splats: SplatCPU[] = new Array(vertexCount);
  const SH_C0 = 0.28209479177387814;

  for (let i = 0; i < vertexCount; i++) {
    const base = i * stride;

    // 读取位置
    const x = readProperty(dataView, base, props.x, littleEndian);
    const y = readProperty(dataView, base, props.y, littleEndian);
    const z = readProperty(dataView, base, props.z, littleEndian);

    // 读取缩放 (exp 转换，因为 3DGS 存储的是 log scale)
    const scale_0 = Math.exp(readProperty(dataView, base, props.scale_0, littleEndian));
    const scale_1 = Math.exp(readProperty(dataView, base, props.scale_1, littleEndian));
    const scale_2 = Math.exp(readProperty(dataView, base, props.scale_2, littleEndian));

    // 读取旋转四元数
    const rot_0 = readProperty(dataView, base, props.rot_0, littleEndian);
    const rot_1 = readProperty(dataView, base, props.rot_1, littleEndian);
    const rot_2 = readProperty(dataView, base, props.rot_2, littleEndian);
    const rot_3 = readProperty(dataView, base, props.rot_3, littleEndian);

    // 归一化四元数
    const qlen = Math.sqrt(
      rot_0 * rot_0 + rot_1 * rot_1 + rot_2 * rot_2 + rot_3 * rot_3
    );
    const qnorm = qlen > 0 ? 1 / qlen : 1;

    // 读取颜色 DC 系数 (SH0)
    // 注意：不要在这里 clamp，因为 SH 贡献可能是负数
    // 最终颜色会在 shader 中 clamp
    const f_dc_0 = readProperty(dataView, base, props.f_dc_0, littleEndian);
    const f_dc_1 = readProperty(dataView, base, props.f_dc_1, littleEndian);
    const f_dc_2 = readProperty(dataView, base, props.f_dc_2, littleEndian);

    const colorR = 0.5 + SH_C0 * f_dc_0;
    const colorG = 0.5 + SH_C0 * f_dc_1;
    const colorB = 0.5 + SH_C0 * f_dc_2;

    // 读取 opacity (sigmoid 转换)
    const rawOpacity = readProperty(dataView, base, props.opacity, littleEndian);
    const opacity = sigmoid(rawOpacity);

    // 读取完整 SH 系数到共享 buffer
    const shOffset = i * 45;
    const shRest = shRestBuffer.subarray(shOffset, shOffset + 45);

    // PLY 文件中 f_rest_* 的顺序是 channel-first (参考 visionary):
    // [R0..R14, G0..G14, B0..B14] - 每通道 15 个系数
    // 
    // 我们转换为 interleaved 格式 (与 visionary 一致):
    // [R0,G0,B0, R1,G1,B1, ...] - 每个基函数的 RGB 连续存储
    //
    // 这样 shader 中可以用 sh_coef(idx) 返回 vec3(R,G,B)
    //
    // 注意：shRestProps 的数量可能是 45 (完整 SH) 或更少
    // 实际 PLY 文件的 f_rest_* 数量 = (shDegree+1)^2 - 1 个系数 × 3 通道
    // 例如 shDegree=3 时: (3+1)^2 - 1 = 15 个系数/通道，共 45 个 f_rest_*

    const totalRestCount = shRestProps.length;
    const perChannel = Math.floor(totalRestCount / 3); // 每通道的系数数量
    
    for (let coefIdx = 0; coefIdx < perChannel; coefIdx++) {
      // PLY 中: R 在 [0..perChannel-1], G 在 [perChannel..2*perChannel-1], B 在 [2*perChannel..3*perChannel-1]
      const srcR = coefIdx;
      const srcG = perChannel + coefIdx;
      const srcB = 2 * perChannel + coefIdx;
      
      // 目标: interleaved [R0,G0,B0, R1,G1,B1, ...]
      const dstBase = coefIdx * 3;
      
      shRest[dstBase + 0] = srcR < shRestProps.length ? readProperty(dataView, base, shRestProps[srcR], littleEndian) : 0;
      shRest[dstBase + 1] = srcG < shRestProps.length ? readProperty(dataView, base, shRestProps[srcG], littleEndian) : 0;
      shRest[dstBase + 2] = srcB < shRestProps.length ? readProperty(dataView, base, shRestProps[srcB], littleEndian) : 0;
    }

    // 打印第一个点的 SH 系数用于调试
    if (i === 0) {
      console.log(`PLY: 第一个点的 L1 SH 系数 (前9个值):`, 
        `[${shRest[0].toFixed(4)}, ${shRest[1].toFixed(4)}, ${shRest[2].toFixed(4)}]`,
        `[${shRest[3].toFixed(4)}, ${shRest[4].toFixed(4)}, ${shRest[5].toFixed(4)}]`,
        `[${shRest[6].toFixed(4)}, ${shRest[7].toFixed(4)}, ${shRest[8].toFixed(4)}]`
      );
      console.log(`PLY: 第一个点的 DC 颜色:`, `[${colorR.toFixed(4)}, ${colorG.toFixed(4)}, ${colorB.toFixed(4)}]`);
    }

    splats[i] = {
      mean: [x, y, z],
      scale: [scale_0, scale_1, scale_2],
      rotation: [
        rot_0 * qnorm,
        rot_1 * qnorm,
        rot_2 * qnorm,
        rot_3 * qnorm,
      ],
      colorDC: [colorR, colorG, colorB],
      opacity,
      shRest,
    };
  }

  return splats;
}
