/**
 * PLYLoader - 加载 3D Gaussian Splatting 的 PLY 文件
 * 只支持 binary_little_endian 格式
 */

/**
 * CPU 端 Splat 数据结构
 */
export type SplatCPU = {
  mean: [number, number, number];
  scale: [number, number, number];
  rotation: [number, number, number, number];
  colorDC: [number, number, number];
  opacity: number;
  shRest?: Float32Array; // 完整 SH 系数: L1(9) + L2(15) + L3(21) = 45
};

/**
 * PLY 属性信息
 */
type PropertyInfo = {
  name: string;
  type: string;
  byteOffset: number;
};

/**
 * 解析 PLY header
 */
function parseHeader(headerText: string): {
  vertexCount: number;
  properties: PropertyInfo[];
  stride: number;
} {
  const lines = headerText.split("\n");
  let vertexCount = 0;
  const properties: PropertyInfo[] = [];
  let currentOffset = 0;

  for (const line of lines) {
    const trimmed = line.trim();

    // 解析 element vertex N
    if (trimmed.startsWith("element vertex")) {
      const parts = trimmed.split(/\s+/);
      vertexCount = parseInt(parts[2], 10);
    }

    // 解析 property <type> <name>
    if (trimmed.startsWith("property")) {
      const parts = trimmed.split(/\s+/);
      const type = parts[1];
      const name = parts[2];

      properties.push({
        name,
        type,
        byteOffset: currentOffset,
      });

      // 假定所有类型为 float32 (4 bytes)
      currentOffset += 4;
    }
  }

  return {
    vertexCount,
    properties,
    stride: currentOffset,
  };
}

/**
 * 从 ArrayBuffer 中提取 header 文本和数据起始位置
 */
function extractHeader(buffer: ArrayBuffer): {
  headerText: string;
  dataOffset: number;
} {
  const bytes = new Uint8Array(buffer);
  const decoder = new TextDecoder("ascii");

  // 查找 "end_header\n" 标记
  const endHeaderMarker = "end_header\n";
  const maxHeaderSize = Math.min(bytes.length, 10000); // header 不应超过 10KB
  const headerBytes = bytes.slice(0, maxHeaderSize);
  const headerText = decoder.decode(headerBytes);

  const endIndex = headerText.indexOf(endHeaderMarker);
  if (endIndex === -1) {
    throw new Error("无法找到 PLY header 结束标记");
  }

  const dataOffset = endIndex + endHeaderMarker.length;
  return {
    headerText: headerText.substring(0, endIndex),
    dataOffset,
  };
}

/**
 * 构建属性名到偏移量的映射
 */
function buildPropertyMap(
  properties: PropertyInfo[]
): Map<string, number> {
  const map = new Map<string, number>();
  for (const prop of properties) {
    map.set(prop.name, prop.byteOffset);
  }
  return map;
}

/**
 * 从 DataView 读取 float32 值
 */
function readFloat(
  dataView: DataView,
  baseOffset: number,
  propertyOffset: number | undefined
): number {
  if (propertyOffset === undefined) {
    return 0;
  }
  return dataView.getFloat32(baseOffset + propertyOffset, true); // little-endian
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
  const { vertexCount, properties, stride } = parseHeader(headerText);

  console.log(`PLY: ${vertexCount} vertices, stride=${stride} bytes`);

  // 构建属性映射
  const propMap = buildPropertyMap(properties);

  // 获取各属性的偏移量
  const offsets = {
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

  // 收集 f_rest_* 字段并按索引排序（用于 L1 SH 系数）
  const shRestProps = properties
    .filter((p) => p.name.startsWith("f_rest_"))
    .sort((a, b) => {
      const idxA = parseInt(a.name.replace("f_rest_", ""), 10);
      const idxB = parseInt(b.name.replace("f_rest_", ""), 10);
      return idxA - idxB;
    });

  console.log(`PLY: 找到 ${shRestProps.length} 个 f_rest_* SH 系数`);

  // 创建 DataView 用于读取二进制数据
  const dataView = new DataView(buffer, dataOffset);

  // 解析每个 vertex
  const splats: SplatCPU[] = [];
  for (let i = 0; i < vertexCount; i++) {
    const base = i * stride;

    // 读取位置
    const x = readFloat(dataView, base, offsets.x);
    const y = readFloat(dataView, base, offsets.y);
    const z = readFloat(dataView, base, offsets.z);

    // 读取缩放 (exp 转换，因为 3DGS 存储的是 log scale)
    const scale_0 = Math.exp(readFloat(dataView, base, offsets.scale_0));
    const scale_1 = Math.exp(readFloat(dataView, base, offsets.scale_1));
    const scale_2 = Math.exp(readFloat(dataView, base, offsets.scale_2));

    // 读取旋转四元数
    const rot_0 = readFloat(dataView, base, offsets.rot_0);
    const rot_1 = readFloat(dataView, base, offsets.rot_1);
    const rot_2 = readFloat(dataView, base, offsets.rot_2);
    const rot_3 = readFloat(dataView, base, offsets.rot_3);

    // 归一化四元数
    const qlen = Math.sqrt(
      rot_0 * rot_0 + rot_1 * rot_1 + rot_2 * rot_2 + rot_3 * rot_3
    );
    const qnorm = qlen > 0 ? 1 / qlen : 1;

    // 读取颜色 DC 系数 (SH0)
    // 3DGS 存储的是球谐系数，需要转换为 RGB
    // SH0 到 RGB: color = 0.5 + SH_C0 * sh_dc
    // 其中 SH_C0 = 0.28209479177387814
    const SH_C0 = 0.28209479177387814;
    const f_dc_0 = readFloat(dataView, base, offsets.f_dc_0);
    const f_dc_1 = readFloat(dataView, base, offsets.f_dc_1);
    const f_dc_2 = readFloat(dataView, base, offsets.f_dc_2);

    const colorR = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_0));
    const colorG = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_1));
    const colorB = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_2));

    // 读取 opacity (sigmoid 转换)
    const rawOpacity = readFloat(dataView, base, offsets.opacity);
    const opacity = sigmoid(rawOpacity);

    // 读取完整 SH 系数: L1(9) + L2(15) + L3(21) = 45
    // 原始 3DGS PLY 存储顺序是按基函数交错 (RGB, RGB, RGB...)
    // 我们的 shader 需要按通道分组 (RRR..., GGG..., BBB...)
    const shRest = new Float32Array(45);
    
    // L1: 3 个基函数 × 3 通道 = 9 系数
    // 原始: [R0,G0,B0, R1,G1,B1, R2,G2,B2]
    // 目标: [R0,R1,R2, G0,G1,G2, B0,B1,B2]
    for (let basis = 0; basis < 3; basis++) {
      for (let channel = 0; channel < 3; channel++) {
        const srcIdx = basis * 3 + channel;  // 原始顺序
        const dstIdx = channel * 3 + basis;  // 目标顺序
        if (srcIdx < shRestProps.length) {
          shRest[dstIdx] = readFloat(dataView, base, shRestProps[srcIdx].byteOffset);
        }
      }
    }
    
    // L2: 5 个基函数 × 3 通道 = 15 系数
    // 原始: [R0,G0,B0, R1,G1,B1, R2,G2,B2, R3,G3,B3, R4,G4,B4]
    // 目标: [R0,R1,R2,R3,R4, G0,G1,G2,G3,G4, B0,B1,B2,B3,B4]
    for (let basis = 0; basis < 5; basis++) {
      for (let channel = 0; channel < 3; channel++) {
        const srcIdx = 9 + basis * 3 + channel;  // 原始顺序 (从 f_rest_9 开始)
        const dstIdx = 9 + channel * 5 + basis;  // 目标顺序
        if (srcIdx < shRestProps.length) {
          shRest[dstIdx] = readFloat(dataView, base, shRestProps[srcIdx].byteOffset);
        }
      }
    }
    
    // L3: 7 个基函数 × 3 通道 = 21 系数
    // 原始: [R0,G0,B0, R1,G1,B1, ...]
    // 目标: [R0,R1,...,R6, G0,G1,...,G6, B0,B1,...,B6]
    for (let basis = 0; basis < 7; basis++) {
      for (let channel = 0; channel < 3; channel++) {
        const srcIdx = 24 + basis * 3 + channel;  // 原始顺序 (从 f_rest_24 开始)
        const dstIdx = 24 + channel * 7 + basis;  // 目标顺序
        if (srcIdx < shRestProps.length) {
          shRest[dstIdx] = readFloat(dataView, base, shRestProps[srcIdx].byteOffset);
        }
      }
    }

    splats.push({
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
    });
  }

  return splats;
}
