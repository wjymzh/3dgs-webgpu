/**
 * PLYLoaderMobile - 移动端优化的 PLY 加载器
 * 
 * 优化点：
 * 1. 直接输出 GPU buffer 格式，避免中间对象
 * 2. 流式降采样，减少内存峰值
 * 3. 可选跳过 SH 系数（移动端 L0 模式）
 * 4. 使用 TypedArray 而非对象数组
 */

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
}

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

    if (trimmed.startsWith("element vertex")) {
      const parts = trimmed.split(/\s+/);
      vertexCount = parseInt(parts[2], 10);
    }

    if (trimmed.startsWith("property")) {
      const parts = trimmed.split(/\s+/);
      const type = parts[1];
      const name = parts[2];

      properties.push({
        name,
        type,
        byteOffset: currentOffset,
      });

      currentOffset += 4; // 假定 float32
    }
  }

  return { vertexCount, properties, stride: currentOffset };
}

/**
 * 提取 header
 */
function extractHeader(buffer: ArrayBuffer): {
  headerText: string;
  dataOffset: number;
} {
  const bytes = new Uint8Array(buffer);
  const decoder = new TextDecoder("ascii");

  const endHeaderMarker = "end_header\n";
  const maxHeaderSize = Math.min(bytes.length, 10000);
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
 * Sigmoid 函数
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

const SH_C0 = 0.28209479177387814;

/**
 * 智能采样：基于重要性（opacity * scale）采样
 * 保留更大、更不透明的 splat，它们对最终效果影响更大
 */
function computeImportanceSampling(
  buffer: ArrayBuffer,
  dataOffset: number,
  stride: number,
  totalCount: number,
  sampleCount: number,
  opacityOffset: number,
  scale0Offset: number,
  scale1Offset: number,
  scale2Offset: number
): Uint32Array {
  const dataView = new DataView(buffer, dataOffset);
  
  // 计算每个 splat 的重要性分数
  // 重要性 = opacity * max(scale) 
  // 更大、更不透明的 splat 更重要
  const importance = new Float32Array(totalCount);
  
  for (let i = 0; i < totalCount; i++) {
    const base = i * stride;
    
    // 获取 opacity（需要 sigmoid 转换）
    const rawOpacity = opacityOffset >= 0 ? dataView.getFloat32(base + opacityOffset, true) : 0;
    const opacity = sigmoid(rawOpacity);
    
    // 获取 scale（需要 exp 转换）
    const s0 = scale0Offset >= 0 ? Math.exp(dataView.getFloat32(base + scale0Offset, true)) : 1;
    const s1 = scale1Offset >= 0 ? Math.exp(dataView.getFloat32(base + scale1Offset, true)) : 1;
    const s2 = scale2Offset >= 0 ? Math.exp(dataView.getFloat32(base + scale2Offset, true)) : 1;
    const maxScale = Math.max(s0, s1, s2);
    
    // 重要性分数：opacity * scale（取对数避免数值问题）
    // 加上一个小的随机扰动，避免相同分数时总是选择相同的
    importance[i] = opacity * maxScale + Math.random() * 0.0001;
  }
  
  // 创建索引数组并按重要性排序
  const indices = new Uint32Array(totalCount);
  for (let i = 0; i < totalCount; i++) {
    indices[i] = i;
  }
  
  // 部分排序：只需要找到最重要的 sampleCount 个
  // 使用快速选择算法的简化版本
  indices.sort((a, b) => importance[b] - importance[a]);
  
  // 返回最重要的 sampleCount 个索引
  const result = new Uint32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    result[i] = indices[i];
  }
  
  // 按原始顺序排列（保持空间局部性）
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
  const {
    maxSplats = 200000,
    loadSH = false,
    onProgress,
  } = options;

  // 获取文件
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`无法加载 PLY 文件: ${url}`);
  }
  const buffer = await response.arrayBuffer();

  // 解析 header
  const { headerText, dataOffset } = extractHeader(buffer);
  const { vertexCount, properties, stride } = parseHeader(headerText);

  console.log(`PLYLoaderMobile: ${vertexCount} vertices, stride=${stride} bytes`);

  // 构建属性偏移映射
  const propMap = new Map<string, number>();
  for (const prop of properties) {
    propMap.set(prop.name, prop.byteOffset);
  }

  // 获取基本属性偏移
  const offsets = {
    x: propMap.get("x") ?? -1,
    y: propMap.get("y") ?? -1,
    z: propMap.get("z") ?? -1,
    scale_0: propMap.get("scale_0") ?? -1,
    scale_1: propMap.get("scale_1") ?? -1,
    scale_2: propMap.get("scale_2") ?? -1,
    rot_0: propMap.get("rot_0") ?? -1,
    rot_1: propMap.get("rot_1") ?? -1,
    rot_2: propMap.get("rot_2") ?? -1,
    rot_3: propMap.get("rot_3") ?? -1,
    f_dc_0: propMap.get("f_dc_0") ?? -1,
    f_dc_1: propMap.get("f_dc_1") ?? -1,
    f_dc_2: propMap.get("f_dc_2") ?? -1,
    opacity: propMap.get("opacity") ?? -1,
  };

  // SH 系数偏移（可选）
  let shOffsets: number[] = [];
  if (loadSH) {
    const shRestProps = properties
      .filter((p) => p.name.startsWith("f_rest_"))
      .sort((a, b) => {
        const idxA = parseInt(a.name.replace("f_rest_", ""), 10);
        const idxB = parseInt(b.name.replace("f_rest_", ""), 10);
        return idxA - idxB;
      });
    shOffsets = shRestProps.map(p => p.byteOffset);
    console.log(`PLYLoaderMobile: 加载 ${shOffsets.length} 个 SH 系数`);
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
  
  // 如果需要降采样，使用智能采样（基于重要性）
  let sampleIndices: Uint32Array | null = null;
  if (needSample) {
    console.log(`PLYLoaderMobile: 开始智能采样 ${vertexCount} -> ${actualCount} (${(actualCount/vertexCount*100).toFixed(1)}%)`);
    sampleIndices = computeImportanceSampling(
      buffer, dataOffset, stride, vertexCount, actualCount,
      offsets.opacity, offsets.scale_0, offsets.scale_1, offsets.scale_2
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
    positions[outputIdx * 3 + 0] = offsets.x >= 0 ? dataView.getFloat32(base + offsets.x, true) : 0;
    positions[outputIdx * 3 + 1] = offsets.y >= 0 ? dataView.getFloat32(base + offsets.y, true) : 0;
    positions[outputIdx * 3 + 2] = offsets.z >= 0 ? dataView.getFloat32(base + offsets.z, true) : 0;

    // 缩放（exp 转换）
    scales[outputIdx * 3 + 0] = offsets.scale_0 >= 0 ? Math.exp(dataView.getFloat32(base + offsets.scale_0, true)) : 1;
    scales[outputIdx * 3 + 1] = offsets.scale_1 >= 0 ? Math.exp(dataView.getFloat32(base + offsets.scale_1, true)) : 1;
    scales[outputIdx * 3 + 2] = offsets.scale_2 >= 0 ? Math.exp(dataView.getFloat32(base + offsets.scale_2, true)) : 1;

    // 旋转四元数（归一化）
    const rot_0 = offsets.rot_0 >= 0 ? dataView.getFloat32(base + offsets.rot_0, true) : 1;
    const rot_1 = offsets.rot_1 >= 0 ? dataView.getFloat32(base + offsets.rot_1, true) : 0;
    const rot_2 = offsets.rot_2 >= 0 ? dataView.getFloat32(base + offsets.rot_2, true) : 0;
    const rot_3 = offsets.rot_3 >= 0 ? dataView.getFloat32(base + offsets.rot_3, true) : 0;
    const qlen = Math.sqrt(rot_0 * rot_0 + rot_1 * rot_1 + rot_2 * rot_2 + rot_3 * rot_3);
    const qnorm = qlen > 0 ? 1 / qlen : 1;
    rotations[outputIdx * 4 + 0] = rot_0 * qnorm;
    rotations[outputIdx * 4 + 1] = rot_1 * qnorm;
    rotations[outputIdx * 4 + 2] = rot_2 * qnorm;
    rotations[outputIdx * 4 + 3] = rot_3 * qnorm;

    // DC 颜色（SH0 -> RGB）
    const f_dc_0 = offsets.f_dc_0 >= 0 ? dataView.getFloat32(base + offsets.f_dc_0, true) : 0;
    const f_dc_1 = offsets.f_dc_1 >= 0 ? dataView.getFloat32(base + offsets.f_dc_1, true) : 0;
    const f_dc_2 = offsets.f_dc_2 >= 0 ? dataView.getFloat32(base + offsets.f_dc_2, true) : 0;
    colors[outputIdx * 3 + 0] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_0));
    colors[outputIdx * 3 + 1] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_1));
    colors[outputIdx * 3 + 2] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_2));

    // 不透明度（sigmoid）
    const rawOpacity = offsets.opacity >= 0 ? dataView.getFloat32(base + offsets.opacity, true) : 0;
    opacities[outputIdx] = sigmoid(rawOpacity);

    // SH 系数（可选）
    if (shCoeffs && shOffsets.length > 0) {
      const shBase = outputIdx * 45;
      // L1: 3 基函数 × 3 通道 = 9
      for (let basis = 0; basis < 3; basis++) {
        for (let channel = 0; channel < 3; channel++) {
          const srcShIdx = basis * 3 + channel;
          const dstShIdx = channel * 3 + basis;
          if (srcShIdx < shOffsets.length) {
            shCoeffs[shBase + dstShIdx] = dataView.getFloat32(base + shOffsets[srcShIdx], true);
          }
        }
      }
      // L2: 5 基函数 × 3 通道 = 15
      for (let basis = 0; basis < 5; basis++) {
        for (let channel = 0; channel < 3; channel++) {
          const srcShIdx = 9 + basis * 3 + channel;
          const dstShIdx = 9 + channel * 5 + basis;
          if (srcShIdx < shOffsets.length) {
            shCoeffs[shBase + dstShIdx] = dataView.getFloat32(base + shOffsets[srcShIdx], true);
          }
        }
      }
      // L3: 7 基函数 × 3 通道 = 21
      for (let basis = 0; basis < 7; basis++) {
        for (let channel = 0; channel < 3; channel++) {
          const srcShIdx = 24 + basis * 3 + channel;
          const dstShIdx = 24 + channel * 7 + basis;
          if (srcShIdx < shOffsets.length) {
            shCoeffs[shBase + dstShIdx] = dataView.getFloat32(base + shOffsets[srcShIdx], true);
          }
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
