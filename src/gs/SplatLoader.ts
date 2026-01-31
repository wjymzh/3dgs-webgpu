/**
 * SplatLoader - 加载 .splat 格式的 3D Gaussian Splatting 文件
 * 
 * .splat 格式是一种紧凑的 3DGS 数据格式：
 * - 每个 splat 固定 32 字节，无文件头
 * - 数据布局: position(12) + scale(12) + color(3) + opacity(1) + rotation(4)
 * - 不包含高阶球谐系数，仅 DC 颜色
 * 
 * 四元数顺序: [w, x, y, z]（与 PLYLoader 保持一致）
 * 
 * 参考: supersplat/src/loaders/splat.ts
 */

import { SplatCPU } from "./PLYLoader";

/** .splat 文件每个 splat 的字节大小 */
const SPLAT_SIZE = 32;

/** 最小有效文件大小（至少包含一个 splat） */
const MIN_FILE_SIZE = SPLAT_SIZE;

/** 最大合理文件大小（防止内存溢出，约 1000 万 splats） */
const MAX_FILE_SIZE = SPLAT_SIZE * 10_000_000;

/**
 * 验证 .splat 文件格式
 * 由于 .splat 没有魔数，只能通过文件大小和数据合理性来验证
 */
function validateSplatFile(data: ArrayBufferLike): void {
  if (data.byteLength < MIN_FILE_SIZE) {
    throw new Error(`无效的 Splat 文件: 文件太小 (${data.byteLength} bytes)，至少需要 ${MIN_FILE_SIZE} bytes`);
  }

  if (data.byteLength > MAX_FILE_SIZE) {
    throw new Error(`Splat 文件过大 (${(data.byteLength / 1024 / 1024).toFixed(1)} MB)，超过最大限制`);
  }

  if (data.byteLength % SPLAT_SIZE !== 0) {
    throw new Error(`无效的 Splat 文件: 文件大小 (${data.byteLength} bytes) 不是 ${SPLAT_SIZE} 的整数倍`);
  }

  // 抽样检查数据合理性（检查前几个 splat 的位置是否为有效浮点数）
  const dataView = new DataView(data);
  const samplesToCheck = Math.min(10, Math.floor(data.byteLength / SPLAT_SIZE));

  for (let i = 0; i < samplesToCheck; i++) {
    const off = i * SPLAT_SIZE;
    const x = dataView.getFloat32(off + 0, true);
    const y = dataView.getFloat32(off + 4, true);
    const z = dataView.getFloat32(off + 8, true);

    // 检查是否为有效浮点数（非 NaN、非 Infinity）
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new Error(`无效的 Splat 文件: 第 ${i} 个 splat 包含无效的位置数据`);
    }

    // 检查位置是否在合理范围内（-10000 到 10000）
    const MAX_POS = 10000;
    if (Math.abs(x) > MAX_POS || Math.abs(y) > MAX_POS || Math.abs(z) > MAX_POS) {
      console.warn(`Splat 文件警告: 第 ${i} 个 splat 位置超出常规范围 (${x}, ${y}, ${z})`);
    }
  }
}

/**
 * Sigmoid 函数，与 PLYLoader 保持一致
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * 逆 Sigmoid 函数
 * 将 [0, 1] 范围的值转换回原始 logit 值
 */
function inverseSigmoid(y: number): number {
  // 避免 log(0) 和 log(负数)
  const clamped = Math.max(0.0001, Math.min(0.9999, y));
  return Math.log(clamped / (1 - clamped));
}

/**
 * 加载并解析 .splat 文件
 * @param url .splat 文件的 URL
 * @returns SplatCPU 数组
 */
export async function loadSplat(url: string): Promise<SplatCPU[]> {
  // 获取文件
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`无法加载 Splat 文件: ${url}`);
  }
  const buffer = await response.arrayBuffer();

  return deserializeSplat(buffer);
}

/**
 * 从 ArrayBuffer 解析 splat 数据
 * @param data 文件的 ArrayBuffer
 * @returns SplatCPU 数组
 */
export function deserializeSplat(data: ArrayBufferLike): SplatCPU[] {
  // 验证文件格式
  validateSplatFile(data);

  const totalSplats = Math.floor(data.byteLength / SPLAT_SIZE);
  console.log(`Splat: ${totalSplats} splats, ${data.byteLength} bytes`);

  const dataView = new DataView(data);
  const splats: SplatCPU[] = new Array(totalSplats);

  // 预分配共享的 SH 系数数组（全为 0，因为 .splat 不包含 SH）
  const shRestBuffer = new Float32Array(totalSplats * 45);

  for (let i = 0; i < totalSplats; i++) {
    const off = i * SPLAT_SIZE;

    // 读取位置 (float32 × 3 = 12 bytes)
    const x = dataView.getFloat32(off + 0, true);
    const y = dataView.getFloat32(off + 4, true);
    const z = dataView.getFloat32(off + 8, true);

    // 读取缩放 (float32 × 3 = 12 bytes)
    // splat 格式存储原始缩放值，直接使用
    const scale_0 = dataView.getFloat32(off + 12, true);
    const scale_1 = dataView.getFloat32(off + 16, true);
    const scale_2 = dataView.getFloat32(off + 20, true);

    // 读取颜色 (uint8 × 3 = 3 bytes)
    // splat 格式直接存储 RGB 颜色 [0-255]，归一化到 [0-1]
    const colorR = dataView.getUint8(off + 24) / 255;
    const colorG = dataView.getUint8(off + 25) / 255;
    const colorB = dataView.getUint8(off + 26) / 255;

    // 读取透明度 (uint8 × 1 = 1 byte)
    // .splat 格式存储的是 sigmoid 后的值 [0-255] 映射到 [0-1]
    // 为了与 PLYLoader 的 sigmoid 处理保持一致，我们：
    // 1. 先将 uint8 转换为 [0, 1] 范围
    // 2. 应用逆 sigmoid 得到原始 logit 值
    // 3. 再应用 sigmoid（这样与 PLYLoader 的处理流程一致）
    // 
    // 实际上这等价于直接使用线性归一化，但保持代码逻辑一致性
    const opacityUint8 = dataView.getUint8(off + 27);
    const opacityNormalized = opacityUint8 / 255;
    // 直接使用归一化值作为最终 opacity（因为 .splat 已经是 sigmoid 后的值）
    const opacity = opacityNormalized;

    // 读取旋转四元数 (uint8 × 4 = 4 bytes)
    // 从 [0, 255] 映射到 [-1, 1]
    // 四元数顺序: [w, x, y, z]
    const rot_w = (dataView.getUint8(off + 28) - 128) / 128;
    const rot_x = (dataView.getUint8(off + 29) - 128) / 128;
    const rot_y = (dataView.getUint8(off + 30) - 128) / 128;
    const rot_z = (dataView.getUint8(off + 31) - 128) / 128;

    // 归一化四元数
    const qlen = Math.sqrt(
      rot_w * rot_w + rot_x * rot_x + rot_y * rot_y + rot_z * rot_z
    );
    const qnorm = qlen > 0 ? 1 / qlen : 1;

    // 使用共享 buffer 的子数组
    const shOffset = i * 45;
    const shRest = shRestBuffer.subarray(shOffset, shOffset + 45);

    splats[i] = {
      mean: [x, y, z],
      scale: [scale_0, scale_1, scale_2],
      rotation: [
        rot_w * qnorm,
        rot_x * qnorm,
        rot_y * qnorm,
        rot_z * qnorm,
      ],
      colorDC: [colorR, colorG, colorB],
      opacity,
      shRest,
    };
  }

  return splats;
}
