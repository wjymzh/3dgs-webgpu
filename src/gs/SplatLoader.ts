/**
 * SplatLoader - 加载 .splat 格式的 3D Gaussian Splatting 文件
 * 
 * .splat 格式是一种紧凑的 3DGS 数据格式：
 * - 每个 splat 固定 32 字节，无文件头
 * - 数据布局: position(12) + scale(12) + color(3) + opacity(1) + rotation(4)
 * - 不包含高阶球谐系数，仅 DC 颜色
 * 
 * 参考: supersplat/src/loaders/splat.ts
 */

import { SplatCPU } from "./PLYLoader";

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
  // .splat 文件没有头部，直接通过文件大小计算 splat 数量
  const SPLAT_SIZE = 32; // 每个 splat 32 字节
  const totalSplats = Math.floor(data.byteLength / SPLAT_SIZE);

  if (totalSplats === 0) {
    console.warn("Splat 文件为空或格式无效");
    return [];
  }

  console.log(`Splat: ${totalSplats} splats, ${data.byteLength} bytes`);

  const dataView = new DataView(data);
  const splats: SplatCPU[] = [];

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
    // 逆 sigmoid: opacity_raw = -log(255/uint8 - 1)
    // 然后应用 sigmoid 得到最终 opacity
    const opacityRaw = dataView.getUint8(off + 27);
    // 避免除以 0 或负数
    const opacity = opacityRaw === 0 ? 0 : 
                    opacityRaw >= 255 ? 1 : 
                    opacityRaw / 255;

    // 读取旋转四元数 (uint8 × 4 = 4 bytes)
    // 从 [0, 255] 映射到 [-1, 1]
    const rot_0 = (dataView.getUint8(off + 28) - 128) / 128;
    const rot_1 = (dataView.getUint8(off + 29) - 128) / 128;
    const rot_2 = (dataView.getUint8(off + 30) - 128) / 128;
    const rot_3 = (dataView.getUint8(off + 31) - 128) / 128;

    // 归一化四元数
    const qlen = Math.sqrt(
      rot_0 * rot_0 + rot_1 * rot_1 + rot_2 * rot_2 + rot_3 * rot_3
    );
    const qnorm = qlen > 0 ? 1 / qlen : 1;

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
      // splat 格式不包含高阶 SH 系数，使用空数组
      // 渲染时会自动使用 L0 模式
      shRest: new Float32Array(45),
    });
  }

  return splats;
}
