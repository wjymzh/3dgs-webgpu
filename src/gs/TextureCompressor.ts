/**
 * TextureCompressor - 移动端纹理压缩工具
 * 将 splat 数据压缩为纹理格式，大幅减少 GPU 内存占用
 *
 * 内存对比：
 * - 原始 Storage Buffer: 256 bytes/splat
 * - 纹理压缩: ~52 bytes/splat (约 5x 压缩)
 * 
 * 数据布局（保证精度）：
 * - positionTexture (RGBA32Float): x, y, z, unused - 16 bytes
 * - scaleRotTexture1 (RGBA32Float): scale_x, scale_y, scale_z, rot_w - 16 bytes
 * - scaleRotTexture2 (RGBA32Float): rot_x, rot_y, rot_z, unused - 16 bytes
 * - colorTexture (RGBA8Unorm): r, g, b, opacity - 4 bytes
 * 总计: 52 bytes/splat
 * 
 * 注意：使用 RGBA32Float 替代 RGBA16Float 以保证 scale 和 rotation 的精度，
 * 避免平面等细节渲染出现块状伪影。
 */

import { CompactSplatData } from "./PLYLoaderMobile";

/**
 * 压缩后的纹理数据
 */
export interface CompressedSplatTextures {
  // 纹理尺寸
  width: number;
  height: number;
  count: number;

  // 位置纹理 (RGBA32Float) - 完整精度
  positionTexture: GPUTexture;

  // 缩放+旋转纹理 (RGBA32Float) - 保证精度
  // R: scale_x, G: scale_y, B: scale_z, A: rot_w
  scaleRotTexture1: GPUTexture;
  // R: rot_x, G: rot_y, B: rot_z, A: unused
  scaleRotTexture2: GPUTexture;

  // 颜色+不透明度纹理 (RGBA8Unorm)
  // R: color_r, G: color_g, B: color_b, A: opacity
  colorTexture: GPUTexture;

  // Bounding box (用于剔除优化)
  boundingBox: {
    min: [number, number, number];
    max: [number, number, number];
  };
}

/**
 * 计算纹理尺寸
 * 将 splat 数量映射到 2D 纹理尺寸
 * @param count splat 数量
 * @returns 纹理宽度和高度（向上取整到 4 的倍数）
 */
export function calculateTextureDimensions(count: number): { width: number; height: number } {
  if (count <= 0) {
    return { width: 4, height: 4 };
  }

  // 计算近似的正方形边长
  const side = Math.ceil(Math.sqrt(count));
  
  // 向上取整到 4 的倍数（GPU 纹理对齐优化）
  const alignedSide = Math.ceil(side / 4) * 4;
  
  // 确保能容纳所有 splat
  let width = alignedSide;
  let height = alignedSide;
  
  // 如果正方形不够，增加高度
  while (width * height < count) {
    height += 4;
  }

  return { width, height };
}

/**
 * 计算 bounding box
 */
function computeBoundingBox(positions: Float32Array, count: number): {
  min: [number, number, number];
  max: [number, number, number];
} {
  if (count === 0) {
    return {
      min: [0, 0, 0],
      max: [0, 0, 0],
    };
  }

  const min: [number, number, number] = [positions[0], positions[1], positions[2]];
  const max: [number, number, number] = [positions[0], positions[1], positions[2]];

  for (let i = 1; i < count; i++) {
    const x = positions[i * 3 + 0];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];

    min[0] = Math.min(min[0], x);
    min[1] = Math.min(min[1], y);
    min[2] = Math.min(min[2], z);
    max[0] = Math.max(max[0], x);
    max[1] = Math.max(max[1], y);
    max[2] = Math.max(max[2], z);
  }

  return { min, max };
}

/**
 * 将 splat 数据压缩为纹理格式
 * @param device GPU 设备
 * @param data 紧凑格式的 splat 数据
 * @returns 压缩后的纹理数据
 */
export function compressSplatsToTextures(
  device: GPUDevice,
  data: CompactSplatData
): CompressedSplatTextures {
  const count = data.count;
  const { width, height } = calculateTextureDimensions(count);
  const totalPixels = width * height;

  console.log(`TextureCompressor: 压缩 ${count} 个 splat 到 ${width}x${height} 纹理`);

  // 计算 bounding box
  const boundingBox = computeBoundingBox(data.positions, count);
  console.log(`TextureCompressor: BoundingBox min=[${boundingBox.min.map(v => v.toFixed(2)).join(', ')}], max=[${boundingBox.max.map(v => v.toFixed(2)).join(', ')}]`);

  // ============================================
  // 准备 CPU 端数据
  // ============================================

  // 位置纹理数据 (RGBA32Float) - 使用 Float32Array
  const positionData = new Float32Array(totalPixels * 4);

  // 缩放+旋转纹理数据 (RGBA32Float) - 使用 Float32Array 保证精度
  const scaleRotData1 = new Float32Array(totalPixels * 4);
  const scaleRotData2 = new Float32Array(totalPixels * 4);

  // 颜色纹理数据 (RGBA8Unorm)
  const colorData = new Uint8Array(totalPixels * 4);

  // ============================================
  // 填充数据
  // ============================================
  for (let i = 0; i < count; i++) {
    const pixelOffset = i * 4;

    // 位置数据 - 直接存储 float32
    positionData[pixelOffset + 0] = data.positions[i * 3 + 0];
    positionData[pixelOffset + 1] = data.positions[i * 3 + 1];
    positionData[pixelOffset + 2] = data.positions[i * 3 + 2];
    positionData[pixelOffset + 3] = 0; // unused

    // scaleRotTexture1: scale_x, scale_y, scale_z, rot_w (直接存储 float32)
    scaleRotData1[pixelOffset + 0] = data.scales[i * 3 + 0];
    scaleRotData1[pixelOffset + 1] = data.scales[i * 3 + 1];
    scaleRotData1[pixelOffset + 2] = data.scales[i * 3 + 2];
    scaleRotData1[pixelOffset + 3] = data.rotations[i * 4 + 0]; // rot_w

    // scaleRotTexture2: rot_x, rot_y, rot_z, unused (直接存储 float32)
    scaleRotData2[pixelOffset + 0] = data.rotations[i * 4 + 1]; // rot_x
    scaleRotData2[pixelOffset + 1] = data.rotations[i * 4 + 2]; // rot_y
    scaleRotData2[pixelOffset + 2] = data.rotations[i * 4 + 3]; // rot_z
    scaleRotData2[pixelOffset + 3] = 0; // unused

    // 颜色数据 (已经是 0-1 范围，转换为 0-255)
    const r = data.colors[i * 3 + 0];
    const g = data.colors[i * 3 + 1];
    const b = data.colors[i * 3 + 2];
    const opacity = data.opacities[i];

    colorData[pixelOffset + 0] = Math.round(Math.max(0, Math.min(1, r)) * 255);
    colorData[pixelOffset + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
    colorData[pixelOffset + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
    colorData[pixelOffset + 3] = Math.round(Math.max(0, Math.min(1, opacity)) * 255);
  }

  // ============================================
  // 创建 GPU 纹理
  // ============================================
  const textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;

  // 位置纹理 (RGBA32Float) - 完整精度
  const positionTexture = device.createTexture({
    size: { width, height },
    format: "rgba32float",
    usage: textureUsage,
  });

  // 缩放+旋转纹理1 (RGBA32Float) - 保证精度
  const scaleRotTexture1 = device.createTexture({
    size: { width, height },
    format: "rgba32float",
    usage: textureUsage,
  });

  // 缩放+旋转纹理2 (RGBA32Float) - 保证精度
  const scaleRotTexture2 = device.createTexture({
    size: { width, height },
    format: "rgba32float",
    usage: textureUsage,
  });

  // 颜色纹理 (RGBA8Unorm)
  const colorTexture = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage: textureUsage,
  });

  // ============================================
  // 上传数据到 GPU
  // ============================================
  
  // 位置纹理 (RGBA32Float = 16 bytes per pixel)
  device.queue.writeTexture(
    { texture: positionTexture },
    positionData,
    { bytesPerRow: width * 16 },
    { width, height }
  );

  // 缩放+旋转纹理1 (RGBA32Float = 16 bytes per pixel)
  device.queue.writeTexture(
    { texture: scaleRotTexture1 },
    scaleRotData1,
    { bytesPerRow: width * 16 },
    { width, height }
  );

  // 缩放+旋转纹理2 (RGBA32Float = 16 bytes per pixel)
  device.queue.writeTexture(
    { texture: scaleRotTexture2 },
    scaleRotData2,
    { bytesPerRow: width * 16 },
    { width, height }
  );

  // 颜色纹理 (RGBA8Unorm = 4 bytes per pixel)
  device.queue.writeTexture(
    { texture: colorTexture },
    colorData,
    { bytesPerRow: width * 4 },
    { width, height }
  );

  // 计算内存占用
  const memoryBytes = 
    width * height * 16 + // positionTexture (RGBA32Float)
    width * height * 16 + // scaleRotTexture1 (RGBA32Float)
    width * height * 16 + // scaleRotTexture2 (RGBA32Float)
    width * height * 4;   // colorTexture (RGBA8Unorm)
  const memoryMB = memoryBytes / (1024 * 1024);
  const bytesPerSplat = memoryBytes / count;

  console.log(`TextureCompressor: GPU 内存占用 = ${memoryMB.toFixed(2)} MB (${bytesPerSplat.toFixed(1)} bytes/splat)`);

  return {
    width,
    height,
    count,
    positionTexture,
    scaleRotTexture1,
    scaleRotTexture2,
    colorTexture,
    boundingBox,
  };
}

/**
 * 销毁压缩纹理资源
 * @param textures 压缩纹理数据
 */
export function destroyCompressedTextures(textures: CompressedSplatTextures): void {
  textures.positionTexture.destroy();
  textures.scaleRotTexture1.destroy();
  textures.scaleRotTexture2.destroy();
  textures.colorTexture.destroy();
  
  console.log("TextureCompressor: 纹理资源已销毁");
}
