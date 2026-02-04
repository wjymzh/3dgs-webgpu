/**
 * 纹理加载工具函数
 */

/**
 * 从 URL 加载纹理
 * @param device GPU 设备
 * @param url 纹理 URL
 * @returns GPU 纹理或 null（加载失败时）
 */
export async function loadTextureFromURL(
  device: GPUDevice,
  url: string
): Promise<GPUTexture | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }

    const blob = await response.blob();
    return loadTextureFromBlob(device, blob);
  } catch (error) {
    console.warn(`Failed to load texture from URL: ${url}`, error);
    return null;
  }
}

/**
 * 从 Blob 加载纹理
 * @param device GPU 设备
 * @param blob 图片 Blob
 * @returns GPU 纹理或 null（加载失败时）
 */
export async function loadTextureFromBlob(
  device: GPUDevice,
  blob: Blob
): Promise<GPUTexture | null> {
  try {
    const imageBitmap = await createImageBitmap(blob);
    return createTextureFromImageBitmap(device, imageBitmap);
  } catch (error) {
    console.warn(`Failed to create texture from blob`, error);
    return null;
  }
}

/**
 * 从 ArrayBuffer 加载纹理
 * @param device GPU 设备
 * @param buffer 图片数据
 * @param mimeType MIME 类型
 * @returns GPU 纹理或 null（加载失败时）
 */
export async function loadTextureFromBuffer(
  device: GPUDevice,
  buffer: ArrayBuffer | Uint8Array,
  mimeType: string = 'image/png'
): Promise<GPUTexture | null> {
  try {
    // 确保是 Uint8Array 类型用于 Blob 构造
    const uint8Array = buffer instanceof Uint8Array 
      ? buffer 
      : new Uint8Array(buffer);
    // 使用类型断言解决 SharedArrayBuffer 兼容性问题
    const blob = new Blob([uint8Array as BlobPart], { type: mimeType });
    return loadTextureFromBlob(device, blob);
  } catch (error) {
    console.warn(`Failed to create texture from buffer`, error);
    return null;
  }
}

/**
 * 从 ImageBitmap 创建 GPU 纹理
 * @param device GPU 设备
 * @param imageBitmap ImageBitmap 对象
 * @returns GPU 纹理
 */
export function createTextureFromImageBitmap(
  device: GPUDevice,
  imageBitmap: ImageBitmap
): GPUTexture {
  const texture = device.createTexture({
    size: [imageBitmap.width, imageBitmap.height, 1],
    format: 'rgba8unorm',
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });

  device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture },
    [imageBitmap.width, imageBitmap.height]
  );

  return texture;
}

/**
 * 纹理缓存管理器
 */
export class TextureCache {
  private cache: Map<string, GPUTexture> = new Map();
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * 获取或加载纹理
   */
  async getOrLoad(url: string): Promise<GPUTexture | null> {
    if (this.cache.has(url)) {
      return this.cache.get(url)!;
    }

    const texture = await loadTextureFromURL(this.device, url);
    if (texture) {
      this.cache.set(url, texture);
    }
    return texture;
  }

  /**
   * 检查缓存中是否存在
   */
  has(url: string): boolean {
    return this.cache.has(url);
  }

  /**
   * 从缓存获取
   */
  get(url: string): GPUTexture | undefined {
    return this.cache.get(url);
  }

  /**
   * 添加到缓存
   */
  set(url: string, texture: GPUTexture): void {
    this.cache.set(url, texture);
  }

  /**
   * 清空缓存
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * 销毁所有纹理并清空缓存
   */
  destroy(): void {
    for (const texture of this.cache.values()) {
      texture.destroy();
    }
    this.cache.clear();
  }
}
