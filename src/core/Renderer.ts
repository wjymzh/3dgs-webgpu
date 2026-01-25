/**
 * Renderer - WebGPU 初始化 + 帧提交
 * 只负责 WebGPU 设备管理和渲染通道
 */
export class Renderer {
  private canvas: HTMLCanvasElement;
  private _device!: GPUDevice;
  private _context!: GPUCanvasContext;
  private _format!: GPUTextureFormat;
  private _depthTexture!: GPUTexture;
  private _depthTextureView!: GPUTextureView;
  
  private commandEncoder!: GPUCommandEncoder;
  private renderPassEncoder!: GPURenderPassEncoder;

  // 背景颜色
  private _clearColor: GPUColorDict = { r: 0.1, g: 0.1, b: 0.15, a: 1.0 };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  /**
   * 设置背景颜色
   */
  setClearColor(r: number, g: number, b: number, a: number = 1.0): void {
    this._clearColor = { r, g, b, a };
  }

  /**
   * 通过十六进制设置背景颜色
   */
  setClearColorHex(hex: string): void {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (result) {
      this._clearColor = {
        r: parseInt(result[1], 16) / 255,
        g: parseInt(result[2], 16) / 255,
        b: parseInt(result[3], 16) / 255,
        a: 1.0,
      };
    }
  }

  /**
   * 获取背景颜色（十六进制）
   */
  getClearColorHex(): string {
    const r = Math.round(this._clearColor.r * 255).toString(16).padStart(2, '0');
    const g = Math.round(this._clearColor.g * 255).toString(16).padStart(2, '0');
    const b = Math.round(this._clearColor.b * 255).toString(16).padStart(2, '0');
    return `#${r}${g}${b}`;
  }

  get device(): GPUDevice {
    return this._device;
  }

  get context(): GPUCanvasContext {
    return this._context;
  }

  get format(): GPUTextureFormat {
    return this._format;
  }

  get depthFormat(): GPUTextureFormat {
    return 'depth24plus';
  }

  /**
   * 获取渲染宽度（像素）
   */
  get width(): number {
    return this.canvas.width;
  }

  /**
   * 获取渲染高度（像素）
   */
  get height(): number {
    return this.canvas.height;
  }

  /**
   * 初始化 WebGPU
   */
  async init(): Promise<void> {
    // 检查 WebGPU 支持
    if (!navigator.gpu) {
      throw new Error('WebGPU 不受支持');
    }

    // 获取适配器
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      throw new Error('无法获取 GPU 适配器');
    }

    // 获取设备，请求更高的缓冲区大小限制以支持大型模型
    const adapterLimits = adapter.limits;
    this._device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: adapterLimits.maxBufferSize,
        maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
      },
    });
    this._device.lost.then((info) => {
      console.error('WebGPU 设备丢失:', info.message);
    });
    
    console.log(`WebGPU: maxBufferSize = ${this._device.limits.maxBufferSize / (1024 * 1024)} MB`);

    // 配置 canvas 上下文
    this._context = this.canvas.getContext('webgpu') as GPUCanvasContext;
    if (!this._context) {
      throw new Error('无法获取 WebGPU 上下文');
    }

    this._format = navigator.gpu.getPreferredCanvasFormat();
    this._context.configure({
      device: this._device,
      format: this._format,
      alphaMode: 'premultiplied',
    });

    // 创建深度纹理
    this.createDepthTexture();

    // 监听 canvas 大小变化
    this.setupResizeObserver();
  }

  /**
   * 创建深度纹理
   */
  private createDepthTexture(): void {
    if (this._depthTexture) {
      this._depthTexture.destroy();
    }

    this._depthTexture = this._device.createTexture({
      size: {
        width: this.canvas.width,
        height: this.canvas.height,
      },
      format: this.depthFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this._depthTextureView = this._depthTexture.createView();
  }

  /**
   * 设置 resize 监听
   */
  private setupResizeObserver(): void {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.floor(width * dpr);
        this.canvas.height = Math.floor(height * dpr);
        this.createDepthTexture();
      }
    });
    observer.observe(this.canvas);
  }

  /**
   * 开始帧 - 创建命令编码器和渲染通道
   */
  beginFrame(): GPURenderPassEncoder {
    const colorTexture = this._context.getCurrentTexture();
    const colorView = colorTexture.createView();

    this.commandEncoder = this._device.createCommandEncoder();
    
    this.renderPassEncoder = this.commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: colorView,
          clearValue: this._clearColor,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this._depthTextureView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    return this.renderPassEncoder;
  }

  /**
   * 结束帧 - 提交命令
   */
  endFrame(): void {
    this.renderPassEncoder.end();
    this._device.queue.submit([this.commandEncoder.finish()]);
  }

  /**
   * 获取 canvas 宽高比
   */
  getAspectRatio(): number {
    return this.canvas.width / this.canvas.height;
  }
}
