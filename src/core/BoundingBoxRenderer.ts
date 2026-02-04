import { Renderer } from "./Renderer";
import { Camera } from "./Camera";
import type { SimpleBoundingBox, BoundingBoxProvider, Vec3Tuple } from "../types";

// 重新导出类型保持向后兼容
export type { BoundingBoxProvider };
export type BoundingBox = SimpleBoundingBox;

/**
 * BoundingBoxRenderer - 包围盒线框渲染器
 * 用于显示选中对象的包围盒，支持动态跟随
 */
export class BoundingBoxRenderer {
  private renderer: Renderer;
  private camera: Camera;
  
  // GPU 资源
  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private vertexBuffer: GPUBuffer | null = null;
  
  // 包围盒数据提供者（动态模式）
  private provider: BoundingBoxProvider | null = null;
  
  // 静态包围盒（备用）
  private staticBoundingBox: BoundingBox | null = null;
  
  // 线条颜色 (白色)
  private lineColor: [number, number, number] = [1.0, 1.0, 1.0];
  
  // 角落线段长度比例 (相对于边长)
  private cornerRatio: number = 0.2;
  
  constructor(renderer: Renderer, camera: Camera) {
    this.renderer = renderer;
    this.camera = camera;
    this.createPipeline();
    this.createVertexBuffer();
  }
  
  /**
   * 创建渲染管线
   */
  private createPipeline(): void {
    const device = this.renderer.device;
    
    const shaderCode = `
      struct Uniforms {
        viewProjection: mat4x4<f32>,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;

      struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) color: vec3<f32>,
      }

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec3<f32>,
      }

      @vertex
      fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.viewProjection * vec4<f32>(input.position, 1.0);
        output.color = input.color;
        return output;
      }

      @fragment
      fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        return vec4<f32>(input.color, 1.0);
      }
    `;
    
    const shaderModule = device.createShaderModule({ code: shaderCode });
    
    // Uniform buffer: viewProjection (64 bytes)
    this.uniformBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      }],
    });
    
    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });
    
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    
    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vertexMain",
        buffers: [{
          arrayStride: 24, // 6 floats * 4 bytes
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ],
        }],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [{
          format: this.renderer.format,
        }],
      },
      primitive: {
        topology: "line-list",
        cullMode: "none",
      },
      depthStencil: {
        format: this.renderer.depthFormat,
        depthWriteEnabled: false,
        depthCompare: "always", // 始终可见
      },
    });
  }
  
  /**
   * 创建顶点缓冲区（预分配）
   */
  private createVertexBuffer(): void {
    const device = this.renderer.device;
    // 48 个顶点 * 6 floats * 4 bytes = 1152 bytes
    this.vertexBuffer = device.createBuffer({
      size: 1152,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }
  
  /**
   * 设置包围盒数据提供者（动态模式）
   * 每帧会从 provider 获取最新的包围盒数据
   */
  setProvider(provider: BoundingBoxProvider | null): void {
    this.provider = provider;
    this.staticBoundingBox = null;
  }
  
  /**
   * 设置静态包围盒（不会自动更新）
   */
  setBoundingBox(box: BoundingBox | null): void {
    this.staticBoundingBox = box;
    this.provider = null;
  }
  
  /**
   * 清除包围盒
   */
  clear(): void {
    this.provider = null;
    this.staticBoundingBox = null;
  }
  
  /**
   * 设置线条颜色
   */
  setLineColor(r: number, g: number, b: number): void {
    this.lineColor = [r, g, b];
  }
  
  /**
   * 生成顶点数据
   */
  private generateVertices(box: BoundingBox): Float32Array {
    const { min, max } = box;
    const [r, g, b] = this.lineColor;
    
    // 计算各边长度
    const dx = max[0] - min[0];
    const dy = max[1] - min[1];
    const dz = max[2] - min[2];
    
    // 角落线段长度
    const lx = dx * this.cornerRatio;
    const ly = dy * this.cornerRatio;
    const lz = dz * this.cornerRatio;
    
    // 8 个角落，每个角落 3 条线段，每条线段 2 个顶点
    // 总共 8 * 3 * 2 = 48 个顶点
    const vertices: number[] = [];
    
    // 辅助函数：添加线段
    const addLine = (x1: number, y1: number, z1: number, x2: number, y2: number, z2: number) => {
      vertices.push(x1, y1, z1, r, g, b);
      vertices.push(x2, y2, z2, r, g, b);
    };
    
    // 角落 0: min, min, min
    addLine(min[0], min[1], min[2], min[0] + lx, min[1], min[2]);
    addLine(min[0], min[1], min[2], min[0], min[1] + ly, min[2]);
    addLine(min[0], min[1], min[2], min[0], min[1], min[2] + lz);
    
    // 角落 1: max, min, min
    addLine(max[0], min[1], min[2], max[0] - lx, min[1], min[2]);
    addLine(max[0], min[1], min[2], max[0], min[1] + ly, min[2]);
    addLine(max[0], min[1], min[2], max[0], min[1], min[2] + lz);
    
    // 角落 2: min, max, min
    addLine(min[0], max[1], min[2], min[0] + lx, max[1], min[2]);
    addLine(min[0], max[1], min[2], min[0], max[1] - ly, min[2]);
    addLine(min[0], max[1], min[2], min[0], max[1], min[2] + lz);
    
    // 角落 3: max, max, min
    addLine(max[0], max[1], min[2], max[0] - lx, max[1], min[2]);
    addLine(max[0], max[1], min[2], max[0], max[1] - ly, min[2]);
    addLine(max[0], max[1], min[2], max[0], max[1], min[2] + lz);
    
    // 角落 4: min, min, max
    addLine(min[0], min[1], max[2], min[0] + lx, min[1], max[2]);
    addLine(min[0], min[1], max[2], min[0], min[1] + ly, max[2]);
    addLine(min[0], min[1], max[2], min[0], min[1], max[2] - lz);
    
    // 角落 5: max, min, max
    addLine(max[0], min[1], max[2], max[0] - lx, min[1], max[2]);
    addLine(max[0], min[1], max[2], max[0], min[1] + ly, max[2]);
    addLine(max[0], min[1], max[2], max[0], min[1], max[2] - lz);
    
    // 角落 6: min, max, max
    addLine(min[0], max[1], max[2], min[0] + lx, max[1], max[2]);
    addLine(min[0], max[1], max[2], min[0], max[1] - ly, max[2]);
    addLine(min[0], max[1], max[2], min[0], max[1], max[2] - lz);
    
    // 角落 7: max, max, max
    addLine(max[0], max[1], max[2], max[0] - lx, max[1], max[2]);
    addLine(max[0], max[1], max[2], max[0], max[1] - ly, max[2]);
    addLine(max[0], max[1], max[2], max[0], max[1], max[2] - lz);
    
    return new Float32Array(vertices);
  }
  
  /**
   * 渲染包围盒
   */
  render(pass: GPURenderPassEncoder): void {
    if (!this.pipeline || !this.bindGroup || !this.vertexBuffer || !this.uniformBuffer) {
      return;
    }
    
    // 获取当前包围盒
    let box: BoundingBox | null = null;
    
    if (this.provider) {
      box = this.provider.getBoundingBox();
    } else {
      box = this.staticBoundingBox;
    }
    
    if (!box) return;
    
    const device = this.renderer.device;
    
    // 每帧都更新顶点缓冲区（动态模式）
    const vertexData = this.generateVertices(box);
    device.queue.writeBuffer(this.vertexBuffer, 0, vertexData.buffer);
    
    // 更新 uniform buffer
    const vpMatrix = new Float32Array(this.camera.viewProjectionMatrix);
    device.queue.writeBuffer(this.uniformBuffer, 0, vpMatrix);
    
    // 渲染
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.draw(48); // 48 个顶点
  }
  
  /**
   * 销毁资源
   */
  destroy(): void {
    if (this.vertexBuffer) {
      this.vertexBuffer.destroy();
      this.vertexBuffer = null;
    }
    if (this.uniformBuffer) {
      this.uniformBuffer.destroy();
      this.uniformBuffer = null;
    }
    this.pipeline = null;
    this.bindGroup = null;
  }
}
