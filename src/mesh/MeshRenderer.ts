import { Renderer } from "../core/Renderer";
import { Camera } from "../core/Camera";
import { Mesh, MeshBoundingBox } from "./Mesh";

/**
 * WGSL Shader - MVP 变换 + 法线着色
 */
const shaderCode = /* wgsl */ `
struct Uniforms {
  viewProjection: mat4x4<f32>,
  model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let worldPos = uniforms.model * vec4<f32>(input.position, 1.0);
  output.position = uniforms.viewProjection * worldPos;
  // 简化法线变换（不考虑非均匀缩放）
  output.normal = (uniforms.model * vec4<f32>(input.normal, 0.0)).xyz;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  // 简单的法线着色
  let normal = normalize(input.normal);
  let color = normal * 0.5 + 0.5;
  return vec4<f32>(color, 1.0);
}
`;

/**
 * MeshRenderer - 网格渲染器
 * 只负责渲染 Mesh[]
 */
export class MeshRenderer {
  private renderer: Renderer;
  private camera: Camera;
  private meshes: Mesh[] = [];

  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private bindGroupLayout!: GPUBindGroupLayout;

  constructor(renderer: Renderer, camera: Camera) {
    this.renderer = renderer;
    this.camera = camera;
    this.createPipeline();
    this.createUniformBuffer();
  }

  /**
   * 创建渲染管线
   */
  private createPipeline(): void {
    const device = this.renderer.device;

    // 创建 shader 模块
    const shaderModule = device.createShaderModule({
      code: shaderCode,
    });

    // 创建 bind group layout
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
      ],
    });

    // 创建 pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    // 顶点缓冲区布局: position(3) + normal(3)
    const vertexBufferLayout: GPUVertexBufferLayout = {
      arrayStride: 24, // 6 * 4 bytes
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" }, // position
        { shaderLocation: 1, offset: 12, format: "float32x3" }, // normal
      ],
    };

    // 创建渲染管线
    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [vertexBufferLayout],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.renderer.format }],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "back",
        frontFace: "ccw",
      },
      depthStencil: {
        format: this.renderer.depthFormat,
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });
  }

  /**
   * 创建 uniform buffer
   * 布局: viewProjection(64) + model(64) = 128 bytes
   */
  private createUniformBuffer(): void {
    this.uniformBuffer = this.renderer.device.createBuffer({
      size: 128, // 2 * mat4x4
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * 添加网格
   */
  addMesh(mesh: Mesh): void {
    this.meshes.push(mesh);
  }

  /**
   * 移除网格
   */
  removeMesh(mesh: Mesh): void {
    const index = this.meshes.indexOf(mesh);
    if (index !== -1) {
      mesh.destroy();
      this.meshes.splice(index, 1);
    }
  }

  /**
   * 按索引移除网格
   */
  removeMeshByIndex(index: number): boolean {
    if (index >= 0 && index < this.meshes.length) {
      const mesh = this.meshes[index];
      mesh.destroy();
      this.meshes.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * 清空所有网格
   */
  clear(): void {
    for (const mesh of this.meshes) {
      mesh.destroy();
    }
    this.meshes = [];
  }

  /**
   * 渲染所有网格
   */
  render(pass: GPURenderPassEncoder): void {
    if (this.meshes.length === 0) return;

    pass.setPipeline(this.pipeline);

    // 更新 viewProjection uniform
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Float32Array(this.camera.viewProjectionMatrix),
    );

    for (const mesh of this.meshes) {
      // 更新 model matrix uniform
      this.renderer.device.queue.writeBuffer(
        this.uniformBuffer,
        64, // offset after viewProjection
        new Float32Array(mesh.modelMatrix),
      );

      // 创建 bind group（每帧重新创建以简化实现）
      const bindGroup = this.renderer.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
      });

      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, mesh.vertexBuffer);

      if (mesh.indexBuffer && mesh.indexCount > 0) {
        pass.setIndexBuffer(mesh.indexBuffer, "uint16");
        pass.drawIndexed(mesh.indexCount);
      } else {
        pass.draw(mesh.vertexCount);
      }
    }
  }

  /**
   * 获取网格数量
   */
  getMeshCount(): number {
    return this.meshes.length;
  }

  /**
   * 按索引获取网格
   */
  getMeshByIndex(index: number): Mesh | null {
    if (index >= 0 && index < this.meshes.length) {
      return this.meshes[index];
    }
    return null;
  }

  /**
   * 获取所有网格的组合 bounding box（世界空间）
   * @returns 组合的 MeshBoundingBox 或 null（如果没有网格）
   */
  getCombinedBoundingBox(): MeshBoundingBox | null {
    if (this.meshes.length === 0) return null;

    let combinedMin: [number, number, number] | null = null;
    let combinedMax: [number, number, number] | null = null;

    for (const mesh of this.meshes) {
      const bbox = mesh.getWorldBoundingBox();
      if (!bbox) continue;

      if (combinedMin === null || combinedMax === null) {
        combinedMin = [...bbox.min];
        combinedMax = [...bbox.max];
      } else {
        combinedMin[0] = Math.min(combinedMin[0], bbox.min[0]);
        combinedMin[1] = Math.min(combinedMin[1], bbox.min[1]);
        combinedMin[2] = Math.min(combinedMin[2], bbox.min[2]);
        combinedMax[0] = Math.max(combinedMax[0], bbox.max[0]);
        combinedMax[1] = Math.max(combinedMax[1], bbox.max[1]);
        combinedMax[2] = Math.max(combinedMax[2], bbox.max[2]);
      }
    }

    if (combinedMin === null || combinedMax === null) return null;

    // 计算中心点和半径
    const center: [number, number, number] = [
      (combinedMin[0] + combinedMax[0]) / 2,
      (combinedMin[1] + combinedMax[1]) / 2,
      (combinedMin[2] + combinedMax[2]) / 2,
    ];
    const dx = combinedMax[0] - combinedMin[0];
    const dy = combinedMax[1] - combinedMin[1];
    const dz = combinedMax[2] - combinedMin[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    return {
      min: combinedMin,
      max: combinedMax,
      center,
      radius,
    };
  }
}
