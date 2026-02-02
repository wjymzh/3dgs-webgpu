import { Renderer } from "../core/Renderer";
import { Camera } from "../core/Camera";
import { Mesh, MeshBoundingBox } from "./Mesh";
import { MaterialData } from "../loaders/GLBLoader";

/**
 * 带纹理的 Shader
 */
const shaderCodeTextured = /* wgsl */ `
struct Uniforms {
  viewProjection: mat4x4<f32>,
  model: mat4x4<f32>,
  baseColorFactor: vec4<f32>,
  lightDir: vec3<f32>,
  ambientIntensity: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var texSampler: sampler;
@group(0) @binding(2) var baseColorTexture: texture_2d<f32>;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let worldPos = uniforms.model * vec4<f32>(input.position, 1.0);
  output.position = uniforms.viewProjection * worldPos;
  output.normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
  output.uv = input.uv;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let texColor = textureSample(baseColorTexture, texSampler, input.uv);
  let baseColor = texColor * uniforms.baseColorFactor;
  
  // Lambert 光照 + 环境光
  let normal = normalize(input.normal);
  let NdotL = max(dot(normal, uniforms.lightDir), 0.0);
  let diffuse = NdotL * (1.0 - uniforms.ambientIntensity);
  let lighting = uniforms.ambientIntensity + diffuse;
  
  return vec4<f32>(baseColor.rgb * lighting, baseColor.a);
}
`;

/**
 * 无纹理的 Shader（使用 baseColorFactor）
 */
const shaderCodeUntextured = /* wgsl */ `
struct Uniforms {
  viewProjection: mat4x4<f32>,
  model: mat4x4<f32>,
  baseColorFactor: vec4<f32>,
  lightDir: vec3<f32>,
  ambientIntensity: f32,
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
  output.normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let normal = normalize(input.normal);
  let NdotL = max(dot(normal, uniforms.lightDir), 0.0);
  let diffuse = NdotL * (1.0 - uniforms.ambientIntensity);
  let lighting = uniforms.ambientIntensity + diffuse;
  
  return vec4<f32>(uniforms.baseColorFactor.rgb * lighting, uniforms.baseColorFactor.a);
}
`;

// Uniform buffer 大小: viewProjection(64) + model(64) + baseColorFactor(16) + lightDir(12) + ambientIntensity(4) = 160 bytes
const UNIFORM_BUFFER_SIZE = 160;

/**
 * 渲染项（Mesh + Material + 独立的 uniform buffer）
 */
interface RenderItem {
  mesh: Mesh;
  material: MaterialData;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
}

/**
 * MeshRenderer - 网格渲染器
 * 支持纹理和材质，每个 mesh 有独立的 uniform buffer
 */
export class MeshRenderer {
  private renderer: Renderer;
  private camera: Camera;
  private items: RenderItem[] = [];

  // 有纹理的管线
  private pipelineTextured!: GPURenderPipeline;
  private pipelineTexturedDoubleSided!: GPURenderPipeline;
  private bindGroupLayoutTextured!: GPUBindGroupLayout;

  // 无纹理的管线
  private pipelineUntextured!: GPURenderPipeline;
  private pipelineUntexturedDoubleSided!: GPURenderPipeline;
  private bindGroupLayoutUntextured!: GPUBindGroupLayout;

  private sampler!: GPUSampler;
  private defaultTexture!: GPUTexture;

  // 光照方向
  private lightDir: Float32Array = new Float32Array([0.5, 0.7, 0.5]);
  // 环境光强度 (0-1)
  private ambientIntensity: number = 0.6;

  constructor(renderer: Renderer, camera: Camera) {
    this.renderer = renderer;
    this.camera = camera;
    this.createResources();
    this.createPipelines();
  }

  private createResources(): void {
    const device = this.renderer.device;

    // 创建采样器
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });

    // 创建默认白色纹理
    this.defaultTexture = device.createTexture({
      size: [1, 1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.defaultTexture },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      [1, 1, 1]
    );
  }

  private createPipelines(): void {
    const device = this.renderer.device;

    // === 有纹理的管线 ===
    const shaderModuleTextured = device.createShaderModule({ code: shaderCodeTextured });

    this.bindGroupLayoutTextured = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });

    const pipelineLayoutTextured = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayoutTextured],
    });

    const vertexBufferLayoutTextured: GPUVertexBufferLayout = {
      arrayStride: 32,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x2" },
      ],
    };

    const basePipelineDescTextured: GPURenderPipelineDescriptor = {
      layout: pipelineLayoutTextured,
      vertex: {
        module: shaderModuleTextured,
        entryPoint: "vs_main",
        buffers: [vertexBufferLayoutTextured],
      },
      fragment: {
        module: shaderModuleTextured,
        entryPoint: "fs_main",
        targets: [{ format: this.renderer.format }],
      },
      primitive: { topology: "triangle-list", frontFace: "ccw" },
      depthStencil: { format: this.renderer.depthFormat, depthWriteEnabled: true, depthCompare: "less" },
    };

    this.pipelineTextured = device.createRenderPipeline({
      ...basePipelineDescTextured,
      primitive: { ...basePipelineDescTextured.primitive, cullMode: "back" },
    });

    this.pipelineTexturedDoubleSided = device.createRenderPipeline({
      ...basePipelineDescTextured,
      primitive: { ...basePipelineDescTextured.primitive, cullMode: "none" },
    });

    // === 无纹理的管线 ===
    const shaderModuleUntextured = device.createShaderModule({ code: shaderCodeUntextured });

    this.bindGroupLayoutUntextured = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });

    const pipelineLayoutUntextured = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayoutUntextured],
    });

    const vertexBufferLayoutUntextured: GPUVertexBufferLayout = {
      arrayStride: 24,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
      ],
    };

    const basePipelineDescUntextured: GPURenderPipelineDescriptor = {
      layout: pipelineLayoutUntextured,
      vertex: {
        module: shaderModuleUntextured,
        entryPoint: "vs_main",
        buffers: [vertexBufferLayoutUntextured],
      },
      fragment: {
        module: shaderModuleUntextured,
        entryPoint: "fs_main",
        targets: [{ format: this.renderer.format }],
      },
      primitive: { topology: "triangle-list", frontFace: "ccw" },
      depthStencil: { format: this.renderer.depthFormat, depthWriteEnabled: true, depthCompare: "less" },
    };

    this.pipelineUntextured = device.createRenderPipeline({
      ...basePipelineDescUntextured,
      primitive: { ...basePipelineDescUntextured.primitive, cullMode: "back" },
    });

    this.pipelineUntexturedDoubleSided = device.createRenderPipeline({
      ...basePipelineDescUntextured,
      primitive: { ...basePipelineDescUntextured.primitive, cullMode: "none" },
    });
  }

  /**
   * 添加网格（带材质）- 每个 mesh 创建独立的 uniform buffer
   */
  addMesh(mesh: Mesh, material?: MaterialData): void {
    const device = this.renderer.device;
    
    const mat = material || {
      baseColorFactor: [0.8, 0.8, 0.8, 1] as [number, number, number, number],
      baseColorTexture: null,
      metallicFactor: 0,
      roughnessFactor: 0.5,
      doubleSided: false,
    };

    // 为每个 mesh 创建独立的 uniform buffer
    const uniformBuffer = device.createBuffer({
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 创建 bind group
    let bindGroup: GPUBindGroup;
    
    if (mesh.hasUV) {
      const texture = mat.baseColorTexture || this.defaultTexture;
      bindGroup = device.createBindGroup({
        layout: this.bindGroupLayoutTextured,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: texture.createView() },
        ],
      });
    } else {
      bindGroup = device.createBindGroup({
        layout: this.bindGroupLayoutUntextured,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
        ],
      });
    }

    this.items.push({ mesh, material: mat, uniformBuffer, bindGroup });
  }

  /**
   * 移除网格
   */
  removeMesh(mesh: Mesh): void {
    const index = this.items.findIndex(item => item.mesh === mesh);
    if (index !== -1) {
      const item = this.items[index];
      item.mesh.destroy();
      item.uniformBuffer.destroy();
      this.items.splice(index, 1);
    }
  }

  /**
   * 按索引移除网格
   */
  removeMeshByIndex(index: number): boolean {
    if (index >= 0 && index < this.items.length) {
      const item = this.items[index];
      item.mesh.destroy();
      item.uniformBuffer.destroy();
      this.items.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * 清空所有网格
   */
  clear(): void {
    for (const item of this.items) {
      item.mesh.destroy();
      item.uniformBuffer.destroy();
    }
    this.items = [];
  }

  /**
   * 设置光照方向
   */
  setLightDirection(x: number, y: number, z: number): void {
    const len = Math.sqrt(x * x + y * y + z * z);
    this.lightDir[0] = x / len;
    this.lightDir[1] = y / len;
    this.lightDir[2] = z / len;
  }

  /**
   * 设置环境光强度
   */
  setAmbientIntensity(intensity: number): void {
    this.ambientIntensity = Math.max(0, Math.min(1, intensity));
  }

  /**
   * 获取环境光强度
   */
  getAmbientIntensity(): number {
    return this.ambientIntensity;
  }

  /**
   * 渲染所有网格
   */
  render(pass: GPURenderPassEncoder): void {
    if (this.items.length === 0) return;

    const device = this.renderer.device;
    const vpMatrix = new Float32Array(this.camera.viewProjectionMatrix);
    const lightData = new Float32Array([
      this.lightDir[0], this.lightDir[1], this.lightDir[2], this.ambientIntensity
    ]);

    for (const item of this.items) {
      const { mesh, material, uniformBuffer, bindGroup } = item;

      // 更新该 mesh 的 uniform buffer
      device.queue.writeBuffer(uniformBuffer, 0, vpMatrix.buffer);
      device.queue.writeBuffer(uniformBuffer, 64, mesh.modelMatrix.buffer);
      const colorData = new Float32Array(material.baseColorFactor);
      device.queue.writeBuffer(uniformBuffer, 128, colorData.buffer);
      device.queue.writeBuffer(uniformBuffer, 144, lightData.buffer);

      // 选择管线
      let pipeline: GPURenderPipeline;
      if (mesh.hasUV) {
        pipeline = material.doubleSided ? this.pipelineTexturedDoubleSided : this.pipelineTextured;
      } else {
        pipeline = material.doubleSided ? this.pipelineUntexturedDoubleSided : this.pipelineUntextured;
      }

      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, mesh.vertexBuffer);

      if (mesh.indexBuffer && mesh.indexCount > 0) {
        pass.setIndexBuffer(mesh.indexBuffer, mesh.indexFormat);
        pass.drawIndexed(mesh.indexCount);
      } else {
        pass.draw(mesh.vertexCount);
      }
    }
  }

  getMeshCount(): number {
    return this.items.length;
  }

  getMeshByIndex(index: number): Mesh | null {
    if (index >= 0 && index < this.items.length) {
      return this.items[index].mesh;
    }
    return null;
  }

  /**
   * 获取指定索引网格的材质颜色
   */
  getMeshColor(index: number): [number, number, number, number] | null {
    if (index >= 0 && index < this.items.length) {
      return [...this.items[index].material.baseColorFactor] as [number, number, number, number];
    }
    return null;
  }

  /**
   * 设置指定索引网格的材质颜色
   */
  setMeshColor(index: number, r: number, g: number, b: number, a: number = 1): boolean {
    if (index >= 0 && index < this.items.length) {
      this.items[index].material.baseColorFactor = [r, g, b, a];
      return true;
    }
    return false;
  }

  /**
   * 设置指定范围内所有网格的材质颜色
   */
  setMeshRangeColor(startIndex: number, count: number, r: number, g: number, b: number, a: number = 1): number {
    let modified = 0;
    for (let i = 0; i < count; i++) {
      if (this.setMeshColor(startIndex + i, r, g, b, a)) {
        modified++;
      }
    }
    return modified;
  }

  getCombinedBoundingBox(): MeshBoundingBox | null {
    if (this.items.length === 0) return null;

    let combinedMin: [number, number, number] | null = null;
    let combinedMax: [number, number, number] | null = null;

    for (const item of this.items) {
      const bbox = item.mesh.getWorldBoundingBox();
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

    const center: [number, number, number] = [
      (combinedMin[0] + combinedMax[0]) / 2,
      (combinedMin[1] + combinedMax[1]) / 2,
      (combinedMin[2] + combinedMax[2]) / 2,
    ];
    const dx = combinedMax[0] - combinedMin[0];
    const dy = combinedMax[1] - combinedMin[1];
    const dz = combinedMax[2] - combinedMin[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    return { min: combinedMin, max: combinedMax, center, radius };
  }

  destroy(): void {
    this.clear();
    if (this.defaultTexture) this.defaultTexture.destroy();
  }
}
