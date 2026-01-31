/**
 * GSSplatRendererMobile - 移动端优化的 3D Gaussian Splatting 渲染器
 *
 * 优化特点：
 * 1. 使用纹理存储 splat 数据，减少 GPU 内存占用 (~52 bytes/splat vs 256 bytes)
 * 2. 仅支持 L0 模式（无 SH 计算）
 * 3. 从纹理采样获取 splat 属性（使用 RGBA32Float 保证精度）
 * 4. 使用简化的排序器
 */

import { Renderer } from "../core/Renderer";
import { Camera } from "../core/Camera";
import { CompactSplatData } from "./PLYLoaderMobile";
import {
  CompressedSplatTextures,
  compressSplatsToTextures,
  destroyCompressedTextures,
} from "./TextureCompressor";
import { GSSplatSorterMobile } from "./GSSplatSorterMobile";

/**
 * Bounding Box 结构
 */
export interface BoundingBox {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  radius: number;
}

// ============================================
// 移动端 L0 Shader - 从纹理采样数据（简化版）
// ============================================
const shaderCodeMobileL0 = /* wgsl */ `
struct Uniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  model: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad: f32,
  screenSize: vec2<f32>,
  _pad2: vec2<f32>,
  textureSize: vec2<f32>,  // 纹理尺寸 (用于坐标计算)
  _pad3: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> sortedIndices: array<u32>;

// 纹理绑定 - 4 张纹理（使用 RGBA32Float 保证精度）
@group(1) @binding(0) var positionTex: texture_2d<f32>;   // RGBA32Float: xyz + unused
@group(1) @binding(1) var scaleRotTex1: texture_2d<f32>;  // RGBA32Float: scale_xyz + rot_w
@group(1) @binding(2) var scaleRotTex2: texture_2d<f32>;  // RGBA32Float: rot_xyz + unused
@group(1) @binding(3) var colorTex: texture_2d<f32>;      // RGBA8Unorm: rgb + opacity

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localUV: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) opacity: f32,
}

const QUAD_POSITIONS = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0,  1.0),
);

const ELLIPSE_SCALE: f32 = 3.0;

// 将索引转换为纹理坐标
fn indexToTexCoord(index: u32) -> vec2<u32> {
  let texWidth = u32(uniforms.textureSize.x);
  let x = index % texWidth;
  let y = index / texWidth;
  return vec2<u32>(x, y);
}

// 四元数转旋转矩阵
fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let w = q[0]; let x = q[1]; let y = q[2]; let z = q[3];
  let x2 = x + x; let y2 = y + y; let z2 = z + z;
  let xx = x * x2; let xy = x * y2; let xz = x * z2;
  let yy = y * y2; let yz = y * z2; let zz = z * z2;
  let wx = w * x2; let wy = w * y2; let wz = w * z2;
  return mat3x3<f32>(
    vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
    vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
    vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy))
  );
}

// 从模型矩阵提取统一缩放因子（取 X 轴向量长度）
fn getModelScale(model: mat4x4<f32>) -> f32 {
  return length(model[0].xyz);
}

// 计算 2D 协方差
fn computeCov2D(mean: vec3<f32>, scale: vec3<f32>, rotation: vec4<f32>, modelView: mat4x4<f32>, proj: mat4x4<f32>, modelScale: f32) -> vec3<f32> {
  let R = quatToMat3(rotation);
  // 应用模型缩放到 splat scale
  let scaledScale = scale * modelScale;
  let s2 = scaledScale * scaledScale;
  let M = mat3x3<f32>(R[0] * s2.x, R[1] * s2.y, R[2] * s2.z);
  let Sigma = M * transpose(R);
  let viewPos = (modelView * vec4<f32>(mean, 1.0)).xyz;
  let viewRot = mat3x3<f32>(modelView[0].xyz, modelView[1].xyz, modelView[2].xyz);
  let SigmaView = viewRot * Sigma * transpose(viewRot);
  let fx = proj[0][0]; let fy = proj[1][1];
  let z = -viewPos.z;
  let z_clamped = max(z, 0.001);
  let z2 = z_clamped * z_clamped;
  // 雅可比矩阵: 从相机坐标 (x_cam, y_cam, z_cam) 到 NDC 的偏导数
  // x_ndc = fx * x_cam / (-z_cam), 所以 dx_ndc/dz_cam = fx * x_cam / z_cam^2 (正号!)
  let j1 = vec3<f32>(fx / z_clamped, 0.0, fx * viewPos.x / z2);
  let j2 = vec3<f32>(0.0, fy / z_clamped, fy * viewPos.y / z2);
  let Sj1 = SigmaView * j1;
  let Sj2 = SigmaView * j2;
  return vec3<f32>(dot(j1, Sj1), dot(j1, Sj2), dot(j2, Sj2));
}

// 计算椭圆轴
fn computeEllipseAxes(cov2D: vec3<f32>) -> mat2x2<f32> {
  let a = cov2D.x; let b = cov2D.y; let c = cov2D.z;
  let trace = a + c;
  let det = a * c - b * b;
  let disc = trace * trace - 4.0 * det;
  let sqrtDisc = sqrt(max(disc, 0.0));
  let lambda1 = max((trace + sqrtDisc) * 0.5, 0.0);
  let lambda2 = max((trace - sqrtDisc) * 0.5, 0.0);
  let r1 = sqrt(lambda1);
  let r2 = sqrt(lambda2);
  var axis1: vec2<f32>; var axis2: vec2<f32>;
  if (abs(b) > 1e-6) {
    axis1 = normalize(vec2<f32>(b, lambda1 - a));
    axis2 = vec2<f32>(-axis1.y, axis1.x);
  } else {
    if (a >= c) { axis1 = vec2<f32>(1.0, 0.0); axis2 = vec2<f32>(0.0, 1.0); }
    else { axis1 = vec2<f32>(0.0, 1.0); axis2 = vec2<f32>(1.0, 0.0); }
  }
  return mat2x2<f32>(axis1 * r1, axis2 * r2);
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var output: VertexOutput;
  
  // 获取排序后的索引
  let splatIndex = sortedIndices[instanceIndex];
  let texCoord = indexToTexCoord(splatIndex);
  
  // 从纹理采样位置数据（RGBA32Float，直接读取）
  let posSample = textureLoad(positionTex, texCoord, 0);
  let mean = posSample.xyz;
  
  // 从纹理采样缩放和旋转（RGBA16Float，GPU 自动转换为 f32）
  let scaleRot1 = textureLoad(scaleRotTex1, texCoord, 0);
  let scaleRot2 = textureLoad(scaleRotTex2, texCoord, 0);
  
  let scale = scaleRot1.xyz;
  let rotation = vec4<f32>(scaleRot1.w, scaleRot2.x, scaleRot2.y, scaleRot2.z);
  
  // 从纹理采样颜色（RGBA8Unorm，GPU 自动归一化到 0-1）
  let colorSample = textureLoad(colorTex, texCoord, 0);
  let color = colorSample.rgb;
  let opacity = colorSample.a;
  
  // 计算顶点位置
  let quadPos = QUAD_POSITIONS[vertexIndex];
  output.localUV = quadPos;
  
  // 计算 modelView 矩阵和模型缩放
  let modelView = uniforms.view * uniforms.model;
  let modelScale = getModelScale(uniforms.model);
  
  let cov2D = computeCov2D(mean, scale, rotation, modelView, uniforms.proj, modelScale);
  let axes = computeEllipseAxes(cov2D);
  let screenOffset = axes[0] * quadPos.x * ELLIPSE_SCALE + axes[1] * quadPos.y * ELLIPSE_SCALE;
  
  // 应用 model 变换到 splat 位置
  let worldPos = uniforms.model * vec4<f32>(mean, 1.0);
  let viewPos = uniforms.view * worldPos;
  var clipPos = uniforms.proj * viewPos;
  clipPos.x = clipPos.x + screenOffset.x * clipPos.w;
  clipPos.y = clipPos.y + screenOffset.y * clipPos.w;
  output.position = clipPos;
  output.color = color;
  output.opacity = opacity;
  
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let r = length(input.localUV);
  if (r > 1.0) { discard; }
  let gaussianWeight = exp(-r * r * 4.0);
  let alpha = input.opacity * gaussianWeight;
  if (alpha < 0.004) { discard; }  // 丢弃几乎透明的像素
  let color = clamp(input.color, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(color * alpha, alpha);
}
`;

/**
 * GSSplatRendererMobile - 移动端优化渲染器
 */
export class GSSplatRendererMobile {
  private renderer: Renderer;
  private camera: Camera;

  // GPU 资源
  private pipeline!: GPURenderPipeline;
  private uniformBindGroupLayout!: GPUBindGroupLayout;
  private textureBindGroupLayout!: GPUBindGroupLayout;
  private uniformBuffer!: GPUBuffer;
  private uniformBindGroup: GPUBindGroup | null = null;
  private textureBindGroup: GPUBindGroup | null = null;

  // 压缩纹理数据
  private compressedTextures: CompressedSplatTextures | null = null;
  private splatCount: number = 0;

  // 排序器
  private sorter: GSSplatSorterMobile | null = null;

  // 位置缓冲区（用于排序）
  private positionsBuffer: GPUBuffer | null = null;

  // Bounding box
  private boundingBox: BoundingBox | null = null;

  // 帧计数（用于排序频率控制）
  private frameCount: number = 0;
  private sortEveryNFrames: number = 1;

  // ============================================
  // 变换相关 (position, rotation, scale)
  // ============================================
  private position: [number, number, number] = [0, 0, 0];
  private rotation: [number, number, number] = [0, 0, 0]; // Euler angles (radians)
  private scaleValue: [number, number, number] = [1, 1, 1];
  private pivot: [number, number, number] = [0, 0, 0]; // 旋转/缩放中心点
  private modelMatrix: Float32Array = new Float32Array(16); // 4x4 model matrix

  constructor(renderer: Renderer, camera: Camera) {
    this.renderer = renderer;
    this.camera = camera;

    this.createPipeline();
    this.createUniformBuffer();
    this.updateModelMatrix(); // 初始化模型矩阵为单位矩阵

    console.log("GSSplatRendererMobile: 移动端纹理压缩渲染器已初始化");
  }

  // ============================================
  // Transform 方法
  // ============================================

  /**
   * 设置位置
   */
  setPosition(x: number, y: number, z: number): void {
    this.position = [x, y, z];
    this.updateModelMatrix();
  }

  /**
   * 获取位置
   */
  getPosition(): [number, number, number] {
    return [...this.position];
  }

  /**
   * 设置旋转 (欧拉角, 弧度)
   */
  setRotation(x: number, y: number, z: number): void {
    this.rotation = [x, y, z];
    this.updateModelMatrix();
  }

  /**
   * 获取旋转
   */
  getRotation(): [number, number, number] {
    return [...this.rotation];
  }

  /**
   * 设置缩放
   */
  setScale(x: number, y: number, z: number): void {
    this.scaleValue = [x, y, z];
    this.updateModelMatrix();
  }

  /**
   * 获取缩放
   */
  getScale(): [number, number, number] {
    return [...this.scaleValue];
  }

  /**
   * 设置旋转/缩放中心点 (pivot)
   */
  setPivot(x: number, y: number, z: number): void {
    this.pivot = [x, y, z];
    this.updateModelMatrix();
  }

  /**
   * 获取旋转/缩放中心点 (pivot)
   */
  getPivot(): [number, number, number] {
    return [...this.pivot];
  }

  /**
   * 更新模型矩阵
   * 变换顺序: T * Tp * R * S * Tp^-1
   * 即: 先移到原点，缩放，旋转，再移回pivot，最后应用用户平移
   */
  private updateModelMatrix(): void {
    const [tx, ty, tz] = this.position;
    const [rx, ry, rz] = this.rotation;
    const [sx, sy, sz] = this.scaleValue;
    const [px, py, pz] = this.pivot;

    // 计算旋转矩阵分量 (Euler XYZ 顺序)
    const cx = Math.cos(rx), sx1 = Math.sin(rx);
    const cy = Math.cos(ry), sy1 = Math.sin(ry);
    const cz = Math.cos(rz), sz1 = Math.sin(rz);

    // 组合旋转矩阵 R = Rz * Ry * Rx
    const r00 = cy * cz;
    const r01 = sx1 * sy1 * cz - cx * sz1;
    const r02 = cx * sy1 * cz + sx1 * sz1;
    const r10 = cy * sz1;
    const r11 = sx1 * sy1 * sz1 + cx * cz;
    const r12 = cx * sy1 * sz1 - sx1 * cz;
    const r20 = -sy1;
    const r21 = sx1 * cy;
    const r22 = cx * cy;

    // RS 矩阵 (旋转 * 缩放)
    const rs00 = r00 * sx, rs01 = r01 * sy, rs02 = r02 * sz;
    const rs10 = r10 * sx, rs11 = r11 * sy, rs12 = r12 * sz;
    const rs20 = r20 * sx, rs21 = r21 * sy, rs22 = r22 * sz;

    // 计算 (I - RS) * pivot
    const dpx = px - (rs00 * px + rs01 * py + rs02 * pz);
    const dpy = py - (rs10 * px + rs11 * py + rs12 * pz);
    const dpz = pz - (rs20 * px + rs21 * py + rs22 * pz);

    // 最终平移 = position + (I - RS) * pivot
    const finalTx = tx + dpx;
    const finalTy = ty + dpy;
    const finalTz = tz + dpz;

    // 模型矩阵 (列主序)
    this.modelMatrix[0] = rs00;
    this.modelMatrix[1] = rs10;
    this.modelMatrix[2] = rs20;
    this.modelMatrix[3] = 0;

    this.modelMatrix[4] = rs01;
    this.modelMatrix[5] = rs11;
    this.modelMatrix[6] = rs21;
    this.modelMatrix[7] = 0;

    this.modelMatrix[8] = rs02;
    this.modelMatrix[9] = rs12;
    this.modelMatrix[10] = rs22;
    this.modelMatrix[11] = 0;

    this.modelMatrix[12] = finalTx;
    this.modelMatrix[13] = finalTy;
    this.modelMatrix[14] = finalTz;
    this.modelMatrix[15] = 1;
  }

  /**
   * 获取当前模型矩阵
   */
  getModelMatrix(): Float32Array {
    return this.modelMatrix;
  }

  /**
   * 创建渲染管线
   */
  private createPipeline(): void {
    const device = this.renderer.device;

    // 创建 shader 模块
    const shaderModule = device.createShaderModule({
      code: shaderCodeMobileL0,
      label: "mobile-splat-shader",
    });

    // Uniform bind group layout (group 0)
    this.uniformBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    // Texture bind group layout (group 1) - 简化版 4 张纹理
    this.textureBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          // positionTex (RGBA32Float)
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          texture: { sampleType: "unfilterable-float" },
        },
        {
          // scaleRotTex1 (RGBA32Float)
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          texture: { sampleType: "unfilterable-float" },
        },
        {
          // scaleRotTex2 (RGBA32Float)
          binding: 2,
          visibility: GPUShaderStage.VERTEX,
          texture: { sampleType: "unfilterable-float" },
        },
        {
          // colorTex (RGBA8Unorm)
          binding: 3,
          visibility: GPUShaderStage.VERTEX,
          texture: { sampleType: "unfilterable-float" },
        },
      ],
    });

    // Pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.uniformBindGroupLayout, this.textureBindGroupLayout],
    });

    // Blend state
    const blendState: GPUBlendState = {
      color: {
        srcFactor: "one",
        dstFactor: "one-minus-src-alpha",
        operation: "add",
      },
      alpha: {
        srcFactor: "one",
        dstFactor: "one-minus-src-alpha",
        operation: "add",
      },
    };

    // 创建管线
    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: this.renderer.format,
            blend: blendState,
          },
        ],
      },
      primitive: {
        topology: "triangle-strip",
      },
      depthStencil: {
        format: this.renderer.depthFormat,
        depthWriteEnabled: false,
        depthCompare: "always",
      },
    });

    console.log("GSSplatRendererMobile: 渲染管线已创建");
  }

  /**
   * 创建 uniform buffer
   * 布局: view (64) + proj (64) + model (64) + cameraPos (12) + pad (4) + screenSize (8) + pad (8) + textureSize (8) + pad (8) = 240 bytes
   */
  private createUniformBuffer(): void {
    this.uniformBuffer = this.renderer.device.createBuffer({
      size: 240,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * 设置紧凑格式的 splat 数据
   * @param data 紧凑格式的 splat 数据
   */
  setCompactData(data: CompactSplatData): void {
    console.log(`GSSplatRendererMobile: 开始处理 ${data.count} 个 splats`);

    try {
      const device = this.renderer.device;

      // 销毁旧资源
      this.destroyInternal();

      this.splatCount = data.count;
      this.frameCount = 0;

      if (this.splatCount === 0) {
        console.warn("GSSplatRendererMobile: splat 数量为 0");
        return;
      }

      // 压缩数据到纹理
      console.log("GSSplatRendererMobile: 压缩数据到纹理...");
      this.compressedTextures = compressSplatsToTextures(device, data);

      // 计算 bounding box
      this.boundingBox = this.computeBoundingBox(data);
      console.log(`GSSplatRendererMobile: BoundingBox center=[${this.boundingBox.center.map(v => v.toFixed(2)).join(", ")}], radius=${this.boundingBox.radius.toFixed(2)}`);

      // 创建位置缓冲区（用于排序）
      console.log("GSSplatRendererMobile: 创建位置缓冲区用于排序...");
      this.positionsBuffer = device.createBuffer({
        size: data.count * 12, // 3 floats per position
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      // 使用 new Float32Array 确保是标准 ArrayBuffer
      device.queue.writeBuffer(this.positionsBuffer, 0, new Float32Array(data.positions));

      // 创建排序器
      console.log("GSSplatRendererMobile: 创建排序器...");
      this.sorter = new GSSplatSorterMobile(
        device,
        this.splatCount,
        this.positionsBuffer,
        this.uniformBuffer
      );

      this.sorter.setScreenSize(this.renderer.width, this.renderer.height);
      this.sorter.setCullingOptions({
        nearPlane: this.camera.near,
        farPlane: this.camera.far,
        pixelThreshold: 1.0,
      });

      // 创建 bind groups
      this.createBindGroups();

      const memoryMB = (
        this.compressedTextures.width *
        this.compressedTextures.height *
        52 / // 约 52 bytes per texel (16+16+16+4)
        (1024 * 1024)
      ).toFixed(2);
      console.log(`GSSplatRendererMobile: 已加载 ${this.splatCount} 个 splats，GPU 内存约 ${memoryMB} MB`);
    } catch (error) {
      console.error("GSSplatRendererMobile.setCompactData 错误:", error);
      this.splatCount = 0;
      this.compressedTextures = null;
      this.sorter = null;
    }
  }

  /**
   * 创建 bind groups
   */
  private createBindGroups(): void {
    if (!this.compressedTextures || !this.sorter) return;

    const device = this.renderer.device;

    // Uniform bind group (group 0)
    this.uniformBindGroup = device.createBindGroup({
      layout: this.uniformBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: this.uniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: this.sorter.getIndicesBuffer() },
        },
      ],
    });

    // Texture bind group (group 1) - 简化版 4 张纹理
    this.textureBindGroup = device.createBindGroup({
      layout: this.textureBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: this.compressedTextures.positionTexture.createView(),
        },
        {
          binding: 1,
          resource: this.compressedTextures.scaleRotTexture1.createView(),
        },
        {
          binding: 2,
          resource: this.compressedTextures.scaleRotTexture2.createView(),
        },
        {
          binding: 3,
          resource: this.compressedTextures.colorTexture.createView(),
        },
      ],
    });
  }

  /**
   * 计算 bounding box
   */
  private computeBoundingBox(data: CompactSplatData): BoundingBox {
    if (data.count === 0) {
      return { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0], radius: 0 };
    }

    const positions = data.positions;
    const min: [number, number, number] = [positions[0], positions[1], positions[2]];
    const max: [number, number, number] = [positions[0], positions[1], positions[2]];

    for (let i = 1; i < data.count; i++) {
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

    const center: [number, number, number] = [
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    ];

    const dx = max[0] - min[0];
    const dy = max[1] - min[1];
    const dz = max[2] - min[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    return { min, max, center, radius };
  }

  /**
   * 渲染
   * @param pass 渲染通道编码器
   */
  render(pass: GPURenderPassEncoder): void {
    if (
      this.splatCount === 0 ||
      !this.uniformBindGroup ||
      !this.textureBindGroup ||
      !this.sorter ||
      !this.compressedTextures
    ) {
      return;
    }

    this.frameCount++;

    // 更新 uniform buffer
    const device = this.renderer.device;
    device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array(this.camera.viewMatrix));
    device.queue.writeBuffer(this.uniformBuffer, 64, new Float32Array(this.camera.projectionMatrix));
    device.queue.writeBuffer(this.uniformBuffer, 128, new Float32Array(this.modelMatrix));
    device.queue.writeBuffer(this.uniformBuffer, 192, new Float32Array(this.camera.position));
    device.queue.writeBuffer(
      this.uniformBuffer,
      208,
      new Float32Array([this.renderer.width, this.renderer.height, 0, 0])
    );
    device.queue.writeBuffer(
      this.uniformBuffer,
      224,
      new Float32Array([this.compressedTextures.width, this.compressedTextures.height, 0, 0])
    );

    // 更新排序器参数
    this.sorter.setScreenSize(this.renderer.width, this.renderer.height);
    this.sorter.setCullingOptions({
      nearPlane: this.camera.near,
      farPlane: this.camera.far,
      pixelThreshold: 1.0,
    });

    // 排序（第一帧必须排序）
    const isFirstFrame = this.frameCount === 1;
    const shouldSort = isFirstFrame || this.frameCount % this.sortEveryNFrames === 0;

    if (shouldSort) {
      this.sorter.sort();
    }

    // 渲染
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.uniformBindGroup);
    pass.setBindGroup(1, this.textureBindGroup);
    pass.drawIndirect(this.sorter.getDrawIndirectBuffer(), 0);
  }

  /**
   * 获取 splat 数量
   */
  getSplatCount(): number {
    return this.splatCount;
  }

  /**
   * 获取 bounding box
   */
  getBoundingBox(): BoundingBox | null {
    return this.boundingBox;
  }

  /**
   * 设置排序频率
   * @param n 每 n 帧排序一次
   */
  setSortFrequency(n: number): void {
    this.sortEveryNFrames = Math.max(1, n);
  }

  /**
   * 内部销毁资源（不销毁管线）
   */
  private destroyInternal(): void {
    if (this.compressedTextures) {
      destroyCompressedTextures(this.compressedTextures);
      this.compressedTextures = null;
    }

    if (this.sorter) {
      this.sorter.destroy();
      this.sorter = null;
    }

    if (this.positionsBuffer) {
      this.positionsBuffer.destroy();
      this.positionsBuffer = null;
    }

    this.uniformBindGroup = null;
    this.textureBindGroup = null;
    this.splatCount = 0;
    this.boundingBox = null;
  }

  /**
   * 销毁资源
   */
  destroy(): void {
    this.destroyInternal();
  }
}
