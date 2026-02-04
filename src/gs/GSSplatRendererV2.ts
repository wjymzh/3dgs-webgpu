/**
 * GSSplatRendererV2 - 优化的 3D Gaussian Splatting 渲染器
 * 
 * 基于 rfs-gsplat-render 的实现进行优化:
 * 1. GPU Radix Sort - O(n) 稳定排序
 * 2. Normalized Gaussian - 消除边缘雾化
 * 3. ClipCorner 优化 - 减少 overdraw
 * 4. MipSplatting 抗锯齿
 * 5. 改进的视锥剔除
 */

import { Renderer } from "../core/Renderer";
import { Camera } from "../core/Camera";
import { SplatCPU } from "./PLYLoader";
import { GSSplatSorter } from "./GSSplatSorter";
import { CompactSplatData, compactDataToGPUBuffer } from "./PLYLoaderMobile";
import {
  IGSSplatRenderer,
  BoundingBox as IBoundingBox,
  SHMode as ISHMode,
  RendererCapabilities,
  IGSSplatRendererWithCapabilities,
} from "./IGSSplatRenderer";

// 优化的 shader (内联)
const gsOptimizedShader = /* wgsl */ `
/**
 * 优化的 3D Gaussian Splatting Shader
 * 参考 rfs-gsplat-render 实现，修复颜色和抗锯齿问题
 */

const SQRT_8: f32 = 2.82842712475;
const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
// Normalized Gaussian 常量 (匹配 SuperSplat)
const EXP_NEG4: f32 = 0.01831563888873418;
const INV_ONE_MINUS_EXP_NEG4: f32 = 1.01865736036377408;
// 低通滤波器 (正则化协方差矩阵)
const LOW_PASS_FILTER: f32 = 0.3;
const ALPHA_CULL_THRESHOLD: f32 = 0.00392156863;

struct Uniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  model: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad: f32,
  screenSize: vec2<f32>,
  _pad2: vec2<f32>,
}

struct Splat {
  mean: vec3<f32>, _pad0: f32,
  scale: vec3<f32>, _pad1: f32,
  rotation: vec4<f32>,
  colorDC: vec3<f32>,
  opacity: f32,
  sh1: array<f32, 9>,
  sh2: array<f32, 15>,
  sh3: array<f32, 21>,
  _pad2: array<f32, 3>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) fragPos: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) opacity: f32,
}

const QUAD_POSITIONS = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
);

// ClipCorner 优化 (精确匹配 PlayCanvas/SuperSplat)
// 从 PlayCanvas: clip = min(1.0, sqrt(-log(1.0 / (255.0 * alpha))) / 2.0)
// 这根据透明度缩小 quad，排除 alpha < 1/255 的 Gaussian 区域
fn computeClipFactor(alpha: f32) -> f32 {
  // 保护非常小的 alpha 值
  // 当 alpha <= 1/255 时，splat 不可见
  if alpha <= ALPHA_CULL_THRESHOLD { return 0.0; }
  // PlayCanvas 公式: clip = min(1.0, sqrt(-log(1.0 / (255.0 * alpha))) / 2.0)
  // 简化: -log(1/(255*a)) = log(255*a)
  return min(1.0, sqrt(log(255.0 * alpha)) / 2.0);
}

// 四元数转旋转矩阵 (PLY 格式: w, x, y, z)
fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let r = q.x; let x = q.y; let y = q.z; let z = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),
    vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),
    vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y))
  );
}

fn computeCovariance3D(scale: vec3<f32>, rotation: vec4<f32>) -> mat3x3<f32> {
  let R = quatToMat3(rotation);
  let S = mat3x3<f32>(vec3<f32>(scale.x, 0.0, 0.0), vec3<f32>(0.0, scale.y, 0.0), vec3<f32>(0.0, 0.0, scale.z));
  let M = R * S;
  return M * transpose(M);
}

// 协方差投影 (匹配参考实现)
// 注意: viewCenter 是 vec4，直接使用 .xyz (不除以 w)
fn projectCovariance(cov3d: mat3x3<f32>, viewCenter: vec4<f32>, focal: vec2<f32>, modelViewMat: mat4x4<f32>) -> vec3<f32> {
  let v = viewCenter.xyz;  // 直接使用，不除以 w
  let s = 1.0 / (v.z * v.z);
  
  // Jacobian 矩阵
  let J = mat3x3<f32>(
    vec3<f32>(focal.x / v.z, 0.0, 0.0),
    vec3<f32>(0.0, focal.y / v.z, 0.0),
    vec3<f32>(-(focal.x * v.x) * s, -(focal.y * v.y) * s, 0.0)
  );
  
  // 从 model-view 矩阵提取 3x3 旋转部分 (匹配参考实现)
  let W = mat3x3<f32>(
    vec3<f32>(modelViewMat[0][0], modelViewMat[0][1], modelViewMat[0][2]),
    vec3<f32>(modelViewMat[1][0], modelViewMat[1][1], modelViewMat[1][2]),
    vec3<f32>(modelViewMat[2][0], modelViewMat[2][1], modelViewMat[2][2])
  );
  
  let T = J * W;
  let cov2d = T * cov3d * transpose(T);
  return vec3<f32>(cov2d[0][0], cov2d[0][1], cov2d[1][1]);
}

struct ExtentResult {
  basis: vec4<f32>,
  adjustedOpacity: f32,
}

// 计算 2D 投影范围
// 精确匹配 PlayCanvas/SuperSplat 实现
// 注意: MipSplatting 抗锯齿默认禁用，因为大多数模型不是用 MipSplatting 训练的
// 如果模型是用 MipSplatting 训练的，可以启用 GSPLAT_AA 模式
fn computeExtentBasisAA(cov2dIn: vec3<f32>, opacity: f32, viewportSize: vec2<f32>) -> ExtentResult {
  var result: ExtentResult;
  var cov2d = cov2dIn;
  var alpha = opacity;
  
  // 添加低通滤波 (正则化) - 匹配 PlayCanvas: +0.3
  // 这避免了非常小的特征值导致的数值问题
  cov2d.x += LOW_PASS_FILTER;
  cov2d.z += LOW_PASS_FILTER;
  
  // 特征值分解 (使用 PlayCanvas 公式)
  let a = cov2d.x;  // diagonal1
  let d = cov2d.z;  // diagonal2
  let b = cov2d.y;  // offDiagonal
  
  let mid = 0.5 * (a + d);
  let radius = length(vec2<f32>((a - d) * 0.5, b));
  
  let lambda1 = mid + radius;
  let lambda2 = max(mid - radius, 0.1);  // PlayCanvas 使用 0.1 最小值
  
  // 检查特征值是否有效
  if lambda2 <= 0.0 { 
    result.basis = vec4<f32>(0.0);
    result.adjustedOpacity = 0.0;
    return result;
  }
  
  // 使用基于视口的最大限制 (匹配 PlayCanvas)
  let vmin = min(1024.0, min(viewportSize.x, viewportSize.y));
  
  // 计算轴长度: l = 2.0 * min(sqrt(2.0 * lambda), vmin)
  // 这等价于 std_dev * sqrt(lambda)，因为 std_dev = sqrt(8) ≈ 2.83
  let l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin);
  let l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin);
  
  // 关键: 剔除小于 2 像素的 Gaussian (匹配 PlayCanvas)
  // 这消除了导致"雾化"伪影的亚像素 splat
  if l1 < 2.0 && l2 < 2.0 { 
    result.basis = vec4<f32>(0.0);
    result.adjustedOpacity = 0.0;
    return result;
  }
  
  // 从 offDiagonal 和特征值差计算特征向量
  // diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1))
  let diagVec = normalize(vec2<f32>(b, lambda1 - a));
  let eigenvector1 = diagVec;
  let eigenvector2 = vec2<f32>(diagVec.y, -diagVec.x);
  
  // 计算基向量 (不应用额外的 splat_scale，因为我们使用默认值 1.0)
  result.basis = vec4<f32>(eigenvector1 * l1, eigenvector2 * l2);
  result.adjustedOpacity = alpha;
  return result;
}

fn getModelScale3(model: mat4x4<f32>) -> vec3<f32> {
  return vec3<f32>(length(model[0].xyz), length(model[1].xyz), length(model[2].xyz));
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var output: VertexOutput;
  let splatIndex = sortedIndices[instanceIndex];
  let splat = splats[splatIndex];
  let quadPos = QUAD_POSITIONS[vertexIndex];
  
  // 透明度剔除
  if splat.opacity < ALPHA_CULL_THRESHOLD { output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0); return output; }
  
  // 四元数有效性检查
  let quatNormSqr = dot(splat.rotation, splat.rotation);
  if quatNormSqr < 1e-6 { output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0); return output; }
  
  // 变换到视图空间 (匹配参考实现: Local -> World -> View -> Clip)
  let worldPos = uniforms.model * vec4<f32>(splat.mean, 1.0);
  let viewPos = uniforms.view * worldPos;  // vec4, 保持 w 分量
  let clipPos = uniforms.proj * viewPos;
  
  // 近平面剔除 (viewPos.z 是负数，相机看向 -Z)
  if viewPos.z >= 0.0 { output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0); return output; }
  
  // NDC 计算
  let pW = 1.0 / (clipPos.w + 0.0000001);
  let ndcPos = clipPos * pW;
  
  // 视锥剔除 (放宽边界以避免 pop-in)
  let clipBound = 1.3;
  if abs(ndcPos.x) > clipBound || abs(ndcPos.y) > clipBound || ndcPos.z < -0.2 || ndcPos.z > 1.0 {
    output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0); return output;
  }
  
  // 计算 3D 协方差 (使用原始 scale，模型缩放通过 model-view 矩阵处理)
  // 关键: 不要在这里应用模型缩放，协方差投影会通过 model-view 矩阵正确处理
  let cov3d = computeCovariance3D(splat.scale, splat.rotation);
  
  // 计算焦距 (匹配参考实现: abs(proj[0][0]) * 0.5 * width)
  let focal = vec2<f32>(
    abs(uniforms.proj[0][0]) * 0.5 * uniforms.screenSize.x,
    abs(uniforms.proj[1][1]) * 0.5 * uniforms.screenSize.y
  );
  
  // 计算 model-view 矩阵 (匹配参考实现)
  let modelViewMat = uniforms.view * uniforms.model;
  
  // 投影协方差到 2D (传入 viewPos 作为 vec4，不除以 w)
  let cov2d = projectCovariance(cov3d, viewPos, focal, modelViewMat);
  
  // 计算范围基向量 (带抗锯齿)
  let extentResult = computeExtentBasisAA(cov2d, splat.opacity, uniforms.screenSize);
  let basis = extentResult.basis;
  let adjustedOpacity = extentResult.adjustedOpacity;
  
  if basis.x == 0.0 && basis.y == 0.0 && basis.z == 0.0 && basis.w == 0.0 {
    output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0); return output;
  }
  
  // 视锥边缘剔除 (匹配 PlayCanvas)
  let maxExtentPixels = max(length(basis.xy), length(basis.zw));
  let pixelToClip = vec2<f32>(clipPos.w, clipPos.w) / uniforms.screenSize;
  let splatExtentClip = vec2<f32>(maxExtentPixels, maxExtentPixels) * pixelToClip;
  if any((abs(clipPos.xy) - splatExtentClip) > vec2<f32>(clipPos.w, clipPos.w)) {
    output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0); return output;
  }
  
  // ClipCorner 优化 (匹配 PlayCanvas/SuperSplat)
  // 根据透明度缩小 quad，排除 alpha < 1/255 的区域
  let clipFactor = computeClipFactor(adjustedOpacity);
  if clipFactor <= 0.0 { output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0); return output; }
  
  // 计算最终顶点位置
  // basis_viewport: 从像素转换到 NDC 空间
  let basisViewport = vec2<f32>(1.0 / uniforms.screenSize.x, 1.0 / uniforms.screenSize.y);
  
  // 用 clipFactor 缩放基向量 (缩小 quad)
  let basisVector1 = basis.xy * clipFactor;
  let basisVector2 = basis.zw * clipFactor;
  
  // 计算 NDC 偏移
  // 注意: quadPos 在 [-1, 1] 范围内，clipFactor 只影响 quad 大小 (basis_vector)
  let ndcOffset = (quadPos.x * basisVector1 + quadPos.y * basisVector2) * basisViewport * 2.0;
  output.position = vec4<f32>(ndcPos.xy + ndcOffset, ndcPos.z, 1.0);
  
  // UV 输出 - 用 clipFactor 缩放以获得正确的 Gaussian 权重
  output.fragPos = quadPos * clipFactor;
  
  // 颜色已在 CPU 端预处理为 (dc * SH_C0 + 0.5)，直接使用
  // 这是 3DGS 的标准颜色格式，在 sRGB 空间中
  output.color = splat.colorDC;
  output.opacity = adjustedOpacity;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  if input.opacity <= 0.0 { discard; }
  
  // A = 到中心的平方距离，在 UV 空间中
  // 由于 clipCorner 优化，fragPos 在 [-clip, clip] 范围内
  let A = dot(input.fragPos, input.fragPos);
  
  // 丢弃单位圆外的片段
  if A > 1.0 { discard; }
  
  // Normalized Gaussian 衰减 (精确匹配 SuperSplat normExp)
  // 关键修复: 在 A=1 (边界) 时返回精确的 0.0，消除边缘雾化
  // 在 A=0 (中心): weight = 1.0
  // 在 A=1 (边界): weight = 精确的 0.0 (而不是标准 exp(-4) ≈ 0.018)
  let weight = (exp(-4.0 * A) - EXP_NEG4) * INV_ONE_MINUS_EXP_NEG4;
  
  // 组合 splat 透明度
  let opacity = weight * input.opacity;
  
  // Alpha 阈值丢弃 (匹配 SuperSplat: if (alpha < 1.0 / 255.0) discard)
  if opacity < ALPHA_CULL_THRESHOLD { discard; }
  
  // 颜色 clamp 到有效范围 (防止负值)
  let color = max(input.color, vec3<f32>(0.0));
  
  // 预乘 alpha 输出 (匹配 blend mode: src=ONE, dst=ONE_MINUS_SRC_ALPHA)
  // 这是 3DGS 渲染的标准混合模式
  return vec4<f32>(color * opacity, opacity);
}
`;

/**
 * SH 模式枚举
 */
export enum SHMode {
  L0 = 0,
  L1 = 1,
  L2 = 2,
  L3 = 3,
}

/**
 * Bounding Box 结构
 */
export interface BoundingBox {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  radius: number;
}

const SPLAT_BYTE_SIZE = 256;
const SPLAT_FLOAT_COUNT = 64;

/**
 * GSSplatRendererV2 - 优化的渲染器
 */
export class GSSplatRendererV2 implements IGSSplatRendererWithCapabilities {
  private renderer: Renderer;
  private camera: Camera;

  private pipeline!: GPURenderPipeline;
  private bindGroupLayout!: GPUBindGroupLayout;
  private uniformBuffer!: GPUBuffer;

  private splatBuffer: GPUBuffer | null = null;
  private splatCount: number = 0;
  private bindGroup: GPUBindGroup | null = null;

  private sorter: GSSplatSorter | null = null;
  private shMode: SHMode = SHMode.L0;
  private boundingBox: BoundingBox | null = null;

  // Transform
  private position: [number, number, number] = [0, 0, 0];
  private rotation: [number, number, number] = [0, 0, 0];
  private scale: [number, number, number] = [1, 1, 1];
  private pivot: [number, number, number] = [0, 0, 0];
  private modelMatrix: Float32Array = new Float32Array(16);

  // 剔除选项
  private pixelCullThreshold: number = 1.0;

  constructor(renderer: Renderer, camera: Camera) {
    this.renderer = renderer;
    this.camera = camera;
    this.createPipeline();
    this.createUniformBuffer();
    this.updateModelMatrix();
  }

  private createPipeline(): void {
    const device = this.renderer.device;

    const shaderModule = device.createShaderModule({
      code: gsOptimizedShader,
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

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
        targets: [{
          format: this.renderer.format,
          blend: {
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
          },
        }],
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
  }

  private createUniformBuffer(): void {
    // view (64) + proj (64) + model (64) + cameraPos (12) + pad (4) + screenSize (8) + pad (8) = 224
    this.uniformBuffer = this.renderer.device.createBuffer({
      size: 224,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  setPosition(x: number, y: number, z: number): void {
    this.position = [x, y, z];
    this.updateModelMatrix();
  }

  getPosition(): [number, number, number] {
    return [...this.position];
  }

  setRotation(x: number, y: number, z: number): void {
    this.rotation = [x, y, z];
    this.updateModelMatrix();
  }

  getRotation(): [number, number, number] {
    return [...this.rotation];
  }

  setScale(x: number, y: number, z: number): void {
    this.scale = [x, y, z];
    this.updateModelMatrix();
  }

  getScale(): [number, number, number] {
    return [...this.scale];
  }

  setPivot(x: number, y: number, z: number): void {
    this.pivot = [x, y, z];
    this.updateModelMatrix();
  }

  getPivot(): [number, number, number] {
    return [...this.pivot];
  }

  private updateModelMatrix(): void {
    const [tx, ty, tz] = this.position;
    const [rx, ry, rz] = this.rotation;
    const [sx, sy, sz] = this.scale;
    const [px, py, pz] = this.pivot;

    const cx = Math.cos(rx), sx1 = Math.sin(rx);
    const cy = Math.cos(ry), sy1 = Math.sin(ry);
    const cz = Math.cos(rz), sz1 = Math.sin(rz);

    const r00 = cy * cz;
    const r01 = sx1 * sy1 * cz - cx * sz1;
    const r02 = cx * sy1 * cz + sx1 * sz1;
    const r10 = cy * sz1;
    const r11 = sx1 * sy1 * sz1 + cx * cz;
    const r12 = cx * sy1 * sz1 - sx1 * cz;
    const r20 = -sy1;
    const r21 = sx1 * cy;
    const r22 = cx * cy;

    const rs00 = r00 * sx, rs01 = r01 * sy, rs02 = r02 * sz;
    const rs10 = r10 * sx, rs11 = r11 * sy, rs12 = r12 * sz;
    const rs20 = r20 * sx, rs21 = r21 * sy, rs22 = r22 * sz;

    const dpx = px - (rs00 * px + rs01 * py + rs02 * pz);
    const dpy = py - (rs10 * px + rs11 * py + rs12 * pz);
    const dpz = pz - (rs20 * px + rs21 * py + rs22 * pz);

    const finalTx = tx + dpx;
    const finalTy = ty + dpy;
    const finalTz = tz + dpz;

    this.modelMatrix[0] = rs00; this.modelMatrix[1] = rs10; this.modelMatrix[2] = rs20; this.modelMatrix[3] = 0;
    this.modelMatrix[4] = rs01; this.modelMatrix[5] = rs11; this.modelMatrix[6] = rs21; this.modelMatrix[7] = 0;
    this.modelMatrix[8] = rs02; this.modelMatrix[9] = rs12; this.modelMatrix[10] = rs22; this.modelMatrix[11] = 0;
    this.modelMatrix[12] = finalTx; this.modelMatrix[13] = finalTy; this.modelMatrix[14] = finalTz; this.modelMatrix[15] = 1;
  }

  getModelMatrix(): Float32Array {
    return this.modelMatrix;
  }

  setSHMode(mode: SHMode): void {
    this.shMode = mode;
  }

  getSHMode(): SHMode {
    return this.shMode;
  }

  setPixelCullThreshold(threshold: number): void {
    this.pixelCullThreshold = threshold;
  }

  setData(splats: SplatCPU[]): void {
    const device = this.renderer.device;

    if (this.splatBuffer) {
      this.splatBuffer.destroy();
    }
    if (this.sorter) {
      this.sorter.destroy();
      this.sorter = null;
    }

    this.splatCount = splats.length;

    if (this.splatCount === 0) {
      this.splatBuffer = null;
      this.bindGroup = null;
      this.boundingBox = null;
      return;
    }

    this.boundingBox = this.computeBoundingBox(splats);

    const data = new Float32Array(this.splatCount * SPLAT_FLOAT_COUNT);

    for (let i = 0; i < this.splatCount; i++) {
      const splat = splats[i];
      const offset = i * SPLAT_FLOAT_COUNT;

      data[offset + 0] = splat.mean[0];
      data[offset + 1] = splat.mean[1];
      data[offset + 2] = splat.mean[2];
      data[offset + 3] = 0;

      data[offset + 4] = splat.scale[0];
      data[offset + 5] = splat.scale[1];
      data[offset + 6] = splat.scale[2];
      data[offset + 7] = 0;

      data[offset + 8] = splat.rotation[0];
      data[offset + 9] = splat.rotation[1];
      data[offset + 10] = splat.rotation[2];
      data[offset + 11] = splat.rotation[3];

      data[offset + 12] = splat.colorDC[0];
      data[offset + 13] = splat.colorDC[1];
      data[offset + 14] = splat.colorDC[2];
      data[offset + 15] = splat.opacity;

      const shRest = splat.shRest;
      for (let j = 0; j < 9; j++) {
        data[offset + 16 + j] = shRest ? shRest[j] : 0;
      }
      for (let j = 0; j < 15; j++) {
        data[offset + 25 + j] = shRest ? shRest[9 + j] : 0;
      }
      for (let j = 0; j < 21; j++) {
        data[offset + 40 + j] = shRest ? shRest[24 + j] : 0;
      }
      data[offset + 61] = 0;
      data[offset + 62] = 0;
      data[offset + 63] = 0;
    }

    this.splatBuffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(this.splatBuffer, 0, data);

    this.sorter = new GSSplatSorter(
      device,
      this.splatCount,
      this.splatBuffer,
      this.uniformBuffer,
    );

    this.sorter.setScreenSize(this.renderer.width, this.renderer.height);
    this.sorter.setCullingOptions({
      nearPlane: this.camera.near,
      farPlane: this.camera.far,
      pixelThreshold: this.pixelCullThreshold,
    });

    this.bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.splatBuffer } },
        { binding: 2, resource: { buffer: this.sorter.getIndicesBuffer() } },
      ],
    });
  }

  setCompactData(compactData: CompactSplatData): void {
    const device = this.renderer.device;

    if (this.splatBuffer) {
      this.splatBuffer.destroy();
    }
    if (this.sorter) {
      this.sorter.destroy();
      this.sorter = null;
    }

    this.splatCount = compactData.count;

    if (this.splatCount === 0) {
      this.splatBuffer = null;
      this.bindGroup = null;
      this.boundingBox = null;
      return;
    }

    this.boundingBox = this.computeBoundingBoxFromCompact(compactData);

    const includeSH = compactData.shCoeffs !== undefined;
    const gpuData = compactDataToGPUBuffer(compactData, includeSH);

    this.splatBuffer = device.createBuffer({
      size: gpuData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(this.splatBuffer, 0, gpuData.buffer);

    this.sorter = new GSSplatSorter(
      device,
      this.splatCount,
      this.splatBuffer,
      this.uniformBuffer,
    );

    this.sorter.setScreenSize(this.renderer.width, this.renderer.height);
    this.sorter.setCullingOptions({
      nearPlane: this.camera.near,
      farPlane: this.camera.far,
      pixelThreshold: this.pixelCullThreshold,
    });

    this.bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.splatBuffer } },
        { binding: 2, resource: { buffer: this.sorter.getIndicesBuffer() } },
      ],
    });
  }

  render(pass: GPURenderPassEncoder): void {
    if (this.splatCount === 0 || !this.bindGroup || !this.sorter) {
      return;
    }

    // 更新 uniforms
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer, 0,
      new Float32Array(this.camera.viewMatrix),
    );
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer, 64,
      new Float32Array(this.camera.projectionMatrix),
    );
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer, 128,
      new Float32Array(this.modelMatrix),
    );
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer, 192,
      new Float32Array(this.camera.position),
    );
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer, 208,
      new Float32Array([this.renderer.width, this.renderer.height, 0, 0]),
    );

    // 更新排序器参数
    this.sorter.setScreenSize(this.renderer.width, this.renderer.height);
    this.sorter.setCullingOptions({
      nearPlane: this.camera.near,
      farPlane: this.camera.far,
      pixelThreshold: this.pixelCullThreshold,
    });

    // 执行 GPU 排序
    this.sorter.sort();

    // 渲染
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.drawIndirect(this.sorter.getDrawIndirectBuffer(), 0);
  }

  getSplatCount(): number {
    return this.splatCount;
  }

  getBoundingBox(): BoundingBox | null {
    return this.boundingBox;
  }

  private computeBoundingBox(splats: SplatCPU[]): BoundingBox {
    if (splats.length === 0) {
      return { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0], radius: 0 };
    }

    const min: [number, number, number] = [splats[0].mean[0], splats[0].mean[1], splats[0].mean[2]];
    const max: [number, number, number] = [splats[0].mean[0], splats[0].mean[1], splats[0].mean[2]];

    for (let i = 1; i < splats.length; i++) {
      const [x, y, z] = splats[i].mean;
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

  private computeBoundingBoxFromCompact(data: CompactSplatData): BoundingBox {
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

  supportsSHMode(mode: ISHMode): boolean {
    return mode >= ISHMode.L0 && mode <= ISHMode.L3;
  }

  getCapabilities(): RendererCapabilities {
    return {
      maxSHMode: ISHMode.L3,
      supportsRawData: true,
      isMobileOptimized: false,
      maxSplatCount: 0,
    };
  }

  destroy(): void {
    if (this.splatBuffer) {
      this.splatBuffer.destroy();
      this.splatBuffer = null;
    }
    if (this.sorter) {
      this.sorter.destroy();
      this.sorter = null;
    }
    this.uniformBuffer.destroy();
    this.splatCount = 0;
    this.bindGroup = null;
  }
}
