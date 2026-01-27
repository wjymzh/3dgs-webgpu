import { Renderer } from "../core/Renderer";
import { Camera } from "../core/Camera";
import { SplatCPU } from "./PLYLoader";
import { GSSplatSorter } from "./GSSplatSorter";
import { CompactSplatData, compactDataToGPUBuffer } from "./PLYLoaderMobile";
import {
  compressSplatsToTextures,
  destroyCompressedTextures,
  CompressedSplatTextures,
} from "./TextureCompressor";

/**
 * 检测是否为移动设备
 */
function isMobileDevice(): boolean {
  if (typeof navigator === "undefined") return false;
  const ua =
    navigator.userAgent || navigator.vendor || (window as any).opera || "";

  // 检测移动设备
  const isMobileUA =
    /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(
      ua.toLowerCase(),
    );

  // 额外检测：触摸屏 + 小屏幕 = 移动设备
  const hasTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0;
  const isSmallScreen = window.innerWidth <= 768;

  // 检测 iPad 伪装成 Mac 的情况
  const isIPadAsMac =
    navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;

  const result = isMobileUA || isIPadAsMac || (hasTouch && isSmallScreen);

  console.log(
    `isMobileDevice: UA检测=${isMobileUA}, 触摸=${hasTouch}, 小屏=${isSmallScreen}, iPad伪装=${isIPadAsMac} => ${result}`,
  );

  return result;
}

/**
 * 性能等级
 */
export enum PerformanceTier {
  HIGH = "high", // 桌面高端 GPU
  MEDIUM = "medium", // 桌面中端
  LOW = "low", // 移动设备（所有移动设备都用 LOW！）
}

/**
 * 根据设备能力检测性能等级
 */
function detectPerformanceTier(device: GPUDevice): PerformanceTier {
  const isMobile = isMobileDevice();
  const maxBufferSize = device.limits.maxBufferSize;
  const maxStorageBufferBindingSize = device.limits.maxStorageBufferBindingSize;

  console.log(
    `detectPerformanceTier: isMobile=${isMobile}, maxBufferSize=${(maxBufferSize / 1024 / 1024).toFixed(0)}MB, maxStorageBuffer=${(maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)}MB`,
  );

  // 移动设备一律使用 LOW！
  // 即使 GPU 报告支持大 buffer，移动端的散热和功耗限制也会导致崩溃
  if (isMobile) {
    console.log("移动设备检测到，强制使用 LOW 性能等级");
    return PerformanceTier.LOW;
  }

  // 桌面设备根据 buffer 大小判断
  if (maxStorageBufferBindingSize >= 1024 * 1024 * 1024) {
    // 1GB
    return PerformanceTier.HIGH;
  } else if (maxStorageBufferBindingSize >= 256 * 1024 * 1024) {
    // 256MB
    return PerformanceTier.MEDIUM;
  }
  return PerformanceTier.LOW;
}

/**
 * 移动端优化配置
 */
export interface MobileOptimizationConfig {
  // 最大渲染 splat 数量
  maxVisibleSplats: number;
  // 是否启用排序（关闭可大幅提升性能）
  enableSorting: boolean;
  // 排序频率（每 N 帧排序一次）
  sortEveryNFrames: number;
  // 是否使用紧凑数据格式
  useCompactFormat: boolean;
  // 像素剔除阈值（更大的值剔除更多小 splat）
  pixelCullThreshold: number;
  // SH 模式（移动端建议 L0）
  defaultSHMode: SHMode;
}

// ============================================
// L0 Shader (仅 DC 颜色，无 SH 计算 - 高性能模式)
// ============================================
const shaderCodeL0 = /* wgsl */ `
struct Uniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad: f32,
  screenSize: vec2<f32>,
  _pad2: vec2<f32>,
}

struct Splat {
  mean:     vec3<f32>,
  _pad0:    f32,
  scale:    vec3<f32>,
  _pad1:    f32,
  rotation: vec4<f32>,
  colorDC:  vec3<f32>,
  opacity:  f32,
  sh1:      array<f32, 9>,
  sh2:      array<f32, 15>,
  sh3:      array<f32, 21>,
  _pad2:    array<f32, 3>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;

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

fn computeCov2D(mean: vec3<f32>, scale: vec3<f32>, rotation: vec4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>) -> vec3<f32> {
  let R = quatToMat3(rotation);
  let s2 = scale * scale;
  let M = mat3x3<f32>(R[0] * s2.x, R[1] * s2.y, R[2] * s2.z);
  let Sigma = M * transpose(R);
  let viewPos = (view * vec4<f32>(mean, 1.0)).xyz;
  let viewRot = mat3x3<f32>(view[0].xyz, view[1].xyz, view[2].xyz);
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
  let splatIndex = sortedIndices[instanceIndex];
  let splat = splats[splatIndex];
  let quadPos = QUAD_POSITIONS[vertexIndex];
  output.localUV = quadPos;
  
  let cov2D = computeCov2D(splat.mean, splat.scale, splat.rotation, uniforms.view, uniforms.proj);
  let axes = computeEllipseAxes(cov2D);
  let screenOffset = axes[0] * quadPos.x * ELLIPSE_SCALE + axes[1] * quadPos.y * ELLIPSE_SCALE;
  
  let viewPos = uniforms.view * vec4<f32>(splat.mean, 1.0);
  var clipPos = uniforms.proj * viewPos;
  clipPos.x = clipPos.x + screenOffset.x * clipPos.w;
  clipPos.y = clipPos.y + screenOffset.y * clipPos.w;
  output.position = clipPos;
  output.color = splat.colorDC;
  output.opacity = splat.opacity;
  
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let r = length(input.localUV);
  if (r > 1.0) { discard; }
  let gaussianWeight = exp(-r * r * 4.0);
  let alpha = input.opacity * gaussianWeight;
  return vec4<f32>(input.color * alpha, alpha);
}
`;

// ============================================
// L1 Shader (DC + L1 SH)
// ============================================
const shaderCodeL1 = /* wgsl */ `
struct Uniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad: f32,
  screenSize: vec2<f32>,
  _pad2: vec2<f32>,
}

struct Splat {
  mean:     vec3<f32>,
  _pad0:    f32,
  scale:    vec3<f32>,
  _pad1:    f32,
  rotation: vec4<f32>,
  colorDC:  vec3<f32>,
  opacity:  f32,
  sh1:      array<f32, 9>,
  sh2:      array<f32, 15>,
  sh3:      array<f32, 21>,
  _pad2:    array<f32, 3>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;

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
const SH_C1: f32 = 0.4886025119029199;

fn evalSH1(dir: vec3<f32>, sh1: array<f32, 9>) -> vec3<f32> {
  let basis = vec3<f32>(dir.y, dir.z, dir.x) * SH_C1;
  let r = sh1[0] * basis.x + sh1[1] * basis.y + sh1[2] * basis.z;
  let g = sh1[3] * basis.x + sh1[4] * basis.y + sh1[5] * basis.z;
  let b = sh1[6] * basis.x + sh1[7] * basis.y + sh1[8] * basis.z;
  return vec3<f32>(r, g, b);
}

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

fn computeCov2D(mean: vec3<f32>, scale: vec3<f32>, rotation: vec4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>) -> vec3<f32> {
  let R = quatToMat3(rotation);
  let s2 = scale * scale;
  let M = mat3x3<f32>(R[0] * s2.x, R[1] * s2.y, R[2] * s2.z);
  let Sigma = M * transpose(R);
  let viewPos = (view * vec4<f32>(mean, 1.0)).xyz;
  let viewRot = mat3x3<f32>(view[0].xyz, view[1].xyz, view[2].xyz);
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
  let splatIndex = sortedIndices[instanceIndex];
  let splat = splats[splatIndex];
  let quadPos = QUAD_POSITIONS[vertexIndex];
  output.localUV = quadPos;
  
  let cov2D = computeCov2D(splat.mean, splat.scale, splat.rotation, uniforms.view, uniforms.proj);
  let axes = computeEllipseAxes(cov2D);
  let screenOffset = axes[0] * quadPos.x * ELLIPSE_SCALE + axes[1] * quadPos.y * ELLIPSE_SCALE;
  
  let viewPos = uniforms.view * vec4<f32>(splat.mean, 1.0);
  var clipPos = uniforms.proj * viewPos;
  clipPos.x = clipPos.x + screenOffset.x * clipPos.w;
  clipPos.y = clipPos.y + screenOffset.y * clipPos.w;
  output.position = clipPos;
  
  let viewDir = normalize(uniforms.cameraPos - splat.mean);
  let shColor = evalSH1(viewDir, splat.sh1);
  output.color = splat.colorDC + shColor;
  output.opacity = splat.opacity;
  
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let r = length(input.localUV);
  if (r > 1.0) { discard; }
  let gaussianWeight = exp(-r * r * 4.0);
  let alpha = input.opacity * gaussianWeight;
  return vec4<f32>(input.color * alpha, alpha);
}
`;

// ============================================
// L2 Shader (DC + L1 + L2 SH)
// ============================================
const shaderCodeL2 = /* wgsl */ `
struct Uniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad: f32,
  screenSize: vec2<f32>,
  _pad2: vec2<f32>,
}

struct Splat {
  mean:     vec3<f32>,
  _pad0:    f32,
  scale:    vec3<f32>,
  _pad1:    f32,
  rotation: vec4<f32>,
  colorDC:  vec3<f32>,
  opacity:  f32,
  sh1:      array<f32, 9>,
  sh2:      array<f32, 15>,
  sh3:      array<f32, 21>,
  _pad2:    array<f32, 3>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;

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

// SH 常数
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;

fn evalSH1(dir: vec3<f32>, sh1: array<f32, 9>) -> vec3<f32> {
  let basis = vec3<f32>(dir.y, dir.z, dir.x) * SH_C1;
  let r = sh1[0] * basis.x + sh1[1] * basis.y + sh1[2] * basis.z;
  let g = sh1[3] * basis.x + sh1[4] * basis.y + sh1[5] * basis.z;
  let b = sh1[6] * basis.x + sh1[7] * basis.y + sh1[8] * basis.z;
  return vec3<f32>(r, g, b);
}

fn evalSH2(dir: vec3<f32>, sh2: array<f32, 15>) -> vec3<f32> {
  let x = dir.x; let y = dir.y; let z = dir.z;
  let xx = x * x; let yy = y * y; let zz = z * z;
  let xy = x * y; let yz = y * z; let xz = x * z;
  
  var basis: array<f32, 5>;
  basis[0] = SH_C2_0 * xy;
  basis[1] = SH_C2_1 * yz;
  basis[2] = SH_C2_2 * (2.0 * zz - xx - yy);
  basis[3] = SH_C2_3 * xz;
  basis[4] = SH_C2_4 * (xx - yy);
  
  let r = sh2[0] * basis[0] + sh2[1] * basis[1] + sh2[2] * basis[2] + sh2[3] * basis[3] + sh2[4] * basis[4];
  let g = sh2[5] * basis[0] + sh2[6] * basis[1] + sh2[7] * basis[2] + sh2[8] * basis[3] + sh2[9] * basis[4];
  let b = sh2[10] * basis[0] + sh2[11] * basis[1] + sh2[12] * basis[2] + sh2[13] * basis[3] + sh2[14] * basis[4];
  return vec3<f32>(r, g, b);
}

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

fn computeCov2D(mean: vec3<f32>, scale: vec3<f32>, rotation: vec4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>) -> vec3<f32> {
  let R = quatToMat3(rotation);
  let s2 = scale * scale;
  let M = mat3x3<f32>(R[0] * s2.x, R[1] * s2.y, R[2] * s2.z);
  let Sigma = M * transpose(R);
  let viewPos = (view * vec4<f32>(mean, 1.0)).xyz;
  let viewRot = mat3x3<f32>(view[0].xyz, view[1].xyz, view[2].xyz);
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
  let splatIndex = sortedIndices[instanceIndex];
  let splat = splats[splatIndex];
  let quadPos = QUAD_POSITIONS[vertexIndex];
  output.localUV = quadPos;
  
  let cov2D = computeCov2D(splat.mean, splat.scale, splat.rotation, uniforms.view, uniforms.proj);
  let axes = computeEllipseAxes(cov2D);
  let screenOffset = axes[0] * quadPos.x * ELLIPSE_SCALE + axes[1] * quadPos.y * ELLIPSE_SCALE;
  
  let viewPos = uniforms.view * vec4<f32>(splat.mean, 1.0);
  var clipPos = uniforms.proj * viewPos;
  clipPos.x = clipPos.x + screenOffset.x * clipPos.w;
  clipPos.y = clipPos.y + screenOffset.y * clipPos.w;
  output.position = clipPos;
  
  let viewDir = normalize(uniforms.cameraPos - splat.mean);
  let shColor1 = evalSH1(viewDir, splat.sh1);
  let shColor2 = evalSH2(viewDir, splat.sh2);
  output.color = splat.colorDC + shColor1 + shColor2;
  output.opacity = splat.opacity;
  
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let r = length(input.localUV);
  if (r > 1.0) { discard; }
  let gaussianWeight = exp(-r * r * 4.0);
  let alpha = input.opacity * gaussianWeight;
  return vec4<f32>(input.color * alpha, alpha);
}
`;

// ============================================
// L3 Shader (DC + L1 + L2 + L3 SH - 完整)
// ============================================
const shaderCodeL3 = /* wgsl */ `
struct Uniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad: f32,
  screenSize: vec2<f32>,
  _pad2: vec2<f32>,
}

struct Splat {
  mean:     vec3<f32>,
  _pad0:    f32,
  scale:    vec3<f32>,
  _pad1:    f32,
  rotation: vec4<f32>,
  colorDC:  vec3<f32>,
  opacity:  f32,
  sh1:      array<f32, 9>,
  sh2:      array<f32, 15>,
  sh3:      array<f32, 21>,
  _pad2:    array<f32, 3>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;

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

// SH 常数
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.445305721320277;
const SH_C3_6: f32 = -0.5900435899266435;

fn evalSH1(dir: vec3<f32>, sh1: array<f32, 9>) -> vec3<f32> {
  let basis = vec3<f32>(dir.y, dir.z, dir.x) * SH_C1;
  let r = sh1[0] * basis.x + sh1[1] * basis.y + sh1[2] * basis.z;
  let g = sh1[3] * basis.x + sh1[4] * basis.y + sh1[5] * basis.z;
  let b = sh1[6] * basis.x + sh1[7] * basis.y + sh1[8] * basis.z;
  return vec3<f32>(r, g, b);
}

fn evalSH2(dir: vec3<f32>, sh2: array<f32, 15>) -> vec3<f32> {
  let x = dir.x; let y = dir.y; let z = dir.z;
  let xx = x * x; let yy = y * y; let zz = z * z;
  let xy = x * y; let yz = y * z; let xz = x * z;
  
  var basis: array<f32, 5>;
  basis[0] = SH_C2_0 * xy;
  basis[1] = SH_C2_1 * yz;
  basis[2] = SH_C2_2 * (2.0 * zz - xx - yy);
  basis[3] = SH_C2_3 * xz;
  basis[4] = SH_C2_4 * (xx - yy);
  
  let r = sh2[0] * basis[0] + sh2[1] * basis[1] + sh2[2] * basis[2] + sh2[3] * basis[3] + sh2[4] * basis[4];
  let g = sh2[5] * basis[0] + sh2[6] * basis[1] + sh2[7] * basis[2] + sh2[8] * basis[3] + sh2[9] * basis[4];
  let b = sh2[10] * basis[0] + sh2[11] * basis[1] + sh2[12] * basis[2] + sh2[13] * basis[3] + sh2[14] * basis[4];
  return vec3<f32>(r, g, b);
}

fn evalSH3(dir: vec3<f32>, sh3: array<f32, 21>) -> vec3<f32> {
  let x = dir.x; let y = dir.y; let z = dir.z;
  let xx = x * x; let yy = y * y; let zz = z * z;
  let xy = x * y; let yz = y * z; let xz = x * z;
  
  var basis: array<f32, 7>;
  basis[0] = SH_C3_0 * y * (3.0 * xx - yy);
  basis[1] = SH_C3_1 * xy * z;
  basis[2] = SH_C3_2 * y * (4.0 * zz - xx - yy);
  basis[3] = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
  basis[4] = SH_C3_4 * x * (4.0 * zz - xx - yy);
  basis[5] = SH_C3_5 * z * (xx - yy);
  basis[6] = SH_C3_6 * x * (xx - 3.0 * yy);
  
  let r = sh3[0] * basis[0] + sh3[1] * basis[1] + sh3[2] * basis[2] + sh3[3] * basis[3] + sh3[4] * basis[4] + sh3[5] * basis[5] + sh3[6] * basis[6];
  let g = sh3[7] * basis[0] + sh3[8] * basis[1] + sh3[9] * basis[2] + sh3[10] * basis[3] + sh3[11] * basis[4] + sh3[12] * basis[5] + sh3[13] * basis[6];
  let b = sh3[14] * basis[0] + sh3[15] * basis[1] + sh3[16] * basis[2] + sh3[17] * basis[3] + sh3[18] * basis[4] + sh3[19] * basis[5] + sh3[20] * basis[6];
  return vec3<f32>(r, g, b);
}

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

fn computeCov2D(mean: vec3<f32>, scale: vec3<f32>, rotation: vec4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>) -> vec3<f32> {
  let R = quatToMat3(rotation);
  let s2 = scale * scale;
  let M = mat3x3<f32>(R[0] * s2.x, R[1] * s2.y, R[2] * s2.z);
  let Sigma = M * transpose(R);
  let viewPos = (view * vec4<f32>(mean, 1.0)).xyz;
  let viewRot = mat3x3<f32>(view[0].xyz, view[1].xyz, view[2].xyz);
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
  let splatIndex = sortedIndices[instanceIndex];
  let splat = splats[splatIndex];
  let quadPos = QUAD_POSITIONS[vertexIndex];
  output.localUV = quadPos;
  
  let cov2D = computeCov2D(splat.mean, splat.scale, splat.rotation, uniforms.view, uniforms.proj);
  let axes = computeEllipseAxes(cov2D);
  let screenOffset = axes[0] * quadPos.x * ELLIPSE_SCALE + axes[1] * quadPos.y * ELLIPSE_SCALE;
  
  let viewPos = uniforms.view * vec4<f32>(splat.mean, 1.0);
  var clipPos = uniforms.proj * viewPos;
  clipPos.x = clipPos.x + screenOffset.x * clipPos.w;
  clipPos.y = clipPos.y + screenOffset.y * clipPos.w;
  output.position = clipPos;
  
  let viewDir = normalize(uniforms.cameraPos - splat.mean);
  let shColor1 = evalSH1(viewDir, splat.sh1);
  let shColor2 = evalSH2(viewDir, splat.sh2);
  let shColor3 = evalSH3(viewDir, splat.sh3);
  output.color = splat.colorDC + shColor1 + shColor2 + shColor3;
  output.opacity = splat.opacity;
  
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let r = length(input.localUV);
  if (r > 1.0) { discard; }
  let gaussianWeight = exp(-r * r * 4.0);
  let alpha = input.opacity * gaussianWeight;
  return vec4<f32>(input.color * alpha, alpha);
}
`;

/**
 * SH 模式枚举
 */
export enum SHMode {
  L0 = 0, // 仅 DC 颜色 (高性能)
  L1 = 1, // DC + L1 SH
  L2 = 2, // DC + L1 + L2 SH
  L3 = 3, // DC + L1 + L2 + L3 SH (完整)
}

/**
 * Bounding Box 结构
 */
export interface BoundingBox {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  radius: number; // bounding sphere 半径
}

/**
 * GPU Splat 结构的字节大小
 * 256 bytes per splat (64 floats), 对齐到 WGSL struct 布局
 * 包含: mean(3) + pad(1) + scale(3) + pad(1) + rotation(4) + colorDC(3) + opacity(1)
 *      + sh1(9) + sh2(15) + sh3(21) + pad(3) = 64 floats
 */
const SPLAT_BYTE_SIZE = 256;
const SPLAT_FLOAT_COUNT = 64;

/**
 * 紧凑 Splat 结构（移动端优化）
 * 64 bytes per splat (16 floats), 仅包含基本渲染数据
 * 包含: mean(3) + pad(1) + scale(3) + pad(1) + rotation(4) + colorDC(3) + opacity(1) = 16 floats
 */
const SPLAT_COMPACT_BYTE_SIZE = 64;
const SPLAT_COMPACT_FLOAT_COUNT = 16;

/**
 * 不同性能等级的默认配置
 *
 * 重要：排序对渲染效果至关重要！
 * 没有正确排序会导致 alpha blending 错误，看起来像"椭球叠加"
 *
 * 注意：maxVisibleSplats 会影响显示完整度
 * - 如果场景有 300 万 splat，设置 30 万只会显示 10%
 * - 移动端可以通过 setOptimizationConfig 调整
 */
const PERFORMANCE_CONFIGS: Record<PerformanceTier, MobileOptimizationConfig> = {
  [PerformanceTier.HIGH]: {
    maxVisibleSplats: Infinity,
    enableSorting: true,
    sortEveryNFrames: 1,
    useCompactFormat: false,
    pixelCullThreshold: 1.0,
    defaultSHMode: SHMode.L1,
  },
  [PerformanceTier.MEDIUM]: {
    maxVisibleSplats: Infinity,
    enableSorting: true,
    sortEveryNFrames: 1,
    useCompactFormat: false,
    pixelCullThreshold: 1.0,
    defaultSHMode: SHMode.L1,
  },
  [PerformanceTier.LOW]: {
    maxVisibleSplats: Infinity, // 15 万（移动端安全值，避免 GPU 内存溢出）
    enableSorting: true,
    sortEveryNFrames: 1,
    useCompactFormat: false,
    pixelCullThreshold: 1.0,
    defaultSHMode: SHMode.L0,
  },
};

/**
 * GSSplatRenderer - 3D Gaussian Splatting 渲染器
 * 使用 instanced quad 方式渲染 splats
 *
 * 优化功能：
 * - GPU 可见性剔除 (视锥/近平面/屏幕尺寸)
 * - DrawIndirect 避免 CPU submit 开销
 * - 仅对可见 splat 排序
 * - 移动端自动优化（降低排序频率、使用紧凑格式）
 */
export class GSSplatRenderer {
  private renderer: Renderer;
  private camera: Camera;

  private pipelineL0!: GPURenderPipeline;
  private pipelineL1!: GPURenderPipeline;
  private pipelineL2!: GPURenderPipeline;
  private pipelineL3!: GPURenderPipeline;
  private pipelineL0Compact!: GPURenderPipeline; // 移动端紧凑格式管线
  private bindGroupLayout!: GPUBindGroupLayout;
  private bindGroupLayoutCompact!: GPUBindGroupLayout; // 紧凑格式的 layout
  private uniformBuffer!: GPUBuffer;

  private splatBuffer: GPUBuffer | null = null;
  private splatCount: number = 0;
  private bindGroup: GPUBindGroup | null = null;

  // 深度排序器（含剔除功能）- 使用 V2 分桶稳定排序
  private sorter: GSSplatSorter | null = null;

  // 是否启用 DrawIndirect (剔除优化)
  // 注意：在某些移动设备上可能有问题，可以禁用作为备用
  private useDrawIndirect: boolean = true;

  // 是否为移动设备（用于调试）
  private isMobile: boolean = false;

  // 像素剔除阈值 (小于此像素的 splat 会被剔除)
  private pixelCullThreshold: number = 1.0;

  // SH 模式：L0/L1/L2/L3
  private shMode: SHMode = SHMode.L1;

  // 点云的 bounding box（在 setData 时计算）
  private boundingBox: BoundingBox | null = null;

  // ============================================
  // 移动端优化相关
  // ============================================
  private performanceTier: PerformanceTier;
  private optimizationConfig: MobileOptimizationConfig;
  private frameCount: number = 0;
  private useCompactFormat: boolean = false;

  constructor(renderer: Renderer, camera: Camera) {
    this.renderer = renderer;
    this.camera = camera;

    // 检测设备类型
    this.isMobile = isMobileDevice();

    // 检测性能等级
    this.performanceTier = detectPerformanceTier(renderer.device);
    this.optimizationConfig = { ...PERFORMANCE_CONFIGS[this.performanceTier] };

    // 应用默认配置
    this.pixelCullThreshold = this.optimizationConfig.pixelCullThreshold;
    this.shMode = this.optimizationConfig.defaultSHMode;
    this.useCompactFormat = this.optimizationConfig.useCompactFormat;

    console.log(
      `GSSplatRenderer: isMobile=${this.isMobile}, 性能等级=${this.performanceTier}`,
    );
    console.log(`GSSplatRenderer: 优化配置 =`, this.optimizationConfig);
    console.log(
      `GSSplatRenderer: GPU limits - maxBufferSize=${renderer.device.limits.maxBufferSize}, maxStorageBufferBindingSize=${renderer.device.limits.maxStorageBufferBindingSize}`,
    );

    this.createPipelines();
    this.createUniformBuffer();
  }

  /**
   * 获取当前性能等级
   */
  getPerformanceTier(): PerformanceTier {
    return this.performanceTier;
  }

  /**
   * 手动设置优化配置
   */
  setOptimizationConfig(config: Partial<MobileOptimizationConfig>): void {
    this.optimizationConfig = { ...this.optimizationConfig, ...config };
    this.pixelCullThreshold = this.optimizationConfig.pixelCullThreshold;
    if (config.defaultSHMode !== undefined) {
      this.shMode = config.defaultSHMode;
    }
    console.log(`GSSplatRenderer: 更新优化配置 =`, this.optimizationConfig);
  }

  /**
   * 获取当前优化配置
   */
  getOptimizationConfig(): MobileOptimizationConfig {
    return { ...this.optimizationConfig };
  }

  /**
   * 设置 SH 模式
   * @param mode L0/L1/L2/L3
   */
  setSHMode(mode: SHMode): void {
    this.shMode = mode;
    const modeNames = [
      "L0 (仅DC)",
      "L1 (DC+SH1)",
      "L2 (DC+SH1+SH2)",
      "L3 (完整SH)",
    ];
    console.log(`GSSplatRenderer: SH 模式切换为 ${modeNames[mode]}`);
  }

  /**
   * 获取当前 SH 模式
   */
  getSHMode(): SHMode {
    return this.shMode;
  }

  /**
   * 设置是否启用 DrawIndirect (剔除优化)
   * 启用后会在 GPU 上进行可见性剔除，仅绘制可见 splat
   */
  setUseDrawIndirect(enabled: boolean): void {
    this.useDrawIndirect = enabled;
  }

  /**
   * 设置像素剔除阈值
   * 屏幕上小于此像素数的 splat 会被剔除
   * @param threshold 像素阈值，默认 1.0
   */
  setPixelCullThreshold(threshold: number): void {
    this.pixelCullThreshold = threshold;
  }

  /**
   * 创建渲染管线 (L0/L1/L2/L3 四个版本)
   */
  private createPipelines(): void {
    const device = this.renderer.device;

    // 创建 shader 模块
    const shaderModuleL0 = device.createShaderModule({ code: shaderCodeL0 });
    const shaderModuleL1 = device.createShaderModule({ code: shaderCodeL1 });
    const shaderModuleL2 = device.createShaderModule({ code: shaderCodeL2 });
    const shaderModuleL3 = device.createShaderModule({ code: shaderCodeL3 });

    // 创建 bind group layout
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          // uniform buffer (view + proj matrices)
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
        {
          // storage buffer (splats array)
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
        {
          // storage buffer (sorted indices)
          binding: 2,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    // 创建 pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    // 共享的管线描述符基础配置
    const basePipelineDesc = {
      layout: pipelineLayout,
      primitive: {
        topology: "triangle-strip" as GPUPrimitiveTopology,
      },
      depthStencil: {
        format: this.renderer.depthFormat,
        depthWriteEnabled: false,
        depthCompare: "always" as GPUCompareFunction,
      },
    };

    const blendState = {
      color: {
        srcFactor: "one" as GPUBlendFactor,
        dstFactor: "one-minus-src-alpha" as GPUBlendFactor,
        operation: "add" as GPUBlendOperation,
      },
      alpha: {
        srcFactor: "one" as GPUBlendFactor,
        dstFactor: "one-minus-src-alpha" as GPUBlendFactor,
        operation: "add" as GPUBlendOperation,
      },
    };

    // 创建各级别渲染管线
    const createPipeline = (module: GPUShaderModule) =>
      device.createRenderPipeline({
        ...basePipelineDesc,
        vertex: { module, entryPoint: "vs_main", buffers: [] },
        fragment: {
          module,
          entryPoint: "fs_main",
          targets: [{ format: this.renderer.format, blend: blendState }],
        },
      });

    this.pipelineL0 = createPipeline(shaderModuleL0);
    this.pipelineL1 = createPipeline(shaderModuleL1);
    this.pipelineL2 = createPipeline(shaderModuleL2);
    this.pipelineL3 = createPipeline(shaderModuleL3);

    console.log("GSSplatRenderer: 已创建 L0/L1/L2/L3 渲染管线");
  }

  /**
   * 创建 uniform buffer
   * 布局: view (64 bytes) + proj (64 bytes) + cameraPos (12 bytes) + padding (4 bytes) + screenSize (8 bytes) + padding (8 bytes) = 160 bytes
   */
  private createUniformBuffer(): void {
    this.uniformBuffer = this.renderer.device.createBuffer({
      size: 160, // view (64) + proj (64) + cameraPos (12) + padding (4) + screenSize (8) + padding (8)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * 设置 splat 数据
   * @param splats CPU 端的 splat 数组
   */
  setData(splats: SplatCPU[]): void {
    const device = this.renderer.device;

    // 销毁旧的资源
    if (this.splatBuffer) {
      this.splatBuffer.destroy();
    }
    if (this.sorter) {
      this.sorter.destroy();
      this.sorter = null;
    }

    // ============================================
    // 移动端优化：限制最大 splat 数量
    // ============================================
    const originalCount = splats.length;
    const maxSplats = this.optimizationConfig.maxVisibleSplats;

    if (splats.length > maxSplats && maxSplats !== Infinity) {
      console.warn(
        `GSSplatRenderer: Splat 数量 (${splats.length}) 超过移动端限制 (${maxSplats})，将进行降采样`,
      );
      // 均匀降采样
      const step = splats.length / maxSplats;
      const sampledSplats: SplatCPU[] = [];
      for (let i = 0; i < maxSplats; i++) {
        const idx = Math.floor(i * step);
        sampledSplats.push(splats[idx]);
      }
      splats = sampledSplats;
      console.log(`GSSplatRenderer: 降采样后 splat 数量 = ${splats.length}`);
    }

    this.splatCount = splats.length;
    this.frameCount = 0; // 重置帧计数，确保第一帧会排序

    if (this.splatCount === 0) {
      this.splatBuffer = null;
      this.bindGroup = null;
      this.boundingBox = null;
      return;
    }

    // 计算 bounding box
    this.boundingBox = this.computeBoundingBox(splats);

    // ============================================
    // 根据配置选择数据格式
    // ============================================
    const useCompact = this.useCompactFormat;
    const floatCount = useCompact
      ? SPLAT_COMPACT_FLOAT_COUNT
      : SPLAT_FLOAT_COUNT;
    const byteSize = useCompact ? SPLAT_COMPACT_BYTE_SIZE : SPLAT_BYTE_SIZE;

    // 创建 CPU 端数据数组
    const data = new Float32Array(this.splatCount * floatCount);

    for (let i = 0; i < this.splatCount; i++) {
      const splat = splats[i];
      const offset = i * floatCount;

      // mean (vec3) + padding
      data[offset + 0] = splat.mean[0];
      data[offset + 1] = splat.mean[1];
      data[offset + 2] = splat.mean[2];
      data[offset + 3] = 0; // padding

      // scale (vec3) + padding
      data[offset + 4] = splat.scale[0];
      data[offset + 5] = splat.scale[1];
      data[offset + 6] = splat.scale[2];
      data[offset + 7] = 0; // padding

      // rotation (vec4)
      data[offset + 8] = splat.rotation[0];
      data[offset + 9] = splat.rotation[1];
      data[offset + 10] = splat.rotation[2];
      data[offset + 11] = splat.rotation[3];

      // colorDC (vec3) + opacity
      data[offset + 12] = splat.colorDC[0];
      data[offset + 13] = splat.colorDC[1];
      data[offset + 14] = splat.colorDC[2];
      data[offset + 15] = splat.opacity;

      // 紧凑格式不包含 SH 系数
      if (!useCompact) {
        const shRest = splat.shRest;

        // sh1 (array<f32, 9>) - L1 SH 系数
        for (let j = 0; j < 9; j++) {
          data[offset + 16 + j] = shRest ? shRest[j] : 0;
        }

        // sh2 (array<f32, 15>) - L2 SH 系数
        for (let j = 0; j < 15; j++) {
          data[offset + 25 + j] = shRest ? shRest[9 + j] : 0;
        }

        // sh3 (array<f32, 21>) - L3 SH 系数
        for (let j = 0; j < 21; j++) {
          data[offset + 40 + j] = shRest ? shRest[24 + j] : 0;
        }

        // padding (array<f32, 3>)
        data[offset + 61] = 0;
        data[offset + 62] = 0;
        data[offset + 63] = 0;
      }
    }

    // 创建 GPU buffer
    this.splatBuffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // 上传数据
    device.queue.writeBuffer(this.splatBuffer, 0, data);

    // 创建深度排序器（含剔除功能 + 量化深度稳定排序）
    this.sorter = new GSSplatSorter(
      device,
      this.splatCount,
      this.splatBuffer,
      this.uniformBuffer,
    );

    // 初始化排序器参数
    this.sorter.setScreenSize(this.renderer.width, this.renderer.height);
    this.sorter.setCullingOptions({
      nearPlane: this.camera.near,
      farPlane: this.camera.far,
      pixelThreshold: this.pixelCullThreshold,
    });

    // 创建 bind group (包含排序后的索引 buffer)
    this.bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: this.uniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: this.splatBuffer },
        },
        {
          binding: 2,
          resource: { buffer: this.sorter.getIndicesBuffer() },
        },
      ],
    });

    const memoryMB = (data.byteLength / (1024 * 1024)).toFixed(2);
    console.log(
      `GSSplatRenderer: 已上传 ${this.splatCount} 个 splats (原始 ${originalCount} 个)`,
    );
    console.log(
      `GSSplatRenderer: 数据格式 = ${useCompact ? "紧凑 (64B/splat)" : "完整 (256B/splat)"}, 内存占用 = ${memoryMB} MB`,
    );
  }

  /**
   * 设置紧凑格式的 splat 数据（移动端优化）
   * 直接接受 CompactSplatData，避免创建中间对象
   * @param compactData 紧凑格式的 splat 数据
   */
  setCompactData(compactData: CompactSplatData): void {
    console.log(`setCompactData: 开始处理 ${compactData.count} 个 splats`);

    try {
      const device = this.renderer.device;

      // 销毁旧的资源
      if (this.splatBuffer) {
        this.splatBuffer.destroy();
      }
      if (this.sorter) {
        this.sorter.destroy();
        this.sorter = null;
      }

      this.splatCount = compactData.count;
      this.frameCount = 0; // 重置帧计数，确保第一帧会排序

      if (this.splatCount === 0) {
        console.warn("setCompactData: splat 数量为 0，跳过初始化");
        this.splatBuffer = null;
        this.bindGroup = null;
        this.boundingBox = null;
        return;
      }

      // 计算 bounding box
      this.boundingBox = this.computeBoundingBoxFromCompact(compactData);
      console.log(`setCompactData: boundingBox =`, this.boundingBox);

      // 转换为 GPU buffer 格式
      const includeSH =
        compactData.shCoeffs !== undefined && this.shMode !== SHMode.L0;
      console.log(`setCompactData: 转换为 GPU 格式, includeSH=${includeSH}`);
      const gpuData = compactDataToGPUBuffer(compactData, includeSH);
      console.log(
        `setCompactData: GPU 数据大小 = ${(gpuData.byteLength / 1024 / 1024).toFixed(2)} MB`,
      );

      // 创建 GPU buffer
      this.splatBuffer = device.createBuffer({
        size: gpuData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      // 上传数据
      device.queue.writeBuffer(this.splatBuffer, 0, gpuData.buffer);

      // 创建深度排序器
      this.sorter = new GSSplatSorter(
        device,
        this.splatCount,
        this.splatBuffer,
        this.uniformBuffer,
      );

      // 初始化排序器参数
      this.sorter.setScreenSize(this.renderer.width, this.renderer.height);
      this.sorter.setCullingOptions({
        nearPlane: this.camera.near,
        farPlane: this.camera.far,
        pixelThreshold: this.pixelCullThreshold,
      });

      // 创建 bind group
      this.bindGroup = device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.splatBuffer } },
          { binding: 2, resource: { buffer: this.sorter.getIndicesBuffer() } },
        ],
      });

      const memoryMB = (gpuData.byteLength / (1024 * 1024)).toFixed(2);
      console.log(
        `GSSplatRenderer: 已上传 ${this.splatCount} 个 splats (紧凑格式)`,
      );
      console.log(`GSSplatRenderer: GPU 内存占用 = ${memoryMB} MB`);
    } catch (error) {
      console.error("setCompactData 错误:", error);
      // 重置状态避免后续渲染崩溃
      this.splatCount = 0;
      this.splatBuffer = null;
      this.bindGroup = null;
      this.sorter = null;
    }
  }

  /**
   * 从紧凑数据计算 bounding box
   */
  private computeBoundingBoxFromCompact(data: CompactSplatData): BoundingBox {
    if (data.count === 0) {
      return { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0], radius: 0 };
    }

    const positions = data.positions;
    const min: [number, number, number] = [
      positions[0],
      positions[1],
      positions[2],
    ];
    const max: [number, number, number] = [
      positions[0],
      positions[1],
      positions[2],
    ];

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
   * 渲染 splats
   * @param pass 渲染通道编码器
   */
  render(pass: GPURenderPassEncoder): void {
    if (this.splatCount === 0 || !this.bindGroup || !this.sorter) {
      return;
    }

    this.frameCount++;

    // 更新 view 和 proj uniform
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Float32Array(this.camera.viewMatrix),
    );
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer,
      64,
      new Float32Array(this.camera.projectionMatrix),
    );
    // 更新 cameraPos uniform
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer,
      128,
      new Float32Array(this.camera.position),
    );
    // 更新 screenSize uniform (用于抗锯齿)
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer,
      144,
      new Float32Array([this.renderer.width, this.renderer.height, 0, 0]),
    );

    // 更新排序器参数（屏幕尺寸和剔除选项）
    this.sorter.setScreenSize(this.renderer.width, this.renderer.height);
    this.sorter.setCullingOptions({
      nearPlane: this.camera.near,
      farPlane: this.camera.far,
      pixelThreshold: this.pixelCullThreshold,
    });

    // ============================================
    // 深度排序（对 3DGS 效果至关重要）
    // 注意：第一帧必须排序，否则渲染顺序错误
    // ============================================
    const isFirstFrame = this.frameCount === 1;
    const shouldSort =
      this.optimizationConfig.enableSorting &&
      (isFirstFrame ||
        this.frameCount % this.optimizationConfig.sortEveryNFrames === 0);

    // 调试：前几帧输出排序状态
    if (this.frameCount <= 3) {
      console.log(
        `GSSplatRenderer: frame=${this.frameCount}, isFirstFrame=${isFirstFrame}, shouldSort=${shouldSort}, enableSorting=${this.optimizationConfig.enableSorting}, sortEveryNFrames=${this.optimizationConfig.sortEveryNFrames}`,
      );
    }

    if (shouldSort) {
      // 执行 GPU 剔除和深度排序 (在 render pass 开始前)
      // 注意: sort() 会在内部创建并提交 compute command buffer
      this.sorter.sort();
    }

    // 根据 SH 模式选择管线
    const pipelines = [
      this.pipelineL0,
      this.pipelineL1,
      this.pipelineL2,
      this.pipelineL3,
    ];
    const pipeline = pipelines[this.shMode];

    // 设置管线和绑定组
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup);

    // 绘制
    if (this.useDrawIndirect) {
      // 使用 DrawIndirect：仅绘制可见 splat
      // DrawIndirect buffer 由 GPU 剔除 pass 更新
      pass.drawIndirect(this.sorter.getDrawIndirectBuffer(), 0);
    } else {
      // 传统方式：绘制所有 splat
      pass.draw(4, this.splatCount);
    }
  }

  /**
   * 获取 splat 数量
   */
  getSplatCount(): number {
    return this.splatCount;
  }

  /**
   * 获取点云的 bounding box
   * @returns BoundingBox 或 null（如果没有点云数据）
   */
  getBoundingBox(): BoundingBox | null {
    return this.boundingBox;
  }

  /**
   * 计算点云的 bounding box
   * @param splats splat 数组
   * @returns BoundingBox
   */
  private computeBoundingBox(splats: SplatCPU[]): BoundingBox {
    if (splats.length === 0) {
      return {
        min: [0, 0, 0],
        max: [0, 0, 0],
        center: [0, 0, 0],
        radius: 0,
      };
    }

    // 初始化为第一个点
    const min: [number, number, number] = [
      splats[0].mean[0],
      splats[0].mean[1],
      splats[0].mean[2],
    ];
    const max: [number, number, number] = [
      splats[0].mean[0],
      splats[0].mean[1],
      splats[0].mean[2],
    ];

    // 遍历所有点，计算 min/max
    for (let i = 1; i < splats.length; i++) {
      const [x, y, z] = splats[i].mean;
      min[0] = Math.min(min[0], x);
      min[1] = Math.min(min[1], y);
      min[2] = Math.min(min[2], z);
      max[0] = Math.max(max[0], x);
      max[1] = Math.max(max[1], y);
      max[2] = Math.max(max[2], z);
    }

    // 计算中心点
    const center: [number, number, number] = [
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    ];

    // 计算 bounding sphere 半径（从中心到最远角）
    const dx = max[0] - min[0];
    const dy = max[1] - min[1];
    const dz = max[2] - min[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    console.log(
      `GSSplatRenderer: BoundingBox computed - center: [${center[0].toFixed(2)}, ${center[1].toFixed(2)}, ${center[2].toFixed(2)}], radius: ${radius.toFixed(2)}`,
    );

    return { min, max, center, radius };
  }

  /**
   * 销毁资源
   */
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
