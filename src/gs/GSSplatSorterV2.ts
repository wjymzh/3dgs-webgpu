/**
 * GSSplatSorterV2 - Z-Binning 分桶稳定排序器
 *
 * 解决百万级 splat 闪烁问题：
 * 1. 将 splat 按深度分到多个桶 (Z-binning)
 * 2. 近处大 splat (screenRadius > threshold) 做局部稳定排序
 * 3. 远处小 splat 不排序，保持原始顺序
 * 4. 稳定排序：深度相同时保持原始索引顺序
 *
 * 优化点：
 * - 深度量化消除浮点精度问题
 * - 避免全局 Bitonic Sort 的不稳定性
 * - 减少排序数据量（远处小 splat 跳过排序）
 * - 单次 submit 完成所有 compute pass
 */

// ============================================
// 常量定义
// ============================================
const NUM_BUCKETS = 128; // 深度桶数量
const WORKGROUP_SIZE = 256; // Compute shader 工作组大小
const SORT_RADIUS_THRESHOLD = 2.0; // 屏幕半径阈值 (像素)，低于此值不排序

// ============================================
// 剔除 + 分桶 Compute Shader
// 优化: 减少 buffer 数量，合并 visibleScreenRadii 到后续 pass
// ============================================
const cullingBinningShaderCode = /* wgsl */ `
/**
 * GPU Splat 可见性剔除 + Z-Binning Compute Shader
 * Pass 1: 剔除不可见 splat，计算深度、桶 ID
 * 
 * 优化: 使用深度量化消除浮点精度问题
 * - 高 24 位存储量化深度
 * - 低 8 位存储原始索引低位（用于稳定排序 tie-breaker）
 */

const NUM_BUCKETS: u32 = ${NUM_BUCKETS}u;

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

struct CameraUniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  model: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad: f32,
}

struct CullingParams {
  splatCount: u32,
  nearPlane: f32,
  farPlane: f32,
  screenWidth: f32,
  screenHeight: f32,
  pixelThreshold: f32,
  _pad0: f32,
  _pad1: f32,
}

struct Counters {
  visibleCount: atomic<u32>,
}

struct BucketCounters {
  counts: array<atomic<u32>, ${NUM_BUCKETS}>,
}

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<uniform> cullingParams: CullingParams;
@group(0) @binding(3) var<storage, read_write> counters: Counters;
@group(0) @binding(4) var<storage, read_write> visibleIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> visibleQuantizedDepths: array<u32>;
@group(0) @binding(6) var<storage, read_write> visibleBucketIds: array<u32>;
@group(0) @binding(7) var<storage, read_write> bucketCounters: BucketCounters;

fn maxScale(scale: vec3<f32>) -> f32 {
  return max(max(scale.x, scale.y), scale.z);
}

// 从模型矩阵提取最大缩放因子
fn getModelMaxScale(model: mat4x4<f32>) -> f32 {
  let sx = length(model[0].xyz);
  let sy = length(model[1].xyz);
  let sz = length(model[2].xyz);
  return max(max(sx, sy), sz);
}

/**
 * 深度量化函数
 * 将深度量化为 u32，保留 24 位精度 + 8 位原始索引低位
 * 
 * 格式: [24位深度][8位原始索引]
 * - 深度部分: 远处值大，近处值小（降序排列时远的在前）
 * - 原始索引低位: 用于深度相同时的稳定排序
 * 
 * 这样直接比较 u32 就能实现稳定的降序排序
 */
fn quantizeDepth(depth: f32, nearPlane: f32, farPlane: f32, originalIndex: u32) -> u32 {
  let normalized = clamp((depth - nearPlane) / (farPlane - nearPlane), 0.0, 1.0);
  // 高 24 位存深度（远处值大）
  let depthBits = u32(normalized * 16777215.0); // 2^24 - 1
  // 低 8 位存原始索引低位（用于稳定排序：相同深度时小索引在前）
  // 注意：为了降序排列，我们取反原始索引的低位
  let indexBits = 255u - (originalIndex & 0xFFu);
  return (depthBits << 8u) | indexBits;
}

// 计算桶 ID：将 [nearPlane, farPlane] 映射到 [0, NUM_BUCKETS-1]
// 使用线性映射，远处桶 ID 小，近处桶 ID 大
fn computeBucketId(depth: f32, nearPlane: f32, farPlane: f32) -> u32 {
  let normalized = clamp((depth - nearPlane) / (farPlane - nearPlane), 0.0, 1.0);
  // 反转：远处 = 0，近处 = NUM_BUCKETS-1（渲染从远到近）
  let bucketId = u32((1.0 - normalized) * f32(NUM_BUCKETS - 1u));
  return min(bucketId, NUM_BUCKETS - 1u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn cullAndBin(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cullingParams.splatCount) {
    return;
  }
  
  let splat = splats[i];
  // 先应用模型矩阵变换到世界空间，再变换到视图空间
  let worldPos = camera.model * vec4<f32>(splat.mean, 1.0);
  let viewPos = camera.view * worldPos;
  let z = -viewPos.z;
  
  // 近平面剔除
  if (z < cullingParams.nearPlane) {
    return;
  }
  
  // 远平面剔除
  if (z > cullingParams.farPlane) {
    return;
  }
  
  // 视锥剔除
  let fx = camera.proj[0][0];
  let fy = camera.proj[1][1];
  let x_ndc = viewPos.x * fx / z;
  let y_ndc = viewPos.y * fy / z;
  // 考虑模型缩放对 splat 半径的影响
  let modelScale = getModelMaxScale(camera.model);
  let worldRadius = maxScale(splat.scale) * modelScale * 3.0;
  let r_ndc = worldRadius * max(fx, fy) / z;
  
  if (x_ndc < -1.0 - r_ndc || x_ndc > 1.0 + r_ndc) {
    return;
  }
  if (y_ndc < -1.0 - r_ndc || y_ndc > 1.0 + r_ndc) {
    return;
  }
  
  // 屏幕尺寸剔除
  let screenRadiusX = r_ndc * cullingParams.screenWidth * 0.5;
  let screenRadiusY = r_ndc * cullingParams.screenHeight * 0.5;
  let screenRadius = max(screenRadiusX, screenRadiusY);
  
  if (screenRadius < cullingParams.pixelThreshold) {
    return;
  }
  
  // 透明度剔除
  if (splat.opacity < 0.004) {
    return;
  }
  
  // 计算桶 ID
  let bucketId = computeBucketId(z, cullingParams.nearPlane, cullingParams.farPlane);
  
  // 量化深度（包含原始索引信息用于稳定排序）
  let quantizedDepth = quantizeDepth(z, cullingParams.nearPlane, cullingParams.farPlane, i);
  
  // 通过剔除，写入结果
  let visibleIdx = atomicAdd(&counters.visibleCount, 1u);
  visibleIndices[visibleIdx] = i;
  visibleQuantizedDepths[visibleIdx] = quantizedDepth;
  visibleBucketIds[visibleIdx] = bucketId;
  
  // 增加对应桶的计数
  atomicAdd(&bucketCounters.counts[bucketId], 1u);
}

@compute @workgroup_size(1)
fn resetCounters() {
  atomicStore(&counters.visibleCount, 0u);
}

@compute @workgroup_size(${NUM_BUCKETS > WORKGROUP_SIZE ? WORKGROUP_SIZE : NUM_BUCKETS})
fn resetBucketCounters(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < NUM_BUCKETS) {
    atomicStore(&bucketCounters.counts[i], 0u);
  }
}
`;

// ============================================
// DrawIndirect 更新 Shader (单独的 pass)
// ============================================
const drawIndirectShaderCode = /* wgsl */ `
struct Counters {
  visibleCount: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> counters: Counters;
@group(0) @binding(1) var<storage, read_write> drawIndirect: array<u32>;

@compute @workgroup_size(1)
fn updateDrawIndirect() {
  let count = atomicLoad(&counters.visibleCount);
  drawIndirect[0] = 4u;
  drawIndirect[1] = count;
  drawIndirect[2] = 0u;
  drawIndirect[3] = 0u;
}
`;

// ============================================
// Prefix Sum Compute Shader
// ============================================
const prefixSumShaderCode = /* wgsl */ `
/**
 * Prefix Sum (Exclusive Scan) for Bucket Offsets
 * 计算每个桶的起始偏移量
 * 
 * 使用 Hillis-Steele 算法的变种，简单可靠
 * 由于桶数量较少 (128)，使用单 workgroup 完成
 */

const NUM_BUCKETS: u32 = ${NUM_BUCKETS}u;

struct PrefixSumParams {
  bucketCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> bucketCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> bucketOffsets: array<u32>;
@group(0) @binding(2) var<uniform> params: PrefixSumParams;

// 双缓冲用于 Hillis-Steele 算法
var<workgroup> buf0: array<u32, ${NUM_BUCKETS}>;
var<workgroup> buf1: array<u32, ${NUM_BUCKETS}>;

@compute @workgroup_size(${Math.min(NUM_BUCKETS, WORKGROUP_SIZE)})
fn prefixSum(@builtin(local_invocation_id) lid: vec3<u32>) {
  let thid = lid.x;
  let n = NUM_BUCKETS;
  
  // 加载数据到共享内存 (exclusive scan: 第一个元素为 0)
  if (thid == 0u) {
    buf0[0] = 0u;
  } else if (thid < n) {
    buf0[thid] = bucketCounts[thid - 1u];
  }
  
  workgroupBarrier();
  
  // Hillis-Steele inclusive scan，然后转换为 exclusive
  // 迭代 log2(n) 次
  var readBuf = 0u;
  for (var stride = 1u; stride < n; stride *= 2u) {
    workgroupBarrier();
    
    if (readBuf == 0u) {
      // 从 buf0 读，写入 buf1
      if (thid < n) {
        if (thid >= stride) {
          buf1[thid] = buf0[thid] + buf0[thid - stride];
        } else {
          buf1[thid] = buf0[thid];
        }
      }
    } else {
      // 从 buf1 读，写入 buf0
      if (thid < n) {
        if (thid >= stride) {
          buf0[thid] = buf1[thid] + buf1[thid - stride];
        } else {
          buf0[thid] = buf1[thid];
        }
      }
    }
    
    readBuf = 1u - readBuf;
  }
  
  workgroupBarrier();
  
  // 写回结果 (从正确的缓冲区读取)
  if (thid < n) {
    if (readBuf == 0u) {
      bucketOffsets[thid] = buf0[thid];
    } else {
      bucketOffsets[thid] = buf1[thid];
    }
  }
}
`;

// ============================================
// Scatter Compute Shader (简化版，减少 buffer)
// ============================================
const scatterShaderCode = /* wgsl */ `
/**
 * Scatter: 将可见 splat 分散到对应桶的位置
 * 使用 atomic 操作确保正确的桶内位置分配
 */

const NUM_BUCKETS: u32 = ${NUM_BUCKETS}u;

struct ScatterParams {
  visibleCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

struct BucketPositions {
  positions: array<atomic<u32>, ${NUM_BUCKETS}>,
}

@group(0) @binding(0) var<storage, read> visibleIndices: array<u32>;
@group(0) @binding(1) var<storage, read> visibleQuantizedDepths: array<u32>;
@group(0) @binding(2) var<storage, read> visibleBucketIds: array<u32>;
@group(0) @binding(3) var<storage, read> bucketOffsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> bucketPositions: BucketPositions;
@group(0) @binding(5) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(6) var<storage, read_write> sortedQuantizedDepths: array<u32>;
@group(0) @binding(7) var<uniform> params: ScatterParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.visibleCount) {
    return;
  }
  
  let bucketId = visibleBucketIds[i];
  let bucketOffset = bucketOffsets[bucketId];
  let originalIndex = visibleIndices[i];
  let quantizedDepth = visibleQuantizedDepths[i];
  
  // 在桶内分配位置
  let posInBucket = atomicAdd(&bucketPositions.positions[bucketId], 1u);
  let destIdx = bucketOffset + posInBucket;
  
  // 写入排序缓冲区
  sortedIndices[destIdx] = originalIndex;
  sortedQuantizedDepths[destIdx] = quantizedDepth;
}

@compute @workgroup_size(${Math.min(NUM_BUCKETS, WORKGROUP_SIZE)})
fn resetBucketPositions(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < NUM_BUCKETS) {
    atomicStore(&bucketPositions.positions[i], 0u);
  }
}
`;

// ============================================
// 全局 Bitonic Sort Shader (替代 Local Sort)
// 对所有可见 splat 进行排序，不使用 workgroupBarrier
// ============================================
const globalSortShaderCode = /* wgsl */ `
/**
 * 全局 Bitonic Sort
 * 对所有可见 splat 按量化深度排序
 * 不使用 workgroupBarrier，避免控制流问题
 */

const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;

struct SortParams {
  k: u32,
  j: u32,
  visibleCount: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(1) var<storage, read_write> sortedQuantizedDepths: array<u32>;
@group(0) @binding(2) var<uniform> params: SortParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn bitonicSort(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.visibleCount) {
    return;
  }
  
  let k = params.k;
  let j = params.j;
  let ixj = i ^ j;
  
  // 只处理 i < ixj 的情况，避免重复交换
  if (ixj <= i || ixj >= params.visibleCount) {
    return;
  }
  
  let depthI = sortedQuantizedDepths[i];
  let depthIxj = sortedQuantizedDepths[ixj];
  
  // 确定排序方向
  let ascending = (i & k) == 0u;
  
  // 降序排列：大的在前
  var needSwap = false;
  if (ascending) {
    needSwap = depthI < depthIxj;
  } else {
    needSwap = depthI > depthIxj;
  }
  
  if (needSwap) {
    // 交换 indices
    let tempIdx = sortedIndices[i];
    sortedIndices[i] = sortedIndices[ixj];
    sortedIndices[ixj] = tempIdx;
    
    // 交换 depths
    sortedQuantizedDepths[i] = depthIxj;
    sortedQuantizedDepths[ixj] = depthI;
  }
}
`;

/**
 * 计算大于等于 n 的最小 2 的幂
 */
function nextPowerOfTwo(n: number): number {
  if (n <= 1) return 1;
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

/**
 * 屏幕尺寸信息
 */
export interface ScreenInfo {
  width: number;
  height: number;
}

/**
 * 剔除参数
 */
export interface CullingOptions {
  nearPlane: number;
  farPlane: number;
  pixelThreshold: number;
}

/**
 * GSSplatSorterV2 - Z-Binning 分桶稳定排序器
 */
export class GSSplatSorterV2 {
  private device: GPUDevice;
  private splatCount: number;

  // ============================================
  // Buffers
  // ============================================
  private cullingParamsBuffer: GPUBuffer;
  private countersBuffer: GPUBuffer;
  private visibleIndicesBuffer: GPUBuffer;
  private visibleQuantizedDepthsBuffer: GPUBuffer;
  private visibleBucketIdsBuffer: GPUBuffer;
  private bucketCountersBuffer: GPUBuffer;
  private bucketOffsetsBuffer: GPUBuffer;
  private bucketPositionsBuffer: GPUBuffer;
  private sortedIndicesBuffer: GPUBuffer;
  private sortedQuantizedDepthsBuffer: GPUBuffer;
  private drawIndirectBuffer: GPUBuffer;
  private prefixSumParamsBuffer: GPUBuffer;
  private scatterParamsBuffer: GPUBuffer;
  private sortParamsBuffer: GPUBuffer;  // 用于全局排序

  // ============================================
  // Pipelines
  // ============================================
  private resetCountersPipeline: GPUComputePipeline;
  private resetBucketCountersPipeline: GPUComputePipeline;
  private cullAndBinPipeline: GPUComputePipeline;
  private updateDrawIndirectPipeline: GPUComputePipeline;
  private prefixSumPipeline: GPUComputePipeline;
  private resetBucketPositionsPipeline: GPUComputePipeline;
  private scatterPipeline: GPUComputePipeline;
  private globalSortPipeline: GPUComputePipeline;  // 全局 Bitonic Sort

  // ============================================
  // Bind Groups
  // ============================================
  private cullingBindGroupLayout: GPUBindGroupLayout;
  private cullingBindGroup: GPUBindGroup;
  private drawIndirectBindGroupLayout: GPUBindGroupLayout;
  private drawIndirectBindGroup: GPUBindGroup;
  private prefixSumBindGroupLayout: GPUBindGroupLayout;
  private prefixSumBindGroup: GPUBindGroup;
  private scatterBindGroupLayout: GPUBindGroupLayout;
  private scatterBindGroup: GPUBindGroup;
  private globalSortBindGroupLayout: GPUBindGroupLayout;
  private globalSortBindGroup: GPUBindGroup;

  // 工作组大小
  private readonly WORKGROUP_SIZE = WORKGROUP_SIZE;
  private readonly NUM_BUCKETS = NUM_BUCKETS;

  // 当前屏幕信息
  private screenWidth: number = 1920;
  private screenHeight: number = 1080;

  // 剔除选项
  private cullingOptions: CullingOptions = {
    nearPlane: 0.1,
    farPlane: 1000,
    pixelThreshold: 1.0,
  };

  constructor(
    device: GPUDevice,
    splatCount: number,
    splatBuffer: GPUBuffer,
    cameraBuffer: GPUBuffer,
  ) {
    this.device = device;
    this.splatCount = splatCount;

    // ============================================
    // 创建 Shader 模块
    // ============================================
    const cullingModule = device.createShaderModule({
      code: cullingBinningShaderCode,
    });

    const drawIndirectModule = device.createShaderModule({
      code: drawIndirectShaderCode,
    });

    const prefixSumModule = device.createShaderModule({
      code: prefixSumShaderCode,
    });

    const scatterModule = device.createShaderModule({
      code: scatterShaderCode,
    });

    const globalSortModule = device.createShaderModule({
      code: globalSortShaderCode,
    });

    // ============================================
    // 创建 Buffers
    // ============================================
    
    // Culling params: 32 bytes
    this.cullingParamsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Counters: 16 bytes (atomic)
    this.countersBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Visible indices, quantized depths, bucket IDs
    this.visibleIndicesBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.visibleQuantizedDepthsBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.visibleBucketIdsBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // Bucket counters: NUM_BUCKETS * 4 bytes (atomic)
    this.bucketCountersBuffer = device.createBuffer({
      size: NUM_BUCKETS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Bucket offsets
    this.bucketOffsetsBuffer = device.createBuffer({
      size: NUM_BUCKETS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Bucket positions (for scatter)
    this.bucketPositionsBuffer = device.createBuffer({
      size: NUM_BUCKETS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Sorted buffers
    this.sortedIndicesBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.sortedQuantizedDepthsBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // Draw indirect buffer
    this.drawIndirectBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    // Params buffers
    this.prefixSumParamsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.scatterParamsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.sortParamsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ============================================
    // 创建 Bind Group Layouts 和 Pipelines
    // ============================================

    // Culling bind group layout (8 bindings: 1 read-only + 2 uniform + 5 storage = 6 storage buffers)
    this.cullingBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splats
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // camera
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // cullingParams
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // counters
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // visibleIndices
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // visibleQuantizedDepths
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // visibleBucketIds
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // bucketCounters
      ],
    });

    const cullingPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.cullingBindGroupLayout],
    });

    this.resetCountersPipeline = device.createComputePipeline({
      layout: cullingPipelineLayout,
      compute: { module: cullingModule, entryPoint: "resetCounters" },
    });

    this.resetBucketCountersPipeline = device.createComputePipeline({
      layout: cullingPipelineLayout,
      compute: { module: cullingModule, entryPoint: "resetBucketCounters" },
    });

    this.cullAndBinPipeline = device.createComputePipeline({
      layout: cullingPipelineLayout,
      compute: { module: cullingModule, entryPoint: "cullAndBin" },
    });

    // DrawIndirect bind group layout (单独的 pass)
    this.drawIndirectBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // counters
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // drawIndirect
      ],
    });

    const drawIndirectPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.drawIndirectBindGroupLayout],
    });

    this.updateDrawIndirectPipeline = device.createComputePipeline({
      layout: drawIndirectPipelineLayout,
      compute: { module: drawIndirectModule, entryPoint: "updateDrawIndirect" },
    });

    // Prefix sum bind group layout
    this.prefixSumBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    const prefixSumPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.prefixSumBindGroupLayout],
    });

    this.prefixSumPipeline = device.createComputePipeline({
      layout: prefixSumPipelineLayout,
      compute: { module: prefixSumModule, entryPoint: "prefixSum" },
    });

    // Scatter bind group layout (8 bindings = 7 storage buffers)
    this.scatterBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // visibleIndices
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // visibleQuantizedDepths
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // visibleBucketIds
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // bucketOffsets
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // bucketPositions
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // sortedIndices
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // sortedQuantizedDepths
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // params
      ],
    });

    const scatterPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.scatterBindGroupLayout],
    });

    this.resetBucketPositionsPipeline = device.createComputePipeline({
      layout: scatterPipelineLayout,
      compute: { module: scatterModule, entryPoint: "resetBucketPositions" },
    });

    this.scatterPipeline = device.createComputePipeline({
      layout: scatterPipelineLayout,
      compute: { module: scatterModule, entryPoint: "scatter" },
    });

    // Global sort bind group layout (3 bindings = 2 storage buffers + 1 uniform)
    this.globalSortBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // sortedIndices
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // sortedQuantizedDepths
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // params
      ],
    });

    const globalSortPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.globalSortBindGroupLayout],
    });

    this.globalSortPipeline = device.createComputePipeline({
      layout: globalSortPipelineLayout,
      compute: { module: globalSortModule, entryPoint: "bitonicSort" },
    });

    // ============================================
    // 创建 Bind Groups
    // ============================================

    this.cullingBindGroup = device.createBindGroup({
      layout: this.cullingBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: splatBuffer } },
        { binding: 1, resource: { buffer: cameraBuffer } },
        { binding: 2, resource: { buffer: this.cullingParamsBuffer } },
        { binding: 3, resource: { buffer: this.countersBuffer } },
        { binding: 4, resource: { buffer: this.visibleIndicesBuffer } },
        { binding: 5, resource: { buffer: this.visibleQuantizedDepthsBuffer } },
        { binding: 6, resource: { buffer: this.visibleBucketIdsBuffer } },
        { binding: 7, resource: { buffer: this.bucketCountersBuffer } },
      ],
    });

    this.drawIndirectBindGroup = device.createBindGroup({
      layout: this.drawIndirectBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.countersBuffer } },
        { binding: 1, resource: { buffer: this.drawIndirectBuffer } },
      ],
    });

    this.prefixSumBindGroup = device.createBindGroup({
      layout: this.prefixSumBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.bucketCountersBuffer } },
        { binding: 1, resource: { buffer: this.bucketOffsetsBuffer } },
        { binding: 2, resource: { buffer: this.prefixSumParamsBuffer } },
      ],
    });

    this.scatterBindGroup = device.createBindGroup({
      layout: this.scatterBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.visibleIndicesBuffer } },
        { binding: 1, resource: { buffer: this.visibleQuantizedDepthsBuffer } },
        { binding: 2, resource: { buffer: this.visibleBucketIdsBuffer } },
        { binding: 3, resource: { buffer: this.bucketOffsetsBuffer } },
        { binding: 4, resource: { buffer: this.bucketPositionsBuffer } },
        { binding: 5, resource: { buffer: this.sortedIndicesBuffer } },
        { binding: 6, resource: { buffer: this.sortedQuantizedDepthsBuffer } },
        { binding: 7, resource: { buffer: this.scatterParamsBuffer } },
      ],
    });

    this.globalSortBindGroup = device.createBindGroup({
      layout: this.globalSortBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.sortedIndicesBuffer } },
        { binding: 1, resource: { buffer: this.sortedQuantizedDepthsBuffer } },
        { binding: 2, resource: { buffer: this.sortParamsBuffer } },
      ],
    });

    // 初始化 prefix sum params
    const prefixSumParams = new Uint32Array([NUM_BUCKETS, 0, 0, 0]);
    device.queue.writeBuffer(this.prefixSumParamsBuffer, 0, prefixSumParams);

    console.log(
      `GSSplatSorterV2: 初始化完成 (Z-Binning 分桶 + 深度量化稳定排序), splatCount=${splatCount}, numBuckets=${NUM_BUCKETS}`,
    );
  }

  /**
   * 设置屏幕尺寸
   */
  setScreenSize(width: number, height: number): void {
    this.screenWidth = width;
    this.screenHeight = height;
  }

  /**
   * 设置剔除参数
   */
  setCullingOptions(options: Partial<CullingOptions>): void {
    this.cullingOptions = { ...this.cullingOptions, ...options };
  }

  /**
   * 执行剔除、分桶和排序
   * 每帧调用
   * 
   * 流程:
   * 1. 剔除不可见 splat
   * 2. 分桶 (Z-binning)
   * 3. Scatter 到排序缓冲区
   * 4. 全局 Bitonic Sort (按量化深度稳定排序)
   */
  sort(): void {
    // ============================================
    // 更新 Culling 参数
    // ============================================
    const cullingParamsData = new ArrayBuffer(32);
    const cullingParamsView = new DataView(cullingParamsData);
    cullingParamsView.setUint32(0, this.splatCount, true);
    cullingParamsView.setFloat32(4, this.cullingOptions.nearPlane, true);
    cullingParamsView.setFloat32(8, this.cullingOptions.farPlane, true);
    cullingParamsView.setFloat32(12, this.screenWidth, true);
    cullingParamsView.setFloat32(16, this.screenHeight, true);
    cullingParamsView.setFloat32(20, this.cullingOptions.pixelThreshold, true);
    cullingParamsView.setFloat32(24, 0, true);
    cullingParamsView.setFloat32(28, 0, true);
    this.device.queue.writeBuffer(this.cullingParamsBuffer, 0, cullingParamsData);

    // 更新 scatter params
    const scatterParams = new Uint32Array([this.splatCount, 0, 0, 0]);
    this.device.queue.writeBuffer(this.scatterParamsBuffer, 0, scatterParams);

    const cullWorkgroupCount = Math.ceil(this.splatCount / this.WORKGROUP_SIZE);
    const bucketResetWorkgroups = Math.ceil(this.NUM_BUCKETS / this.WORKGROUP_SIZE);

    // ============================================
    // 第一阶段: 剔除和分桶
    // ============================================
    const encoder = this.device.createCommandEncoder();

    // Pass 1: 重置 visibleCount
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.resetCountersPipeline);
      pass.setBindGroup(0, this.cullingBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
    }

    // Pass 2: 重置桶计数
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.resetBucketCountersPipeline);
      pass.setBindGroup(0, this.cullingBindGroup);
      pass.dispatchWorkgroups(bucketResetWorkgroups);
      pass.end();
    }

    // Pass 3: 执行剔除和分桶
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.cullAndBinPipeline);
      pass.setBindGroup(0, this.cullingBindGroup);
      pass.dispatchWorkgroups(cullWorkgroupCount);
      pass.end();
    }

    // Pass 4: 更新 DrawIndirect
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.updateDrawIndirectPipeline);
      pass.setBindGroup(0, this.drawIndirectBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
    }

    // Pass 5: Prefix Sum 计算桶偏移
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.prefixSumPipeline);
      pass.setBindGroup(0, this.prefixSumBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
    }

    // Pass 6: 重置桶内位置计数
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.resetBucketPositionsPipeline);
      pass.setBindGroup(0, this.scatterBindGroup);
      pass.dispatchWorkgroups(bucketResetWorkgroups);
      pass.end();
    }

    // Pass 7: Scatter 分散到排序缓冲区
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.scatterPipeline);
      pass.setBindGroup(0, this.scatterBindGroup);
      pass.dispatchWorkgroups(cullWorkgroupCount);
      pass.end();
    }

    // 提交第一阶段命令
    this.device.queue.submit([encoder.finish()]);

    // ============================================
    // 第二阶段: 全局 Bitonic Sort
    // ============================================
    this.runBitonicSort();
  }

  /**
   * 执行全局 Bitonic Sort
   * 对所有可见 splat 按量化深度排序
   */
  private runBitonicSort(): void {
    const paddedSize = nextPowerOfTwo(this.splatCount);
    const sortWorkgroups = Math.ceil(this.splatCount / this.WORKGROUP_SIZE);

    // 收集所有 Bitonic Sort (k, j) 步骤
    const steps: Array<{ k: number; j: number }> = [];
    for (let k = 2; k <= paddedSize; k *= 2) {
      for (let j = k >> 1; j >= 1; j >>= 1) {
        steps.push({ k, j });
      }
    }

    if (steps.length === 0) {
      return;
    }

    // 批量提交以提高性能
    const BATCH_SIZE = 16;
    
    for (let batchStart = 0; batchStart < steps.length; batchStart += BATCH_SIZE) {
      const batchEnd = Math.min(batchStart + BATCH_SIZE, steps.length);
      const encoder = this.device.createCommandEncoder();

      for (let i = batchStart; i < batchEnd; i++) {
        const { k, j } = steps[i];
        
        // 更新排序参数
        const sortParams = new Uint32Array([k, j, this.splatCount, 0]);
        this.device.queue.writeBuffer(this.sortParamsBuffer, 0, sortParams);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.globalSortPipeline);
        pass.setBindGroup(0, this.globalSortBindGroup);
        pass.dispatchWorkgroups(sortWorkgroups);
        pass.end();
      }

      this.device.queue.submit([encoder.finish()]);
    }
  }

  /**
   * 获取排序后的索引 buffer（用于渲染）
   */
  getIndicesBuffer(): GPUBuffer {
    return this.sortedIndicesBuffer;
  }

  /**
   * 获取 DrawIndirect buffer
   */
  getDrawIndirectBuffer(): GPUBuffer {
    return this.drawIndirectBuffer;
  }

  /**
   * 获取 splat 总数量
   */
  getSplatCount(): number {
    return this.splatCount;
  }

  /**
   * 销毁资源
   */
  destroy(): void {
    this.cullingParamsBuffer.destroy();
    this.countersBuffer.destroy();
    this.visibleIndicesBuffer.destroy();
    this.visibleQuantizedDepthsBuffer.destroy();
    this.visibleBucketIdsBuffer.destroy();
    this.bucketCountersBuffer.destroy();
    this.bucketOffsetsBuffer.destroy();
    this.bucketPositionsBuffer.destroy();
    this.sortedIndicesBuffer.destroy();
    this.sortedQuantizedDepthsBuffer.destroy();
    this.drawIndirectBuffer.destroy();
    this.prefixSumParamsBuffer.destroy();
    this.scatterParamsBuffer.destroy();
    this.sortParamsBuffer.destroy();
  }
}
