/**
 * GSSplatSorter - GPU Radix Sort 深度排序器
 *
 * 基于 rfs-gsplat-render 的 3-Pass Radix Sort 架构实现:
 * - 4 个 pass (8-bit 增量，总共 32 位)
 * - 每个 pass 包含: Upsweep -> Spine -> Downsweep
 * - 稳定排序，解决远距离闪烁问题
 *
 * 参考: rfs-gsplat-render/assets/shaders/radix_sort.wgsl
 */

const WORKGROUP_SIZE = 256;
const RADIX_BITS = 8;
const RADIX_SIZE = 256; // 2^8
const ELEMENTS_PER_THREAD = 4;
const BLOCK_SIZE = WORKGROUP_SIZE * ELEMENTS_PER_THREAD; // 1024

/**
 * 生成 Culling Shader 代码
 * 基于 rfs-gsplat-render/assets/shaders/gaussian_splat_cull.wgsl
 */
function generateCullingShaderCode(): string {
  return /* wgsl */ `
/**
 * Project & Cull Shader
 * 基于 rfs-gsplat-render 实现
 */

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
  frustumDilation: f32,
  pixelThreshold: f32,
  _pad1: f32,
}

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<uniform> params: CullingParams;
@group(0) @binding(3) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> visibleIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirectBuffer: array<atomic<u32>, 4>;

fn maxScale(scale: vec3<f32>) -> f32 {
  return max(max(scale.x, scale.y), scale.z);
}

fn getModelMaxScale(model: mat4x4<f32>) -> f32 {
  let sx = length(model[0].xyz);
  let sy = length(model[1].xyz);
  let sz = length(model[2].xyz);
  return max(max(sx, sy), sz);
}

// IEEE 754 位操作编码浮点数为可排序的 u32
// 参考 rfs-gsplat-render 的 encode_min_max_fp32 实现
fn encodeDepthKey(val: f32) -> u32 {
  var bits = bitcast<u32>(val);
  bits ^= bitcast<u32>(bitcast<i32>(bits) >> 31) | 0x80000000u;
  return bits;
}

// 视锥剔除检查
// 基于 rfs-gsplat-render 的 is_in_frustum 实现
fn isInFrustum(clipPos: vec4<f32>, frustumDilation: f32) -> bool {
  let clip = (1.0 + frustumDilation) * clipPos.w;
  
  if abs(clipPos.x) > clip { return false; }
  if abs(clipPos.y) > clip { return false; }
  
  let nearThreshold = (0.0 - frustumDilation) * clipPos.w;
  if clipPos.z < nearThreshold || clipPos.z > clipPos.w {
    return false;
  }
  
  return true;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn projectAndCull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.splatCount { return; }
  
  let splat = splats[i];
  
  // 透明度剔除
  if splat.opacity < 0.004 { return; }
  
  // 变换: Local -> World -> View -> Clip
  let worldPos = camera.model * vec4<f32>(splat.mean, 1.0);
  let viewPos = camera.view * worldPos;
  let clipPos = camera.proj * viewPos;
  
  // 视锥剔除
  if !isInFrustum(clipPos, params.frustumDilation) { return; }
  
  // 深度编码 (viewPos.z 是负数)
  let depth = viewPos.z;
  let sortableDepth = encodeDepthKey(depth);
  
  // 原子增加可见计数并获取索引
  // indirectBuffer[1] 是 instance_count
  let visibleIdx = atomicAdd(&indirectBuffer[1], 1u);
  
  // 写入可见点列表
  depthKeys[visibleIdx] = sortableDepth;
  visibleIndices[visibleIdx] = i;
}

@compute @workgroup_size(1)
fn initIndirectBuffer() {
  // [vertex_count, instance_count, first_vertex, first_instance]
  atomicStore(&indirectBuffer[0], 4u);
  atomicStore(&indirectBuffer[1], 0u);  // instance_count 由 cull shader 填充
  atomicStore(&indirectBuffer[2], 0u);
  atomicStore(&indirectBuffer[3], 0u);
}
`;
}

/**
 * 生成 Radix Sort Shader 代码
 * 完整移植自 rfs-gsplat-render/assets/shaders/radix_sort.wgsl
 */
function generateRadixSortShaderCode(): string {
  return /* wgsl */ `
/**
 * GPU Radix Sort - 3-Pass Architecture
 * 基于 rfs-gsplat-render 实现
 * 
 * Pass 1: Upsweep - 构建局部直方图并累加到全局
 * Pass 2: Spine - 对分区和全局直方图进行前缀和
 * Pass 3: Downsweep - 使用计算的偏移量散射元素 (稳定排序)
 */

const WG: u32 = ${WORKGROUP_SIZE}u;
const RADIX_BITS: u32 = ${RADIX_BITS}u;
const RADIX_SIZE: u32 = ${RADIX_SIZE}u;
const RADIX_MASK: u32 = ${RADIX_SIZE - 1}u;
const ELEMENTS_PER_THREAD: u32 = ${ELEMENTS_PER_THREAD}u;
const BLOCK_SIZE: u32 = ${BLOCK_SIZE}u;

fn divCeil(a: u32, b: u32) -> u32 {
  return (a + b - 1u) / b;
}

struct SortParams {
  maxElementCount: u32,
  bitShift: u32,
  passIndex: u32,
  _padding: u32,
}

// ============================================================================
// Pass 1: Upsweep - 计数局部直方图并累加到全局
// ============================================================================

@group(0) @binding(0) var<uniform> upsweepParams: SortParams;
@group(0) @binding(1) var<storage, read> indirectBufferUpsweep: array<u32>;
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> globalHistogram: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> partitionHistogram: array<u32>;

var<workgroup> localHistogram: array<atomic<u32>, RADIX_SIZE>;

@compute @workgroup_size(256, 1, 1)
fn upsweep(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  // 从 indirectBuffer[1] 读取动态可见数量 (instance_count)
  let numKeys = indirectBufferUpsweep[1];
  let numPartitions = divCeil(numKeys, BLOCK_SIZE);
  let partitionId = workgroupId.x;
  
  if partitionId >= numPartitions { return; }
  
  let tid = localId.x;
  let partitionStart = partitionId * BLOCK_SIZE;
  let shift = upsweepParams.bitShift;
  let passIdx = upsweepParams.passIndex;
  
  // 初始化局部直方图
  if tid < RADIX_SIZE {
    atomicStore(&localHistogram[tid], 0u);
  }
  workgroupBarrier();
  
  // 构建局部直方图
  for (var j = 0u; j < ELEMENTS_PER_THREAD; j++) {
    let keyIdx = partitionStart + tid * ELEMENTS_PER_THREAD + j;
    if keyIdx < numKeys {
      let key = keysIn[keyIdx];
      let bin = (key >> shift) & RADIX_MASK;
      atomicAdd(&localHistogram[bin], 1u);
    }
  }
  
  workgroupBarrier();
  
  // 写入分区直方图并累加到全局直方图
  if tid < RADIX_SIZE {
    let count = atomicLoad(&localHistogram[tid]);
    partitionHistogram[RADIX_SIZE * partitionId + tid] = count;
    atomicAdd(&globalHistogram[RADIX_SIZE * passIdx + tid], count);
  }
}

// ============================================================================
// Pass 2: Spine - 对分区和全局直方图进行前缀和
// ============================================================================

@group(0) @binding(0) var<storage, read> indirectBufferSpine: array<u32>;
@group(0) @binding(1) var<storage, read_write> globalHistogramSpine: array<u32>;
@group(0) @binding(2) var<storage, read_write> partitionHistogramSpine: array<u32>;
@group(0) @binding(3) var<uniform> spineParams: SortParams;

// 双缓冲用于无数据竞争的 Hillis-Steele scan
var<workgroup> scanA: array<u32, 256>;
var<workgroup> scanB: array<u32, 256>;
var<workgroup> reductionShared: u32;

@compute @workgroup_size(256, 1, 1)
fn spine(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let numKeys = indirectBufferSpine[1];
  let numPartitions = divCeil(numKeys, BLOCK_SIZE);
  let bin = workgroupId.x;
  let tid = localId.x;
  
  if bin >= RADIX_SIZE { return; }
  
  // 初始化共享 reduction
  if tid == 0u {
    reductionShared = 0u;
  }
  workgroupBarrier();
  
  // 处理此 bin 的所有分区（分批处理）
  let MAX_BATCH_SIZE = 256u;
  for (var batchStart = 0u; batchStart < numPartitions; batchStart += MAX_BATCH_SIZE) {
    let partitionIdx = batchStart + tid;
    let batchSize = min(MAX_BATCH_SIZE, numPartitions - batchStart);
    
    // 加载此批次的值
    if tid < batchSize && partitionIdx < numPartitions {
      scanA[tid] = partitionHistogramSpine[RADIX_SIZE * partitionIdx + bin];
    } else {
      scanA[tid] = 0u;
    }
    workgroupBarrier();
    
    // Hillis-Steele inclusive prefix sum (双缓冲避免数据竞争)
    var useA = true;
    var offset = 1u;
    for (var d = 0u; d < 8u; d++) {
      if useA {
        if tid >= offset {
          scanB[tid] = scanA[tid] + scanA[tid - offset];
        } else {
          scanB[tid] = scanA[tid];
        }
      } else {
        if tid >= offset {
          scanA[tid] = scanB[tid] + scanB[tid - offset];
        } else {
          scanA[tid] = scanB[tid];
        }
      }
      workgroupBarrier();
      useA = !useA;
      offset <<= 1u;
    }
    
    // 8 次迭代后结果在 scanA 中
    
    // 写回为 exclusive prefix sum（加上 reduction）
    if tid < batchSize && partitionIdx < numPartitions {
      var exclusive = reductionShared;
      if tid > 0u {
        exclusive += scanA[tid - 1u];
      }
      partitionHistogramSpine[RADIX_SIZE * partitionIdx + bin] = exclusive;
    }
    
    // 更新下一批的 reduction
    workgroupBarrier();
    if tid == 0u && batchSize > 0u {
      reductionShared += scanA[batchSize - 1u];
    }
    workgroupBarrier();
  }
  
  // Bin 0 的工作组同时处理全局直方图前缀和
  if bin == 0u {
    let passIdx = spineParams.passIndex;
    scanA[tid] = globalHistogramSpine[RADIX_SIZE * passIdx + tid];
    workgroupBarrier();
    
    // Hillis-Steele inclusive scan (双缓冲)
    var useA = true;
    var offset = 1u;
    for (var d = 0u; d < 8u; d++) {
      if useA {
        if tid >= offset {
          scanB[tid] = scanA[tid] + scanA[tid - offset];
        } else {
          scanB[tid] = scanA[tid];
        }
      } else {
        if tid >= offset {
          scanA[tid] = scanB[tid] + scanB[tid - offset];
        } else {
          scanA[tid] = scanB[tid];
        }
      }
      workgroupBarrier();
      useA = !useA;
      offset <<= 1u;
    }
    
    // 转换为 exclusive (结果在 scanA 中)
    var exclusive = 0u;
    if tid > 0u {
      exclusive = scanA[tid - 1u];
    }
    globalHistogramSpine[RADIX_SIZE * passIdx + tid] = exclusive;
  }
}

// ============================================================================
// Pass 3: Downsweep - 使用偏移量散射元素 (稳定排序)
// ============================================================================

@group(0) @binding(0) var<uniform> downsweepParams: SortParams;
@group(0) @binding(1) var<storage, read> indirectBufferDownsweep: array<u32>;
@group(0) @binding(2) var<storage, read> globalHistogramDownsweep: array<u32>;
@group(0) @binding(3) var<storage, read> partitionHistogramDownsweep: array<u32>;
@group(0) @binding(4) var<storage, read> downsweepKeysIn: array<u32>;
@group(0) @binding(5) var<storage, read> downsweepValuesIn: array<u32>;
@group(0) @binding(6) var<storage, read_write> downsweepKeysOut: array<u32>;
@group(0) @binding(7) var<storage, read_write> downsweepValuesOut: array<u32>;

var<workgroup> localKeys: array<u32, BLOCK_SIZE>;
var<workgroup> localValues: array<u32, BLOCK_SIZE>;
var<workgroup> localBins: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(256, 1, 1)
fn downsweep(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let numKeys = indirectBufferDownsweep[1];
  let numPartitions = divCeil(numKeys, BLOCK_SIZE);
  let partitionId = workgroupId.x;
  
  if partitionId >= numPartitions { return; }
  
  let tid = localId.x;
  let partitionStart = partitionId * BLOCK_SIZE;
  let shift = downsweepParams.bitShift;
  
  // 加载元素到共享内存
  for (var j = 0u; j < ELEMENTS_PER_THREAD; j++) {
    let keyIdx = partitionStart + tid * ELEMENTS_PER_THREAD + j;
    let localIdx = tid * ELEMENTS_PER_THREAD + j;
    
    if keyIdx < numKeys {
      let key = downsweepKeysIn[keyIdx];
      localKeys[localIdx] = key;
      localValues[localIdx] = downsweepValuesIn[keyIdx];
      localBins[localIdx] = (key >> shift) & RADIX_MASK;
    } else {
      localBins[localIdx] = 0xFFFFFFFFu;
    }
  }
  
  workgroupBarrier();
  
  // 线程 0 执行顺序散射以保持稳定性
  // 这是 rfs-gsplat-render 的关键设计，确保稳定排序
  if tid == 0u {
    var binWritePos: array<u32, RADIX_SIZE>;
    
    let passIdx = downsweepParams.passIndex;
    
    // 从全局 + 分区偏移初始化写入位置
    for (var b = 0u; b < RADIX_SIZE; b++) {
      binWritePos[b] = globalHistogramDownsweep[RADIX_SIZE * passIdx + b] + 
                       partitionHistogramDownsweep[RADIX_SIZE * partitionId + b];
    }
    
    // 按输入顺序顺序写入 (稳定)
    let partitionEnd = min(partitionStart + BLOCK_SIZE, numKeys);
    for (var k = 0u; k < BLOCK_SIZE; k++) {
      let keyIdx = partitionStart + k;
      if keyIdx < partitionEnd {
        let b = localBins[k];
        if b != 0xFFFFFFFFu {
          let writePos = binWritePos[b];
          if writePos < numKeys {
            downsweepKeysOut[writePos] = localKeys[k];
            downsweepValuesOut[writePos] = localValues[k];
            binWritePos[b]++;
          }
        }
      }
    }
  }
}
`;
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
  frustumDilation?: number;
}

/**
 * 排序器配置选项
 */
export interface SorterOptions {
  /** 暂时保留，Radix Sort 不使用桶配置 */
  numBuckets?: number;
}

/**
 * GSSplatSorter - GPU Radix Sort 排序器
 * 基于 rfs-gsplat-render 的 3-Pass Radix Sort 实现
 */
export class GSSplatSorter {
  private device: GPUDevice;
  private splatCount: number;

  // Culling Buffers
  private cullingParamsBuffer: GPUBuffer;
  private depthKeysBuffer: GPUBuffer;
  private visibleIndicesBuffer: GPUBuffer;
  private indirectBuffer: GPUBuffer;

  // Radix Sort Buffers
  private globalHistogramBuffer: GPUBuffer;
  private partitionHistogramBuffer: GPUBuffer;
  private keysTempBuffer: GPUBuffer;
  private valuesTempBuffer: GPUBuffer;
  // 每个 pass 独立的参数 buffer (避免竞争)
  private sortParamsBuffers: GPUBuffer[] = [];

  // Sorted output
  private sortedIndicesBuffer: GPUBuffer;

  // Culling Pipelines
  private initIndirectPipeline: GPUComputePipeline;
  private projectCullPipeline: GPUComputePipeline;
  private cullingBindGroupLayout: GPUBindGroupLayout;
  private cullingBindGroup: GPUBindGroup;

  // Radix Sort Pipelines
  private upsweepPipeline: GPUComputePipeline;
  private spinePipeline: GPUComputePipeline;
  private downsweepPipeline: GPUComputePipeline;
  private upsweepBindGroupLayout: GPUBindGroupLayout;
  private spineBindGroupLayout: GPUBindGroupLayout;
  private downsweepBindGroupLayout: GPUBindGroupLayout;

  // Bind groups for each pass (4 passes)
  private upsweepBindGroups: GPUBindGroup[] = [];
  private spineBindGroups: GPUBindGroup[] = [];
  private downsweepBindGroups: GPUBindGroup[] = [];

  private numPartitions: number;

  // 屏幕信息和剔除选项
  private screenWidth: number = 1920;
  private screenHeight: number = 1080;
  private cullingOptions: CullingOptions = {
    nearPlane: 0.1,
    farPlane: 1000,
    pixelThreshold: 0,
    frustumDilation: 0.2,
  };

  constructor(
    device: GPUDevice,
    splatCount: number,
    splatBuffer: GPUBuffer,
    cameraBuffer: GPUBuffer,
    _options: SorterOptions = {},
  ) {
    this.device = device;
    this.splatCount = splatCount;
    this.numPartitions = Math.ceil(splatCount / BLOCK_SIZE);

    // ============================================
    // 创建 Shader 模块
    // ============================================
    const cullingModule = device.createShaderModule({
      code: generateCullingShaderCode(),
      label: "culling-shader",
    });

    const radixSortModule = device.createShaderModule({
      code: generateRadixSortShaderCode(),
      label: "radix-sort-shader",
    });

    // ============================================
    // 创建 Buffers
    // ============================================

    // Culling params: splatCount, nearPlane, farPlane, screenWidth, screenHeight, frustumDilation, pixelThreshold, _pad
    this.cullingParamsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "culling-params",
    });

    this.depthKeysBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "depth-keys",
    });

    this.visibleIndicesBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "visible-indices",
    });

    // Indirect buffer: [vertex_count, instance_count, first_vertex, first_instance]
    this.indirectBuffer = device.createBuffer({
      size: 16,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.COPY_DST,
      label: "indirect-buffer",
    });

    // Global histogram: 4 passes * RADIX_SIZE bins
    this.globalHistogramBuffer = device.createBuffer({
      size: RADIX_SIZE * 4 * 4, // 4 passes * 256 bins * 4 bytes
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "global-histogram",
    });

    // Partition histogram: numPartitions * RADIX_SIZE
    this.partitionHistogramBuffer = device.createBuffer({
      size: this.numPartitions * RADIX_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "partition-histogram",
    });

    // Temp buffers for ping-pong
    this.keysTempBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "keys-temp",
    });

    this.valuesTempBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "values-temp",
    });

    // 为每个 pass 创建独立的参数 buffer (4 个 pass)
    for (let i = 0; i < 4; i++) {
      const paramsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: `sort-params-${i}`,
      });
      this.sortParamsBuffers.push(paramsBuffer);
      
      // 预填充参数
      const sortParams = new ArrayBuffer(16);
      const sortView = new DataView(sortParams);
      sortView.setUint32(0, splatCount, true); // maxElementCount
      sortView.setUint32(4, i * RADIX_BITS, true); // bitShift
      sortView.setUint32(8, i, true); // passIndex
      sortView.setUint32(12, 0, true); // padding
      device.queue.writeBuffer(paramsBuffer, 0, sortParams);
    }

    // Final sorted indices
    this.sortedIndicesBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: "sorted-indices",
    });

    // ============================================
    // 创建 Culling Pipelines
    // ============================================
    this.cullingBindGroupLayout = device.createBindGroupLayout({
      label: "culling-bind-group-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const cullingPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.cullingBindGroupLayout],
    });

    this.initIndirectPipeline = device.createComputePipeline({
      layout: cullingPipelineLayout,
      compute: { module: cullingModule, entryPoint: "initIndirectBuffer" },
      label: "init-indirect-pipeline",
    });

    this.projectCullPipeline = device.createComputePipeline({
      layout: cullingPipelineLayout,
      compute: { module: cullingModule, entryPoint: "projectAndCull" },
      label: "project-cull-pipeline",
    });

    this.cullingBindGroup = device.createBindGroup({
      layout: this.cullingBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: splatBuffer } },
        { binding: 1, resource: { buffer: cameraBuffer } },
        { binding: 2, resource: { buffer: this.cullingParamsBuffer } },
        { binding: 3, resource: { buffer: this.depthKeysBuffer } },
        { binding: 4, resource: { buffer: this.visibleIndicesBuffer } },
        { binding: 5, resource: { buffer: this.indirectBuffer } },
      ],
      label: "culling-bind-group",
    });

    // ============================================
    // 创建 Radix Sort Pipelines
    // ============================================

    // Upsweep layout: params, indirect, keys_in, global_histogram, partition_histogram
    this.upsweepBindGroupLayout = device.createBindGroupLayout({
      label: "upsweep-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // Spine layout: indirect, global_histogram, partition_histogram, params
    this.spineBindGroupLayout = device.createBindGroupLayout({
      label: "spine-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    // Downsweep layout: params, indirect, global_histogram, partition_histogram, keys_in, values_in, keys_out, values_out
    this.downsweepBindGroupLayout = device.createBindGroupLayout({
      label: "downsweep-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    this.upsweepPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.upsweepBindGroupLayout] }),
      compute: { module: radixSortModule, entryPoint: "upsweep" },
      label: "upsweep-pipeline",
    });

    this.spinePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.spineBindGroupLayout] }),
      compute: { module: radixSortModule, entryPoint: "spine" },
      label: "spine-pipeline",
    });

    this.downsweepPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.downsweepBindGroupLayout] }),
      compute: { module: radixSortModule, entryPoint: "downsweep" },
      label: "downsweep-pipeline",
    });

    // ============================================
    // 为 4 个 pass 创建 bind groups (ping-pong buffers)
    // ============================================
    this.createRadixSortBindGroups();
  }

  /**
   * 创建 Radix Sort 的 bind groups
   * 4 个 pass，使用 ping-pong buffers
   * 
   * Ping-pong 模式:
   * - Pass 0: depthKeys/visibleIndices -> keysTempBuffer/valuesTempBuffer
   * - Pass 1: keysTempBuffer/valuesTempBuffer -> depthKeys/visibleIndices
   * - Pass 2: depthKeys/visibleIndices -> keysTempBuffer/valuesTempBuffer
   * - Pass 3: keysTempBuffer/valuesTempBuffer -> (depthKeys)/sortedIndicesBuffer
   */
  private createRadixSortBindGroups(): void {
    for (let passIdx = 0; passIdx < 4; passIdx++) {
      const isEvenPass = passIdx % 2 === 0;

      // Ping-pong: 偶数 pass 从原始 buffer 读取，奇数 pass 从临时 buffer 读取
      const keysIn = isEvenPass ? this.depthKeysBuffer : this.keysTempBuffer;
      const valuesIn = isEvenPass ? this.visibleIndicesBuffer : this.valuesTempBuffer;
      
      // 输出: 偶数 pass 写入临时 buffer，奇数 pass 写回原始 buffer
      // 最后一个 pass (pass 3) 的 values 输出到 sortedIndicesBuffer
      let keysOut: GPUBuffer;
      let valuesOut: GPUBuffer;
      
      if (isEvenPass) {
        keysOut = this.keysTempBuffer;
        valuesOut = this.valuesTempBuffer;
      } else {
        keysOut = this.depthKeysBuffer;
        // Pass 3 输出最终排序结果
        valuesOut = passIdx === 3 ? this.sortedIndicesBuffer : this.visibleIndicesBuffer;
      }

      this.upsweepBindGroups[passIdx] = this.device.createBindGroup({
        layout: this.upsweepBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.sortParamsBuffers[passIdx] } },
          { binding: 1, resource: { buffer: this.indirectBuffer } },
          { binding: 2, resource: { buffer: keysIn } },
          { binding: 3, resource: { buffer: this.globalHistogramBuffer } },
          { binding: 4, resource: { buffer: this.partitionHistogramBuffer } },
        ],
        label: `upsweep-bind-group-${passIdx}`,
      });

      this.spineBindGroups[passIdx] = this.device.createBindGroup({
        layout: this.spineBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.indirectBuffer } },
          { binding: 1, resource: { buffer: this.globalHistogramBuffer } },
          { binding: 2, resource: { buffer: this.partitionHistogramBuffer } },
          { binding: 3, resource: { buffer: this.sortParamsBuffers[passIdx] } },
        ],
        label: `spine-bind-group-${passIdx}`,
      });

      this.downsweepBindGroups[passIdx] = this.device.createBindGroup({
        layout: this.downsweepBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.sortParamsBuffers[passIdx] } },
          { binding: 1, resource: { buffer: this.indirectBuffer } },
          { binding: 2, resource: { buffer: this.globalHistogramBuffer } },
          { binding: 3, resource: { buffer: this.partitionHistogramBuffer } },
          { binding: 4, resource: { buffer: keysIn } },
          { binding: 5, resource: { buffer: valuesIn } },
          { binding: 6, resource: { buffer: keysOut } },
          { binding: 7, resource: { buffer: valuesOut } },
        ],
        label: `downsweep-bind-group-${passIdx}`,
      });
    }
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
   * 执行剔除和排序
   * 每帧调用
   */
  sort(): void {
    // ============================================
    // 更新 Culling 参数
    // ============================================
    const cullingParamsData = new ArrayBuffer(32);
    const view = new DataView(cullingParamsData);
    view.setUint32(0, this.splatCount, true);
    view.setFloat32(4, this.cullingOptions.nearPlane, true);
    view.setFloat32(8, this.cullingOptions.farPlane, true);
    view.setFloat32(12, this.screenWidth, true);
    view.setFloat32(16, this.screenHeight, true);
    view.setFloat32(20, this.cullingOptions.frustumDilation ?? 0.2, true);
    view.setFloat32(24, this.cullingOptions.pixelThreshold, true);
    view.setFloat32(28, 0, true);
    this.device.queue.writeBuffer(this.cullingParamsBuffer, 0, cullingParamsData);

    const encoder = this.device.createCommandEncoder({ label: "splat-sort-encoder" });

    // ============================================
    // 清理 buffers (关键! 防止上一帧数据导致闪烁)
    // ============================================
    encoder.clearBuffer(this.depthKeysBuffer);
    encoder.clearBuffer(this.visibleIndicesBuffer);
    encoder.clearBuffer(this.keysTempBuffer);
    encoder.clearBuffer(this.valuesTempBuffer);
    encoder.clearBuffer(this.globalHistogramBuffer);
    encoder.clearBuffer(this.partitionHistogramBuffer);

    // ============================================
    // Pass 0: 初始化 Indirect Buffer
    // ============================================
    {
      const pass = encoder.beginComputePass({ label: "init-indirect" });
      pass.setPipeline(this.initIndirectPipeline);
      pass.setBindGroup(0, this.cullingBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
    }

    // ============================================
    // Pass 1: Project & Cull
    // ============================================
    {
      const pass = encoder.beginComputePass({ label: "project-cull" });
      pass.setPipeline(this.projectCullPipeline);
      pass.setBindGroup(0, this.cullingBindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.splatCount / WORKGROUP_SIZE));
      pass.end();
    }

    // ============================================
    // Radix Sort: 4 passes (8-bit increments)
    // 每个 pass 在独立的 compute pass 中以确保内存同步
    // 参数已在构造函数中预填充到独立的 buffer
    // ============================================
    for (let passIdx = 0; passIdx < 4; passIdx++) {
      // Upsweep
      {
        const pass = encoder.beginComputePass({ label: `upsweep-p${passIdx}` });
        pass.setPipeline(this.upsweepPipeline);
        pass.setBindGroup(0, this.upsweepBindGroups[passIdx]);
        pass.dispatchWorkgroups(this.numPartitions);
        pass.end();
      }

      // Spine
      {
        const pass = encoder.beginComputePass({ label: `spine-p${passIdx}` });
        pass.setPipeline(this.spinePipeline);
        pass.setBindGroup(0, this.spineBindGroups[passIdx]);
        pass.dispatchWorkgroups(RADIX_SIZE);
        pass.end();
      }

      // Downsweep
      {
        const pass = encoder.beginComputePass({ label: `downsweep-p${passIdx}` });
        pass.setPipeline(this.downsweepPipeline);
        pass.setBindGroup(0, this.downsweepBindGroups[passIdx]);
        pass.dispatchWorkgroups(this.numPartitions);
        pass.end();
      }
    }

    this.device.queue.submit([encoder.finish()]);
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
    return this.indirectBuffer;
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
    this.depthKeysBuffer.destroy();
    this.visibleIndicesBuffer.destroy();
    this.indirectBuffer.destroy();
    this.globalHistogramBuffer.destroy();
    this.partitionHistogramBuffer.destroy();
    this.keysTempBuffer.destroy();
    this.valuesTempBuffer.destroy();
    for (const buffer of this.sortParamsBuffers) {
      buffer.destroy();
    }
    this.sortedIndicesBuffer.destroy();
  }
}
