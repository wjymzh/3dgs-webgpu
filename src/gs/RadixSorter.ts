/**
 * RadixSorter - GPU Radix Sort 深度排序器
 * 
 * 基于 rfs-gsplat-render 的 3-Pass 架构实现:
 * Pass 1: Upsweep - 构建局部直方图并累加到全局
 * Pass 2: Spine - 对分区和全局直方图进行前缀和
 * Pass 3: Downsweep - 使用计算的偏移量散射元素
 * 
 * 优势:
 * - O(n) 时间复杂度
 * - 稳定排序（保持相同深度元素的相对顺序）
 * - 适合大规模 splat 排序
 */

const RADIX_BITS = 8;
const RADIX_SIZE = 256;
const RADIX_MASK = 255;
const WORKGROUP_SIZE = 256;
const ELEMENTS_PER_THREAD = 4;
const BLOCK_SIZE = WORKGROUP_SIZE * ELEMENTS_PER_THREAD; // 1024

// ============================================
// Shader 代码
// ============================================

const projectCullShader = /* wgsl */ `
struct CameraUniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  model: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad: f32,
  screenSize: vec2<f32>,
  _pad2: vec2<f32>,
}

struct CullParams {
  splatCount: u32,
  nearPlane: f32,
  farPlane: f32,
  screenWidth: f32,
  screenHeight: f32,
  pixelThreshold: f32,
  frustumDilation: f32,
  _pad: f32,
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

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: CullParams;
@group(0) @binding(2) var<storage, read> splats: array<Splat>;
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

// 将浮点数编码为可排序的 u32 (IEEE 754 位操作)
fn encodeDepth(val: f32) -> u32 {
  var bits = bitcast<u32>(val);
  bits ^= bitcast<u32>(bitcast<i32>(bits) >> 31) | 0x80000000u;
  return bits;
}

fn isInFrustum(clipPos: vec4<f32>, dilation: f32) -> bool {
  let clip = (1.0 + dilation) * clipPos.w;
  if abs(clipPos.x) > clip { return false; }
  if abs(clipPos.y) > clip { return false; }
  let nearThreshold = (0.0 - dilation) * clipPos.w;
  if clipPos.z < nearThreshold || clipPos.z > clipPos.w { return false; }
  return true;
}

@compute @workgroup_size(256, 1, 1)
fn projectAndCull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.splatCount { return; }
  
  let splat = splats[i];
  
  // 透明度剔除
  if splat.opacity < 0.004 { return; }
  
  // 变换到裁剪空间
  let localPos = vec4<f32>(splat.mean, 1.0);
  let worldPos = camera.model * localPos;
  let viewPos = camera.view * worldPos;
  let clipPos = camera.proj * viewPos;
  
  // 视锥剔除
  if !isInFrustum(clipPos, params.frustumDilation) { return; }
  
  // 屏幕尺寸剔除
  let z = -viewPos.z;
  if z < params.nearPlane { return; }
  
  let fx = camera.proj[0][0];
  let fy = camera.proj[1][1];
  let modelScale = getModelMaxScale(camera.model);
  let worldRadius = maxScale(splat.scale) * modelScale * 3.0;
  let rNdc = worldRadius * max(fx, fy) / z;
  let screenRadius = max(rNdc * params.screenWidth * 0.5, rNdc * params.screenHeight * 0.5);
  
  if screenRadius < params.pixelThreshold { return; }
  
  // 通过剔除，计算深度键
  let depth = viewPos.z; // 保持负值用于排序
  let sortableDepth = encodeDepth(depth);
  
  // 原子递增获取唯一索引
  let visibleIdx = atomicAdd(&indirectBuffer[1], 1u);
  
  // 写入可见点列表
  depthKeys[visibleIdx] = sortableDepth;
  visibleIndices[visibleIdx] = i;
}

@compute @workgroup_size(1)
fn resetIndirect() {
  atomicStore(&indirectBuffer[0], 4u);  // vertexCount
  atomicStore(&indirectBuffer[1], 0u);  // instanceCount (will be incremented)
  atomicStore(&indirectBuffer[2], 0u);  // firstVertex
  atomicStore(&indirectBuffer[3], 0u);  // firstInstance
}
`;

const upsweepShader = /* wgsl */ `
const RADIX_SIZE: u32 = 256u;
const RADIX_MASK: u32 = 255u;
const ELEMENTS_PER_THREAD: u32 = 4u;
const BLOCK_SIZE: u32 = 1024u;

struct SortParams {
  maxElementCount: u32,
  bitShift: u32,
  passIndex: u32,
  _padding: u32,
}

fn divCeil(a: u32, b: u32) -> u32 {
  return (a + b - 1u) / b;
}

@group(0) @binding(0) var<uniform> sortParams: SortParams;
@group(0) @binding(1) var<storage, read> indirectBuffer: array<u32>;
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> globalHistogram: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> partitionHistogram: array<u32>;

var<workgroup> localHistogram: array<atomic<u32>, RADIX_SIZE>;

@compute @workgroup_size(256, 1, 1)
fn upsweep(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let numKeys = indirectBuffer[1];
  let numPartitions = divCeil(numKeys, BLOCK_SIZE);
  let partitionId = workgroupId.x;
  
  if partitionId >= numPartitions { return; }
  
  let tid = localId.x;
  let partitionStart = partitionId * BLOCK_SIZE;
  let shift = sortParams.bitShift;
  let passIdx = sortParams.passIndex;
  
  // 初始化局部直方图
  if tid < RADIX_SIZE {
    atomicStore(&localHistogram[tid], 0u);
  }
  workgroupBarrier();
  
  // 构建局部直方图
  for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
    let keyIdx = partitionStart + tid * ELEMENTS_PER_THREAD + i;
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
`;

const spineShader = /* wgsl */ `
const RADIX_SIZE: u32 = 256u;
const BLOCK_SIZE: u32 = 1024u;

struct SortParams {
  maxElementCount: u32,
  bitShift: u32,
  passIndex: u32,
  _padding: u32,
}

fn divCeil(a: u32, b: u32) -> u32 {
  return (a + b - 1u) / b;
}

@group(0) @binding(0) var<storage, read> indirectBuffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> globalHistogram: array<u32>;
@group(0) @binding(2) var<storage, read_write> partitionHistogram: array<u32>;
@group(0) @binding(3) var<uniform> sortParams: SortParams;

var<workgroup> scanA: array<u32, 256>;
var<workgroup> scanB: array<u32, 256>;
var<workgroup> reductionShared: u32;

@compute @workgroup_size(256, 1, 1)
fn spine(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let numKeys = indirectBuffer[1];
  let numPartitions = divCeil(numKeys, BLOCK_SIZE);
  let bin = workgroupId.x;
  let tid = localId.x;
  
  if bin >= RADIX_SIZE { return; }
  
  // 初始化共享归约
  if tid == 0u {
    reductionShared = 0u;
  }
  workgroupBarrier();
  
  // 批量处理所有分区
  let MAX_BATCH_SIZE = 256u;
  for (var batchStart = 0u; batchStart < numPartitions; batchStart += MAX_BATCH_SIZE) {
    let partitionIdx = batchStart + tid;
    let batchSize = min(MAX_BATCH_SIZE, numPartitions - batchStart);
    
    // 加载批次值
    if tid < batchSize && partitionIdx < numPartitions {
      scanA[tid] = partitionHistogram[RADIX_SIZE * partitionIdx + bin];
    } else {
      scanA[tid] = 0u;
    }
    workgroupBarrier();
    
    // Hillis-Steele 包含前缀和（双缓冲避免数据竞争）
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
    
    // 写回为排他前缀和
    if tid < batchSize && partitionIdx < numPartitions {
      var exclusive = reductionShared;
      if tid > 0u {
        exclusive += scanA[tid - 1u];
      }
      partitionHistogram[RADIX_SIZE * partitionIdx + bin] = exclusive;
    }
    
    // 更新归约值
    workgroupBarrier();
    if tid == 0u && batchSize > 0u {
      reductionShared += scanA[batchSize - 1u];
    }
    workgroupBarrier();
  }
  
  // Bin 0 工作组也做全局直方图前缀和
  if bin == 0u {
    let passIdx = sortParams.passIndex;
    scanA[tid] = globalHistogram[RADIX_SIZE * passIdx + tid];
    workgroupBarrier();
    
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
    
    var exclusive = 0u;
    if tid > 0u {
      exclusive = scanA[tid - 1u];
    }
    globalHistogram[RADIX_SIZE * passIdx + tid] = exclusive;
  }
}
`;

const downsweepShader = /* wgsl */ `
const RADIX_SIZE: u32 = 256u;
const RADIX_MASK: u32 = 255u;
const ELEMENTS_PER_THREAD: u32 = 4u;
const BLOCK_SIZE: u32 = 1024u;

struct SortParams {
  maxElementCount: u32,
  bitShift: u32,
  passIndex: u32,
  _padding: u32,
}

fn divCeil(a: u32, b: u32) -> u32 {
  return (a + b - 1u) / b;
}

@group(0) @binding(0) var<uniform> sortParams: SortParams;
@group(0) @binding(1) var<storage, read> indirectBuffer: array<u32>;
@group(0) @binding(2) var<storage, read> globalHistogram: array<u32>;
@group(0) @binding(3) var<storage, read> partitionHistogram: array<u32>;
@group(0) @binding(4) var<storage, read> keysIn: array<u32>;
@group(0) @binding(5) var<storage, read> valuesIn: array<u32>;
@group(0) @binding(6) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(7) var<storage, read_write> valuesOut: array<u32>;

var<workgroup> localKeys: array<u32, BLOCK_SIZE>;
var<workgroup> localValues: array<u32, BLOCK_SIZE>;
var<workgroup> localBins: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(256, 1, 1)
fn downsweep(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let numKeys = indirectBuffer[1];
  let numPartitions = divCeil(numKeys, BLOCK_SIZE);
  let partitionId = workgroupId.x;
  
  if partitionId >= numPartitions { return; }
  
  let tid = localId.x;
  let partitionStart = partitionId * BLOCK_SIZE;
  let shift = sortParams.bitShift;
  
  // 加载元素到共享内存
  for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
    let keyIdx = partitionStart + tid * ELEMENTS_PER_THREAD + i;
    let localIdx = tid * ELEMENTS_PER_THREAD + i;
    
    if keyIdx < numKeys {
      let key = keysIn[keyIdx];
      localKeys[localIdx] = key;
      localValues[localIdx] = valuesIn[keyIdx];
      localBins[localIdx] = (key >> shift) & RADIX_MASK;
    } else {
      localBins[localIdx] = 0xFFFFFFFFu;
    }
  }
  
  workgroupBarrier();
  
  // 线程 0 顺序散射以保持稳定性
  if tid == 0u {
    var binWritePos: array<u32, RADIX_SIZE>;
    let passIdx = sortParams.passIndex;
    
    // 从全局 + 分区偏移初始化写入位置
    for (var bin = 0u; bin < RADIX_SIZE; bin++) {
      binWritePos[bin] = globalHistogram[RADIX_SIZE * passIdx + bin] + 
                         partitionHistogram[RADIX_SIZE * partitionId + bin];
    }
    
    // 按输入顺序顺序写入（稳定）
    let partitionEnd = min(partitionStart + BLOCK_SIZE, numKeys);
    for (var i = 0u; i < BLOCK_SIZE; i++) {
      let keyIdx = partitionStart + i;
      if keyIdx < partitionEnd {
        let bin = localBins[i];
        if bin != 0xFFFFFFFFu {
          let writePos = binWritePos[bin];
          if writePos < numKeys {
            keysOut[writePos] = localKeys[i];
            valuesOut[writePos] = localValues[i];
            binWritePos[bin]++;
          }
        }
      }
    }
  }
}
`;

export interface CullingOptions {
  nearPlane: number;
  farPlane: number;
  pixelThreshold: number;
  frustumDilation?: number;
}

export interface ScreenInfo {
  width: number;
  height: number;
}

/**
 * RadixSorter - GPU Radix Sort 深度排序器
 */
export class RadixSorter {
  private device: GPUDevice;
  private maxSplatCount: number;

  // Buffers
  private cullParamsBuffer: GPUBuffer;
  private depthKeysBuffer: GPUBuffer;
  private depthKeysTempBuffer: GPUBuffer;
  private visibleIndicesBuffer: GPUBuffer;
  private visibleIndicesTempBuffer: GPUBuffer;
  private sortedIndicesBuffer: GPUBuffer;
  private indirectBuffer: GPUBuffer;
  private globalHistogramBuffer: GPUBuffer;
  private partitionHistogramBuffer: GPUBuffer;
  private sortParamsBuffer: GPUBuffer;

  // Pipelines
  private resetIndirectPipeline: GPUComputePipeline;
  private projectCullPipeline: GPUComputePipeline;
  private upsweepPipeline: GPUComputePipeline;
  private spinePipeline: GPUComputePipeline;
  private downsweepPipeline: GPUComputePipeline;

  // Bind Groups
  private cullBindGroup: GPUBindGroup;
  private upsweepBindGroups: GPUBindGroup[] = [];
  private spineBindGroups: GPUBindGroup[] = [];
  private downsweepBindGroups: GPUBindGroup[] = [];

  // State
  private screenWidth: number = 1920;
  private screenHeight: number = 1080;
  private cullingOptions: CullingOptions = {
    nearPlane: 0.1,
    farPlane: 1000,
    pixelThreshold: 1.0,
    frustumDilation: 0.2,
  };

  constructor(
    device: GPUDevice,
    maxSplatCount: number,
    splatBuffer: GPUBuffer,
    cameraBuffer: GPUBuffer,
  ) {
    this.device = device;
    this.maxSplatCount = maxSplatCount;

    const numPartitions = Math.ceil(maxSplatCount / BLOCK_SIZE);

    // 创建 Buffers
    this.cullParamsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.depthKeysBuffer = device.createBuffer({
      size: maxSplatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.depthKeysTempBuffer = device.createBuffer({
      size: maxSplatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.visibleIndicesBuffer = device.createBuffer({
      size: maxSplatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.visibleIndicesTempBuffer = device.createBuffer({
      size: maxSplatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.sortedIndicesBuffer = device.createBuffer({
      size: maxSplatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.indirectBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.globalHistogramBuffer = device.createBuffer({
      size: RADIX_SIZE * 4 * 4, // 4 passes
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.partitionHistogramBuffer = device.createBuffer({
      size: RADIX_SIZE * numPartitions * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.sortParamsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 创建 Shader Modules
    const projectCullModule = device.createShaderModule({ code: projectCullShader });
    const upsweepModule = device.createShaderModule({ code: upsweepShader });
    const spineModule = device.createShaderModule({ code: spineShader });
    const downsweepModule = device.createShaderModule({ code: downsweepShader });

    // 创建 Bind Group Layouts
    const cullBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const upsweepBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const spineBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    const downsweepBindGroupLayout = device.createBindGroupLayout({
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

    // 创建 Pipelines
    this.resetIndirectPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [cullBindGroupLayout] }),
      compute: { module: projectCullModule, entryPoint: "resetIndirect" },
    });

    this.projectCullPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [cullBindGroupLayout] }),
      compute: { module: projectCullModule, entryPoint: "projectAndCull" },
    });

    this.upsweepPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [upsweepBindGroupLayout] }),
      compute: { module: upsweepModule, entryPoint: "upsweep" },
    });

    this.spinePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [spineBindGroupLayout] }),
      compute: { module: spineModule, entryPoint: "spine" },
    });

    this.downsweepPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [downsweepBindGroupLayout] }),
      compute: { module: downsweepModule, entryPoint: "downsweep" },
    });

    // 创建 Bind Groups
    this.cullBindGroup = device.createBindGroup({
      layout: cullBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: this.cullParamsBuffer } },
        { binding: 2, resource: { buffer: splatBuffer } },
        { binding: 3, resource: { buffer: this.depthKeysBuffer } },
        { binding: 4, resource: { buffer: this.visibleIndicesBuffer } },
        { binding: 5, resource: { buffer: this.indirectBuffer } },
      ],
    });

    // 为 4 个 pass 创建 bind groups (ping-pong buffers)
    for (let pass = 0; pass < 4; pass++) {
      const isEvenPass = pass % 2 === 0;
      const keysIn = isEvenPass ? this.depthKeysBuffer : this.depthKeysTempBuffer;
      const keysOut = isEvenPass ? this.depthKeysTempBuffer : this.depthKeysBuffer;
      const valuesIn = isEvenPass ? this.visibleIndicesBuffer : this.visibleIndicesTempBuffer;
      const valuesOut = isEvenPass ? this.visibleIndicesTempBuffer : this.visibleIndicesBuffer;

      this.upsweepBindGroups.push(device.createBindGroup({
        layout: upsweepBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.sortParamsBuffer } },
          { binding: 1, resource: { buffer: this.indirectBuffer } },
          { binding: 2, resource: { buffer: keysIn } },
          { binding: 3, resource: { buffer: this.globalHistogramBuffer } },
          { binding: 4, resource: { buffer: this.partitionHistogramBuffer } },
        ],
      }));

      this.spineBindGroups.push(device.createBindGroup({
        layout: spineBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.indirectBuffer } },
          { binding: 1, resource: { buffer: this.globalHistogramBuffer } },
          { binding: 2, resource: { buffer: this.partitionHistogramBuffer } },
          { binding: 3, resource: { buffer: this.sortParamsBuffer } },
        ],
      }));

      this.downsweepBindGroups.push(device.createBindGroup({
        layout: downsweepBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.sortParamsBuffer } },
          { binding: 1, resource: { buffer: this.indirectBuffer } },
          { binding: 2, resource: { buffer: this.globalHistogramBuffer } },
          { binding: 3, resource: { buffer: this.partitionHistogramBuffer } },
          { binding: 4, resource: { buffer: keysIn } },
          { binding: 5, resource: { buffer: valuesIn } },
          { binding: 6, resource: { buffer: keysOut } },
          { binding: 7, resource: { buffer: valuesOut } },
        ],
      }));
    }
  }

  setScreenSize(width: number, height: number): void {
    this.screenWidth = width;
    this.screenHeight = height;
  }

  setCullingOptions(options: Partial<CullingOptions>): void {
    this.cullingOptions = { ...this.cullingOptions, ...options };
  }

  sort(): void {
    const numPartitions = Math.ceil(this.maxSplatCount / BLOCK_SIZE);
    const cullWorkgroups = Math.ceil(this.maxSplatCount / WORKGROUP_SIZE);

    // 更新剔除参数
    const cullParamsData = new Float32Array([
      this.maxSplatCount,
      this.cullingOptions.nearPlane,
      this.cullingOptions.farPlane,
      this.screenWidth,
      this.screenHeight,
      this.cullingOptions.pixelThreshold,
      this.cullingOptions.frustumDilation ?? 0.2,
      0, // padding
    ]);
    // 修正: splatCount 是 u32，需要用 Uint32Array
    const cullParamsU32 = new Uint32Array(cullParamsData.buffer);
    cullParamsU32[0] = this.maxSplatCount;
    this.device.queue.writeBuffer(this.cullParamsBuffer, 0, cullParamsData);

    // 清零全局直方图
    this.device.queue.writeBuffer(
      this.globalHistogramBuffer,
      0,
      new Uint32Array(RADIX_SIZE * 4),
    );

    const encoder = this.device.createCommandEncoder();

    // Pass 0: Reset indirect buffer
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.resetIndirectPipeline);
      pass.setBindGroup(0, this.cullBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
    }

    // Pass 1: Project and Cull
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.projectCullPipeline);
      pass.setBindGroup(0, this.cullBindGroup);
      pass.dispatchWorkgroups(cullWorkgroups);
      pass.end();
    }

    // Radix Sort: 4 passes (32-bit key, 8-bit per pass)
    for (let passIdx = 0; passIdx < 4; passIdx++) {
      const bitShift = passIdx * 8;

      // 更新排序参数
      const sortParamsData = new Uint32Array([
        this.maxSplatCount,
        bitShift,
        passIdx,
        0, // padding
      ]);
      this.device.queue.writeBuffer(this.sortParamsBuffer, 0, sortParamsData);

      // Upsweep
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.upsweepPipeline);
        pass.setBindGroup(0, this.upsweepBindGroups[passIdx]);
        pass.dispatchWorkgroups(numPartitions);
        pass.end();
      }

      // Spine
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.spinePipeline);
        pass.setBindGroup(0, this.spineBindGroups[passIdx]);
        pass.dispatchWorkgroups(RADIX_SIZE);
        pass.end();
      }

      // Downsweep
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.downsweepPipeline);
        pass.setBindGroup(0, this.downsweepBindGroups[passIdx]);
        pass.dispatchWorkgroups(numPartitions);
        pass.end();
      }
    }

    // 最终结果在 depthKeysTempBuffer 和 visibleIndicesTempBuffer (4 passes 后)
    // 复制到 sortedIndicesBuffer
    encoder.copyBufferToBuffer(
      this.visibleIndicesBuffer, 0,
      this.sortedIndicesBuffer, 0,
      this.maxSplatCount * 4,
    );

    this.device.queue.submit([encoder.finish()]);
  }

  getIndicesBuffer(): GPUBuffer {
    return this.sortedIndicesBuffer;
  }

  getDrawIndirectBuffer(): GPUBuffer {
    return this.indirectBuffer;
  }

  getSplatCount(): number {
    return this.maxSplatCount;
  }

  destroy(): void {
    this.cullParamsBuffer.destroy();
    this.depthKeysBuffer.destroy();
    this.depthKeysTempBuffer.destroy();
    this.visibleIndicesBuffer.destroy();
    this.visibleIndicesTempBuffer.destroy();
    this.sortedIndicesBuffer.destroy();
    this.indirectBuffer.destroy();
    this.globalHistogramBuffer.destroy();
    this.partitionHistogramBuffer.destroy();
    this.sortParamsBuffer.destroy();
  }
}
