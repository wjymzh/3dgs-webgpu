/**
 * GSSplatSorter - GPU Counting Sort 深度排序器
 *
 * 基于 PlayCanvas 的排序算法实现，使用 Counting Sort
 * 时间复杂度 O(n)，远比 Bitonic Sort O(n log²n) 更快
 *
 * 流程：
 * 1. 剔除 + 计算深度 + 统计计数
 * 2. 前缀和计算偏移
 * 3. 散射到最终位置
 */

// 深度桶数量 (2^16 = 65536)
// 平衡精度和性能
const NUM_BUCKETS = 65536;
const WORKGROUP_SIZE = 256;
const PREFIX_SUM_WORKGROUP_SIZE = 256; // 前缀和每个 workgroup 处理 256 个桶

// ============================================
// 剔除 + 深度计算 + 计数 Compute Shader
// ============================================
const cullingCountShaderCode = /* wgsl */ `
/**
 * Pass 1: 剔除 + 深度计算 + 桶计数
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

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<uniform> params: CullingParams;
@group(0) @binding(3) var<storage, read_write> counters: Counters;
@group(0) @binding(4) var<storage, read_write> visibleIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(6) var<storage, read_write> bucketCounts: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> drawIndirect: array<u32>;

fn maxScale(scale: vec3<f32>) -> f32 {
  return max(max(scale.x, scale.y), scale.z);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn cullAndCount(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.splatCount) {
    return;
  }
  
  let splat = splats[i];
  let viewPos = camera.view * vec4<f32>(splat.mean, 1.0);
  let z = -viewPos.z;
  
  // 近平面剔除
  if (z < params.nearPlane) {
    return;
  }
  
  // 远平面剔除
  if (z > params.farPlane) {
    return;
  }
  
  // 视锥剔除
  let fx = camera.proj[0][0];
  let fy = camera.proj[1][1];
  let x_ndc = viewPos.x * fx / z;
  let y_ndc = viewPos.y * fy / z;
  let worldRadius = maxScale(splat.scale) * 3.0;
  let r_ndc = worldRadius * max(fx, fy) / z;
  
  if (x_ndc < -1.0 - r_ndc || x_ndc > 1.0 + r_ndc) {
    return;
  }
  if (y_ndc < -1.0 - r_ndc || y_ndc > 1.0 + r_ndc) {
    return;
  }
  
  // 屏幕尺寸剔除
  let screenRadiusX = r_ndc * params.screenWidth * 0.5;
  let screenRadiusY = r_ndc * params.screenHeight * 0.5;
  let screenRadius = max(screenRadiusX, screenRadiusY);
  
  if (screenRadius < params.pixelThreshold) {
    return;
  }
  
  // 透明度剔除
  if (splat.opacity < 0.004) {
    return;
  }
  
  // 通过剔除，计算深度桶
  // 使用复合 key: 高16位深度 + 低16位原始索引，确保稳定排序
  let depthRange = params.farPlane - params.nearPlane;
  let normalizedDepth = clamp((z - params.nearPlane) / depthRange, 0.0, 1.0);
  // 反转：让远处的桶 ID 更小，这样 counting sort 后远处的在前面
  let depthBucket = 65535u - u32(normalizedDepth * 65535.0);
  // 复合 key: 深度桶 + 原始索引低16位作为 tie-breaker
  let depthKey = (depthBucket << 16u) | (i & 0xFFFFu);
  
  // 分配可见索引位置
  let visibleIdx = atomicAdd(&counters.visibleCount, 1u);
  visibleIndices[visibleIdx] = i;
  depthKeys[visibleIdx] = depthKey;
  
  // 统计桶计数（只用高16位深度桶）
  atomicAdd(&bucketCounts[depthBucket], 1u);
}

@compute @workgroup_size(1)
fn resetCounters() {
  atomicStore(&counters.visibleCount, 0u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn resetBucketCounts(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < 65536u) {
    atomicStore(&bucketCounts[i], 0u);
  }
}

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
// 前缀和 Compute Shader (简单串行版本)
// ============================================
const prefixSumShaderCode = /* wgsl */ `
/**
 * Pass 2: 串行前缀和
 * 65536 个桶，单线程循环足够快
 */

const NUM_BUCKETS: u32 = 65536u;

@group(0) @binding(0) var<storage, read_write> bucketCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> bucketOffsets: array<u32>;

@compute @workgroup_size(1)
fn prefixSum() {
  var sum = 0u;
  for (var i = 0u; i < NUM_BUCKETS; i++) {
    bucketOffsets[i] = sum;
    sum += bucketCounts[i];
  }
}
`;

// ============================================
// 散射 Compute Shader (稳定排序版本)
// ============================================
const scatterShaderCode = /* wgsl */ `
/**
 * Pass 3: 散射到最终排序位置
 * 
 * depthKey 格式: 高16位是深度桶，低16位是原始索引
 * 使用深度桶查找偏移，原始索引确保桶内稳定排序
 */

const NUM_BUCKETS: u32 = 65536u;

struct Counters {
  visibleCount: u32,
}

@group(0) @binding(0) var<storage, read> visibleIndices: array<u32>;
@group(0) @binding(1) var<storage, read> depthKeys: array<u32>;
@group(0) @binding(2) var<storage, read> bucketOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> bucketPositions: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(5) var<storage, read> counters: Counters;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  // 使用 GPU 上计算的 visibleCount
  if (i >= counters.visibleCount) {
    return;
  }
  
  let depthKey = depthKeys[i];
  let originalIndex = visibleIndices[i];
  
  // 从复合 key 提取深度桶 (高16位)
  let depthBucket = depthKey >> 16u;
  
  // 在桶内分配位置
  let bucketOffset = bucketOffsets[depthBucket];
  let posInBucket = atomicAdd(&bucketPositions[depthBucket], 1u);
  let destIdx = bucketOffset + posInBucket;
  
  // 写入最终排序位置
  sortedIndices[destIdx] = originalIndex;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn resetBucketPositions(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < NUM_BUCKETS) {
    atomicStore(&bucketPositions[i], 0u);
  }
}
`;

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
 * GSSplatSorter - GPU Counting Sort 排序器
 */
export class GSSplatSorter {
  private device: GPUDevice;
  private splatCount: number;

  // ============================================
  // Buffers
  // ============================================
  private cullingParamsBuffer: GPUBuffer;
  private countersBuffer: GPUBuffer;
  private visibleIndicesBuffer: GPUBuffer;
  private depthKeysBuffer: GPUBuffer;
  private bucketCountsBuffer: GPUBuffer;
  private bucketOffsetsBuffer: GPUBuffer;
  private bucketPositionsBuffer: GPUBuffer;
  private sortedIndicesBuffer: GPUBuffer;
  private drawIndirectBuffer: GPUBuffer;

  // ============================================
  // Pipelines
  // ============================================
  private resetCountersPipeline: GPUComputePipeline;
  private resetBucketCountsPipeline: GPUComputePipeline;
  private cullAndCountPipeline: GPUComputePipeline;
  private updateDrawIndirectPipeline: GPUComputePipeline;
  private prefixSumPipeline: GPUComputePipeline;
  private resetBucketPositionsPipeline: GPUComputePipeline;
  private scatterPipeline: GPUComputePipeline;

  // ============================================
  // Bind Groups
  // ============================================
  private cullingBindGroupLayout: GPUBindGroupLayout;
  private cullingBindGroup: GPUBindGroup;
  private prefixSumBindGroupLayout: GPUBindGroupLayout;
  private prefixSumBindGroup: GPUBindGroup;
  private scatterBindGroupLayout: GPUBindGroupLayout;
  private scatterBindGroup: GPUBindGroup;

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
      code: cullingCountShaderCode,
    });

    const prefixSumModule = device.createShaderModule({
      code: prefixSumShaderCode,
    });

    const scatterModule = device.createShaderModule({
      code: scatterShaderCode,
    });

    // ============================================
    // 创建 Buffers
    // ============================================

    this.cullingParamsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.countersBuffer = device.createBuffer({
      size: 16,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    this.visibleIndicesBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    this.depthKeysBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    this.bucketCountsBuffer = device.createBuffer({
      size: NUM_BUCKETS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.bucketOffsetsBuffer = device.createBuffer({
      size: NUM_BUCKETS * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    this.bucketPositionsBuffer = device.createBuffer({
      size: NUM_BUCKETS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.sortedIndicesBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.drawIndirectBuffer = device.createBuffer({
      size: 16,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.COPY_DST,
    });

    // scatterParamsBuffer 不再需要，scatter 直接从 countersBuffer 读取 visibleCount

    // ============================================
    // 创建 Bind Group Layouts 和 Pipelines
    // ============================================

    // Culling bind group layout (8 bindings)
    this.cullingBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 7,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const cullingPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.cullingBindGroupLayout],
    });

    this.resetCountersPipeline = device.createComputePipeline({
      layout: cullingPipelineLayout,
      compute: { module: cullingModule, entryPoint: "resetCounters" },
    });

    this.resetBucketCountsPipeline = device.createComputePipeline({
      layout: cullingPipelineLayout,
      compute: { module: cullingModule, entryPoint: "resetBucketCounts" },
    });

    this.cullAndCountPipeline = device.createComputePipeline({
      layout: cullingPipelineLayout,
      compute: { module: cullingModule, entryPoint: "cullAndCount" },
    });

    this.updateDrawIndirectPipeline = device.createComputePipeline({
      layout: cullingPipelineLayout,
      compute: { module: cullingModule, entryPoint: "updateDrawIndirect" },
    });

    // Prefix sum bind group layout (2 bindings)
    this.prefixSumBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const prefixSumPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.prefixSumBindGroupLayout],
    });

    this.prefixSumPipeline = device.createComputePipeline({
      layout: prefixSumPipelineLayout,
      compute: { module: prefixSumModule, entryPoint: "prefixSum" },
    });

    // Scatter bind group layout (6 bindings)
    this.scatterBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        }, // countersBuffer
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
        { binding: 5, resource: { buffer: this.depthKeysBuffer } },
        { binding: 6, resource: { buffer: this.bucketCountsBuffer } },
        { binding: 7, resource: { buffer: this.drawIndirectBuffer } },
      ],
    });

    this.prefixSumBindGroup = device.createBindGroup({
      layout: this.prefixSumBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.bucketCountsBuffer } },
        { binding: 1, resource: { buffer: this.bucketOffsetsBuffer } },
      ],
    });

    this.scatterBindGroup = device.createBindGroup({
      layout: this.scatterBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.visibleIndicesBuffer } },
        { binding: 1, resource: { buffer: this.depthKeysBuffer } },
        { binding: 2, resource: { buffer: this.bucketOffsetsBuffer } },
        { binding: 3, resource: { buffer: this.bucketPositionsBuffer } },
        { binding: 4, resource: { buffer: this.sortedIndicesBuffer } },
        { binding: 5, resource: { buffer: this.countersBuffer } }, // 使用 GPU countersBuffer
      ],
    });

    console.log(
      `GSSplatSorter: 初始化完成 (GPU Counting Sort), splatCount=${splatCount}, numBuckets=${NUM_BUCKETS}`,
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
   * 执行剔除和排序
   * 每帧调用
   */
  sort(): void {
    // ============================================
    // 更新参数
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
    this.device.queue.writeBuffer(
      this.cullingParamsBuffer,
      0,
      cullingParamsData,
    );

    // scatter 直接从 GPU countersBuffer 读取 visibleCount，不需要 CPU 传入

    const cullWorkgroupCount = Math.ceil(this.splatCount / this.WORKGROUP_SIZE);
    const bucketResetWorkgroups = Math.ceil(
      this.NUM_BUCKETS / this.WORKGROUP_SIZE,
    );

    // ============================================
    // Pass 1: 重置 + 剔除 + 计数
    // ============================================
    {
      const encoder = this.device.createCommandEncoder();

      // 重置 visibleCount
      const resetPass = encoder.beginComputePass();
      resetPass.setPipeline(this.resetCountersPipeline);
      resetPass.setBindGroup(0, this.cullingBindGroup);
      resetPass.dispatchWorkgroups(1);
      resetPass.end();

      // 重置桶计数
      const resetBucketPass = encoder.beginComputePass();
      resetBucketPass.setPipeline(this.resetBucketCountsPipeline);
      resetBucketPass.setBindGroup(0, this.cullingBindGroup);
      resetBucketPass.dispatchWorkgroups(bucketResetWorkgroups);
      resetBucketPass.end();

      // 剔除 + 深度计算 + 桶计数
      const cullPass = encoder.beginComputePass();
      cullPass.setPipeline(this.cullAndCountPipeline);
      cullPass.setBindGroup(0, this.cullingBindGroup);
      cullPass.dispatchWorkgroups(cullWorkgroupCount);
      cullPass.end();

      // 更新 DrawIndirect
      const indirectPass = encoder.beginComputePass();
      indirectPass.setPipeline(this.updateDrawIndirectPipeline);
      indirectPass.setBindGroup(0, this.cullingBindGroup);
      indirectPass.dispatchWorkgroups(1);
      indirectPass.end();

      this.device.queue.submit([encoder.finish()]);
    }

    // ============================================
    // Pass 2: 前缀和 (简单串行版本)
    // ============================================
    {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.prefixSumPipeline);
      pass.setBindGroup(0, this.prefixSumBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }

    // ============================================
    // Pass 3: 重置桶位置 + 散射
    // ============================================
    {
      const encoder = this.device.createCommandEncoder();

      // 重置桶位置计数
      const resetPass = encoder.beginComputePass();
      resetPass.setPipeline(this.resetBucketPositionsPipeline);
      resetPass.setBindGroup(0, this.scatterBindGroup);
      resetPass.dispatchWorkgroups(bucketResetWorkgroups);
      resetPass.end();

      // 散射到最终位置
      const scatterPass = encoder.beginComputePass();
      scatterPass.setPipeline(this.scatterPipeline);
      scatterPass.setBindGroup(0, this.scatterBindGroup);
      scatterPass.dispatchWorkgroups(cullWorkgroupCount);
      scatterPass.end();

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
    this.depthKeysBuffer.destroy();
    this.bucketCountsBuffer.destroy();
    this.bucketOffsetsBuffer.destroy();
    this.bucketPositionsBuffer.destroy();
    this.sortedIndicesBuffer.destroy();
    this.drawIndirectBuffer.destroy();
  }
}
