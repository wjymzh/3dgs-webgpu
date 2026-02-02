/**
 * GSSplatSorterMobile - 移动端优化的 GPU 排序器
 *
 * 与主排序器的区别：
 * 1. 使用紧凑的位置数据（仅 xyz，无 scale/rotation）
 * 2. 简化的剔除逻辑（移除屏幕尺寸剔除）
 * 3. 针对 iOS 优化的桶数量
 */

// 默认配置
const DEFAULT_NUM_BUCKETS = 65536;
const IOS_NUM_BUCKETS = 4096;
const WORKGROUP_SIZE = 256;

/**
 * 排序器配置
 */
export interface SorterOptions {
  numBuckets?: number;
}

/**
 * 剔除选项
 */
export interface CullingOptions {
  nearPlane: number;
  farPlane: number;
  pixelThreshold: number;
}

/**
 * 检测是否为 iOS 设备
 */
function isIOSDevice(): boolean {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent || "";
  return (
    /iphone|ipad|ipod/i.test(ua.toLowerCase()) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1)
  );
}

/**
 * 生成剔除 Shader（简化版，仅使用位置数据）
 */
function generateCullingShaderCode(numBuckets: number): string {
  const bucketBits = Math.log2(numBuckets);

  return /* wgsl */ `
/**
 * Pass 1: 剔除 + 深度计算 + 桶计数（移动端简化版）
 * 仅使用位置数据，不包含 scale 剔除
 */

const NUM_BUCKETS: u32 = ${numBuckets}u;
const BUCKET_MAX: u32 = ${numBuckets - 1}u;

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

// 位置数据：紧密打包格式 [x0,y0,z0,x1,y1,z1,...]
// 注意：不使用 array<vec3<f32>> 因为 vec3 有 16 字节对齐要求
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<uniform> params: CullingParams;
@group(0) @binding(3) var<storage, read_write> counters: Counters;
@group(0) @binding(4) var<storage, read_write> visibleIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(6) var<storage, read_write> bucketCounts: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> drawIndirect: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn cullAndCount(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.splatCount) {
    return;
  }
  
  // 手动读取位置（避免 vec3 对齐问题）
  let base = i * 3u;
  let position = vec3<f32>(positions[base], positions[base + 1u], positions[base + 2u]);
  // 先应用模型矩阵变换到世界空间，再变换到视图空间
  let worldPos = camera.model * vec4<f32>(position, 1.0);
  let viewPos = camera.view * worldPos;
  let z = -viewPos.z;
  
  // 近平面剔除
  if (z < params.nearPlane) {
    return;
  }
  
  // 远平面剔除
  if (z > params.farPlane) {
    return;
  }
  
  // 视锥剔除（简化版，不考虑 splat 半径）
  let fx = camera.proj[0][0];
  let fy = camera.proj[1][1];
  let x_ndc = viewPos.x * fx / z;
  let y_ndc = viewPos.y * fy / z;
  
  // 放宽边界以避免边缘裁剪
  let margin: f32 = 0.5;
  if (x_ndc < -1.0 - margin || x_ndc > 1.0 + margin) {
    return;
  }
  if (y_ndc < -1.0 - margin || y_ndc > 1.0 + margin) {
    return;
  }
  
  // 通过剔除，计算深度桶
  let depthRange = params.farPlane - params.nearPlane;
  let normalizedDepth = clamp((z - params.nearPlane) / depthRange, 0.0, 1.0);
  let depthBucket = BUCKET_MAX - u32(normalizedDepth * f32(BUCKET_MAX));
  let depthKey = (depthBucket << ${32 - bucketBits}u) | (i & ${(1 << (32 - bucketBits)) - 1}u);
  
  // 分配可见索引位置
  let visibleIdx = atomicAdd(&counters.visibleCount, 1u);
  visibleIndices[visibleIdx] = i;
  depthKeys[visibleIdx] = depthKey;
  
  // 统计桶计数
  atomicAdd(&bucketCounts[depthBucket], 1u);
}

@compute @workgroup_size(1)
fn resetCounters() {
  atomicStore(&counters.visibleCount, 0u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn resetBucketCounts(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < NUM_BUCKETS) {
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
}

/**
 * 生成前缀和 Shader
 */
function generatePrefixSumShaderCode(numBuckets: number): string {
  return /* wgsl */ `
const NUM_BUCKETS: u32 = ${numBuckets}u;

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
}

/**
 * 生成散射 Shader
 */
function generateScatterShaderCode(numBuckets: number): string {
  const bucketBits = Math.log2(numBuckets);

  return /* wgsl */ `
const NUM_BUCKETS: u32 = ${numBuckets}u;

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
  if (i >= counters.visibleCount) {
    return;
  }
  
  let depthKey = depthKeys[i];
  let bucket = depthKey >> ${32 - bucketBits}u;
  let baseOffset = bucketOffsets[bucket];
  let localOffset = atomicAdd(&bucketPositions[bucket], 1u);
  let finalIdx = baseOffset + localOffset;
  
  sortedIndices[finalIdx] = visibleIndices[i];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn resetBucketPositions(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < NUM_BUCKETS) {
    atomicStore(&bucketPositions[i], 0u);
  }
}
`;
}

/**
 * GSSplatSorterMobile - 移动端排序器
 */
export class GSSplatSorterMobile {
  private device: GPUDevice;
  private splatCount: number;

  // Buffers
  private cullingParamsBuffer: GPUBuffer;
  private countersBuffer: GPUBuffer;
  private visibleIndicesBuffer: GPUBuffer;
  private depthKeysBuffer: GPUBuffer;
  private bucketCountsBuffer: GPUBuffer;
  private bucketOffsetsBuffer: GPUBuffer;
  private bucketPositionsBuffer: GPUBuffer;
  private sortedIndicesBuffer: GPUBuffer;
  private drawIndirectBuffer: GPUBuffer;

  // Pipelines
  private resetCountersPipeline: GPUComputePipeline;
  private resetBucketCountsPipeline: GPUComputePipeline;
  private cullAndCountPipeline: GPUComputePipeline;
  private updateDrawIndirectPipeline: GPUComputePipeline;
  private prefixSumPipeline: GPUComputePipeline;
  private resetBucketPositionsPipeline: GPUComputePipeline;
  private scatterPipeline: GPUComputePipeline;

  // Bind Groups
  private cullingBindGroupLayout: GPUBindGroupLayout;
  private cullingBindGroup: GPUBindGroup;
  private prefixSumBindGroupLayout: GPUBindGroupLayout;
  private prefixSumBindGroup: GPUBindGroup;
  private scatterBindGroupLayout: GPUBindGroupLayout;
  private scatterBindGroup: GPUBindGroup;

  private readonly WORKGROUP_SIZE = WORKGROUP_SIZE;
  private readonly numBuckets: number;

  // 屏幕信息
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
    positionsBuffer: GPUBuffer, // 紧凑位置数据
    cameraBuffer: GPUBuffer,
    options: SorterOptions = {}
  ) {
    this.device = device;
    this.splatCount = splatCount;

    const isIOS = isIOSDevice();
    this.numBuckets = options.numBuckets ?? (isIOS ? IOS_NUM_BUCKETS : DEFAULT_NUM_BUCKETS);

    // 创建 Shader 模块
    const cullingModule = device.createShaderModule({
      code: generateCullingShaderCode(this.numBuckets),
      label: "mobile-culling-shader",
    });

    const prefixSumModule = device.createShaderModule({
      code: generatePrefixSumShaderCode(this.numBuckets),
      label: "mobile-prefix-sum-shader",
    });

    const scatterModule = device.createShaderModule({
      code: generateScatterShaderCode(this.numBuckets),
      label: "mobile-scatter-shader",
    });

    // 创建 Buffers
    this.cullingParamsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.countersBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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
      size: this.numBuckets * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.bucketOffsetsBuffer = device.createBuffer({
      size: this.numBuckets * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    this.bucketPositionsBuffer = device.createBuffer({
      size: this.numBuckets * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.sortedIndicesBuffer = device.createBuffer({
      size: splatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.drawIndirectBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    // 创建 Bind Group Layouts 和 Pipelines
    // Culling bind group layout
    this.cullingBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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

    // Prefix sum bind group layout
    this.prefixSumBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const prefixSumPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.prefixSumBindGroupLayout],
    });

    this.prefixSumPipeline = device.createComputePipeline({
      layout: prefixSumPipelineLayout,
      compute: { module: prefixSumModule, entryPoint: "prefixSum" },
    });

    // Scatter bind group layout
    this.scatterBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const scatterPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.scatterBindGroupLayout],
    });

    this.scatterPipeline = device.createComputePipeline({
      layout: scatterPipelineLayout,
      compute: { module: scatterModule, entryPoint: "scatter" },
    });

    this.resetBucketPositionsPipeline = device.createComputePipeline({
      layout: scatterPipelineLayout,
      compute: { module: scatterModule, entryPoint: "resetBucketPositions" },
    });

    // 创建 Bind Groups
    this.cullingBindGroup = device.createBindGroup({
      layout: this.cullingBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionsBuffer } },
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
        { binding: 5, resource: { buffer: this.countersBuffer } },
      ],
    });
  }

  /**
   * 设置屏幕尺寸
   */
  setScreenSize(width: number, height: number): void {
    this.screenWidth = width;
    this.screenHeight = height;
  }

  /**
   * 设置剔除选项
   */
  setCullingOptions(options: Partial<CullingOptions>): void {
    this.cullingOptions = { ...this.cullingOptions, ...options };
  }

  /**
   * 执行排序
   */
  sort(): void {
    try {
      // 更新剔除参数
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

      const cullWorkgroupCount = Math.ceil(this.splatCount / this.WORKGROUP_SIZE);
      const bucketResetWorkgroups = Math.ceil(this.numBuckets / this.WORKGROUP_SIZE);

      const encoder = this.device.createCommandEncoder();

      // Pass 1: 重置计数器
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
        pass.setPipeline(this.resetBucketCountsPipeline);
        pass.setBindGroup(0, this.cullingBindGroup);
        pass.dispatchWorkgroups(bucketResetWorkgroups);
        pass.end();
      }

      // Pass 3: 剔除 + 深度计算 + 桶计数
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.cullAndCountPipeline);
        pass.setBindGroup(0, this.cullingBindGroup);
        pass.dispatchWorkgroups(cullWorkgroupCount);
        pass.end();
      }

      // Pass 4: 更新 DrawIndirect
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.updateDrawIndirectPipeline);
        pass.setBindGroup(0, this.cullingBindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
      }

      // Pass 5: 前缀和
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.prefixSumPipeline);
        pass.setBindGroup(0, this.prefixSumBindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
      }

      // Pass 6: 重置桶位置计数
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.resetBucketPositionsPipeline);
        pass.setBindGroup(0, this.scatterBindGroup);
        pass.dispatchWorkgroups(bucketResetWorkgroups);
        pass.end();
      }

      // Pass 7: 散射到最终位置
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.scatterPipeline);
        pass.setBindGroup(0, this.scatterBindGroup);
        pass.dispatchWorkgroups(cullWorkgroupCount);
        pass.end();
      }

      this.device.queue.submit([encoder.finish()]);
    } catch (error) {
      // 排序错误（静默处理）
    }
  }

  /**
   * 获取排序后的索引 buffer
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
   * 获取 splat 数量
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
