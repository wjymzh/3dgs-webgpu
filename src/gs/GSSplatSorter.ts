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
 * 
 * iOS 兼容性：
 * - iOS WebGPU 对大型原子数组支持有限
 * - 默认使用 65536 个桶（桌面/Android）
 * - iOS 使用 4096 个桶（减少原子操作压力）
 */

// 默认配置
const DEFAULT_NUM_BUCKETS = 65536;
const IOS_NUM_BUCKETS = 4096; // iOS 使用更少的桶
const WORKGROUP_SIZE = 256;

/**
 * 检测是否为 iOS 设备
 */
function isIOSDevice(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent || '';
  return /iphone|ipad|ipod/i.test(ua.toLowerCase()) || 
         (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1); // iPad 伪装为 Mac
}

/**
 * 生成 Culling Shader 代码（支持可配置桶数量）
 */
function generateCullingShaderCode(numBuckets: number): string {
  const bucketMask = numBuckets - 1; // 用于位运算
  const bucketBits = Math.log2(numBuckets); // 桶 ID 占用的位数
  const depthShift = 32 - bucketBits; // 深度值移位量
  
  return /* wgsl */ `
/**
 * Pass 1: 剔除 + 深度计算 + 桶计数
 * 桶数量: ${numBuckets}
 */

const NUM_BUCKETS: u32 = ${numBuckets}u;
const BUCKET_MAX: u32 = ${numBuckets - 1}u;

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

// 从模型矩阵提取最大缩放因子
fn getModelMaxScale(model: mat4x4<f32>) -> f32 {
  let sx = length(model[0].xyz);
  let sy = length(model[1].xyz);
  let sz = length(model[2].xyz);
  return max(max(sx, sy), sz);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn cullAndCount(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.splatCount) {
    return;
  }
  
  let splat = splats[i];
  // 先应用模型矩阵变换到世界空间，再变换到视图空间
  let worldPos = camera.model * vec4<f32>(splat.mean, 1.0);
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
  let depthRange = params.farPlane - params.nearPlane;
  let normalizedDepth = clamp((z - params.nearPlane) / depthRange, 0.0, 1.0);
  // 反转：让远处的桶 ID 更小，这样 counting sort 后远处的在前面
  let depthBucket = BUCKET_MAX - u32(normalizedDepth * f32(BUCKET_MAX));
  // 复合 key: 深度桶 + 原始索引低位作为 tie-breaker
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
 * 生成前缀和 Shader 代码
 */
function generatePrefixSumShaderCode(numBuckets: number): string {
  return /* wgsl */ `
/**
 * Pass 2: 串行前缀和
 * ${numBuckets} 个桶
 */

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
 * 生成散射 Shader 代码
 */
function generateScatterShaderCode(numBuckets: number): string {
  const bucketBits = Math.log2(numBuckets);
  const depthShift = 32 - bucketBits;
  
  return /* wgsl */ `
/**
 * Pass 3: 散射到最终排序位置
 * 桶数量: ${numBuckets}
 */

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
  let originalIndex = visibleIndices[i];
  
  // 从复合 key 提取深度桶
  let depthBucket = depthKey >> ${depthShift}u;
  
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
 * 排序器配置选项
 */
export interface SorterOptions {
  /** 深度桶数量（必须是 2 的幂次），默认根据平台自动选择 */
  numBuckets?: number;
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

  // 工作组大小和桶数量
  private readonly WORKGROUP_SIZE = WORKGROUP_SIZE;
  private readonly numBuckets: number;

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
    options: SorterOptions = {},
  ) {
    this.device = device;
    this.splatCount = splatCount;
    
    // 根据平台选择桶数量
    // iOS 使用较少的桶以避免原子操作问题
    const isIOS = isIOSDevice();
    this.numBuckets = options.numBuckets ?? (isIOS ? IOS_NUM_BUCKETS : DEFAULT_NUM_BUCKETS);
    
    console.log(`GSSplatSorter: 平台=${isIOS ? 'iOS' : '其他'}, 桶数量=${this.numBuckets}`);

    // ============================================
    // 创建 Shader 模块（使用动态生成的代码）
    // 添加编译错误检查
    // ============================================
    const cullingCode = generateCullingShaderCode(this.numBuckets);
    const prefixSumCode = generatePrefixSumShaderCode(this.numBuckets);
    const scatterCode = generateScatterShaderCode(this.numBuckets);
    
    // 调试：在移动端输出 shader 代码用于诊断
    if (isIOS) {
      console.log('GSSplatSorter: iOS Culling Shader (前 500 字符):', cullingCode.substring(0, 500));
    }
    
    const cullingModule = device.createShaderModule({
      code: cullingCode,
      label: 'culling-shader',
    });

    const prefixSumModule = device.createShaderModule({
      code: prefixSumCode,
      label: 'prefix-sum-shader',
    });

    const scatterModule = device.createShaderModule({
      code: scatterCode,
      label: 'scatter-shader',
    });
    
    // 检查 shader 编译错误
    cullingModule.getCompilationInfo().then(info => {
      if (info.messages.length > 0) {
        console.warn('GSSplatSorter: Culling shader 编译信息:', info.messages);
      }
    });
    prefixSumModule.getCompilationInfo().then(info => {
      if (info.messages.length > 0) {
        console.warn('GSSplatSorter: PrefixSum shader 编译信息:', info.messages);
      }
    });
    scatterModule.getCompilationInfo().then(info => {
      if (info.messages.length > 0) {
        console.warn('GSSplatSorter: Scatter shader 编译信息:', info.messages);
      }
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
      `GSSplatSorter: 初始化完成 (GPU Counting Sort), splatCount=${splatCount}, numBuckets=${this.numBuckets}`,
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
   * 
   * 优化：合并所有 compute pass 到单次 GPU 提交
   * WebGPU 保证同一 command buffer 中的命令按顺序执行
   */
  sort(): void {
    try {
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

    const cullWorkgroupCount = Math.ceil(this.splatCount / this.WORKGROUP_SIZE);
    const bucketResetWorkgroups = Math.ceil(
      this.numBuckets / this.WORKGROUP_SIZE,
    );

    // ============================================
    // 单次提交所有 Compute Pass
    // WebGPU 保证同一 encoder 中的 pass 按顺序执行
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

    // 单次提交
    this.device.queue.submit([encoder.finish()]);
    } catch (error) {
      console.error('GSSplatSorter.sort() 错误:', error);
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
