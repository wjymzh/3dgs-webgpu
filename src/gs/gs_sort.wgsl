/**
 * GPU Splat 深度排序 Compute Shaders
 * - computeDepths: 计算每个 splat 的视图空间深度
 * - bitonicSort: Bitonic 排序步骤
 */

// Splat 数据结构 (256 bytes, 与渲染 shader 一致)
struct Splat {
  mean:     vec3<f32>,       // offset 0
  _pad0:    f32,             // offset 12
  scale:    vec3<f32>,       // offset 16
  _pad1:    f32,             // offset 28
  rotation: vec4<f32>,       // offset 32
  colorDC:  vec3<f32>,       // offset 48
  opacity:  f32,             // offset 60
  sh1:      array<f32, 9>,   // offset 64 (L1 SH 系数)
  sh2:      array<f32, 15>,  // offset 100 (L2 SH 系数)
  sh3:      array<f32, 21>,  // offset 160 (L3 SH 系数)
  _pad2:    array<f32, 3>,   // offset 244 (padding to 256)
}

// 相机 uniform (144 bytes, 与渲染 shader 一致)
struct CameraUniforms {
  view: mat4x4<f32>,        // 64 bytes
  proj: mat4x4<f32>,        // 64 bytes
  cameraPos: vec3<f32>,     // 12 bytes
  _pad: f32,                // 4 bytes
}

// 排序参数 uniform
struct SortParams {
  splatCount: u32,
  k: u32,
  j: u32,
  _pad: u32,
}

// ============================================
// computeDepths Shader Bindings
// ============================================
@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<storage, read_write> depths: array<f32>;
@group(0) @binding(3) var<storage, read_write> indices: array<u32>;
@group(0) @binding(4) var<uniform> params: SortParams;

/**
 * 计算深度并初始化索引
 * depths[i] = viewPos.z (负值，越小表示越远)
 * indices[i] = i
 */
@compute @workgroup_size(256)
fn computeDepths(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.splatCount) {
    return;
  }

  // 获取 splat 的世界坐标
  let mean = splats[i].mean;
  
  // 变换到视图空间
  let viewPos = camera.view * vec4<f32>(mean, 1.0);
  
  // 存储 z 值 (负值，越远越小)
  // 我们要从远到近排序，即 depths 降序排列
  depths[i] = viewPos.z;
  
  // 初始化索引
  indices[i] = i;
}

/**
 * Bitonic Sort 单步
 * 每次调用处理一个 (k, j) 对
 */
@compute @workgroup_size(256)
fn bitonicSort(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.splatCount) {
    return;
  }

  let k = params.k;
  let j = params.j;
  
  // 计算配对的索引
  let ixj = i ^ j;
  
  // 只处理 i < ixj 的情况，避免重复交换
  if (ixj > i && ixj < params.splatCount) {
    let depthI = depths[i];
    let depthIxj = depths[ixj];
    
    // 确定排序方向
    // (i & k) == 0 时升序，否则降序
    // 但我们要从远到近（降序），所以逻辑取反
    let ascending = (i & k) == 0u;
    
    // 比较并决定是否交换
    // 降序排列：远的(小的z)在前，近的(大的z)在后
    // 因为 viewPos.z 是负值，越远越小
    var shouldSwap = false;
    if (ascending) {
      // 升序块：如果 depths[i] > depths[ixj]，需要交换
      shouldSwap = depthI > depthIxj;
    } else {
      // 降序块：如果 depths[i] < depths[ixj]，需要交换
      shouldSwap = depthI < depthIxj;
    }
    
    if (shouldSwap) {
      // 交换深度值
      depths[i] = depthIxj;
      depths[ixj] = depthI;
      
      // 交换索引
      let indexI = indices[i];
      let indexIxj = indices[ixj];
      indices[i] = indexIxj;
      indices[ixj] = indexI;
    }
  }
}
