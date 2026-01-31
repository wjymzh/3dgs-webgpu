/**
 * GPU Splat 可见性剔除 Compute Shader
 * 功能：
 * 1. 近平面剔除 (Near Plane Culling)
 * 2. 视锥剔除 (Frustum Culling)
 * 3. 屏幕尺寸剔除 (Screen Size / Pixel Culling)
 * 
 * 输出：visibleIndices, visibleDepths, visibleCount
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

// 相机 uniform (扩展版本)
struct CameraUniforms {
  view: mat4x4<f32>,        // 64 bytes
  proj: mat4x4<f32>,        // 64 bytes
  model: mat4x4<f32>,       // 64 bytes
  cameraPos: vec3<f32>,     // 12 bytes
  _pad: f32,                // 4 bytes
}

// 剔除参数 uniform
struct CullingParams {
  splatCount: u32,          // 总 splat 数量
  nearPlane: f32,           // 近平面距离
  farPlane: f32,            // 远平面距离
  screenWidth: f32,         // 屏幕宽度
  screenHeight: f32,        // 屏幕高度
  pixelThreshold: f32,      // 像素剔除阈值 (默认 1.0)
  _pad0: f32,
  _pad1: f32,
}

// 原子计数器结构
struct AtomicCounters {
  visibleCount: atomic<u32>,
}

// Bindings
@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<uniform> params: CullingParams;
@group(0) @binding(3) var<storage, read_write> counters: AtomicCounters;
@group(0) @binding(4) var<storage, read_write> visibleIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> visibleDepths: array<f32>;

// DrawIndirect buffer 结构
// [vertexCount, instanceCount, firstVertex, firstInstance]
@group(0) @binding(6) var<storage, read_write> drawIndirect: array<u32>;

/**
 * 获取 splat 的最大缩放值 (用于计算包围球半径)
 */
fn maxScale(scale: vec3<f32>) -> f32 {
  return max(max(scale.x, scale.y), scale.z);
}

/**
 * 从模型矩阵提取最大缩放因子
 */
fn getModelMaxScale(model: mat4x4<f32>) -> f32 {
  let sx = length(model[0].xyz);
  let sy = length(model[1].xyz);
  let sz = length(model[2].xyz);
  return max(max(sx, sy), sz);
}

/**
 * 主剔除函数
 * 每个线程处理一个 splat
 */
@compute @workgroup_size(256)
fn cullSplats(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  
  // 边界检查
  if (i >= params.splatCount) {
    return;
  }
  
  let splat = splats[i];
  
  // ============================================
  // Step 1: 变换到视图空间
  // 先应用模型矩阵变换到世界空间，再变换到视图空间
  // ============================================
  let worldPos = camera.model * vec4<f32>(splat.mean, 1.0);
  let viewPos = camera.view * worldPos;
  
  // 视图空间中 z 是负值（相机看向 -Z），取正值表示深度
  let z = -viewPos.z;
  
  // ============================================
  // Step 2: 近平面剔除
  // 如果 splat 中心在近平面之前，剔除
  // ============================================
  if (z < params.nearPlane) {
    return;
  }
  
  // ============================================
  // Step 3: 远平面剔除 (可选，通常 splat 场景不需要)
  // ============================================
  if (z > params.farPlane) {
    return;
  }
  
  // ============================================
  // Step 4: 视锥剔除
  // 考虑 splat 的 3-sigma 半径，保守剔除
  // ============================================
  
  // 获取投影矩阵参数
  let fx = camera.proj[0][0];  // focal_x / aspect
  let fy = camera.proj[1][1];  // focal_y
  
  // 计算 NDC 坐标 (未裁剪)
  let x_ndc = viewPos.x * fx / z;
  let y_ndc = viewPos.y * fy / z;
  
  // 计算 splat 在 NDC 空间的半径 (3-sigma 保守估计)
  // 考虑模型缩放对 splat 半径的影响
  let modelScale = getModelMaxScale(camera.model);
  let worldRadius = maxScale(splat.scale) * modelScale * 3.0;
  let r_ndc = worldRadius * max(fx, fy) / z;
  
  // 视锥剔除：如果 splat 完全在视锥外，剔除
  // 考虑 splat 半径进行保守剔除
  if (x_ndc < -1.0 - r_ndc || x_ndc > 1.0 + r_ndc) {
    return;
  }
  if (y_ndc < -1.0 - r_ndc || y_ndc > 1.0 + r_ndc) {
    return;
  }
  
  // ============================================
  // Step 5: 屏幕尺寸剔除
  // 如果 splat 在屏幕上小于指定像素阈值，剔除
  // ============================================
  
  // 计算屏幕空间半径 (像素)
  let screenRadiusX = r_ndc * params.screenWidth * 0.5;
  let screenRadiusY = r_ndc * params.screenHeight * 0.5;
  let screenRadius = max(screenRadiusX, screenRadiusY);
  
  if (screenRadius < params.pixelThreshold) {
    return;
  }
  
  // ============================================
  // Step 6: 透明度剔除 (可选)
  // 如果 opacity 太低，剔除
  // ============================================
  if (splat.opacity < 0.004) {  // 约 1/255
    return;
  }
  
  // ============================================
  // Step 7: 通过所有剔除测试，写入结果
  // ============================================
  
  // 原子递增计数器，获取写入位置
  let visibleIdx = atomicAdd(&counters.visibleCount, 1u);
  
  // 写入可见 splat 的原始索引和深度
  visibleIndices[visibleIdx] = i;
  visibleDepths[visibleIdx] = z;
}

/**
 * 重置计数器
 * 每帧开始时调用
 */
@compute @workgroup_size(1)
fn resetCounters() {
  atomicStore(&counters.visibleCount, 0u);
}

/**
 * 更新 DrawIndirect buffer
 * 在剔除完成后调用
 */
@compute @workgroup_size(1)
fn updateDrawIndirect() {
  let count = atomicLoad(&counters.visibleCount);
  
  // DrawIndirect 布局: [vertexCount, instanceCount, firstVertex, firstInstance]
  drawIndirect[0] = 4u;     // vertexCount: 4 个顶点 (triangle strip quad)
  drawIndirect[1] = count;  // instanceCount: 可见 splat 数量
  drawIndirect[2] = 0u;     // firstVertex
  drawIndirect[3] = 0u;     // firstInstance
}
