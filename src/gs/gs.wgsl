/**
 * 3D Gaussian Splatting Shader
 * 椭圆投影版本 - 使用协方差矩阵计算屏幕空间椭圆
 * 
 * 参考: MipSplatting 抗锯齿技术
 * https://niujinshuchong.github.io/mip-splatting/
 */

// Camera uniform
struct Uniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  screenSize: vec2<f32>,  // 屏幕尺寸 (width, height)
  _pad: vec2<f32>,
}

// Splat 数据结构 (256 bytes, 对齐)
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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;

// 顶点着色器输出
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) fragPos: vec2<f32>,      // 高斯空间坐标 (已乘 sqrt8)
  @location(1) color: vec3<f32>,
  @location(2) opacity: f32,            // 包含 AA 补偿后的透明度
}

// Quad 顶点位置 (triangle strip: 0-1-2, 1-3-2)
const QUAD_POSITIONS = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>( 1.0,  1.0),
);

// sqrt(8) ≈ 2.828, 用于椭圆展开
// 在 A = dot(fragPos, fragPos) > 8.0 时截断
const SQRT8: f32 = 2.82842712475;

// 低通滤波器常数 (用于抗锯齿)
const LOW_PASS_FILTER: f32 = 0.3;

// Alpha 剔除阈值
const ALPHA_CULL_THRESHOLD: f32 = 0.00392156863; // 1/255

/**
 * 四元数转 3x3 旋转矩阵
 * 使用 HLSL 参考实现的格式: q = (r, x, y, z) 其中 r 是实部
 */
fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let r = q.x;  // 实部 (w)
  let x = q.y;
  let y = q.z;
  let z = q.w;
  
  // 构建旋转矩阵 (与 HLSL 参考保持一致)
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),
    vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),
    vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y))
  );
}

/**
 * 计算 3D 协方差矩阵
 * 返回紧凑表示: covA = (xx, xy, xz), covB = (yy, yz, zz)
 */
fn computeCovariance3D(scale: vec3<f32>, rotation: vec4<f32>) -> array<vec3<f32>, 2> {
  let R = quatToMat3(rotation);
  
  // S = scale 对角矩阵
  let S = mat3x3<f32>(
    vec3<f32>(scale.x, 0.0, 0.0),
    vec3<f32>(0.0, scale.y, 0.0),
    vec3<f32>(0.0, 0.0, scale.z)
  );
  
  // M = R * S
  let M = R * S;
  
  // 协方差 = M * M^T
  // 只需要上三角部分 (对称矩阵)
  let covA = vec3<f32>(dot(M[0], M[0]), dot(M[0], M[1]), dot(M[0], M[2]));
  let covB = vec3<f32>(dot(M[1], M[1]), dot(M[1], M[2]), dot(M[2], M[2]));
  
  return array<vec3<f32>, 2>(covA, covB);
}

/**
 * 将 3D 协方差投影到 2D 屏幕空间
 * 基于 EWA splatting (Zwicker et al. 2002)
 */
fn projectCovariance(
  cov3D: mat3x3<f32>,
  viewCenter: vec3<f32>,
  focal: vec2<f32>,
  modelView: mat3x3<f32>
) -> vec3<f32> {
  let v = viewCenter;
  let s = 1.0 / (v.z * v.z);
  
  // Jacobian 矩阵 (透视投影的线性近似)
  let J = mat3x3<f32>(
    vec3<f32>(focal.x / v.z, 0.0, 0.0),
    vec3<f32>(0.0, focal.y / v.z, 0.0),
    vec3<f32>(-(focal.x * v.x) * s, -(focal.y * v.y) * s, 0.0)
  );
  
  // T = J * W (W 是模型视图矩阵的旋转部分)
  let T = J * modelView;
  
  // cov2D = T * cov3D * T^T
  let cov2D = T * cov3D * transpose(T);
  
  // 返回 (xx, xy, yy)
  return vec3<f32>(cov2D[0][0], cov2D[0][1], cov2D[1][1]);
}

/**
 * 计算椭圆的特征值和特征向量，返回基向量
 * 使用 MipSplatting 抗锯齿: 添加低通滤波并补偿 alpha
 */
fn computeExtentBasis(
  cov2Dv: vec3<f32>,
  stdDev: f32,
  splatScale: f32,
  opacity: ptr<function, f32>
) -> array<vec2<f32>, 2> {
  
  // 原始行列式 (用于 AA 补偿)
  let detOrig = cov2Dv.x * cov2Dv.z - cov2Dv.y * cov2Dv.y;
  
  // 添加低通滤波器 (MipSplatting 抗锯齿)
  // 这相当于将高斯与一个像素大小的滤波器卷积
  var cov = cov2Dv;
  cov.x += LOW_PASS_FILTER;
  cov.z += LOW_PASS_FILTER;
  
  // 滤波后的行列式
  let detBlur = cov.x * cov.z - cov.y * cov.y;
  
  // Alpha 补偿: 保持总能量不变
  if (detBlur > 0.0 && detOrig > 0.0) {
    *opacity *= sqrt(detOrig / detBlur);
  }
  
  // 计算特征值
  let a = cov.x;
  let b = cov.y;
  let d = cov.z;
  
  let D = a * d - b * b;
  let trace = a + d;
  let traceOver2 = 0.5 * trace;
  
  // 使用较大的最小值确保数值稳定
  let term2 = sqrt(max(0.1, traceOver2 * traceOver2 - D));
  let eigenValue1 = traceOver2 + term2;
  let eigenValue2 = traceOver2 - term2;
  
  // 如果特征值无效，返回零向量 (将被剔除)
  if (eigenValue2 <= 0.0) {
    return array<vec2<f32>, 2>(vec2<f32>(0.0), vec2<f32>(0.0));
  }
  
  // 计算特征向量
  let eigenVector1 = normalize(vec2<f32>(b, eigenValue1 - a));
  let eigenVector2 = vec2<f32>(eigenVector1.y, -eigenVector1.x);
  
  // 基向量 = 特征向量 * splatScale * min(stdDev * sqrt(eigenValue), 2048)
  // 限制最大尺寸防止数值问题
  let basisVector1 = eigenVector1 * splatScale * min(stdDev * sqrt(eigenValue1), 2048.0);
  let basisVector2 = eigenVector2 * splatScale * min(stdDev * sqrt(eigenValue2), 2048.0);
  
  return array<vec2<f32>, 2>(basisVector1, basisVector2);
}

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  var output: VertexOutput;
  
  // 获取当前 splat 数据
  let splat = splats[instanceIndex];
  
  // 早期透明度剔除
  if (splat.opacity < ALPHA_CULL_THRESHOLD) {
    output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return output;
  }
  
  // 检查四元数有效性
  let quatNormSqr = dot(splat.rotation, splat.rotation);
  if (quatNormSqr < 1e-6) {
    output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return output;
  }
  
  // 计算视图空间位置
  let worldPos = vec4<f32>(splat.mean, 1.0);
  let viewPos = uniforms.view * worldPos;
  
  // 近平面剔除 (视图空间 z 为负)
  if (viewPos.z >= 0.0) {
    output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return output;
  }
  
  // 透视投影
  let clipPos = uniforms.proj * viewPos;
  let pW = 1.0 / (clipPos.w + 0.0000001);
  let ndcPos = clipPos * pW;
  
  // 视锥剔除 (带边界扩展防止边缘闪烁)
  let frustumDilation = 0.15;
  let clipBound = 1.0 + frustumDilation;
  if (abs(ndcPos.x) > clipBound || abs(ndcPos.y) > clipBound ||
      ndcPos.z < -frustumDilation || ndcPos.z > 1.0) {
    output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return output;
  }
  
  // Quad 顶点位置
  let fragPos = QUAD_POSITIONS[vertexIndex] * SQRT8;
  output.fragPos = fragPos;
  
  // 计算 3D 协方差
  let covParts = computeCovariance3D(splat.scale, splat.rotation);
  let cov3D = mat3x3<f32>(
    vec3<f32>(covParts[0].x, covParts[0].y, covParts[0].z),
    vec3<f32>(covParts[0].y, covParts[1].x, covParts[1].y),
    vec3<f32>(covParts[0].z, covParts[1].y, covParts[1].z)
  );
  
  // 提取模型视图矩阵的旋转部分
  let modelView = mat3x3<f32>(
    uniforms.view[0].xyz,
    uniforms.view[1].xyz,
    uniforms.view[2].xyz
  );
  
  // 计算焦距
  let tanFovX = 1.0 / uniforms.proj[0][0];
  let tanFovY = 1.0 / abs(uniforms.proj[1][1]);
  let focalX = uniforms.screenSize.x / (2.0 * tanFovX);
  let focalY = uniforms.screenSize.y / (2.0 * tanFovY);
  let focal = vec2<f32>(focalX, focalY);
  
  // 视图空间中心
  let centerView = viewPos.xyz / viewPos.w;
  
  // 投影 2D 协方差
  let cov2D = projectCovariance(cov3D, centerView, focal, modelView);
  
  // 计算椭圆基向量 (包含抗锯齿处理)
  var opacity = splat.opacity;
  let basis = computeExtentBasis(cov2D, SQRT8, 1.0, &opacity);
  let basisVector1 = basis[0];
  let basisVector2 = basis[1];
  
  // 检查基向量是否有效
  if (length(basisVector1) < 0.001 && length(basisVector2) < 0.001) {
    output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return output;
  }
  
  // 计算 NDC 偏移
  let basisViewport = 1.0 / uniforms.screenSize;
  let ndcOffset = (fragPos.x * basisVector1 + fragPos.y * basisVector2) * basisViewport * 2.0;
  
  // 最终位置
  output.position = vec4<f32>(ndcPos.xy + ndcOffset, ndcPos.z, 1.0);
  
  // 传递颜色和透明度 (包含 AA 补偿)
  output.color = splat.colorDC;
  output.opacity = min(0.999, opacity); // 限制最大透明度
  
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  // 丢弃无效片段
  if (input.opacity <= 0.0) {
    discard;
  }
  
  // 计算到中心的距离平方
  // fragPos 已经乘以 sqrt(8)，所以 A = dot(fragPos, fragPos) 
  // 当 A > 8.0 时超出边界 (对应约 2.83 sigma)
  let A = dot(input.fragPos, input.fragPos);
  
  if (A > 8.0) {
    discard;
  }
  
  // 高斯衰减: exp(-0.5 * A)
  // 注意: A 已经是标准化的距离平方
  let power = -0.5 * A;
  var alpha = min(0.999, input.opacity * exp(power));
  
  // 极低透明度剔除
  if (alpha < ALPHA_CULL_THRESHOLD) {
    discard;
  }
  
  // 输出颜色 (预乘 alpha)
  return vec4<f32>(input.color * alpha, alpha);
}
