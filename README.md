# @d5techs/3dgs-lib

可扩展的 WebGPU 3D 渲染引擎，核心特性是支持 **3D Gaussian Splatting (3DGS)** 技术。

![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue?logo=typescript)
![WebGPU](https://img.shields.io/badge/WebGPU-Supported-green)
![Vite](https://img.shields.io/badge/Vite-5.0-purple?logo=vite)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 功能特性

### 3D Gaussian Splatting

- PLY / Splat 文件加载与解析
- GPU 加速排序（基于 Radix Sort Compute Shader）
- 球谐函数 (SH) 多级支持：L0 / L1 / L2 / L3
- 桌面端/移动端自适应渲染器
- Normalized Gaussian 抗锯齿
- ClipCorner 优化减少 overdraw

### 传统 3D 模型

- GLB/GLTF 模型加载（支持 PBR 材质和纹理）
- OBJ/MTL 模型加载（支持材质和纹理）
- 自动计算 Bounding Box

### 交互系统

- 轨道控制器 (OrbitControls) - 支持鼠标和触摸
- 变换 Gizmo - 平移/旋转/缩放三种模式
- 视口坐标轴指示器
- 选中对象包围盒显示

### 场景管理

- 多对象管理
- 材质颜色编辑
- 自动 Frame Model 功能
- Splat 变换（位置/旋转/缩放）

---

## 系统要求

### 浏览器支持

| 浏览器 | 最低版本 | 备注 |
|--------|----------|------|
| Chrome | 113+ | 推荐 |
| Edge | 113+ | 推荐 |
| Safari | 17+ | macOS/iOS |
| Firefox | Nightly | 实验性支持 |

### 运行环境

- **HTTPS** 或 **localhost**（WebGPU 安全要求）
- Node.js 18+（仅开发构建需要）

---

## 快速开始

### 安装

```bash
# yarn
yarn add @d5techs/3dgs-lib

# npm
npm install @d5techs/3dgs-lib

# pnpm
pnpm add @d5techs/3dgs-lib
```

### 基本用法

```typescript
import { App } from '@d5techs/3dgs-lib';

// 1. 创建应用
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const app = new App(canvas);

// 2. 初始化（异步）
await app.init();

// 3. 加载模型
// 3DGS 模型
await app.addPLY('model.ply', (progress, stage) => {
  console.log(`${stage}: ${progress.toFixed(1)}%`);
});

// 或传统 3D 模型
await app.addGLB('model.glb');
await app.addOBJ('model.obj');

// 4. 自动调整相机视角
app.frameCurrentModel();

// 5. 启动渲染循环
app.start();
```

### HTML 设置

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    body { margin: 0; overflow: hidden; }
    canvas { width: 100vw; height: 100vh; display: block; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <script type="module" src="./main.ts"></script>
</body>
</html>
```

---

## 核心 API

### App 类

主入口类，提供完整的应用生命周期管理。

```typescript
const app = new App(canvas);

// 生命周期
await app.init();           // 初始化 WebGPU
app.start();                // 开始渲染循环
app.stop();                 // 停止渲染循环
app.destroy();              // 销毁所有资源

// 模型加载
await app.addPLY(url, onProgress?);    // 加载 PLY (3DGS)
await app.addSplat(url, onProgress?);  // 加载 Splat (3DGS)
await app.addGLB(url);                 // 加载 GLB
await app.addOBJ(url);                 // 加载 OBJ

// 场景查询
app.getMeshCount();          // Mesh 数量
app.getSplatCount();         // Splat 数量
app.getSplatBoundingBox();   // Splat 包围盒

// 场景操作
app.clearMeshes();           // 清空所有 Mesh
app.clearSplats();           // 清空 Splat

// Splat 变换
app.setSplatPosition(x, y, z);
app.setSplatRotation(x, y, z);  // 弧度
app.setSplatScale(x, y, z);

// SH 模式 (球谐函数级别)
app.setSHMode(0);  // L0 - 仅 DC 颜色
app.setSHMode(1);  // L1 - DC + 一阶 SH
app.setSHMode(2);  // L2 - DC + 一二阶 SH
app.setSHMode(3);  // L3 - 完整 SH

// 相机控制
app.frameCurrentModel(animate?);  // 自动调整视角
app.getCamera();                  // 获取相机实例
app.getControls();                // 获取控制器实例

// Gizmo 控制
app.setGizmoMode('translate');  // 平移模式
app.setGizmoMode('rotate');     // 旋转模式
app.setGizmoMode('scale');      // 缩放模式
app.setGizmoTarget(object);     // 设置操作目标

// 子系统访问
app.getRenderer();       // WebGPU 渲染器
app.getMeshRenderer();   // Mesh 渲染器
app.getGSRenderer();     // 3DGS 渲染器
```

### 进度回调

```typescript
type ProgressCallback = (
  progress: number,           // 0-100
  stage: 'download' | 'parse' | 'upload'
) => void;

await app.addPLY('model.ply', (progress, stage) => {
  if (stage === 'download') {
    console.log(`下载中: ${progress.toFixed(1)}%`);
  } else if (stage === 'parse') {
    console.log(`解析中: ${progress.toFixed(1)}%`);
  } else {
    console.log(`上传 GPU: ${progress.toFixed(1)}%`);
  }
});
```

---

## 交互控制

### 鼠标

| 操作 | 功能 |
|------|------|
| 左键拖拽 | 旋转视角 |
| 右键拖拽 | 平移视角 |
| 滚轮 | 缩放 |

### 触摸（移动端）

| 操作 | 功能 |
|------|------|
| 单指拖拽 | 旋转视角 |
| 双指捏合 | 缩放 |
| 双指拖拽 | 平移视角 |

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| W | 切换到平移模式 |
| E | 切换到旋转模式 |
| R | 切换到缩放模式 |

---

## 项目架构

```
@d5techs/3dgs-lib/
├── src/
│   ├── index.ts              # 库入口，导出所有公共 API
│   ├── App.ts                # 统一调度入口
│   │
│   ├── types/                # 统一类型定义
│   │   ├── geometry.ts       # BoundingBox, Vec3Tuple, Transform
│   │   ├── material.ts       # MaterialData
│   │   └── splat.ts          # SHMode, RendererCapabilities
│   │
│   ├── utils/                # 公共工具函数
│   │   ├── device.ts         # 设备检测 (isMobileDevice)
│   │   ├── geometry.ts       # 几何计算 (computeBoundingBox)
│   │   └── texture.ts        # 纹理加载工具
│   │
│   ├── core/                 # 核心渲染模块
│   │   ├── Renderer.ts       # WebGPU 渲染器
│   │   ├── Camera.ts         # 透视相机
│   │   ├── OrbitControls.ts  # 轨道控制器
│   │   ├── ViewportGizmo.ts  # 视口坐标轴
│   │   ├── BoundingBoxRenderer.ts
│   │   ├── gizmo/            # 变换 Gizmo 组件
│   │   └── math/             # 数学库 (Vec3, Mat4, Quat, Ray)
│   │
│   ├── gs/                   # 3D Gaussian Splatting
│   │   ├── IGSSplatRenderer.ts      # 渲染器接口
│   │   ├── GSSplatRenderer.ts       # 桌面端渲染器
│   │   ├── GSSplatRendererMobile.ts # 移动端渲染器
│   │   ├── GSSplatSorter.ts         # GPU 排序器
│   │   ├── PLYLoader.ts             # PLY 加载器
│   │   └── SplatLoader.ts           # Splat 加载器
│   │
│   ├── mesh/                 # 网格渲染
│   │   ├── Mesh.ts           # 网格数据结构
│   │   └── MeshRenderer.ts   # 网格渲染器
│   │
│   ├── loaders/              # 模型加载器
│   │   ├── GLBLoader.ts      # GLB/GLTF 加载
│   │   ├── OBJLoader.ts      # OBJ 加载
│   │   ├── OBJParser.ts      # OBJ 解析
│   │   └── MTLParser.ts      # MTL 材质解析
│   │
│   ├── scene/                # 场景管理
│   │   ├── SceneManager.ts   # 场景管理器
│   │   └── proxies/          # 变换代理类
│   │
│   └── interaction/          # 交互管理
│       └── GizmoManager.ts   # Gizmo 管理器
│
├── demo/                     # Demo 应用
│   ├── index.html
│   └── main.ts
│
└── dist/                     # 构建输出
    ├── 3dgs-lib.js           # ESM
    ├── 3dgs-lib.cjs          # CommonJS
    └── index.d.ts            # 类型声明
```

---

## 高级用法

### 自定义渲染器颜色

```typescript
const renderer = app.getRenderer();
renderer.setClearColor(0.1, 0.1, 0.1);      // RGB
renderer.setClearColorHex('#1a1a2e');       // Hex
```

### 访问底层相机

```typescript
const camera = app.getCamera();
camera.fov = 60;                    // 视场角
camera.near = 0.1;                  // 近裁剪面
camera.far = 1000;                  // 远裁剪面
camera.setPosition(0, 5, 10);       // 相机位置
camera.lookAt(0, 0, 0);             // 看向目标
```

### 配置轨道控制器

```typescript
const controls = app.getControls();
controls.minDistance = 1;           // 最小距离
controls.maxDistance = 100;         // 最大距离
controls.enableDamping = true;      // 启用阻尼
controls.dampingFactor = 0.1;       // 阻尼系数
```

### 使用 Gizmo 变换对象

```typescript
import { GizmoMode } from '@d5techs/3dgs-lib';

// 获取 Splat 变换代理
const proxy = app.getSplatTransformProxy();
if (proxy) {
  app.setGizmoTarget(proxy);
  app.setGizmoMode(GizmoMode.Translate);
  
  // 设置包围盒显示
  const bbProvider = app.createSplatBoundingBoxProvider();
  app.setSelectionBoundingBoxProvider(bbProvider);
}
```

### 检测 WebGPU 支持

```typescript
import { isWebGPUSupported } from '@d5techs/3dgs-lib';

if (!isWebGPUSupported()) {
  alert('您的浏览器不支持 WebGPU');
}
```

---

## 开发

```bash
# 安装依赖
yarn install

# 启动开发服务器
yarn dev

# 构建库
yarn build:lib

# 构建 Demo
yarn build:demo

# 运行测试
yarn test
```

---

## 技术细节

### 3D Gaussian Splatting 渲染管线

1. **数据加载**: PLY/Splat → CPU 解析 → GPU Buffer
2. **视锥剔除**: Compute Shader 剔除不可见 Splat
3. **深度排序**: GPU Radix Sort 按深度排序
4. **渲染**: 实例化渲染 Quad，2D 高斯椭圆投影

### 着色器技术

- WGSL (WebGPU Shading Language)
- Normalized Gaussian 消除边缘雾化
- 多 SH 级别优化变体
- GPU 排序 Compute Shader

### 内存优化

- 桌面端: 256 bytes/splat（完整 SH）
- 移动端: ~52 bytes/splat（纹理压缩）

---

## 许可证

MIT License

---

## 相关链接

- [3D Gaussian Splatting 论文](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [WebGPU 规范](https://www.w3.org/TR/webgpu/)
- [WGSL 规范](https://www.w3.org/TR/WGSL/)
