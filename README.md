# WebGPU 3D Gaussian Splatting 渲染引擎

一个可扩展的 WebGPU 3D 渲染引擎，核心特性是支持 **3D Gaussian Splatting (3DGS)** 技术。

![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue?logo=typescript)
![WebGPU](https://img.shields.io/badge/WebGPU-Supported-green)
![Vite](https://img.shields.io/badge/Vite-5.0-purple?logo=vite)

## 功能特性

- **WebGPU 高性能渲染** - 利用现代 GPU API 实现高效渲染
- **3D Gaussian Splatting 支持**
  - PLY / Splat 文件加载
  - GPU 加速排序（基于 Radix Sort）
  - 球谐函数 (SH) 多级支持：L0 / L1 / L2 / L3
  - 移动端自动优化
- **多格式模型加载**
  - GLB/GLTF 模型
  - OBJ/MTL 模型（支持材质和纹理）
- **完整交互系统**
  - 轨道控制器 (OrbitControls)
  - 变换 Gizmo（平移/旋转/缩放）
  - 视口坐标轴指示器
  - 选中对象包围盒显示
- **场景管理**
  - 多对象管理
  - 材质颜色编辑
  - 自动 Frame Model 功能
- **移动端支持**
  - 触摸手势控制
  - 自动性能优化
  - 响应式 UI

## 文档

📖 **[完整使用手册](./USAGE_GUIDE.md)** - 详细的 API 文档和使用示例

## 系统要求

### 浏览器支持

| 浏览器 | 最低版本 |
|--------|----------|
| Chrome | 113+ |
| Edge | 113+ |
| Safari | 17+ |
| Firefox | 实验性支持 |

### 其他要求

- 需要在 **HTTPS** 或 **localhost** 环境下运行
- Node.js 18+（用于开发构建）

## 快速开始

### 安装

```bash
yarn install
```

### 启动开发服务器

```bash
yarn dev
```

访问 `https://localhost:3000` 查看 Demo。

### 构建

```bash
# 构建 Demo
yarn build:demo

# 构建库（类型检查）
yarn build:lib
```

## 基本用法

```typescript
import { App } from 'webgpu-3dgs';

// 创建应用
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const app = new App(canvas);

// 初始化
await app.init();

// 加载 3D Gaussian Splatting 模型
await app.addPLY('model.ply', (progress, stage) => {
  console.log(`${stage}: ${progress.toFixed(1)}%`);
});

// 或加载传统 3D 模型
await app.addGLB('model.glb');
await app.addOBJ('model.obj');

// 自动调整相机
app.frameCurrentModel();

// 启动渲染
app.start();
```

## 项目结构

```
webgpu-3dgs/
├── src/                    # 引擎源代码
│   ├── index.ts           # 库入口
│   ├── App.ts             # 统一调度入口
│   ├── core/              # 核心模块
│   │   ├── Renderer.ts    # WebGPU 渲染器
│   │   ├── Camera.ts      # 相机
│   │   ├── OrbitControls.ts # 轨道控制器
│   │   ├── ViewportGizmo.ts # 视口 Gizmo
│   │   ├── BoundingBoxRenderer.ts # 包围盒渲染
│   │   ├── gizmo/         # 变换 Gizmo
│   │   └── math/          # 数学工具
│   ├── gs/                # 3D Gaussian Splatting
│   │   ├── GSSplatRenderer.ts  # 桌面端渲染器
│   │   ├── GSSplatRendererMobile.ts # 移动端渲染器
│   │   ├── GSSplatSorter.ts    # GPU 排序器
│   │   ├── PLYLoader.ts        # PLY 加载器
│   │   ├── SplatLoader.ts      # Splat 加载器
│   │   └── *.wgsl              # WGSL 着色器
│   ├── mesh/              # 网格渲染
│   ├── loaders/           # 模型加载器
│   ├── scene/             # 场景管理
│   └── interaction/       # 交互管理
├── demo/                  # Demo 应用
├── USAGE_GUIDE.md         # 使用手册
└── package.json
```

## 核心 API

### App 类

```typescript
// 初始化
await app.init();
app.start();
app.stop();
app.destroy();

// 模型加载
await app.addPLY(url, onProgress?);
await app.addSplat(url, onProgress?);
await app.addGLB(url);
await app.addOBJ(url);

// 场景管理
app.getMeshCount();
app.getSplatCount();
app.clearMeshes();
app.clearSplats();

// SH 模式 (0-3)
app.setSHMode(mode);
app.getSHMode();

// 相机控制
app.frameCurrentModel(animate?);
app.getCamera();
app.getControls();

// Gizmo
app.setGizmoMode(mode);
app.setGizmoTarget(object);
```

### 导出类

| 类名 | 说明 |
|------|------|
| `App` | 统一调度入口 |
| `Renderer` | WebGPU 渲染器 |
| `Camera` | 透视相机 |
| `OrbitControls` | 轨道控制器 |
| `Mesh` | 网格数据结构 |
| `MeshRenderer` | 网格渲染器 |
| `GSSplatRenderer` | 3DGS 渲染器 |
| `GLBLoader` | GLB 加载器 |
| `OBJLoader` | OBJ 加载器 |
| `TransformGizmoV2` | 变换 Gizmo |
| `ViewportGizmo` | 视口 Gizmo |
| `SceneManager` | 场景管理器 |

## 交互控制

### 鼠标

| 操作 | 功能 |
|------|------|
| 左键拖拽 | 旋转视角 |
| 右键拖拽 | 平移视角 |
| 滚轮 | 缩放 |

### 触摸

| 操作 | 功能 |
|------|------|
| 单指拖拽 | 旋转视角 |
| 双指捏合 | 缩放 |
| 双指拖拽 | 平移视角 |

### 键盘

| 按键 | 功能 |
|------|------|
| W | 平移模式 |
| E | 旋转模式 |
| R | 缩放模式 |

## 技术细节

### 3D Gaussian Splatting

- **排序**: GPU Radix Sort (Compute Shader)
- **渲染**: 基于 Quad 的 2D 高斯椭圆投影
- **协方差**: 3D → 2D 屏幕空间投影
- **球谐函数**: 0-3 阶 SH 系数，视角相关颜色

### 着色器

- WGSL (WebGPU Shading Language)
- 多 SH 级别优化变体
- GPU 排序 Compute Shader

## 许可证

MIT License
