# WebGPU 3D Gaussian Splatting 渲染引擎

一个可扩展的 WebGPU 3D 渲染引擎，核心特性是支持 **3D Gaussian Splatting (3DGS)** 技术。

![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue?logo=typescript)
![WebGPU](https://img.shields.io/badge/WebGPU-Supported-green)
![Vite](https://img.shields.io/badge/Vite-5.0-purple?logo=vite)

## 功能特性

- **WebGPU 高性能渲染** - 利用现代 GPU API 实现高效渲染
- **3D Gaussian Splatting 支持**
  - PLY 文件加载（支持标准 3DGS 格式）
  - GPU 加速排序（基于 Radix Sort）
  - 球谐函数 (SH) 多级支持：L0 / L1 / L2 / L3
- **GLB/GLTF 模型加载** - 支持标准 3D 模型格式
- **相机系统**
  - 轨道控制器 (OrbitControls)
  - 视口 Gizmo 坐标轴指示器
  - 自动 Frame Model 功能
- **内置基础几何体** - 立方体、球体等测试用几何体
- **完整 Demo 应用** - 包含场景树、属性面板、文件拖放等功能

## 系统要求

### 浏览器支持

WebGPU 需要现代浏览器支持：

| 浏览器 | 最低版本 |
|--------|----------|
| Chrome | 113+ |
| Edge | 113+ |
| Safari | 17+ |
| Firefox | 实验性支持 |

### 其他要求

- 需要在 **HTTPS** 或 **localhost** 环境下运行（WebGPU 安全上下文要求）
- Node.js 18+（用于开发构建）

## 快速开始

### 安装依赖

```bash
yarn install
```

### 启动开发服务器

```bash
yarn dev
```

访问 `https://localhost:3000` 查看 Demo（注意是 HTTPS）。

### 构建

```bash
# 构建 Demo
yarn build:demo

# 构建库（类型检查）
yarn build:lib
```

## 项目结构

```
webgpu-3dgs/
├── src/                    # 引擎源代码
│   ├── index.ts           # 库入口，导出所有公共 API
│   ├── App.ts             # 统一调度入口类
│   ├── core/              # 核心模块
│   │   ├── Renderer.ts    # WebGPU 渲染器
│   │   ├── Camera.ts      # 相机
│   │   ├── OrbitControls.ts # 轨道控制器
│   │   └── ViewportGizmo.ts # 视口坐标轴指示器
│   ├── gs/                # 3D Gaussian Splatting 模块
│   │   ├── GSSplatRenderer.ts  # Splat 渲染器
│   │   ├── GSSplatSorter.ts    # GPU 排序器
│   │   ├── PLYLoader.ts        # PLY 文件加载器
│   │   └── *.wgsl              # WGSL 着色器
│   ├── mesh/              # 网格渲染模块
│   │   ├── Mesh.ts        # 网格数据结构
│   │   └── MeshRenderer.ts # 网格渲染器
│   └── loaders/           # 加载器
│       └── GLBLoader.ts   # GLB/GLTF 加载器
├── demo/                  # Demo 应用
│   ├── index.html         # 入口 HTML
│   └── main.ts            # Demo 主逻辑
├── gaussian/              # HLSL 着色器参考（来自原始 3DGS）
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## 使用示例

### 基本用法

```typescript
import { App } from 'webgpu-3dgs';

// 创建应用
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const app = new App(canvas);

// 初始化
await app.init();

// 加载 PLY 文件（3D Gaussian Splatting）
await app.addPLY('path/to/model.ply');

// 或加载 GLB 模型
await app.addGLB('path/to/model.glb');

// 自动调整相机到模型
app.frameCurrentModel();

// 启动渲染循环
app.start();
```

### 使用单独模块

```typescript
import { 
  Renderer, 
  Camera, 
  OrbitControls,
  GSSplatRenderer,
  loadPLY 
} from 'webgpu-3dgs';

// 初始化渲染器
const renderer = new Renderer(canvas);
await renderer.init();

// 创建相机
const camera = new Camera();
camera.setAspect(canvas.width / canvas.height);

// 添加轨道控制
const controls = new OrbitControls(camera, canvas);

// 加载并渲染 3DGS
const splats = await loadPLY('model.ply');
const gsRenderer = new GSSplatRenderer(renderer, camera);
gsRenderer.setData(splats);

// 渲染循环
function render() {
  const pass = renderer.beginFrame();
  gsRenderer.render(pass);
  renderer.endFrame();
  requestAnimationFrame(render);
}
render();
```

### 设置 SH 模式

```typescript
// 设置球谐函数等级
// 0 = L0 (仅 DC 颜色，最快)
// 1 = L1 (默认)
// 2 = L2
// 3 = L3 (完整 SH，最高质量)
app.setSHMode(2);
```

## API 概览

### 导出类

| 类名 | 说明 |
|------|------|
| `App` | 统一调度入口，管理所有子系统 |
| `Renderer` | WebGPU 设备管理和渲染通道 |
| `Camera` | 透视相机 |
| `OrbitControls` | 轨道相机控制器 |
| `ViewportGizmo` | 视口坐标轴指示器 |
| `Mesh` | 网格数据结构 |
| `MeshRenderer` | 网格渲染器 |
| `GSSplatRenderer` | 3DGS Splat 渲染器 |
| `GLBLoader` | GLB/GLTF 模型加载器 |

### 导出函数

| 函数 | 说明 |
|------|------|
| `loadPLY(url)` | 加载 PLY 文件，返回 `SplatCPU[]` |

### 导出类型

| 类型 | 说明 |
|------|------|
| `SplatCPU` | Splat 数据结构（位置、缩放、旋转、颜色、SH 系数） |
| `SHMode` | 球谐函数模式枚举 (0-3) |

### App 主要方法

```typescript
class App {
  // 初始化
  init(): Promise<void>
  
  // 加载模型
  addPLY(url: string): Promise<number>    // 返回 splat 数量
  addGLB(url: string): Promise<number>    // 返回网格数量
  
  // 测试几何体
  addTestCube(): void
  addTestSphere(): void
  
  // 渲染控制
  start(): void
  stop(): void
  
  // 相机控制
  frameCurrentModel(animate?: boolean): boolean
  
  // SH 模式
  setSHMode(mode: 0 | 1 | 2 | 3): void
  getSHMode(): number
  
  // 获取子系统
  getRenderer(): Renderer
  getCamera(): Camera
  getControls(): OrbitControls
  getMeshRenderer(): MeshRenderer
  getGSRenderer(): GSSplatRenderer | undefined
  
  // 场景管理
  clearMeshes(): void
  clearSplats(): void
  getMeshCount(): number
  getSplatCount(): number
}
```

## 技术细节

### 3D Gaussian Splatting 实现

- **排序算法**: GPU Radix Sort，在 Compute Shader 中实现
- **渲染方式**: 基于 Quad 的 2D 高斯椭圆投影
- **协方差计算**: 3D 协方差矩阵投影到 2D 屏幕空间
- **球谐函数**: 支持 0-3 阶 SH 系数，用于视角相关的颜色

### 着色器

- 使用 WGSL (WebGPU Shading Language)
- 包含多个 SH 级别的优化着色器变体
- GPU 排序使用 Compute Shader 实现

## 交互控制

| 操作 | 功能 |
|------|------|
| 鼠标左键拖拽 | 旋转视角 |
| 鼠标右键拖拽 | 平移视角 |
| 鼠标滚轮 | 缩放 |
| 点击 Gizmo 轴 | 切换到正交视图 |

## 许可证

MIT License
