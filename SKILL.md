# @d5techs/3dgs-lib 使用指南

本文档为 AI 助手提供关于 `@d5techs/3dgs-lib` 库的使用指南。

## 概述

`@d5techs/3dgs-lib` 是一个基于 WebGPU 的 3D 渲染库，核心特性是支持 **3D Gaussian Splatting (3DGS)** 技术渲染。同时也支持传统的 GLB/OBJ 模型加载和渲染。

## 何时使用此技能

当用户需要以下功能时，应使用此库：

- 在网页中渲染 3D Gaussian Splatting 模型（.ply, .splat 文件）
- 在网页中渲染传统 3D 模型（.glb, .gltf, .obj 文件）
- 需要 WebGPU 高性能渲染
- 需要 3D 场景交互（旋转、平移、缩放、Gizmo 变换）

## 基本使用模式

### 1. 最简模式 - 使用 App 类

这是最推荐的方式，适合大多数场景：

```typescript
import { App } from '@d5techs/3dgs-lib';

async function main() {
  // 获取 canvas 元素
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  
  // 创建应用
  const app = new App(canvas);
  
  // 初始化（必须 await）
  await app.init();
  
  // 加载模型（选择其一）
  // 3DGS 模型
  await app.addPLY('/models/scene.ply');
  // 或传统模型
  await app.addGLB('/models/model.glb');
  
  // 自动调整相机
  app.frameCurrentModel();
  
  // 开始渲染
  app.start();
}

main();
```

### 2. 带进度回调的加载

```typescript
await app.addPLY('/models/scene.ply', (progress, stage) => {
  // progress: 0-100
  // stage: 'download' | 'parse' | 'upload'
  updateLoadingUI(progress, stage);
});
```

### 3. 从 ArrayBuffer 加载（本地文件）

```typescript
// 用户选择文件
const file = event.target.files[0];
const buffer = await file.arrayBuffer();

// 加载（第三个参数表示是本地文件）
await app.addPLY(buffer, onProgress, true);
```

## 常用操作

### 模型管理

```typescript
// 查询
app.getMeshCount();           // Mesh 数量
app.getSplatCount();          // Splat 数量

// 清空
app.clearMeshes();            // 清空所有 Mesh
app.clearSplats();            // 清空 Splat

// 获取包围盒
app.getSplatBoundingBox();    // Splat 包围盒
app.getMeshRangeBoundingBox(startIndex, count);  // Mesh 范围包围盒
```

### Splat 变换

```typescript
// 设置位置
app.setSplatPosition(x, y, z);

// 设置旋转（欧拉角，弧度）
app.setSplatRotation(rx, ry, rz);

// 设置缩放
app.setSplatScale(sx, sy, sz);

// 获取当前值
const pos = app.getSplatPosition();     // [x, y, z] | null
const rot = app.getSplatRotation();     // [rx, ry, rz] | null
const scale = app.getSplatScale();      // [sx, sy, sz] | null
```

### 球谐函数模式 (SH Mode)

控制 3DGS 的渲染质量和性能：

```typescript
// L0: 仅 DC 颜色（最快，移动端默认）
app.setSHMode(0);

// L1: DC + 一阶 SH
app.setSHMode(1);

// L2: DC + 一二阶 SH
app.setSHMode(2);

// L3: 完整 SH（最高质量，桌面端默认）
app.setSHMode(3);
```

### 相机控制

```typescript
// 自动调整视角以显示整个模型
app.frameCurrentModel();        // 带动画
app.frameCurrentModel(false);   // 不带动画

// 访问相机
const camera = app.getCamera();
camera.fov = 60;                // 视场角
camera.setPosition(0, 5, 10);   // 位置
camera.lookAt(0, 0, 0);         // 看向目标

// 访问控制器
const controls = app.getControls();
controls.minDistance = 1;       // 最小距离
controls.maxDistance = 100;     // 最大距离
```

### Gizmo 变换工具

```typescript
import { GizmoMode } from '@d5techs/3dgs-lib';

// 设置模式
app.setGizmoMode(GizmoMode.Translate);  // 平移
app.setGizmoMode(GizmoMode.Rotate);     // 旋转
app.setGizmoMode(GizmoMode.Scale);      // 缩放

// 设置目标对象
// 对于 Splat
const splatProxy = app.getSplatTransformProxy();
if (splatProxy) {
  app.setGizmoTarget(splatProxy);
}

// 对于 Mesh 组
const meshProxy = app.createMeshGroupProxy(startIndex, count);
if (meshProxy) {
  app.setGizmoTarget(meshProxy);
}

// 显示包围盒
const bbProvider = app.createSplatBoundingBoxProvider();
app.setSelectionBoundingBoxProvider(bbProvider);

// 清除
app.setGizmoTarget(null);
app.clearSelectionBoundingBox();
```

### 渲染器配置

```typescript
const renderer = app.getRenderer();

// 设置背景颜色
renderer.setClearColor(0.1, 0.1, 0.1);     // RGB (0-1)
renderer.setClearColorHex('#1a1a2e');      // Hex

// 获取背景颜色
const hexColor = renderer.getClearColorHex();  // '#rrggbb'
```

## 生命周期管理

```typescript
// 初始化
await app.init();

// 开始/停止渲染
app.start();
app.stop();

// 销毁（释放所有资源）
app.destroy();
```

## 检测 WebGPU 支持

```typescript
import { isWebGPUSupported } from '@d5techs/3dgs-lib';

if (!isWebGPUSupported()) {
  // 显示不支持提示
  showFallbackUI();
}
```

## 工具函数

库导出了一些有用的工具函数：

```typescript
import {
  isMobileDevice,           // 检测移动设备
  getRecommendedDPR,        // 获取推荐的设备像素比
  computeBoundingBox,       // 计算包围盒
  mergeBoundingBoxes,       // 合并多个包围盒
} from '@d5techs/3dgs-lib';
```

## 类型定义

主要类型：

```typescript
import type {
  BoundingBox,              // 包围盒 { min, max, center, radius }
  Vec3Tuple,                // [number, number, number]
  MaterialData,             // 材质数据
  ProgressCallback,         // 进度回调
  TransformableObject,      // 可变换对象接口
} from '@d5techs/3dgs-lib';
```

## HTML Canvas 设置

确保 canvas 正确设置：

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    body { margin: 0; overflow: hidden; }
    #canvas { 
      width: 100vw; 
      height: 100vh; 
      display: block; 
    }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <script type="module" src="./main.ts"></script>
</body>
</html>
```

## 常见问题处理

### WebGPU 不支持

```typescript
if (!navigator.gpu) {
  console.error('WebGPU 不支持');
  // 建议用户使用 Chrome 113+ 或 Safari 17+
}
```

### 非 HTTPS 环境

WebGPU 需要安全上下文，确保在以下环境运行：
- `https://` 域名
- `localhost` 或 `127.0.0.1`

### 大模型加载卡顿

对于大型 PLY 模型，使用进度回调显示加载状态：

```typescript
await app.addPLY(url, (progress, stage) => {
  if (stage === 'parse') {
    // 解析阶段可能较慢
    showProgress(`解析中 ${progress.toFixed(0)}%`);
  }
});
```

### 移动端性能

移动端会自动使用优化的渲染器：
- 使用纹理存储数据（~52 bytes/splat）
- 仅支持 L0 SH 模式
- 限制 DPI 为 1.5

## 高级：直接使用底层 API

如果需要更细粒度的控制，可以直接使用底层类：

```typescript
import {
  Renderer,
  Camera,
  OrbitControls,
  GSSplatRenderer,
  GLBLoader,
  MeshRenderer,
} from '@d5techs/3dgs-lib';

// 手动初始化各组件
const renderer = new Renderer(canvas);
await renderer.init();

const camera = new Camera();
const controls = new OrbitControls(camera, canvas);
const meshRenderer = new MeshRenderer(renderer, camera);

// ...自定义渲染循环
```

## 示例代码模板

### 完整的查看器

```typescript
import { App, isWebGPUSupported, GizmoMode } from '@d5techs/3dgs-lib';

class Viewer {
  private app: App;
  
  constructor(canvas: HTMLCanvasElement) {
    this.app = new App(canvas);
  }
  
  async init() {
    if (!isWebGPUSupported()) {
      throw new Error('WebGPU not supported');
    }
    await this.app.init();
    this.app.start();
  }
  
  async loadModel(url: string, onProgress?: (p: number) => void) {
    const ext = url.split('.').pop()?.toLowerCase();
    
    if (ext === 'ply') {
      await this.app.addPLY(url, (progress) => onProgress?.(progress));
    } else if (ext === 'splat') {
      await this.app.addSplat(url, (progress) => onProgress?.(progress));
    } else if (ext === 'glb' || ext === 'gltf') {
      await this.app.addGLB(url);
    } else if (ext === 'obj') {
      await this.app.addOBJ(url);
    }
    
    this.app.frameCurrentModel();
  }
  
  setGizmoMode(mode: 'translate' | 'rotate' | 'scale') {
    const modeMap = {
      translate: GizmoMode.Translate,
      rotate: GizmoMode.Rotate,
      scale: GizmoMode.Scale,
    };
    this.app.setGizmoMode(modeMap[mode]);
  }
  
  destroy() {
    this.app.destroy();
  }
}

// 使用
const viewer = new Viewer(document.getElementById('canvas') as HTMLCanvasElement);
await viewer.init();
await viewer.loadModel('/models/scene.ply', (p) => console.log(`${p}%`));
```

## 依赖关系

此库无运行时依赖，仅需：
- 支持 WebGPU 的浏览器
- TypeScript 5.3+（开发时）

## 版本兼容性

- `@webgpu/types`: >=0.1.40（可选，用于类型提示）
