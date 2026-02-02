# WebGPU 3D Gaussian Splatting 渲染引擎 - 使用手册

本手册详细介绍如何使用 webgpu-3dgs 渲染引擎，包括基础用法、高级功能和 API 参考。

## 目录

- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [基础用法](#基础用法)
- [3D Gaussian Splatting](#3d-gaussian-splatting)
- [网格渲染](#网格渲染)
- [相机与控制器](#相机与控制器)
- [变换 Gizmo](#变换-gizmo)
- [场景管理](#场景管理)
- [移动端优化](#移动端优化)
- [API 参考](#api-参考)

---

## 快速开始

### 安装

```bash
# 使用 yarn
yarn add webgpu-3dgs

# 或使用 npm
npm install webgpu-3dgs
```

### 最简示例

```typescript
import { App } from 'webgpu-3dgs';

async function main() {
  // 获取 canvas 元素
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  
  // 创建应用实例
  const app = new App(canvas);
  
  // 初始化（必须 await）
  await app.init();
  
  // 加载 3D Gaussian Splatting 模型
  await app.addPLY('path/to/model.ply');
  
  // 自动调整相机到模型
  app.frameCurrentModel();
  
  // 启动渲染循环
  app.start();
}

main();
```

---

## 核心概念

### 架构概览

引擎采用模块化设计，主要组件包括：

```
App (统一调度入口)
├── Renderer (WebGPU 渲染器)
├── Camera (相机)
├── OrbitControls (轨道控制器)
├── MeshRenderer (网格渲染器)
├── GSSplatRenderer (3DGS 渲染器)
├── SceneManager (场景管理)
└── GizmoManager (Gizmo 交互)
```

### 坐标系统

- 使用右手坐标系
- Y 轴向上
- Z 轴指向观察者

---

## 基础用法

### 初始化应用

```typescript
import { App } from 'webgpu-3dgs';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const app = new App(canvas);

// 初始化 WebGPU
await app.init();

// 设置背景颜色
app.getRenderer().setClearColorHex('#1a1a26');

// 启动渲染
app.start();
```

### 加载模型

引擎支持多种 3D 格式：

```typescript
// 加载 PLY 文件 (3D Gaussian Splatting)
const splatCount = await app.addPLY('model.ply');

// 加载 Splat 文件 (紧凑格式)
const splatCount = await app.addSplat('model.splat');

// 加载 GLB/GLTF 模型
const meshCount = await app.addGLB('model.glb');

// 加载 OBJ 模型
const meshes = await app.addOBJ('model.obj');
```

### 带进度回调的加载

```typescript
await app.addPLY('large-model.ply', (progress, stage) => {
  console.log(`${stage}: ${progress.toFixed(1)}%`);
  // stage: 'download' | 'parse' | 'upload'
});
```

### 添加测试几何体

```typescript
// 添加立方体
app.addTestCube();

// 添加球体
app.addTestSphere();
```

---

## 3D Gaussian Splatting

### 球谐函数 (SH) 模式

引擎支持 4 种 SH 模式，影响渲染质量和性能：

| 模式 | 说明 | 性能 | 质量 |
|------|------|------|------|
| L0 | 仅 DC 颜色 | 最快 | 基础 |
| L1 | DC + L1 SH | 快 | 良好 |
| L2 | DC + L1 + L2 SH | 中等 | 高 |
| L3 | 完整 SH | 较慢 | 最高 |

```typescript
// 设置 SH 模式 (0-3)
app.setSHMode(2);

// 获取当前 SH 模式
const mode = app.getSHMode();
```

### Splat 变换

```typescript
// 设置位置
app.setSplatPosition(1, 0, 0);

// 设置旋转（欧拉角，弧度）
app.setSplatRotation(0, Math.PI / 4, 0);

// 设置缩放
app.setSplatScale(1.5, 1.5, 1.5);

// 获取当前变换
const pos = app.getSplatPosition();
const rot = app.getSplatRotation();
const scale = app.getSplatScale();
```

### 获取包围盒

```typescript
const bbox = app.getSplatBoundingBox();
if (bbox) {
  console.log('中心:', bbox.center);
  console.log('半径:', bbox.radius);
  console.log('最小点:', bbox.min);
  console.log('最大点:', bbox.max);
}
```

---

## 网格渲染

### 网格管理

```typescript
// 获取网格数量
const count = app.getMeshCount();

// 获取指定索引的网格
const mesh = app.getMeshByIndex(0);

// 获取指定范围的网格
const meshes = app.getMeshRange(0, 5);

// 移除指定索引的网格
app.removeMeshByIndex(0);

// 清空所有网格
app.clearMeshes();
```

### 材质颜色

```typescript
// 获取网格颜色 [r, g, b, a]
const color = app.getMeshColor(0);

// 设置网格颜色
app.setMeshColor(0, 1.0, 0.5, 0.2, 1.0);

// 批量设置颜色
app.setMeshRangeColor(0, 5, 0.8, 0.2, 0.2, 1.0);
```

### 网格变换

```typescript
const mesh = app.getMeshByIndex(0);
if (mesh) {
  // 设置位置
  mesh.setPosition(1, 0, 0);
  
  // 设置旋转（欧拉角，弧度）
  mesh.setRotation(0, Math.PI / 2, 0);
  
  // 设置缩放
  mesh.setScale(2, 2, 2);
  
  // 重置变换
  mesh.resetTransform();
}
```

---

## 相机与控制器

### 相机参数

```typescript
const camera = app.getCamera();

// 设置视野角度（弧度）
camera.fov = Math.PI / 4;  // 45度

// 设置近/远裁剪面
camera.near = 0.001;
camera.far = 10000;

// 设置宽高比
camera.setAspect(16 / 9);

// 更新矩阵
camera.updateMatrix();
```

### 轨道控制器

```typescript
const controls = app.getControls();

// 设置距离
controls.distance = 5;

// 设置旋转角度
controls.theta = Math.PI / 4;  // 水平角
controls.phi = Math.PI / 4;    // 垂直角

// 设置目标点
controls.setTarget(0, 1, 0);

// 设置灵敏度
controls.rotateSpeed = 0.005;
controls.zoomSpeed = 0.001;
controls.panSpeed = 0.005;

// 启用/禁用控制
controls.enabled = true;

// 更新相机
controls.update();
```

### 视图切换

```typescript
const controls = app.getControls();

// 切换到标准视图
controls.setViewAxis('X', true);   // 从 +X 方向看
controls.setViewAxis('Y', false);  // 从 -Y 方向看
controls.setViewAxis('Z', true);   // 从 +Z 方向看
```

### 自动对焦模型

```typescript
// 自动调整相机到模型（带动画）
app.frameCurrentModel(true);

// 无动画
app.frameCurrentModel(false);
```

---

## 变换 Gizmo

### Gizmo 模式

```typescript
import { GizmoMode } from 'webgpu-3dgs';

// 设置 Gizmo 模式
app.setGizmoMode(GizmoMode.Translate);  // 平移
app.setGizmoMode(GizmoMode.Rotate);     // 旋转
app.setGizmoMode(GizmoMode.Scale);      // 缩放
```

### 设置 Gizmo 目标

```typescript
// 为 Splat 创建变换代理
const splatProxy = app.getSplatTransformProxy();
if (splatProxy) {
  app.setGizmoTarget(splatProxy);
}

// 为 Mesh 组创建变换代理
const meshProxy = app.createMeshGroupProxy(0, 5);
if (meshProxy) {
  app.setGizmoTarget(meshProxy);
}

// 清除目标
app.setGizmoTarget(null);
```

### 包围盒显示

```typescript
// 设置静态包围盒
app.setSelectionBoundingBox({
  min: [-1, -1, -1],
  max: [1, 1, 1]
});

// 设置动态包围盒提供者
const provider = app.createSplatBoundingBoxProvider();
app.setSelectionBoundingBoxProvider(provider);

// 清除包围盒
app.clearSelectionBoundingBox();
```

### 直接访问 Gizmo

```typescript
// 获取变换 Gizmo
const transformGizmo = app.getTransformGizmo();
transformGizmo.snap = true;
transformGizmo.snapIncrement = 0.5;

// 获取视口 Gizmo
const viewportGizmo = app.getViewportGizmo();
viewportGizmo.setSize(150);
```

---

## 场景管理

### 清空场景

```typescript
// 清空所有网格
app.clearMeshes();

// 清空所有 Splat
app.clearSplats();
```

### 获取场景信息

```typescript
// 获取网格数量
const meshCount = app.getMeshCount();

// 获取 Splat 数量
const splatCount = app.getSplatCount();

// 检查是否使用移动端渲染器
const isMobile = app.isUsingMobileRenderer();
```

---

## 移动端优化

引擎会自动检测移动设备并启用优化：

- 使用紧凑数据格式减少内存占用
- 限制 DPI 避免 GPU 过载
- 默认使用 L0 SH 模式提升性能

### 手动检测

```typescript
// 检查是否使用移动端渲染器
if (app.isUsingMobileRenderer()) {
  console.log('使用移动端优化渲染器');
}

// 获取移动端渲染器
const mobileRenderer = app.getGSRendererMobile();
```

---

## API 参考

### App 类

#### 初始化

| 方法 | 说明 |
|------|------|
| `constructor(canvas)` | 创建应用实例 |
| `init()` | 初始化 WebGPU |
| `start()` | 启动渲染循环 |
| `stop()` | 停止渲染循环 |
| `destroy()` | 销毁应用及资源 |

#### 模型加载

| 方法 | 说明 |
|------|------|
| `addPLY(url, onProgress?)` | 加载 PLY 文件 |
| `addSplat(url, onProgress?)` | 加载 Splat 文件 |
| `addGLB(url)` | 加载 GLB 文件 |
| `addOBJ(url)` | 加载 OBJ 文件 |
| `addTestCube()` | 添加测试立方体 |
| `addTestSphere()` | 添加测试球体 |

#### 场景管理

| 方法 | 说明 |
|------|------|
| `getMeshCount()` | 获取网格数量 |
| `getMeshByIndex(index)` | 获取指定网格 |
| `getMeshRange(start, count)` | 获取网格范围 |
| `removeMeshByIndex(index)` | 移除指定网格 |
| `clearMeshes()` | 清空所有网格 |
| `getSplatCount()` | 获取 Splat 数量 |
| `clearSplats()` | 清空所有 Splat |

#### Splat 变换

| 方法 | 说明 |
|------|------|
| `setSplatPosition(x, y, z)` | 设置位置 |
| `getSplatPosition()` | 获取位置 |
| `setSplatRotation(x, y, z)` | 设置旋转 |
| `getSplatRotation()` | 获取旋转 |
| `setSplatScale(x, y, z)` | 设置缩放 |
| `getSplatScale()` | 获取缩放 |

#### SH 模式

| 方法 | 说明 |
|------|------|
| `setSHMode(mode)` | 设置 SH 模式 (0-3) |
| `getSHMode()` | 获取当前 SH 模式 |

#### 相机控制

| 方法 | 说明 |
|------|------|
| `frameCurrentModel(animate?)` | 自动对焦模型 |
| `getCamera()` | 获取相机实例 |
| `getControls()` | 获取控制器实例 |

#### Gizmo

| 方法 | 说明 |
|------|------|
| `setGizmoMode(mode)` | 设置 Gizmo 模式 |
| `setGizmoTarget(object)` | 设置 Gizmo 目标 |
| `getTransformGizmo()` | 获取变换 Gizmo |
| `getViewportGizmo()` | 获取视口 Gizmo |
| `createMeshGroupProxy(start, count)` | 创建 Mesh 组代理 |
| `getSplatTransformProxy()` | 获取 Splat 变换代理 |

#### 子系统访问

| 方法 | 说明 |
|------|------|
| `getRenderer()` | 获取渲染器 |
| `getMeshRenderer()` | 获取网格渲染器 |
| `getGSRenderer()` | 获取 GS 渲染器 |
| `getGSRendererMobile()` | 获取移动端 GS 渲染器 |

---

## 交互控制

### 鼠标操作

| 操作 | 功能 |
|------|------|
| 左键拖拽 | 旋转视角 |
| 右键拖拽 | 平移视角 |
| 滚轮 | 缩放 |
| 点击 Gizmo 轴 | 切换正交视图 |

### 触摸操作

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

## 浏览器支持

| 浏览器 | 最低版本 |
|--------|----------|
| Chrome | 113+ |
| Edge | 113+ |
| Safari | 17+ |
| Firefox | 实验性支持 |

> 注意：WebGPU 需要在 HTTPS 或 localhost 环境下运行。

---

## 常见问题

### Q: 为什么模型加载后看不到？

A: 尝试调用 `app.frameCurrentModel()` 自动调整相机位置。

### Q: 如何提升移动端性能？

A: 引擎会自动检测移动设备并启用优化。你也可以手动设置 `app.setSHMode(0)` 使用最快的渲染模式。

### Q: 支持哪些 PLY 格式？

A: 支持 `binary_little_endian` 和 `binary_big_endian` 格式，不支持 ASCII 格式。

### Q: 如何处理大型模型？

A: 使用带进度回调的加载方法，并考虑在移动端限制 Splat 数量。

---

## 许可证

MIT License
