/**
 * GizmoManager - Gizmo 交互管理器
 * 
 * 负责管理所有 Gizmo 相关的交互：
 * - TransformGizmo（变换控制）
 * - ViewportGizmo（视口坐标轴）
 * - BoundingBoxRenderer（选中对象包围盒）
 */

import { Renderer } from "../core/Renderer";
import { Camera } from "../core/Camera";
import { OrbitControls } from "../core/OrbitControls";
import { ViewportGizmo } from "../core/ViewportGizmo";
import { TransformGizmoV2, TransformableObject, GizmoMode } from "../core/gizmo/TransformGizmoV2";
import { BoundingBoxRenderer, BoundingBox as RendererBoundingBox, BoundingBoxProvider } from "../core/BoundingBoxRenderer";
import { IGSSplatRenderer, BoundingBox as GSBoundingBox } from "../gs/IGSSplatRenderer";
import { Mesh } from "../mesh/Mesh";

/**
 * SplatTransformProxy - PLY/Splat 变换代理对象
 * 实现类似 Mesh 的接口，让 TransformGizmo 可以操作 PLY 模型
 */
export class SplatTransformProxy implements TransformableObject {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];

  private renderer: IGSSplatRenderer;
  private center: [number, number, number];

  constructor(renderer: IGSSplatRenderer, center: [number, number, number]) {
    this.renderer = renderer;
    this.center = [...center];

    // 设置渲染器的 pivot 为包围盒中心
    renderer.setPivot(center[0], center[1], center[2]);

    // 初始化为当前渲染器的变换状态
    const pos = renderer.getPosition();
    const rot = renderer.getRotation();
    const scl = renderer.getScale();

    // Gizmo 位置 = 渲染器位置 + 中心点
    this.position = [
      pos[0] + center[0],
      pos[1] + center[1],
      pos[2] + center[2],
    ];
    this.rotation = [...rot];
    this.scale = [...scl];
  }

  setPosition(x: number, y: number, z: number): void {
    this.position = [x, y, z];
    // 渲染器位置 = Gizmo 位置 - 中心点
    this.renderer.setPosition(
      x - this.center[0],
      y - this.center[1],
      z - this.center[2]
    );
  }

  setRotation(x: number, y: number, z: number): void {
    this.rotation = [x, y, z];
    this.renderer.setRotation(x, y, z);
  }

  setScale(x: number, y: number, z: number): void {
    this.scale = [x, y, z];
    this.renderer.setScale(x, y, z);
  }
}

/**
 * MeshGroupProxy - 多 Mesh 组变换代理对象
 * 让 TransformGizmo 可以同时操作多个 Mesh
 */
export class MeshGroupProxy implements TransformableObject {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];

  private meshes: Mesh[];

  constructor(meshes: Mesh[]) {
    this.meshes = meshes;

    if (meshes.length > 0) {
      const firstMesh = meshes[0];
      this.position = [
        firstMesh.position[0],
        firstMesh.position[1],
        firstMesh.position[2],
      ];
      this.rotation = [
        firstMesh.rotation[0],
        firstMesh.rotation[1],
        firstMesh.rotation[2],
      ];
      this.scale = [
        firstMesh.scale[0],
        firstMesh.scale[1],
        firstMesh.scale[2],
      ];
    } else {
      this.position = [0, 0, 0];
      this.rotation = [0, 0, 0];
      this.scale = [1, 1, 1];
    }
  }

  setPosition(x: number, y: number, z: number): void {
    this.position = [x, y, z];
    for (const mesh of this.meshes) {
      mesh.setPosition(x, y, z);
    }
  }

  setRotation(x: number, y: number, z: number): void {
    this.rotation = [x, y, z];
    for (const mesh of this.meshes) {
      mesh.setRotation(x, y, z);
    }
  }

  setScale(x: number, y: number, z: number): void {
    this.scale = [x, y, z];
    for (const mesh of this.meshes) {
      mesh.setScale(x, y, z);
    }
  }

  getBoundingBox(): RendererBoundingBox | null {
    if (this.meshes.length === 0) return null;

    let combinedMin: [number, number, number] | null = null;
    let combinedMax: [number, number, number] | null = null;

    for (const mesh of this.meshes) {
      const bbox = mesh.getWorldBoundingBox();
      if (!bbox) continue;

      if (combinedMin === null || combinedMax === null) {
        combinedMin = [...bbox.min];
        combinedMax = [...bbox.max];
      } else {
        combinedMin[0] = Math.min(combinedMin[0], bbox.min[0]);
        combinedMin[1] = Math.min(combinedMin[1], bbox.min[1]);
        combinedMin[2] = Math.min(combinedMin[2], bbox.min[2]);
        combinedMax[0] = Math.max(combinedMax[0], bbox.max[0]);
        combinedMax[1] = Math.max(combinedMax[1], bbox.max[1]);
        combinedMax[2] = Math.max(combinedMax[2], bbox.max[2]);
      }
    }

    if (combinedMin === null || combinedMax === null) return null;

    return { min: combinedMin, max: combinedMax };
  }
}

/**
 * SplatBoundingBoxProvider - PLY/Splat 包围盒提供者
 * 动态获取 PLY 的包围盒（考虑变换）
 */
export class SplatBoundingBoxProvider implements BoundingBoxProvider {
  private renderer: IGSSplatRenderer;

  constructor(renderer: IGSSplatRenderer) {
    this.renderer = renderer;
  }

  getBoundingBox(): RendererBoundingBox | null {
    const bbox = this.renderer.getBoundingBox();
    if (!bbox) return null;

    // 获取变换参数
    const position = this.renderer.getPosition();
    const rotation = this.renderer.getRotation();
    const scale = this.renderer.getScale();
    const pivot = this.renderer.getPivot();

    // 获取本地包围盒的 8 个角点
    const corners: [number, number, number][] = [
      [bbox.min[0], bbox.min[1], bbox.min[2]],
      [bbox.max[0], bbox.min[1], bbox.min[2]],
      [bbox.min[0], bbox.max[1], bbox.min[2]],
      [bbox.max[0], bbox.max[1], bbox.min[2]],
      [bbox.min[0], bbox.min[1], bbox.max[2]],
      [bbox.max[0], bbox.min[1], bbox.max[2]],
      [bbox.min[0], bbox.max[1], bbox.max[2]],
      [bbox.max[0], bbox.max[1], bbox.max[2]],
    ];

    // 计算变换矩阵
    const [sx, sy, sz] = scale;
    const [rx, ry, rz] = rotation;
    const [tx, ty, tz] = position;
    const [px, py, pz] = pivot;

    const cx = Math.cos(rx), sx1 = Math.sin(rx);
    const cy = Math.cos(ry), sy1 = Math.sin(ry);
    const cz = Math.cos(rz), sz1 = Math.sin(rz);

    // 组合旋转矩阵 R = Rz * Ry * Rx
    const r00 = cy * cz;
    const r01 = sx1 * sy1 * cz - cx * sz1;
    const r02 = cx * sy1 * cz + sx1 * sz1;
    const r10 = cy * sz1;
    const r11 = sx1 * sy1 * sz1 + cx * cz;
    const r12 = cx * sy1 * sz1 - sx1 * cz;
    const r20 = -sy1;
    const r21 = sx1 * cy;
    const r22 = cx * cy;

    // RS 矩阵 (旋转 * 缩放)
    const rs00 = r00 * sx, rs01 = r01 * sy, rs02 = r02 * sz;
    const rs10 = r10 * sx, rs11 = r11 * sy, rs12 = r12 * sz;
    const rs20 = r20 * sx, rs21 = r21 * sy, rs22 = r22 * sz;

    // 计算 (I - RS) * pivot
    const dpx = px - (rs00 * px + rs01 * py + rs02 * pz);
    const dpy = py - (rs10 * px + rs11 * py + rs12 * pz);
    const dpz = pz - (rs20 * px + rs21 * py + rs22 * pz);

    // 最终平移 = position + (I - RS) * pivot
    const finalTx = tx + dpx;
    const finalTy = ty + dpy;
    const finalTz = tz + dpz;

    // 变换所有角点
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (const [x, y, z] of corners) {
      const wx = rs00 * x + rs01 * y + rs02 * z + finalTx;
      const wy = rs10 * x + rs11 * y + rs12 * z + finalTy;
      const wz = rs20 * x + rs21 * y + rs22 * z + finalTz;

      minX = Math.min(minX, wx);
      minY = Math.min(minY, wy);
      minZ = Math.min(minZ, wz);
      maxX = Math.max(maxX, wx);
      maxY = Math.max(maxY, wy);
      maxZ = Math.max(maxZ, wz);
    }

    return {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    };
  }
}

/**
 * GizmoManager - Gizmo 交互管理器
 */
export class GizmoManager {
  private renderer: Renderer;
  private camera: Camera;
  private canvas: HTMLCanvasElement;
  private controls: OrbitControls;

  private viewportGizmo: ViewportGizmo;
  private transformGizmo: TransformGizmoV2;
  private boundingBoxRenderer: BoundingBoxRenderer;

  // 事件处理函数引用（用于移除监听器）
  private boundOnClick: (e: MouseEvent) => void;
  private boundOnPointerMove: (e: PointerEvent) => void;
  private boundOnPointerDown: (e: PointerEvent) => void;
  private boundOnPointerUp: (e: PointerEvent) => void;

  constructor(
    renderer: Renderer,
    camera: Camera,
    canvas: HTMLCanvasElement,
    controls: OrbitControls
  ) {
    this.renderer = renderer;
    this.camera = camera;
    this.canvas = canvas;
    this.controls = controls;

    // 初始化 Gizmo
    this.viewportGizmo = new ViewportGizmo(renderer, camera, canvas);
    this.transformGizmo = new TransformGizmoV2({ renderer, camera, canvas });
    this.transformGizmo.init();
    this.boundingBoxRenderer = new BoundingBoxRenderer(renderer, camera);

    // 设置 Gizmo 拖拽时禁用 OrbitControls
    this.transformGizmo.setOnDragStateChange((isDragging) => {
      this.controls.enabled = !isDragging;
    });

    // 绑定事件处理函数
    this.boundOnClick = this.onCanvasClick.bind(this);
    this.boundOnPointerMove = this.onPointerMove.bind(this);
    this.boundOnPointerDown = this.onPointerDown.bind(this);
    this.boundOnPointerUp = this.onPointerUp.bind(this);

    this.setupEventListeners();
  }

  /**
   * 设置事件监听器
   */
  private setupEventListeners(): void {
    // 设置视口 Gizmo 轴点击回调
    this.viewportGizmo.setOnAxisClick((axis, positive) => {
      this.controls.setViewAxis(axis, positive, true);
    });

    // 监听点击事件
    this.canvas.addEventListener("click", this.boundOnClick);

    // 添加变换 Gizmo 的指针事件监听器
    this.canvas.addEventListener("pointermove", this.boundOnPointerMove);
    this.canvas.addEventListener("pointerdown", this.boundOnPointerDown);
    this.canvas.addEventListener("pointerup", this.boundOnPointerUp);
  }

  private onCanvasClick(e: MouseEvent): void {
    this.viewportGizmo.handleClick(e.clientX, e.clientY);
  }

  private onPointerMove(e: PointerEvent): void {
    this.transformGizmo.onPointerMove(e);
  }

  private onPointerDown(e: PointerEvent): void {
    this.transformGizmo.onPointerDown(e);
  }

  private onPointerUp(e: PointerEvent): void {
    this.transformGizmo.onPointerUp(e);
  }

  // ============================================
  // 渲染
  // ============================================

  /**
   * 渲染所有 Gizmo
   */
  render(pass: GPURenderPassEncoder): void {
    // 渲染包围盒
    this.boundingBoxRenderer.render(pass);
    // 渲染变换 Gizmo
    this.transformGizmo.render(pass);
    // 渲染视口 Gizmo
    this.viewportGizmo.render(pass);
  }

  // ============================================
  // Transform Gizmo
  // ============================================

  /**
   * 获取变换 Gizmo
   */
  getTransformGizmo(): TransformGizmoV2 {
    return this.transformGizmo;
  }

  /**
   * 设置 Gizmo 模式
   */
  setGizmoMode(mode: GizmoMode): void {
    this.transformGizmo.mode = mode;
  }

  /**
   * 设置 Gizmo 目标对象
   */
  setGizmoTarget(object: TransformableObject | null): void {
    this.transformGizmo.setTarget(object);
  }

  // ============================================
  // Viewport Gizmo
  // ============================================

  /**
   * 获取视口 Gizmo
   */
  getViewportGizmo(): ViewportGizmo {
    return this.viewportGizmo;
  }

  // ============================================
  // Bounding Box
  // ============================================

  /**
   * 获取包围盒渲染器
   */
  getBoundingBoxRenderer(): BoundingBoxRenderer {
    return this.boundingBoxRenderer;
  }

  /**
   * 设置选中对象的包围盒（静态模式）
   */
  setSelectionBoundingBox(box: RendererBoundingBox | null): void {
    this.boundingBoxRenderer.setBoundingBox(box);
  }

  /**
   * 设置选中对象的包围盒提供者（动态模式）
   */
  setSelectionBoundingBoxProvider(provider: BoundingBoxProvider | null): void {
    this.boundingBoxRenderer.setProvider(provider);
  }

  /**
   * 清除选中对象的包围盒
   */
  clearSelectionBoundingBox(): void {
    this.boundingBoxRenderer.clear();
  }

  // ============================================
  // 代理对象创建
  // ============================================

  /**
   * 创建 Splat 变换代理
   */
  createSplatTransformProxy(renderer: IGSSplatRenderer): SplatTransformProxy | null {
    const bbox = renderer.getBoundingBox();
    if (!bbox) return null;
    return new SplatTransformProxy(renderer, bbox.center);
  }

  /**
   * 创建 Mesh 组变换代理
   */
  createMeshGroupProxy(meshes: Mesh[]): MeshGroupProxy | null {
    if (meshes.length === 0) return null;
    return new MeshGroupProxy(meshes);
  }

  /**
   * 创建 Splat 包围盒提供者
   */
  createSplatBoundingBoxProvider(renderer: IGSSplatRenderer): SplatBoundingBoxProvider {
    return new SplatBoundingBoxProvider(renderer);
  }

  /**
   * 销毁
   */
  destroy(): void {
    // 移除事件监听器
    this.canvas.removeEventListener("click", this.boundOnClick);
    this.canvas.removeEventListener("pointermove", this.boundOnPointerMove);
    this.canvas.removeEventListener("pointerdown", this.boundOnPointerDown);
    this.canvas.removeEventListener("pointerup", this.boundOnPointerUp);

    // 销毁 Gizmo
    this.transformGizmo.destroy();
    this.boundingBoxRenderer.destroy();
  }
}
