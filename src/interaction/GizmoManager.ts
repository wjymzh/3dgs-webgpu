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
import type { IGSSplatRenderer } from "../gs/IGSSplatRenderer";
import type { Mesh } from "../mesh/Mesh";

// 从 scene/proxies 导入代理类
import { 
  SplatTransformProxy, 
  MeshGroupProxy, 
  SplatBoundingBoxProvider 
} from "../scene/proxies";

// 重新导出代理类保持向后兼容
export { SplatTransformProxy, MeshGroupProxy, SplatBoundingBoxProvider };

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
