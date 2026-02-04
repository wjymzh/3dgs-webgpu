/**
 * App - 统一调度入口
 * 
 * 重构后的职责：
 * - 初始化和协调各子系统
 * - 渲染循环管理
 * - 模型加载（委托给加载器）
 * - 对外提供简洁的 API
 * 
 * 场景管理委托给 SceneManager
 * Gizmo 交互委托给 GizmoManager
 */

import { Renderer } from "./core/Renderer";
import { Camera } from "./core/Camera";
import { OrbitControls } from "./core/OrbitControls";
import { MeshRenderer } from "./mesh/MeshRenderer";
import { GLBLoader } from "./loaders/GLBLoader";
import { OBJLoader } from "./loaders/OBJLoader";
import { Mesh } from "./mesh/Mesh";
import { GSSplatRenderer } from "./gs/GSSplatRenderer";
import { GSSplatRendererV2 } from "./gs/GSSplatRendererV2";
import { GSSplatRendererMobile } from "./gs/GSSplatRendererMobile";
import { IGSSplatRenderer, BoundingBox } from "./gs/IGSSplatRenderer";
import { deserializeSplat } from "./gs/SplatLoader";
import { SceneManager } from "./scene/SceneManager";
import { 
  GizmoManager, 
  SplatTransformProxy, 
  MeshGroupProxy, 
  SplatBoundingBoxProvider 
} from "./interaction/GizmoManager";
import { TransformableObject, GizmoMode } from "./core/gizmo/TransformGizmoV2";
import { BoundingBoxProvider } from "./core/BoundingBoxRenderer";

// 重新导出代理类以保持向后兼容
export { SplatTransformProxy, MeshGroupProxy, SplatBoundingBoxProvider };

/**
 * 统一进度回调类型
 * @param progress 进度值 0-100
 * @param stage 当前阶段: 'download' | 'parse' | 'upload'
 */
export type ProgressCallback = (progress: number, stage: 'download' | 'parse' | 'upload') => void;

/**
 * 检测是否为移动设备
 */
function isMobileDevice(): boolean {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent || navigator.vendor || (window as any).opera || "";
  const isMobileUA = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(ua.toLowerCase());
  const hasTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0;
  const isSmallScreen = window.innerWidth <= 768;
  const isIPadAsMac = navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
  return isMobileUA || isIPadAsMac || (hasTouch && isSmallScreen);
}

/**
 * App - 统一调度入口
 */
export class App {
  private canvas: HTMLCanvasElement;
  private renderer!: Renderer;
  private camera!: Camera;
  private controls!: OrbitControls;
  private meshRenderer!: MeshRenderer;
  private glbLoader!: GLBLoader;
  private objLoader!: OBJLoader;

  // 子系统管理器
  private sceneManager!: SceneManager;
  private gizmoManager!: GizmoManager;

  private isRunning: boolean = false;
  private animationId: number = 0;

  // 是否使用移动端渲染器
  private useMobileRenderer: boolean = false;

  // 绑定的事件处理函数
  private boundOnResize: () => void;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.boundOnResize = this.onResize.bind(this);
  }

  /**
   * 初始化应用
   */
  async init(): Promise<void> {
    // 初始化渲染器
    this.renderer = new Renderer(this.canvas);
    await this.renderer.init();

    // 初始化相机
    this.camera = new Camera();
    this.camera.setAspect(this.renderer.getAspectRatio());

    // 初始化控制器
    this.controls = new OrbitControls(this.camera, this.canvas);

    // 初始化网格渲染器
    this.meshRenderer = new MeshRenderer(this.renderer, this.camera);

    // 初始化加载器
    this.glbLoader = new GLBLoader(this.renderer.device);
    this.objLoader = new OBJLoader(this.renderer.device);

    // 初始化场景管理器
    this.sceneManager = new SceneManager(this.meshRenderer);

    // 初始化 Gizmo 管理器
    this.gizmoManager = new GizmoManager(
      this.renderer,
      this.camera,
      this.canvas,
      this.controls
    );

    // 监听窗口大小变化
    window.addEventListener("resize", this.boundOnResize);
  }

  // ============================================
  // 模型加载
  // ============================================

  /**
   * 加载 GLB 文件
   */
  async addGLB(url: string): Promise<number> {
    try {
      const loadedMeshes = await this.glbLoader.load(url);
      for (const { mesh, material } of loadedMeshes) {
        this.meshRenderer.addMesh(mesh, material);
      }
      return loadedMeshes.length;
    } catch (error) {
      throw error;
    }
  }

  /**
   * 加载 OBJ 文件
   */
  async addOBJ(url: string): Promise<Mesh[]> {
    try {
      const loadedMeshes = await this.objLoader.load(url);
      const meshes: Mesh[] = [];
      for (const { mesh, material } of loadedMeshes) {
        this.meshRenderer.addMesh(mesh, material);
        meshes.push(mesh);
      }
      return meshes;
    } catch (error) {
      throw error;
    }
  }

  /**
   * 加载 PLY 文件 (3D Gaussian Splatting)
   */
  async addPLY(
    urlOrBuffer: string | ArrayBuffer,
    onProgress?: ProgressCallback,
    isLocalFile: boolean = false,
  ): Promise<number> {
    try {
      const isMobile = isMobileDevice();
      let buffer: ArrayBuffer;

      // 下载阶段 (0-50%)
      if (typeof urlOrBuffer === 'string') {
        buffer = await this.fetchWithProgress(urlOrBuffer, (downloadProgress) => {
          if (onProgress) {
            onProgress(downloadProgress * 0.5, 'download');
          }
        });
      } else {
        buffer = urlOrBuffer;
        if (onProgress && isLocalFile) {
          onProgress(50, 'download');
        }
      }

      // 解析阶段 (50-90%)
      const parseProgressCallback = (loaded: number, total: number) => {
        if (onProgress) {
          const parseProgress = (loaded / total) * 40;
          onProgress(50 + parseProgress, 'parse');
        }
      };

      let gsRenderer: IGSSplatRenderer;

      if (isMobile) {
        gsRenderer = new GSSplatRendererMobile(this.renderer, this.camera);
        this.useMobileRenderer = true;

        const compactData = await this.parsePLYBuffer(buffer, {
          maxSplats: Infinity,
          loadSH: false,
          onProgress: parseProgressCallback,
        });

        if (onProgress) onProgress(90, 'upload');
        gsRenderer.setCompactData(compactData);
        if (onProgress) onProgress(100, 'upload');

        this.sceneManager.setGSRenderer(gsRenderer);
        return compactData.count;
      } else {
        // 桌面端使用优化的 V2 渲染器
        gsRenderer = new GSSplatRendererV2(this.renderer, this.camera);
        this.useMobileRenderer = false;

        const compactData = await this.parsePLYBuffer(buffer, {
          maxSplats: Infinity,
          loadSH: true,
          onProgress: parseProgressCallback,
        });

        if (onProgress) onProgress(90, 'upload');
        gsRenderer.setCompactData(compactData);
        if (onProgress) onProgress(100, 'upload');

        this.sceneManager.setGSRenderer(gsRenderer);
        return compactData.count;
      }
    } catch (error) {
      throw error;
    }
  }

  /**
   * 加载 Splat 文件
   */
  async addSplat(
    urlOrBuffer: string | ArrayBuffer,
    onProgress?: ProgressCallback,
    isLocalFile: boolean = false,
  ): Promise<number> {
    try {
      let buffer: ArrayBuffer;

      if (typeof urlOrBuffer === 'string') {
        buffer = await this.fetchWithProgress(urlOrBuffer, (downloadProgress) => {
          if (onProgress) {
            onProgress(downloadProgress * 0.5, 'download');
          }
        });
      } else {
        buffer = urlOrBuffer;
        if (onProgress && isLocalFile) {
          onProgress(50, 'download');
        }
      }

      if (onProgress) onProgress(50, 'parse');
      const splats = deserializeSplat(buffer);
      if (onProgress) onProgress(90, 'parse');

      if (onProgress) onProgress(90, 'upload');
      // 使用优化的 V2 渲染器
      const gsRenderer = new GSSplatRendererV2(this.renderer, this.camera);
      gsRenderer.setData(splats);
      this.sceneManager.setGSRenderer(gsRenderer);
      this.useMobileRenderer = false;
      if (onProgress) onProgress(100, 'upload');

      return splats.length;
    } catch (error) {
      throw error;
    }
  }

  /**
   * 添加测试立方体
   */
  addTestCube(): void {
    const { mesh, material } = this.glbLoader.createTestCube();
    this.meshRenderer.addMesh(mesh, material);
  }

  /**
   * 添加测试球体
   */
  addTestSphere(): void {
    const { mesh, material } = this.glbLoader.createTestSphere();
    this.meshRenderer.addMesh(mesh, material);
  }

  // ============================================
  // 渲染循环
  // ============================================

  /**
   * 开始渲染循环
   */
  start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.animate();
  }

  /**
   * 停止渲染循环
   */
  stop(): void {
    this.isRunning = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = 0;
    }
  }

  private animate(): void {
    if (!this.isRunning) return;
    this.render();
    this.animationId = requestAnimationFrame(this.animate.bind(this));
  }

  private render(): void {
    this.camera.setAspect(this.renderer.getAspectRatio());
    this.camera.updateMatrix();

    const pass = this.renderer.beginFrame();

    // 渲染 3D Gaussian Splatting
    const gsRenderer = this.sceneManager.getGSRenderer();
    if (gsRenderer) {
      gsRenderer.render(pass);
    }

    // 渲染网格
    this.meshRenderer.render(pass);

    // 渲染 Gizmo
    this.gizmoManager.render(pass);

    this.renderer.endFrame();
  }

  private onResize(): void {
    this.camera.setAspect(this.renderer.getAspectRatio());
    this.camera.updateMatrix();
  }

  // ============================================
  // 场景管理（委托给 SceneManager）
  // ============================================

  getMeshCount(): number {
    return this.sceneManager.getMeshCount();
  }

  getMeshByIndex(index: number): Mesh | null {
    return this.sceneManager.getMeshByIndex(index);
  }

  getMeshRange(startIndex: number, count: number): Mesh[] {
    return this.sceneManager.getMeshRange(startIndex, count);
  }

  clearMeshes(): void {
    this.sceneManager.clearMeshes();
  }

  removeMeshByIndex(index: number): boolean {
    const result = this.sceneManager.removeMeshByIndex(index);
    return result;
  }

  getSplatCount(): number {
    return this.sceneManager.getSplatCount();
  }

  clearSplats(): void {
    this.sceneManager.clearSplats();
    this.useMobileRenderer = false;
  }

  // ============================================
  // Splat 变换（委托给 SceneManager）
  // ============================================

  setSplatPosition(x: number, y: number, z: number): void {
    this.sceneManager.setSplatPosition(x, y, z);
  }

  getSplatPosition(): [number, number, number] | null {
    return this.sceneManager.getSplatPosition();
  }

  setSplatRotation(x: number, y: number, z: number): void {
    this.sceneManager.setSplatRotation(x, y, z);
  }

  getSplatRotation(): [number, number, number] | null {
    return this.sceneManager.getSplatRotation();
  }

  setSplatScale(x: number, y: number, z: number): void {
    this.sceneManager.setSplatScale(x, y, z);
  }

  getSplatScale(): [number, number, number] | null {
    return this.sceneManager.getSplatScale();
  }

  // ============================================
  // SH 模式
  // ============================================

  setSHMode(mode: 0 | 1 | 2 | 3): void {
    this.sceneManager.setSHMode(mode);
  }

  getSHMode(): number {
    return this.sceneManager.getSHMode();
  }

  // ============================================
  // Bounding Box
  // ============================================

  getSplatBoundingBox(): BoundingBox | null {
    return this.sceneManager.getSplatBoundingBox();
  }

  getMeshRangeBoundingBox(startIndex: number, count: number): BoundingBox | null {
    return this.sceneManager.getMeshRangeBoundingBox(startIndex, count);
  }

  // ============================================
  // 材质颜色
  // ============================================

  getMeshColor(index: number): [number, number, number, number] | null {
    return this.sceneManager.getMeshColor(index);
  }

  setMeshColor(index: number, r: number, g: number, b: number, a: number = 1): boolean {
    return this.sceneManager.setMeshColor(index, r, g, b, a);
  }

  setMeshRangeColor(startIndex: number, count: number, r: number, g: number, b: number, a: number = 1): number {
    return this.sceneManager.setMeshRangeColor(startIndex, count, r, g, b, a);
  }

  // ============================================
  // 相机控制
  // ============================================

  frameCurrentModel(animate: boolean = true): boolean {
    const bbox = this.sceneManager.getSceneBoundingBox();
    if (!bbox) {
      return false;
    }

    this.controls.frameModel(bbox.center, bbox.radius, animate);
    return true;
  }

  // ============================================
  // Gizmo（委托给 GizmoManager）
  // ============================================

  getTransformGizmo() {
    return this.gizmoManager.getTransformGizmo();
  }

  getViewportGizmo() {
    return this.gizmoManager.getViewportGizmo();
  }

  getBoundingBoxRenderer() {
    return this.gizmoManager.getBoundingBoxRenderer();
  }

  setGizmoMode(mode: GizmoMode): void {
    this.gizmoManager.setGizmoMode(mode);
  }

  setGizmoTarget(object: TransformableObject | null): void {
    this.gizmoManager.setGizmoTarget(object);
  }

  setSelectionBoundingBox(box: BoundingBox | null): void {
    this.gizmoManager.setSelectionBoundingBox(box);
  }

  setSelectionBoundingBoxProvider(provider: BoundingBoxProvider | null): void {
    this.gizmoManager.setSelectionBoundingBoxProvider(provider);
  }

  clearSelectionBoundingBox(): void {
    this.gizmoManager.clearSelectionBoundingBox();
  }

  /**
   * 创建 Mesh 组的变换代理
   */
  createMeshGroupProxy(startIndex: number, count: number): MeshGroupProxy | null {
    const meshes = this.sceneManager.getMeshRange(startIndex, count);
    return this.gizmoManager.createMeshGroupProxy(meshes);
  }

  /**
   * 获取 Splat 的变换代理
   */
  getSplatTransformProxy(): SplatTransformProxy | null {
    const gsRenderer = this.sceneManager.getGSRenderer();
    if (!gsRenderer) return null;
    return this.gizmoManager.createSplatTransformProxy(gsRenderer);
  }

  /**
   * 创建 Splat 包围盒提供者
   */
  createSplatBoundingBoxProvider(): SplatBoundingBoxProvider | null {
    const gsRenderer = this.sceneManager.getGSRenderer();
    if (!gsRenderer) return null;
    return this.gizmoManager.createSplatBoundingBoxProvider(gsRenderer);
  }

  // ============================================
  // 子系统访问
  // ============================================

  getRenderer(): Renderer {
    return this.renderer;
  }

  getCamera(): Camera {
    return this.camera;
  }

  getControls(): OrbitControls {
    return this.controls;
  }

  getMeshRenderer(): MeshRenderer {
    return this.meshRenderer;
  }

  getGSRenderer(): GSSplatRendererV2 | undefined {
    const renderer = this.sceneManager.getGSRenderer();
    if (renderer && !this.useMobileRenderer) {
      return renderer as GSSplatRendererV2;
    }
    return undefined;
  }

  getGSRendererMobile(): GSSplatRendererMobile | undefined {
    const renderer = this.sceneManager.getGSRenderer();
    if (renderer && this.useMobileRenderer) {
      return renderer as GSSplatRendererMobile;
    }
    return undefined;
  }

  isUsingMobileRenderer(): boolean {
    return this.useMobileRenderer;
  }

  // ============================================
  // 内部方法
  // ============================================

  private async fetchWithProgress(
    url: string,
    onProgress?: (progress: number) => void
  ): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`无法加载文件: ${url}`);
    }

    const contentLength = response.headers.get('content-length');
    if (!contentLength || !response.body) {
      const buffer = await response.arrayBuffer();
      if (onProgress) onProgress(100);
      return buffer;
    }

    const total = parseInt(contentLength, 10);
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      loaded += value.length;

      if (onProgress) {
        onProgress((loaded / total) * 100);
      }
    }

    const buffer = new ArrayBuffer(loaded);
    const view = new Uint8Array(buffer);
    let offset = 0;
    for (const chunk of chunks) {
      view.set(chunk, offset);
      offset += chunk.length;
    }

    return buffer;
  }

  private async parsePLYBuffer(
    buffer: ArrayBuffer,
    options: { maxSplats?: number; loadSH?: boolean; onProgress?: (loaded: number, total: number) => void }
  ): Promise<import('./gs/PLYLoaderMobile').CompactSplatData> {
    const { parsePLYBuffer } = await import('./gs/PLYLoaderMobile');
    return parsePLYBuffer(buffer, options);
  }

  /**
   * 销毁应用及所有资源
   */
  destroy(): void {
    this.stop();
    window.removeEventListener("resize", this.boundOnResize);

    this.sceneManager.destroy();
    this.gizmoManager.destroy();

    if (this.meshRenderer) {
      this.meshRenderer.destroy();
    }

    if (this.controls) {
      this.controls.destroy();
    }

    if (this.renderer) {
      this.renderer.destroy();
    }
  }
}
