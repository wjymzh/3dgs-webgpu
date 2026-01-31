import { Renderer } from "./core/Renderer";
import { Camera } from "./core/Camera";
import { OrbitControls } from "./core/OrbitControls";
import { ViewportGizmo } from "./core/ViewportGizmo";
import { TransformGizmo, TransformableObject } from "./core/gizmo/TransformGizmo";
import { GizmoMode } from "./core/gizmo/GizmoAxis";
import { MeshRenderer } from "./mesh/MeshRenderer";
import { GLBLoader } from "./loaders/GLBLoader";
import { Mesh } from "./mesh/Mesh";
import { GSSplatRenderer, PerformanceTier } from "./gs/GSSplatRenderer";
import { GSSplatRendererMobile } from "./gs/GSSplatRendererMobile";
import { loadPLYMobile } from "./gs/PLYLoaderMobile";
import { loadSplat } from "./gs/SplatLoader";

/**
 * SplatTransformProxy - PLY/Splat 变换代理对象
 * 实现类似 Mesh 的接口，让 TransformGizmo 可以操作 PLY 模型
 */
export class SplatTransformProxy {
  // 位置、旋转、缩放 - 使用数组以匹配 Mesh 接口
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];

  // 内部引用渲染器
  private renderer: GSSplatRenderer | GSSplatRendererMobile;
  // 原始中心点（用于计算相对位移）
  private originalCenter: [number, number, number];

  constructor(
    renderer: GSSplatRenderer | GSSplatRendererMobile,
    center: [number, number, number]
  ) {
    this.renderer = renderer;
    this.originalCenter = [...center];

    // 初始化为当前渲染器的变换状态
    const pos = renderer.getPosition();
    const rot = renderer.getRotation();
    const scl = renderer.getScale();

    // 位置需要加上原始中心点（因为渲染器的位置是相对于原点的）
    this.position = [
      pos[0] + center[0],
      pos[1] + center[1],
      pos[2] + center[2],
    ];
    this.rotation = [...rot];
    this.scale = [...scl];
  }

  /**
   * 设置位置（Gizmo 会调用这个方法）
   */
  setPosition(x: number, y: number, z: number): void {
    this.position = [x, y, z];
    // 计算相对于原始中心的位移
    this.renderer.setPosition(
      x - this.originalCenter[0],
      y - this.originalCenter[1],
      z - this.originalCenter[2]
    );
  }

  /**
   * 设置旋转（Gizmo 会调用这个方法）
   */
  setRotation(x: number, y: number, z: number): void {
    this.rotation = [x, y, z];
    this.renderer.setRotation(x, y, z);
  }

  /**
   * 设置缩放（Gizmo 会调用这个方法）
   */
  setScale(x: number, y: number, z: number): void {
    this.scale = [x, y, z];
    this.renderer.setScale(x, y, z);
  }
}

/**
 * MeshGroupProxy - 多 Mesh 组变换代理对象
 * 让 TransformGizmo 可以同时操作多个 Mesh（如 GLB 模型的所有部件）
 */
export class MeshGroupProxy implements TransformableObject {
  // 位置、旋转、缩放
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];

  // 内部引用的 mesh 数组
  private meshes: Mesh[];

  constructor(meshes: Mesh[]) {
    this.meshes = meshes;

    // 初始化为第一个 mesh 的变换状态（假设组内所有 mesh 初始变换一致）
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

  /**
   * 设置位置（Gizmo 会调用这个方法）- 同步更新所有 mesh
   */
  setPosition(x: number, y: number, z: number): void {
    this.position = [x, y, z];
    for (const mesh of this.meshes) {
      mesh.setPosition(x, y, z);
    }
  }

  /**
   * 设置旋转（Gizmo 会调用这个方法）- 同步更新所有 mesh
   */
  setRotation(x: number, y: number, z: number): void {
    this.rotation = [x, y, z];
    for (const mesh of this.meshes) {
      mesh.setRotation(x, y, z);
    }
  }

  /**
   * 设置缩放（Gizmo 会调用这个方法）- 同步更新所有 mesh
   */
  setScale(x: number, y: number, z: number): void {
    this.scale = [x, y, z];
    for (const mesh of this.meshes) {
      mesh.setScale(x, y, z);
    }
  }
}

/**
 * 检测是否为移动设备
 */
function isMobileDevice(): boolean {
  if (typeof navigator === "undefined") return false;
  const ua =
    navigator.userAgent || navigator.vendor || (window as any).opera || "";
  const isMobileUA =
    /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(
      ua.toLowerCase(),
    );
  const hasTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0;
  const isSmallScreen = window.innerWidth <= 768;
  const isIPadAsMac =
    navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
  return isMobileUA || isIPadAsMac || (hasTouch && isSmallScreen);
}

/**
 * App - 统一调度入口
 * 管理 Renderer、Camera、Controls、MeshRenderer
 * 未来扩展点：GSSplatRenderer
 */
export class App {
  private canvas: HTMLCanvasElement;
  private renderer!: Renderer;
  private camera!: Camera;
  private controls!: OrbitControls;
  private meshRenderer!: MeshRenderer;
  private glbLoader!: GLBLoader;
  private viewportGizmo!: ViewportGizmo;
  private transformGizmo!: TransformGizmo;

  private isRunning: boolean = false;
  private animationId: number = 0;

  // 3D Gaussian Splatting 渲染器
  private gsRenderer?: GSSplatRenderer;
  // 移动端纹理压缩渲染器
  private gsRendererMobile?: GSSplatRendererMobile;
  // 是否使用移动端渲染器
  private useMobileRenderer: boolean = false;

  // 绑定的事件处理函数（用于移除监听器）
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

    // 初始化视口 Gizmo
    this.viewportGizmo = new ViewportGizmo(
      this.renderer,
      this.camera,
      this.canvas,
    );

    // 初始化变换 Gizmo
    this.transformGizmo = new TransformGizmo({
      renderer: this.renderer,
      camera: this.camera,
      canvas: this.canvas,
    });
    this.transformGizmo.init();

    // 当 Gizmo 开始/结束拖拽时，禁用/启用 OrbitControls
    this.transformGizmo.setOnDragStateChange((isDragging) => {
      this.controls.enabled = !isDragging;
    });

    this.setupGizmoInteraction();

    // 监听窗口大小变化
    window.addEventListener("resize", this.boundOnResize);

    console.log("WebGPU 3D 渲染引擎已初始化");
  }

  /**
   * 设置 Gizmo 交互
   */
  private setupGizmoInteraction(): void {
    // 设置视口 Gizmo 轴点击回调
    this.viewportGizmo.setOnAxisClick((axis, positive) => {
      this.controls.setViewAxis(axis, positive, true);
    });

    // 监听点击事件
    this.canvas.addEventListener("click", (e) => {
      this.viewportGizmo.handleClick(e.clientX, e.clientY);
    });

    // 添加变换 Gizmo 的指针事件监听器
    this.canvas.addEventListener("pointermove", (e) => {
      this.transformGizmo.onPointerMove(e);
    });

    this.canvas.addEventListener("pointerdown", (e) => {
      this.transformGizmo.onPointerDown(e);
    });

    this.canvas.addEventListener("pointerup", (e) => {
      this.transformGizmo.onPointerUp(e);
    });
  }

  /**
   * 加载 GLB 文件
   * @returns 加载的网格数量
   */
  async addGLB(url: string): Promise<number> {
    try {
      const meshes = await this.glbLoader.load(url);
      for (const mesh of meshes) {
        this.meshRenderer.addMesh(mesh);
      }
      console.log(`已加载 ${meshes.length} 个网格: ${url}`);
      return meshes.length;
    } catch (error) {
      console.error("加载 GLB 文件失败:", error);
      throw error;
    }
  }

  /**
   * 加载 PLY 文件 (3D Gaussian Splatting)
   * 自动根据设备性能选择加载方式
   * - 移动端：使用纹理压缩渲染器 (GSSplatRendererMobile)，支持更多 splat
   * - 桌面端：使用标准渲染器 (GSSplatRenderer)，完整效果
   * @param url PLY 文件 URL
   * @param onProgress 进度回调（可选）
   * @returns 加载的 splat 数量
   */
  async addPLY(
    url: string,
    onProgress?: (loaded: number, total: number) => void,
  ): Promise<number> {
    try {
      // 检测是否为移动设备
      const isMobile = isMobileDevice();

      if (isMobile) {
        // ============================================
        // 移动端：使用纹理压缩渲染器
        // 内存占用从 256 bytes/splat 降低到 ~36 bytes/splat
        // ============================================
        console.log("📱 检测到移动设备，使用纹理压缩渲染器");

        if (!this.gsRendererMobile) {
          this.gsRendererMobile = new GSSplatRendererMobile(
            this.renderer,
            this.camera,
          );
        }
        this.useMobileRenderer = true;

        // 移动端配置：不限制 splat 数量
        // 纹理压缩后约 52 bytes/splat，内存占用大幅降低
        // 让用户自行控制加载的模型大小

        try {
          console.log("开始解析 PLY 文件...");
          const compactData = await loadPLYMobile(url, {
            maxSplats: Infinity, // 不限制数量
            loadSH: false, // 移动端纹理压缩模式不支持 SH
            onProgress,
          });

          console.log(`✅ PLY 解析完成: ${compactData.count} 个 splats`);

          console.log("开始压缩并上传到 GPU（纹理模式）...");
          this.gsRendererMobile.setCompactData(compactData);
          console.log(
            `✅ 已加载 ${compactData.count} 个 Splats (移动端纹理压缩): ${url}`,
          );
          return compactData.count;
        } catch (loadError) {
          console.error("❌ 移动端加载失败:", loadError);
          throw loadError;
        }
      } else {
        // ============================================
        // 桌面端：使用标准渲染器（完整效果）
        // 使用 loadPLYMobile + setCompactData 路径来减少内存使用
        // 旧的 loadPLY + setData 路径会为每个 splat 创建对象，内存使用量是 2-3 倍
        // ============================================
        if (!this.gsRenderer) {
          this.gsRenderer = new GSSplatRenderer(this.renderer, this.camera);
        }
        this.useMobileRenderer = false;

        const tier = this.gsRenderer.getPerformanceTier();
        console.log(`🖥️ 使用标准渲染器 (性能等级: ${tier})`);

        // 使用更高效的加载路径（减少内存峰值）
        // loadSH: true 以支持 SH 光照效果
        const compactData = await loadPLYMobile(url, {
          maxSplats: Infinity,
          loadSH: true, // 桌面端加载 SH 系数以支持完整效果
          onProgress,
        });

        this.gsRenderer.setCompactData(compactData);
        console.log(`已加载 ${compactData.count} 个 Splats: ${url}`);
        return compactData.count;
      }
    } catch (error) {
      console.error("加载 PLY 文件失败:", error);
      throw error;
    }
  }

  /**
   * 加载 Splat 文件 (3D Gaussian Splatting)
   * .splat 是一种紧凑的 3DGS 格式，每个 splat 32 字节
   * @returns 加载的 splat 数量
   */
  async addSplat(url: string): Promise<number> {
    try {
      const splats = await loadSplat(url);
      if (!this.gsRenderer) {
        this.gsRenderer = new GSSplatRenderer(this.renderer, this.camera);
      }
      this.gsRenderer.setData(splats);
      console.log(`已加载 ${splats.length} 个 Splats (splat 格式): ${url}`);
      return splats.length;
    } catch (error) {
      console.error("加载 Splat 文件失败:", error);
      throw error;
    }
  }

  /**
   * 添加测试立方体
   */
  addTestCube(): void {
    const cube = this.glbLoader.createTestCube();
    this.meshRenderer.addMesh(cube);
    console.log("已添加测试立方体");
  }

  /**
   * 添加测试球体
   */
  addTestSphere(): void {
    const sphere = this.glbLoader.createTestSphere();
    this.meshRenderer.addMesh(sphere);
    console.log("已添加测试球体");
  }

  /**
   * 开始渲染循环
   */
  start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.animate();
    console.log("渲染循环已启动");
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

  /**
   * 渲染循环
   */
  private animate(): void {
    if (!this.isRunning) return;

    this.render();
    this.animationId = requestAnimationFrame(this.animate.bind(this));
  }

  /**
   * 单帧渲染
   */
  private render(): void {
    // 更新相机宽高比
    this.camera.setAspect(this.renderer.getAspectRatio());
    this.camera.updateMatrix();

    // 开始帧
    const pass = this.renderer.beginFrame();

    // 渲染 3D Gaussian Splatting (先画，因为无深度排序)
    // 根据设备类型选择渲染器
    if (this.useMobileRenderer && this.gsRendererMobile) {
      this.gsRendererMobile.render(pass);
    } else if (this.gsRenderer) {
      this.gsRenderer.render(pass);
    }

    // 渲染网格
    this.meshRenderer.render(pass);

    // 渲染变换 Gizmo (在网格之后，视口 Gizmo 之前)
    this.transformGizmo.render(pass);

    // 渲染视口 Gizmo
    this.viewportGizmo.render(pass);

    // 结束帧
    this.renderer.endFrame();
  }

  /**
   * 窗口大小变化处理
   */
  private onResize(): void {
    this.camera.setAspect(this.renderer.getAspectRatio());
    this.camera.updateMatrix();
  }

  /**
   * 获取渲染器
   */
  getRenderer(): Renderer {
    return this.renderer;
  }

  /**
   * 获取相机
   */
  getCamera(): Camera {
    return this.camera;
  }

  /**
   * 获取控制器
   */
  getControls(): OrbitControls {
    return this.controls;
  }

  /**
   * 获取网格渲染器
   */
  getMeshRenderer(): MeshRenderer {
    return this.meshRenderer;
  }

  /**
   * 清空场景中的所有网格
   */
  clearMeshes(): void {
    this.meshRenderer.clear();
    console.log("场景已清空");
  }

  /**
   * 按索引移除网格
   */
  removeMeshByIndex(index: number): boolean {
    const result = this.meshRenderer.removeMeshByIndex(index);
    if (result) {
      console.log(`已移除网格: index=${index}`);
    }
    return result;
  }

  /**
   * 获取网格数量
   */
  getMeshCount(): number {
    return this.meshRenderer.getMeshCount();
  }

  /**
   * 获取视口 Gizmo
   */
  getViewportGizmo(): ViewportGizmo {
    return this.viewportGizmo;
  }

  /**
   * 获取指定索引的网格
   */
  getMeshByIndex(index: number): Mesh | null {
    return this.meshRenderer.getMeshByIndex(index);
  }

  /**
   * 获取指定范围的多个网格
   * @param startIndex 起始索引
   * @param count 数量
   * @returns Mesh 数组
   */
  getMeshRange(startIndex: number, count: number): Mesh[] {
    const meshes: Mesh[] = [];
    for (let i = 0; i < count; i++) {
      const mesh = this.meshRenderer.getMeshByIndex(startIndex + i);
      if (mesh) {
        meshes.push(mesh);
      }
    }
    return meshes;
  }

  /**
   * 创建 Mesh 组的变换代理，用于 Gizmo 同时操作多个 Mesh
   * @param startIndex 起始索引
   * @param count 数量
   * @returns MeshGroupProxy 或 null
   */
  createMeshGroupProxy(startIndex: number, count: number): MeshGroupProxy | null {
    const meshes = this.getMeshRange(startIndex, count);
    if (meshes.length === 0) {
      return null;
    }
    return new MeshGroupProxy(meshes);
  }

  /**
   * 获取 GS Splat 渲染器（桌面端）
   */
  getGSRenderer(): GSSplatRenderer | undefined {
    return this.gsRenderer;
  }

  /**
   * 获取 GS Splat 渲染器（移动端纹理压缩）
   */
  getGSRendererMobile(): GSSplatRendererMobile | undefined {
    return this.gsRendererMobile;
  }

  /**
   * 是否正在使用移动端渲染器
   */
  isUsingMobileRenderer(): boolean {
    return this.useMobileRenderer;
  }

  /**
   * 设置 SH 模式
   * @param mode 0=L0(仅DC), 1=L1, 2=L2, 3=L3(完整)
   * 注意：移动端纹理压缩模式仅支持 L0
   */
  setSHMode(mode: 0 | 1 | 2 | 3): void {
    if (this.useMobileRenderer && this.gsRendererMobile) {
      if (mode !== 0) {
        console.warn("setSHMode: 移动端纹理压缩模式仅支持 L0，已忽略");
      }
      return;
    }
    if (this.gsRenderer) {
      this.gsRenderer.setSHMode(mode);
    } else {
      console.warn("setSHMode: 没有加载 Splat 数据");
    }
  }

  /**
   * 获取当前 SH 模式
   * 移动端纹理压缩模式固定返回 0 (L0)
   */
  getSHMode(): number {
    if (this.useMobileRenderer && this.gsRendererMobile) {
      return 0; // 移动端纹理压缩仅支持 L0
    }
    return this.gsRenderer?.getSHMode() ?? 1;
  }

  /**
   * 获取 Splat 数量
   */
  getSplatCount(): number {
    if (this.useMobileRenderer && this.gsRendererMobile) {
      return this.gsRendererMobile.getSplatCount();
    }
    return this.gsRenderer?.getSplatCount() ?? 0;
  }

  /**
   * 清空 Splats
   */
  clearSplats(): void {
    if (this.gsRenderer) {
      this.gsRenderer.destroy();
      this.gsRenderer = undefined;
    }
    if (this.gsRendererMobile) {
      this.gsRendererMobile.destroy();
      this.gsRendererMobile = undefined;
    }
    this.useMobileRenderer = false;
    console.log("Splats 已清空");
  }

  /**
   * 获取变换 Gizmo
   */
  getTransformGizmo(): TransformGizmo {
    return this.transformGizmo;
  }

  /**
   * 设置 Gizmo 模式
   * @param mode - Gizmo 模式 (Translate=0, Rotate=1, Scale=2)
   */
  setGizmoMode(mode: GizmoMode): void {
    this.transformGizmo.setMode(mode);
  }

  /**
   * 设置 Gizmo 目标对象
   * @param object - 要操作的对象（Mesh 或 SplatTransformProxy），或 null 清除目标
   */
  setGizmoTarget(object: TransformableObject | null): void {
    this.transformGizmo.setTarget(object);
  }

  /**
   * 获取 PLY/Splat 的变换代理对象，用于 Gizmo 操作
   * 返回一个类似 Mesh 接口的对象，Gizmo 可以直接操作它
   * @returns 代理对象或 null（如果没有 PLY 数据）
   */
  getSplatTransformProxy(): SplatTransformProxy | null {
    // 获取当前使用的渲染器
    const renderer = this.useMobileRenderer ? this.gsRendererMobile : this.gsRenderer;
    if (!renderer) {
      return null;
    }

    // 获取 bounding box 用于初始化位置
    const bbox = renderer.getBoundingBox();
    if (!bbox) {
      return null;
    }

    // 创建代理对象
    return new SplatTransformProxy(renderer, bbox.center);
  }

  /**
   * 设置 PLY 位置
   */
  setSplatPosition(x: number, y: number, z: number): void {
    if (this.useMobileRenderer && this.gsRendererMobile) {
      this.gsRendererMobile.setPosition(x, y, z);
    } else if (this.gsRenderer) {
      this.gsRenderer.setPosition(x, y, z);
    }
  }

  /**
   * 设置 PLY 旋转（弧度）
   */
  setSplatRotation(x: number, y: number, z: number): void {
    if (this.useMobileRenderer && this.gsRendererMobile) {
      this.gsRendererMobile.setRotation(x, y, z);
    } else if (this.gsRenderer) {
      this.gsRenderer.setRotation(x, y, z);
    }
  }

  /**
   * 设置 PLY 缩放
   */
  setSplatScale(x: number, y: number, z: number): void {
    if (this.useMobileRenderer && this.gsRendererMobile) {
      this.gsRendererMobile.setScale(x, y, z);
    } else if (this.gsRenderer) {
      this.gsRenderer.setScale(x, y, z);
    }
  }

  /**
   * 获取 PLY 位置
   */
  getSplatPosition(): [number, number, number] | null {
    if (this.useMobileRenderer && this.gsRendererMobile) {
      return this.gsRendererMobile.getPosition();
    } else if (this.gsRenderer) {
      return this.gsRenderer.getPosition();
    }
    return null;
  }

  /**
   * 获取 PLY 旋转
   */
  getSplatRotation(): [number, number, number] | null {
    if (this.useMobileRenderer && this.gsRendererMobile) {
      return this.gsRendererMobile.getRotation();
    } else if (this.gsRenderer) {
      return this.gsRenderer.getRotation();
    }
    return null;
  }

  /**
   * 获取 PLY 缩放
   */
  getSplatScale(): [number, number, number] | null {
    if (this.useMobileRenderer && this.gsRendererMobile) {
      return this.gsRendererMobile.getScale();
    } else if (this.gsRenderer) {
      return this.gsRenderer.getScale();
    }
    return null;
  }

  /**
   * 自动调整相机以适应当前场景中的所有模型
   * 计算所有网格和点云的组合 bounding box，并调整相机位置、near/far
   * @param animate 是否使用动画过渡（默认 true）
   * @returns 是否成功（场景为空时返回 false）
   */
  frameCurrentModel(animate: boolean = true): boolean {
    // 收集所有 bounding box
    let combinedMin: [number, number, number] | null = null;
    let combinedMax: [number, number, number] | null = null;

    // 1. 获取网格的组合 bounding box
    const meshBBox = this.meshRenderer.getCombinedBoundingBox();
    if (meshBBox) {
      combinedMin = [...meshBBox.min];
      combinedMax = [...meshBBox.max];
    }

    // 2. 获取点云的 bounding box（支持两种渲染器）
    const splatBBox =
      this.useMobileRenderer && this.gsRendererMobile
        ? this.gsRendererMobile.getBoundingBox()
        : this.gsRenderer?.getBoundingBox();

    if (splatBBox) {
      if (combinedMin === null || combinedMax === null) {
        combinedMin = [...splatBBox.min];
        combinedMax = [...splatBBox.max];
      } else {
        // 合并
        combinedMin[0] = Math.min(combinedMin[0], splatBBox.min[0]);
        combinedMin[1] = Math.min(combinedMin[1], splatBBox.min[1]);
        combinedMin[2] = Math.min(combinedMin[2], splatBBox.min[2]);
        combinedMax[0] = Math.max(combinedMax[0], splatBBox.max[0]);
        combinedMax[1] = Math.max(combinedMax[1], splatBBox.max[1]);
        combinedMax[2] = Math.max(combinedMax[2], splatBBox.max[2]);
      }
    }

    // 3. 检查是否有有效的 bounding box
    if (combinedMin === null || combinedMax === null) {
      console.warn("frameCurrentModel: 场景中没有模型或点云");
      return false;
    }

    // 4. 计算组合的中心点和半径
    const center: [number, number, number] = [
      (combinedMin[0] + combinedMax[0]) / 2,
      (combinedMin[1] + combinedMax[1]) / 2,
      (combinedMin[2] + combinedMax[2]) / 2,
    ];
    const dx = combinedMax[0] - combinedMin[0];
    const dy = combinedMax[1] - combinedMin[1];
    const dz = combinedMax[2] - combinedMin[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    // 5. 调用 OrbitControls 的 frameModel 方法
    this.controls.frameModel(center, radius, animate);

    console.log(
      `App.frameCurrentModel: center=[${center[0].toFixed(2)}, ${center[1].toFixed(2)}, ${center[2].toFixed(2)}], radius=${radius.toFixed(2)}`,
    );

    return true;
  }

  /**
   * 销毁应用及所有资源
   */
  destroy(): void {
    // 停止渲染循环
    this.stop();

    // 移除窗口事件监听
    window.removeEventListener("resize", this.boundOnResize);

    // 销毁 Splat 渲染器
    this.clearSplats();

    // 销毁 Transform Gizmo
    if (this.transformGizmo) {
      this.transformGizmo.destroy();
    }

    // 销毁 Mesh 渲染器（会清空所有网格）
    if (this.meshRenderer) {
      this.meshRenderer.destroy();
    }

    // 销毁控制器
    if (this.controls) {
      this.controls.destroy();
    }

    // 销毁渲染器
    if (this.renderer) {
      this.renderer.destroy();
    }

    console.log("App: 所有资源已销毁");
  }
}
