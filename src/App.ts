import { Renderer } from "./core/Renderer";
import { Camera } from "./core/Camera";
import { OrbitControls } from "./core/OrbitControls";
import { ViewportGizmo } from "./core/ViewportGizmo";
import { MeshRenderer } from "./mesh/MeshRenderer";
import { GLBLoader } from "./loaders/GLBLoader";
import { Mesh } from "./mesh/Mesh";
import { GSSplatRenderer, PerformanceTier } from "./gs/GSSplatRenderer";
import { GSSplatRendererMobile } from "./gs/GSSplatRendererMobile";
import { loadPLYMobile } from "./gs/PLYLoaderMobile";
import { loadSplat } from "./gs/SplatLoader";

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

  private isRunning: boolean = false;
  private animationId: number = 0;

  // 3D Gaussian Splatting 渲染器
  private gsRenderer?: GSSplatRenderer;
  // 移动端纹理压缩渲染器
  private gsRendererMobile?: GSSplatRendererMobile;
  // 是否使用移动端渲染器
  private useMobileRenderer: boolean = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
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

    this.setupGizmoInteraction();

    // 监听窗口大小变化
    window.addEventListener("resize", this.onResize.bind(this));

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
  async addPLY(url: string, onProgress?: (loaded: number, total: number) => void): Promise<number> {
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
          this.gsRendererMobile = new GSSplatRendererMobile(this.renderer, this.camera);
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
          console.log(`✅ 已加载 ${compactData.count} 个 Splats (移动端纹理压缩): ${url}`);
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
          loadSH: true,  // 桌面端加载 SH 系数以支持完整效果
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
    const splatBBox = this.useMobileRenderer && this.gsRendererMobile
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
      `App.frameCurrentModel: center=[${center[0].toFixed(2)}, ${center[1].toFixed(2)}, ${center[2].toFixed(2)}], radius=${radius.toFixed(2)}`
    );

    return true;
  }
}
