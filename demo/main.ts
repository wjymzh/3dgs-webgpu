import { App, SplatTransformProxy, MeshGroupProxy, GizmoMode } from '@lib';

/**
 * 场景对象类型
 */
interface SceneObject {
  id: string;
  name: string;
  type: 'mesh' | 'geometry' | 'ply';
  meshStartIndex: number;  // 起始网格索引
  meshCount: number;       // 包含的网格数量
}

/**
 * Demo 应用 - 实时测试 WebGPU 3D 渲染引擎
 */
class Demo {
  private app!: App;
  
  // UI 元素
  private canvas!: HTMLCanvasElement;
  private errorDiv!: HTMLDivElement;
  
  // 场景对象列表
  private objects: SceneObject[] = [];
  private selectedId: string = 'scene';
  private objectIdCounter = 0;
  
  // 性能统计
  private frameCount = 0;
  private lastTime = performance.now();
  private fps = 0;
  private frameTime = 0;
  
  // 移动端状态
  private isMobile: boolean = false;
  private currentMobilePanel: string | null = null;
  
  // 变换代理（用于 Gizmo 操作）
  private splatProxy: SplatTransformProxy | null = null;
  private meshGroupProxy: MeshGroupProxy | null = null;

  async init(): Promise<void> {
    // 获取 DOM 元素
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement;
    this.errorDiv = document.getElementById('error') as HTMLDivElement;

    // 检查 WebGPU 支持
    if (!navigator.gpu) {
      this.errorDiv.style.display = 'block';
      throw new Error('WebGPU 不受支持');
    }

    // 初始化应用
    this.app = new App(this.canvas);
    await this.app.init();

    // 设置初始背景色
    this.app.getRenderer().setClearColorHex('#1a1a26');

    // 检测是否为移动端
    this.isMobile = window.matchMedia('(max-width: 768px)').matches;
    
    // 设置 UI 事件
    this.setupUI();
    this.setupSceneTree();
    
    // 设置移动端 UI
    this.setupMobileUI();
    
    // 启动渲染和性能监控
    this.app.start();
    this.startPerformanceMonitor();

    console.log('Demo 已初始化', this.isMobile ? '(移动端)' : '(桌面端)');
  }

  private setupUI(): void {
    // 文件选择按钮
    const btnLoad = document.getElementById('btn-load')!;
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    
    btnLoad.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
    
    // URL 加载按钮
    const btnLoadUrl = document.getElementById('btn-load-url')!;
    const urlInput = document.getElementById('url-input') as HTMLInputElement;
    btnLoadUrl.addEventListener('click', () => this.loadFromUrl(urlInput.value));
    urlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.loadFromUrl(urlInput.value);
    });

    // 拖放区域 - 支持整个页面拖放
    const dropZone = document.getElementById('drop-zone')!;
    
    document.body.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    document.body.addEventListener('dragleave', (e) => {
      if (e.relatedTarget === null) {
        dropZone.classList.remove('drag-over');
      }
    });
    document.body.addEventListener('drop', (e) => this.handleFileDrop(e));

    // 添加立方体按钮
    const btnCube = document.getElementById('btn-cube')!;
    btnCube.addEventListener('click', () => {
      this.app.addTestCube();
      this.addObjectToList('立方体', 'geometry');
    });

    // 添加球体按钮
    const btnSphere = document.getElementById('btn-sphere')!;
    btnSphere.addEventListener('click', () => {
      this.app.addTestSphere();
      this.addObjectToList('球体', 'geometry');
    });

    // 相机参数 UI
    this.setupCameraUI();
    
    // 灯光控制 UI
    this.setupLightingUI();

    // 指向模型按钮
    const btnFrameModel = document.getElementById('btn-frame-model')!;
    btnFrameModel.addEventListener('click', () => {
      const success = this.app.frameCurrentModel(true);
      if (!success) {
        console.log('场景中没有模型');
      }
      // 相机参数会通过 syncCameraToUI 自动更新
    });

    // 重置视角按钮
    const btnReset = document.getElementById('btn-reset')!;
    btnReset.addEventListener('click', () => {
      const controls = this.app.getControls();
      const camera = this.app.getCamera();
      
      controls.distance = 5;
      controls.theta = 0;
      controls.phi = Math.PI / 4;
      controls.update();
      
      camera.fov = Math.PI / 4;
      camera.near = 0.1;
      camera.far = 1000;
      camera.updateMatrix();
      
      // 同步 UI
      this.syncCameraToUI();
    });

    // 同步控制器状态到 UI
    this.syncControlsToUI();
    
    // Gizmo 模式切换按钮
    this.setupGizmoModeUI();
  }

  /**
   * 设置 Gizmo 模式切换 UI
   */
  private setupGizmoModeUI(): void {
    // 桌面端按钮
    const btnTranslate = document.getElementById('btn-gizmo-translate')!;
    const btnRotate = document.getElementById('btn-gizmo-rotate')!;
    const btnScale = document.getElementById('btn-gizmo-scale')!;
    
    // 移动端按钮
    const mobileBtnTranslate = document.getElementById('mobile-btn-gizmo-translate');
    const mobileBtnRotate = document.getElementById('mobile-btn-gizmo-rotate');
    const mobileBtnScale = document.getElementById('mobile-btn-gizmo-scale');
    
    // 所有按钮（桌面端 + 移动端）
    const allButtons = [
      btnTranslate, btnRotate, btnScale,
      mobileBtnTranslate, mobileBtnRotate, mobileBtnScale
    ].filter(btn => btn !== null) as HTMLElement[];
    
    // 更新按钮激活状态
    const updateActiveState = (mode: GizmoMode) => {
      allButtons.forEach(btn => btn.classList.remove('active'));
      
      if (mode === GizmoMode.Translate) {
        btnTranslate.classList.add('active');
        mobileBtnTranslate?.classList.add('active');
      } else if (mode === GizmoMode.Rotate) {
        btnRotate.classList.add('active');
        mobileBtnRotate?.classList.add('active');
      } else if (mode === GizmoMode.Scale) {
        btnScale.classList.add('active');
        mobileBtnScale?.classList.add('active');
      }
    };
    
    // 设置模式并更新 UI
    const setGizmoMode = (mode: GizmoMode) => {
      this.app.setGizmoMode(mode);
      updateActiveState(mode);
    };
    
    // 桌面端按钮事件
    btnTranslate.addEventListener('click', () => setGizmoMode(GizmoMode.Translate));
    btnRotate.addEventListener('click', () => setGizmoMode(GizmoMode.Rotate));
    btnScale.addEventListener('click', () => setGizmoMode(GizmoMode.Scale));
    
    // 移动端按钮事件
    mobileBtnTranslate?.addEventListener('click', () => setGizmoMode(GizmoMode.Translate));
    mobileBtnRotate?.addEventListener('click', () => setGizmoMode(GizmoMode.Rotate));
    mobileBtnScale?.addEventListener('click', () => setGizmoMode(GizmoMode.Scale));
    
    // 键盘快捷键 (W/E/R)
    window.addEventListener('keydown', (e) => {
      // 如果焦点在输入框中，不触发快捷键
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      switch (e.key.toLowerCase()) {
        case 'w':
          setGizmoMode(GizmoMode.Translate);
          break;
        case 'e':
          setGizmoMode(GizmoMode.Rotate);
          break;
        case 'r':
          setGizmoMode(GizmoMode.Scale);
          break;
      }
    });
  }

  private setupSceneTree(): void {
    // Scene 项点击
    const sceneItem = document.querySelector('.scene-item')!;
    sceneItem.addEventListener('click', () => {
      this.selectObject('scene');
    });

    // 背景色选择器
    const bgColorInput = document.getElementById('bg-color') as HTMLInputElement;
    const bgColorHex = document.getElementById('bg-color-hex') as HTMLInputElement;

    bgColorInput.addEventListener('input', () => {
      const color = bgColorInput.value;
      bgColorHex.value = color;
      this.app.getRenderer().setClearColorHex(color);
    });

    bgColorHex.addEventListener('change', () => {
      let hex = bgColorHex.value;
      if (!hex.startsWith('#')) {
        hex = '#' + hex;
      }
      if (/^#[0-9A-Fa-f]{6}$/.test(hex)) {
        bgColorInput.value = hex;
        bgColorHex.value = hex;
        this.app.getRenderer().setClearColorHex(hex);
      }
    });

    // 添加按钮菜单
    const btnAddMenu = document.getElementById('btn-add-menu')!;
    btnAddMenu.addEventListener('click', () => {
      // 简单实现：添加立方体
      this.app.addTestCube();
      this.addObjectToList('立方体', 'geometry');
    });
  }

  private selectObject(id: string): void {
    this.selectedId = id;
    
    // 更新选中状态
    document.querySelectorAll('.tree-item').forEach(item => {
      item.classList.remove('selected');
      if (item.getAttribute('data-id') === id) {
        item.classList.add('selected');
      }
    });

    // 更新属性面板
    this.updatePropertiesPanel(id);
    
    // 设置 TransformGizmo 目标和包围盒
    if (id === 'scene') {
      // 选中场景时清除 Gizmo 目标和包围盒
      this.app.setGizmoTarget(null);
      this.app.clearSelectionBoundingBox();
      this.splatProxy = null;
      this.meshGroupProxy = null;
    } else {
      const obj = this.objects.find(o => o.id === id);
      if (obj && obj.type !== 'ply') {
        // 动态计算实际的 mesh 起始索引（因为删除对象后索引会变化）
        let actualStartIndex = 0;
        for (const o of this.objects) {
          if (o.id === obj.id) break;
          if (o.type !== 'ply') {
            actualStartIndex += o.meshCount;
          }
        }
        console.log(`选中对象: ${obj.name}, startIndex=${actualStartIndex}, count=${obj.meshCount}, 总mesh数=${this.app.getMeshCount()}`);
        // 创建 MeshGroupProxy 来同时操作所有相关的 mesh
        this.meshGroupProxy = this.app.createMeshGroupProxy(actualStartIndex, obj.meshCount);
        if (this.meshGroupProxy) {
          console.log(`MeshGroupProxy 创建成功，包含 ${obj.meshCount} 个 mesh`);
          this.app.setGizmoTarget(this.meshGroupProxy);
          // 设置动态包围盒提供者（MeshGroupProxy 实现了 getBoundingBox 方法）
          this.app.setSelectionBoundingBoxProvider(this.meshGroupProxy);
        } else {
          console.log(`MeshGroupProxy 创建失败`);
          this.app.setGizmoTarget(null);
          this.app.clearSelectionBoundingBox();
        }
        this.splatProxy = null;
      } else if (obj && obj.type === 'ply') {
        // PLY 类型：创建变换代理并设置为 Gizmo 目标
        this.splatProxy = this.app.getSplatTransformProxy();
        if (this.splatProxy) {
          this.app.setGizmoTarget(this.splatProxy);
          // 设置 PLY 的动态包围盒提供者
          const bboxProvider = this.app.createSplatBoundingBoxProvider();
          this.app.setSelectionBoundingBoxProvider(bboxProvider);
        } else {
          this.app.setGizmoTarget(null);
          this.app.clearSelectionBoundingBox();
        }
        this.meshGroupProxy = null;
      } else {
        this.app.setGizmoTarget(null);
        this.app.clearSelectionBoundingBox();
        this.splatProxy = null;
        this.meshGroupProxy = null;
      }
    }
  }

  private updatePropertiesPanel(id: string): void {
    const panel = document.getElementById('properties-panel')!;
    
    if (id === 'scene') {
      panel.innerHTML = `
        <div class="prop-title">Scene 属性</div>
        <div class="prop-row">
          <label>背景色</label>
          <input type="color" id="bg-color" value="${this.app.getRenderer().getClearColorHex()}">
          <input type="text" id="bg-color-hex" value="${this.app.getRenderer().getClearColorHex()}" maxlength="7">
        </div>
      `;
      
      // 重新绑定事件
      const bgColorInput = document.getElementById('bg-color') as HTMLInputElement;
      const bgColorHex = document.getElementById('bg-color-hex') as HTMLInputElement;

      bgColorInput.addEventListener('input', () => {
        const color = bgColorInput.value;
        bgColorHex.value = color;
        this.app.getRenderer().setClearColorHex(color);
      });

      bgColorHex.addEventListener('change', () => {
        let hex = bgColorHex.value;
        if (!hex.startsWith('#')) {
          hex = '#' + hex;
        }
        if (/^#[0-9A-Fa-f]{6}$/.test(hex)) {
          bgColorInput.value = hex;
          bgColorHex.value = hex;
          this.app.getRenderer().setClearColorHex(hex);
        }
      });
    } else {
      const obj = this.objects.find(o => o.id === id);
      if (obj) {
        panel.innerHTML = `
          <div class="prop-title">${obj.name} 属性</div>
          <div class="prop-row">
            <label>类型</label>
            <span style="color: #888;">${this.getTypeLabel(obj.type)}</span>
          </div>
          <div class="prop-row">
            <label>网格数</label>
            <span style="color: #667eea;">${obj.meshCount}</span>
          </div>
          <div class="prop-row">
            <label>ID</label>
            <span style="color: #667eea; font-family: monospace;">${obj.id}</span>
          </div>
        `;
      }
    }
  }

  private getTypeLabel(type: string): string {
    switch (type) {
      case 'mesh': return 'GLB 模型';
      case 'geometry': return '几何体';
      case 'ply': return 'PLY 点云';
      default: return type;
    }
  }

  private getTypeIcon(type: string): string {
    switch (type) {
      case 'mesh': return '📦';
      case 'geometry': return '🔷';
      case 'ply': return '☁️';
      default: return '📄';
    }
  }

  private addObjectToList(name: string, type: 'mesh' | 'geometry' | 'ply', meshCount: number = 1): void {
    const id = `obj_${++this.objectIdCounter}`;
    const currentMeshCount = this.app.getMeshCount();
    const obj: SceneObject = {
      id,
      name,
      type,
      meshStartIndex: currentMeshCount - meshCount,  // 起始索引
      meshCount: meshCount,                           // 网格数量
    };
    this.objects.push(obj);
    this.renderObjectList();
    this.updateStats();
    this.selectObject(id);
  }

  private removeObject(id: string): void {
    const objIndex = this.objects.findIndex(o => o.id === id);
    if (objIndex !== -1) {
      const obj = this.objects[objIndex];
      
      // 根据类型选择不同的删除方式
      if (obj.type === 'ply') {
        // PLY/Splat 类型：清除点云数据
        this.app.clearSplats();
      } else {
        // Mesh/Geometry 类型：计算实际起始索引并移除
        let actualStartIndex = 0;
        for (let i = 0; i < objIndex; i++) {
          // 只计算非 ply 类型的 mesh 数量
          if (this.objects[i].type !== 'ply') {
            actualStartIndex += this.objects[i].meshCount;
          }
        }
        
        // 从渲染器中移除所有相关网格（从后往前删除，避免索引变化问题）
        for (let i = obj.meshCount - 1; i >= 0; i--) {
          this.app.removeMeshByIndex(actualStartIndex + i);
        }
      }
      
      // 从列表中移除
      this.objects.splice(objIndex, 1);
      this.renderObjectList();
      this.updateStats();
      
      // 如果删除的是当前选中项，选中 Scene
      if (this.selectedId === id) {
        this.selectObject('scene');
      }
    }
  }

  private renderObjectList(): void {
    const listContainer = document.getElementById('object-list')!;
    const mobileListContainer = document.getElementById('mobile-object-list');
    
    const emptyStateHtml = `
      <div class="empty-state">
        <div class="icon">📭</div>
        <div>场景为空</div>
        <div style="font-size: 11px; margin-top: 4px;">添加模型或几何体开始</div>
      </div>
    `;
    
    if (this.objects.length === 0) {
      listContainer.innerHTML = emptyStateHtml;
      if (mobileListContainer) {
        mobileListContainer.innerHTML = emptyStateHtml;
      }
      return;
    }

    const listHtml = this.objects.map(obj => `
      <div class="tree-item ${this.selectedId === obj.id ? 'selected' : ''}" 
           data-type="${obj.type}" 
           data-id="${obj.id}">
        <span class="icon">${this.getTypeIcon(obj.type)}</span>
        <span class="name">${obj.name}</span>
        <span class="type">${obj.type}</span>
        <span class="actions">
          <button data-delete="${obj.id}" title="删除">×</button>
        </span>
      </div>
    `).join('');
    
    listContainer.innerHTML = listHtml;
    if (mobileListContainer) {
      mobileListContainer.innerHTML = listHtml;
    }

    // 绑定桌面端点击事件
    this.bindObjectListEvents(listContainer);
    
    // 绑定移动端点击事件
    if (mobileListContainer) {
      this.bindObjectListEvents(mobileListContainer);
    }
  }
  
  /**
   * 绑定对象列表的事件
   */
  private bindObjectListEvents(container: HTMLElement): void {
    // 绑定点击事件
    container.querySelectorAll('.tree-item').forEach(item => {
      item.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        // 如果点击的是删除按钮，不选中
        if (target.hasAttribute('data-delete')) {
          return;
        }
        const id = item.getAttribute('data-id')!;
        this.selectObject(id);
      });
    });

    // 绑定删除按钮事件
    container.querySelectorAll('[data-delete]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = (btn as HTMLElement).getAttribute('data-delete')!;
        this.removeObject(id);
      });
    });
  }

  /**
   * 设置移动端 UI 交互
   */
  private setupMobileUI(): void {
    const mobilePanel = document.getElementById('mobile-panel')!;
    const mobileOverlay = document.getElementById('mobile-overlay')!;
    const mobilePanelTitle = document.getElementById('mobile-panel-title')!;
    const mobilePanelClose = document.getElementById('mobile-panel-close')!;
    
    // 面板标题映射
    const panelTitles: Record<string, string> = {
      'scene': '场景',
      'controls': '控制',
      'import': '导入',
      'stats': '状态',
    };
    
    // 工具栏按钮点击事件
    document.querySelectorAll('.mobile-toolbar-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const panelType = btn.getAttribute('data-panel')!;
        
        if (this.currentMobilePanel === panelType) {
          // 再次点击同一按钮，关闭面板
          this.closeMobilePanel();
        } else {
          // 打开面板
          this.openMobilePanel(panelType);
        }
      });
    });
    
    // 关闭按钮
    mobilePanelClose.addEventListener('click', () => {
      this.closeMobilePanel();
    });
    
    // 点击遮罩层关闭面板
    mobileOverlay.addEventListener('click', () => {
      this.closeMobilePanel();
    });
    
    // 移动端文件选择
    const mobileFileInput = document.getElementById('mobile-file-input') as HTMLInputElement;
    const mobileBtnLoad = document.getElementById('mobile-btn-load')!;
    mobileBtnLoad.addEventListener('click', () => mobileFileInput.click());
    mobileFileInput.addEventListener('change', (e) => this.handleFileSelect(e));
    
    // 移动端 URL 加载
    const mobileBtnLoadUrl = document.getElementById('mobile-btn-load-url')!;
    const mobileUrlInput = document.getElementById('mobile-url-input') as HTMLInputElement;
    mobileBtnLoadUrl.addEventListener('click', () => this.loadFromUrl(mobileUrlInput.value, true));
    mobileUrlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.loadFromUrl(mobileUrlInput.value, true);
    });
    
    // 移动端添加几何体按钮
    const mobileBtnCube = document.getElementById('mobile-btn-cube')!;
    mobileBtnCube.addEventListener('click', () => {
      this.app.addTestCube();
      this.addObjectToList('立方体', 'geometry');
    });
    
    const mobileBtnSphere = document.getElementById('mobile-btn-sphere')!;
    mobileBtnSphere.addEventListener('click', () => {
      this.app.addTestSphere();
      this.addObjectToList('球体', 'geometry');
    });
    
    // 移动端背景色
    const mobileBgColor = document.getElementById('mobile-bg-color') as HTMLInputElement;
    mobileBgColor.addEventListener('input', () => {
      const color = mobileBgColor.value;
      this.app.getRenderer().setClearColorHex(color);
      // 同步到桌面端
      const bgColorInput = document.getElementById('bg-color') as HTMLInputElement;
      const bgColorHex = document.getElementById('bg-color-hex') as HTMLInputElement;
      if (bgColorInput) bgColorInput.value = color;
      if (bgColorHex) bgColorHex.value = color;
    });
    
    // 移动端相机控制
    this.setupMobileCameraUI();
    
    // 移动端指向模型按钮
    const mobileBtnFrameModel = document.getElementById('mobile-btn-frame-model')!;
    mobileBtnFrameModel.addEventListener('click', () => {
      const success = this.app.frameCurrentModel(true);
      if (!success) {
        console.log('场景中没有模型');
      }
    });
    
    // 移动端重置视角按钮
    const mobileBtnReset = document.getElementById('mobile-btn-reset')!;
    mobileBtnReset.addEventListener('click', () => {
      const controls = this.app.getControls();
      const camera = this.app.getCamera();
      
      controls.distance = 5;
      controls.theta = 0;
      controls.phi = Math.PI / 4;
      controls.update();
      
      camera.fov = Math.PI / 4;
      camera.near = 0.1;
      camera.far = 1000;
      camera.updateMatrix();
      
      this.syncCameraToUI();
    });
    
    // 监听窗口大小变化
    window.addEventListener('resize', () => {
      const wasMobile = this.isMobile;
      this.isMobile = window.matchMedia('(max-width: 768px)').matches;
      
      // 从桌面端切换到移动端时，关闭任何打开的面板
      if (!wasMobile && this.isMobile) {
        this.closeMobilePanel();
      }
    });
  }
  
  /**
   * 打开移动端面板
   */
  private openMobilePanel(panelType: string): void {
    const mobilePanel = document.getElementById('mobile-panel')!;
    const mobileOverlay = document.getElementById('mobile-overlay')!;
    const mobilePanelTitle = document.getElementById('mobile-panel-title')!;
    
    const panelTitles: Record<string, string> = {
      'scene': '场景',
      'controls': '控制',
      'import': '导入',
      'stats': '状态',
    };
    
    // 更新标题
    mobilePanelTitle.textContent = panelTitles[panelType] || '面板';
    
    // 显示对应的内容区域
    document.querySelectorAll('.mobile-panel-section').forEach(section => {
      section.classList.remove('active');
      if (section.getAttribute('data-section') === panelType) {
        section.classList.add('active');
      }
    });
    
    // 更新工具栏按钮状态
    document.querySelectorAll('.mobile-toolbar-btn').forEach(btn => {
      btn.classList.remove('active');
      if (btn.getAttribute('data-panel') === panelType) {
        btn.classList.add('active');
      }
    });
    
    // 显示面板和遮罩
    mobilePanel.classList.add('open');
    mobileOverlay.classList.add('visible');
    
    this.currentMobilePanel = panelType;
  }
  
  /**
   * 关闭移动端面板
   */
  private closeMobilePanel(): void {
    const mobilePanel = document.getElementById('mobile-panel')!;
    const mobileOverlay = document.getElementById('mobile-overlay')!;
    
    mobilePanel.classList.remove('open');
    mobileOverlay.classList.remove('visible');
    
    // 移除工具栏按钮激活状态
    document.querySelectorAll('.mobile-toolbar-btn').forEach(btn => {
      btn.classList.remove('active');
    });
    
    this.currentMobilePanel = null;
  }
  
  /**
   * 设置移动端相机 UI
   */
  private setupMobileCameraUI(): void {
    const camera = this.app.getCamera();
    
    // 位置输入
    const posX = document.getElementById('mobile-cam-pos-x') as HTMLInputElement;
    const posY = document.getElementById('mobile-cam-pos-y') as HTMLInputElement;
    const posZ = document.getElementById('mobile-cam-pos-z') as HTMLInputElement;
    
    const updatePosition = () => {
      camera.position[0] = parseFloat(posX.value) || 0;
      camera.position[1] = parseFloat(posY.value) || 0;
      camera.position[2] = parseFloat(posZ.value) || 0;
      camera.updateMatrix();
    };
    
    posX.addEventListener('change', updatePosition);
    posY.addEventListener('change', updatePosition);
    posZ.addEventListener('change', updatePosition);
    
    // 目标点输入
    const targetX = document.getElementById('mobile-cam-target-x') as HTMLInputElement;
    const targetY = document.getElementById('mobile-cam-target-y') as HTMLInputElement;
    const targetZ = document.getElementById('mobile-cam-target-z') as HTMLInputElement;
    
    const updateTarget = () => {
      camera.target[0] = parseFloat(targetX.value) || 0;
      camera.target[1] = parseFloat(targetY.value) || 0;
      camera.target[2] = parseFloat(targetZ.value) || 0;
      camera.updateMatrix();
      const controls = this.app.getControls();
      controls.setTarget(camera.target[0], camera.target[1], camera.target[2]);
    };
    
    targetX.addEventListener('change', updateTarget);
    targetY.addEventListener('change', updateTarget);
    targetZ.addEventListener('change', updateTarget);
    
    // FOV 滑块
    const fovSlider = document.getElementById('mobile-fov') as HTMLInputElement;
    const fovValue = document.getElementById('mobile-fov-value')!;
    fovSlider.addEventListener('input', () => {
      const value = parseInt(fovSlider.value);
      fovValue.textContent = `${value}°`;
      camera.fov = (value * Math.PI) / 180;
      camera.updateMatrix();
    });
    
    // 移动端灯光控制
    const mobileAmbientSlider = document.getElementById('mobile-ambient-intensity') as HTMLInputElement;
    const mobileAmbientValue = document.getElementById('mobile-ambient-value')!;
    if (mobileAmbientSlider) {
      mobileAmbientSlider.addEventListener('input', () => {
        const value = parseInt(mobileAmbientSlider.value);
        mobileAmbientValue.textContent = `${value}%`;
        this.app.getMeshRenderer().setAmbientIntensity(value / 100);
        // 同步到桌面端
        const desktopSlider = document.getElementById('ambient-intensity') as HTMLInputElement;
        const desktopValue = document.getElementById('ambient-value');
        if (desktopSlider) desktopSlider.value = value.toString();
        if (desktopValue) desktopValue.textContent = `${value}%`;
      });
    }
  }

  /**
   * 设置灯光控制 UI
   */
  private setupLightingUI(): void {
    // 桌面端环境光滑块
    const ambientSlider = document.getElementById('ambient-intensity') as HTMLInputElement;
    const ambientValue = document.getElementById('ambient-value')!;
    
    ambientSlider.addEventListener('input', () => {
      const value = parseInt(ambientSlider.value);
      ambientValue.textContent = `${value}%`;
      this.app.getMeshRenderer().setAmbientIntensity(value / 100);
      // 同步到移动端
      const mobileSlider = document.getElementById('mobile-ambient-intensity') as HTMLInputElement;
      const mobileValue = document.getElementById('mobile-ambient-value');
      if (mobileSlider) mobileSlider.value = value.toString();
      if (mobileValue) mobileValue.textContent = `${value}%`;
    });
  }

  private setupCameraUI(): void {
    const camera = this.app.getCamera();
    
    // 位置输入
    const posX = document.getElementById('cam-pos-x') as HTMLInputElement;
    const posY = document.getElementById('cam-pos-y') as HTMLInputElement;
    const posZ = document.getElementById('cam-pos-z') as HTMLInputElement;
    
    const updatePosition = () => {
      camera.position[0] = parseFloat(posX.value) || 0;
      camera.position[1] = parseFloat(posY.value) || 0;
      camera.position[2] = parseFloat(posZ.value) || 0;
      camera.updateMatrix();
    };
    
    posX.addEventListener('change', updatePosition);
    posY.addEventListener('change', updatePosition);
    posZ.addEventListener('change', updatePosition);
    
    // 目标点输入
    const targetX = document.getElementById('cam-target-x') as HTMLInputElement;
    const targetY = document.getElementById('cam-target-y') as HTMLInputElement;
    const targetZ = document.getElementById('cam-target-z') as HTMLInputElement;
    
    const updateTarget = () => {
      camera.target[0] = parseFloat(targetX.value) || 0;
      camera.target[1] = parseFloat(targetY.value) || 0;
      camera.target[2] = parseFloat(targetZ.value) || 0;
      camera.updateMatrix();
      // 同步控制器的目标点
      const controls = this.app.getControls();
      controls.setTarget(camera.target[0], camera.target[1], camera.target[2]);
    };
    
    targetX.addEventListener('change', updateTarget);
    targetY.addEventListener('change', updateTarget);
    targetZ.addEventListener('change', updateTarget);
    
    // FOV 滑块
    const fovSlider = document.getElementById('fov') as HTMLInputElement;
    const fovValue = document.getElementById('fov-value')!;
    fovSlider.addEventListener('input', () => {
      const value = parseInt(fovSlider.value);
      fovValue.textContent = `${value}°`;
      camera.fov = (value * Math.PI) / 180;
      camera.updateMatrix();
    });
    
    // Near 输入
    const nearInput = document.getElementById('cam-near') as HTMLInputElement;
    nearInput.addEventListener('change', () => {
      const value = parseFloat(nearInput.value);
      if (value > 0) {
        camera.near = value;
        camera.updateMatrix();
      }
    });
    
    // Far 输入
    const farInput = document.getElementById('cam-far') as HTMLInputElement;
    farInput.addEventListener('change', () => {
      const value = parseFloat(farInput.value);
      if (value > camera.near) {
        camera.far = value;
        camera.updateMatrix();
      }
    });
  }

  private syncCameraToUI(): void {
    const camera = this.app.getCamera();
    const activeEl = document.activeElement;
    
    // 辅助函数：仅在输入框未获得焦点时更新
    const updateIfNotFocused = (id: string, value: string) => {
      const input = document.getElementById(id) as HTMLInputElement;
      if (input && activeEl !== input) {
        input.value = value;
      }
    };
    
    // 桌面端 - 位置
    updateIfNotFocused('cam-pos-x', camera.position[0].toFixed(2));
    updateIfNotFocused('cam-pos-y', camera.position[1].toFixed(2));
    updateIfNotFocused('cam-pos-z', camera.position[2].toFixed(2));
    
    // 桌面端 - 目标点
    updateIfNotFocused('cam-target-x', camera.target[0].toFixed(2));
    updateIfNotFocused('cam-target-y', camera.target[1].toFixed(2));
    updateIfNotFocused('cam-target-z', camera.target[2].toFixed(2));
    
    // 桌面端 - FOV
    const fovDegrees = Math.round((camera.fov * 180) / Math.PI);
    updateIfNotFocused('fov', fovDegrees.toString());
    const fovValueEl = document.getElementById('fov-value');
    if (fovValueEl) fovValueEl.textContent = `${fovDegrees}°`;
    
    // 桌面端 - Near / Far
    updateIfNotFocused('cam-near', camera.near.toString());
    updateIfNotFocused('cam-far', camera.far.toString());
    
    // 移动端 - 位置
    updateIfNotFocused('mobile-cam-pos-x', camera.position[0].toFixed(2));
    updateIfNotFocused('mobile-cam-pos-y', camera.position[1].toFixed(2));
    updateIfNotFocused('mobile-cam-pos-z', camera.position[2].toFixed(2));
    
    // 移动端 - 目标点
    updateIfNotFocused('mobile-cam-target-x', camera.target[0].toFixed(2));
    updateIfNotFocused('mobile-cam-target-y', camera.target[1].toFixed(2));
    updateIfNotFocused('mobile-cam-target-z', camera.target[2].toFixed(2));
    
    // 移动端 - FOV
    updateIfNotFocused('mobile-fov', fovDegrees.toString());
    const mobileFovValueEl = document.getElementById('mobile-fov-value');
    if (mobileFovValueEl) mobileFovValueEl.textContent = `${fovDegrees}°`;
  }

  private syncControlsToUI(): void {
    // 初始同步相机参数
    this.syncCameraToUI();

    // 监听控制器变化（通过轮询），同步相机位置到 UI
    setInterval(() => {
      this.syncCameraToUI();
    }, 100);
  }

  private async handleFileSelect(e: Event): Promise<void> {
    const input = e.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      await this.loadFile(input.files[0]);
      // 重置 input，允许再次选择同一个文件
      input.value = '';
    }
  }

  private async handleFileDrop(e: DragEvent): Promise<void> {
    e.preventDefault();
    const dropZone = document.getElementById('drop-zone')!;
    dropZone.classList.remove('drag-over');

    if (e.dataTransfer?.files) {
      for (const file of Array.from(e.dataTransfer.files)) {
        await this.loadFile(file);
      }
    }
  }

  private async loadFile(file: File): Promise<void> {
    const ext = file.name.split('.').pop()?.toLowerCase();
    
    try {
      if (ext === 'glb') {
        const arrayBuffer = await file.arrayBuffer();
        const url = URL.createObjectURL(new Blob([arrayBuffer]));
        const meshCount = await this.app.addGLB(url);
        URL.revokeObjectURL(url);
        this.addObjectToList(file.name, 'mesh', meshCount);
        console.log(`已加载 GLB: ${file.name}, 包含 ${meshCount} 个网格`);
      } else if (ext === 'ply') {
        // 显示加载进度弹窗
        const progressDiv = document.createElement('div');
        progressDiv.id = 'load-progress';
        progressDiv.style.cssText = `
          position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
          background: rgba(0,0,0,0.85); color: white; padding: 24px 40px;
          border-radius: 12px; font-size: 16px; z-index: 9999;
          min-width: 220px; text-align: center;
        `;
        
        const progressText = document.createElement('div');
        progressText.style.marginBottom = '12px';
        progressText.textContent = '加载中... 0%';
        
        const progressTrack = document.createElement('div');
        progressTrack.style.cssText = 'height: 6px; background: rgba(255,255,255,0.2); border-radius: 3px; overflow: hidden;';
        
        const progressBarInner = document.createElement('div');
        progressBarInner.style.cssText = 'height: 100%; width: 0%; background-color: #667eea; transition: width 0.15s ease;';
        
        progressTrack.appendChild(progressBarInner);
        progressDiv.appendChild(progressText);
        progressDiv.appendChild(progressTrack);
        document.body.appendChild(progressDiv);
        
        try {
          // 等待 DOM 渲染并显示初始状态
          await new Promise(r => setTimeout(r, 50));
          
          // 读取文件阶段 (0-50%)
          const arrayBuffer = await file.arrayBuffer();
          
          progressText.textContent = '加载中... 50%';
          progressBarInner.style.width = '50%';
          
          // 等待进度条动画渲染
          await new Promise(r => setTimeout(r, 50));
          
          // 本地文件，从 50% 开始（跳过下载阶段）
          const splatCount = await this.app.addPLY(arrayBuffer, (progress, _stage) => {
            progressText.textContent = `加载中... ${Math.floor(progress)}%`;
            progressBarInner.style.width = `${progress}%`;
          }, true);
          this.addObjectToList(file.name, 'ply', 1);
          console.log(`已加载 PLY: ${file.name}, 包含 ${splatCount} 个 Splats`);
        } finally {
          progressDiv.remove();
        }
      } else if (ext === 'splat') {
        // 显示加载进度弹窗
        const progressDiv = document.createElement('div');
        progressDiv.id = 'load-progress';
        progressDiv.style.cssText = `
          position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
          background: rgba(0,0,0,0.85); color: white; padding: 24px 40px;
          border-radius: 12px; font-size: 16px; z-index: 9999;
          min-width: 220px; text-align: center;
        `;
        
        const progressText = document.createElement('div');
        progressText.style.marginBottom = '12px';
        progressText.textContent = '加载中... 0%';
        
        const progressTrack = document.createElement('div');
        progressTrack.style.cssText = 'height: 6px; background: rgba(255,255,255,0.2); border-radius: 3px; overflow: hidden;';
        
        const progressBarInner = document.createElement('div');
        progressBarInner.style.cssText = 'height: 100%; width: 0%; background-color: #667eea; transition: width 0.15s ease;';
        
        progressTrack.appendChild(progressBarInner);
        progressDiv.appendChild(progressText);
        progressDiv.appendChild(progressTrack);
        document.body.appendChild(progressDiv);
        
        try {
          // 等待 DOM 渲染并显示初始状态
          await new Promise(r => setTimeout(r, 50));
          
          // 读取文件阶段 (0-50%)
          const arrayBuffer = await file.arrayBuffer();
          
          progressText.textContent = '加载中... 50%';
          progressBarInner.style.width = '50%';
          
          // 等待进度条动画渲染
          await new Promise(r => setTimeout(r, 50));
          
          const splatCount = await this.app.addSplat(arrayBuffer, (progress, _stage) => {
            progressText.textContent = `加载中... ${Math.floor(progress)}%`;
            progressBarInner.style.width = `${progress}%`;
          }, true);
          this.addObjectToList(file.name, 'ply', 1);
          console.log(`已加载 Splat: ${file.name}, 包含 ${splatCount} 个 Splats`);
        } finally {
          progressDiv.remove();
        }
      } else {
        alert(`不支持的文件格式: ${ext}`);
      }
    } catch (error) {
      console.error('加载文件失败:', error);
      alert(`加载失败: ${error}`);
    }
  }

  /**
   * 从 URL 加载 PLY/SPLAT 文件
   */
  private async loadFromUrl(url: string, isMobile: boolean = false): Promise<void> {
    url = url.trim();
    if (!url) {
      alert('请输入有效的 URL');
      return;
    }

    // 获取文件扩展名
    const urlPath = url.split('?')[0];
    const ext = urlPath.split('.').pop()?.toLowerCase();
    
    if (ext !== 'ply' && ext !== 'splat') {
      alert('URL 加载仅支持 PLY 和 SPLAT 格式');
      return;
    }

    // 创建屏幕中央进度弹窗
    const progressDiv = document.createElement('div');
    progressDiv.id = 'load-progress';
    progressDiv.style.cssText = `
      position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
      background: rgba(0,0,0,0.85); color: white; padding: 24px 40px;
      border-radius: 12px; font-size: 16px; z-index: 9999;
      min-width: 220px; text-align: center;
    `;
    
    const progressText = document.createElement('div');
    progressText.style.marginBottom = '12px';
    progressText.textContent = '加载中... 0%';
    
    const progressTrack = document.createElement('div');
    progressTrack.style.cssText = 'height: 6px; background: rgba(255,255,255,0.2); border-radius: 3px; overflow: hidden;';
    
    const progressBarInner = document.createElement('div');
    progressBarInner.style.cssText = 'height: 100%; width: 0%; background-color: #667eea; transition: width 0.15s ease;';
    
    progressTrack.appendChild(progressBarInner);
    progressDiv.appendChild(progressText);
    progressDiv.appendChild(progressTrack);
    document.body.appendChild(progressDiv);

    // 统一进度回调：直接使用 0-100 的进度值
    const updateProgress = (progress: number, _stage: 'download' | 'parse' | 'upload') => {
      progressText.textContent = `加载中... ${Math.floor(progress)}%`;
      progressBarInner.style.width = `${progress}%`;
    };

    try {
      // 从 URL 提取文件名
      const fileName = urlPath.split('/').pop() || `model.${ext}`;
      
      let splatCount: number;
      if (ext === 'ply') {
        splatCount = await this.app.addPLY(url, updateProgress, false);
      } else {
        splatCount = await this.app.addSplat(url, updateProgress, false);
      }
      
      this.addObjectToList(fileName, 'ply', 1);
      console.log(`已从 URL 加载: ${fileName}, 包含 ${splatCount} 个 Splats`);
      
      // 清空输入框
      const urlInput = document.getElementById(isMobile ? 'mobile-url-input' : 'url-input') as HTMLInputElement;
      urlInput.value = '';
      
    } catch (error) {
      console.error('从 URL 加载失败:', error);
      alert(`加载失败: ${error}`);
    } finally {
      progressDiv.remove();
    }
  }

  private startPerformanceMonitor(): void {
    // 桌面端元素
    const fpsDisplay = document.getElementById('fps')!;
    const frameTimeDisplay = document.getElementById('frame-time')!;
    const perfTierDisplay = document.getElementById('perf-tier')!;
    const sortFreqDisplay = document.getElementById('sort-freq')!;
    const splatCountDisplay = document.getElementById('splat-count')!;
    
    // 移动端元素
    const mobileFpsDisplay = document.getElementById('mobile-fps');
    const mobileFrameTimeDisplay = document.getElementById('mobile-frame-time');
    const mobilePerfTierDisplay = document.getElementById('mobile-perf-tier');
    const mobileSortFreqDisplay = document.getElementById('mobile-sort-freq');
    const mobileSplatCountDisplay = document.getElementById('mobile-splat-count');

    // 显示初始性能等级
    const gsRenderer = this.app.getGSRenderer();
    if (gsRenderer) {
      const tier = gsRenderer.getPerformanceTier();
      const config = gsRenderer.getOptimizationConfig();
      perfTierDisplay.textContent = tier;
      sortFreqDisplay.textContent = `1/${config.sortEveryNFrames}`;
      if (mobilePerfTierDisplay) mobilePerfTierDisplay.textContent = tier;
      if (mobileSortFreqDisplay) mobileSortFreqDisplay.textContent = `1/${config.sortEveryNFrames}`;
    } else {
      // 默认显示（可能还未加载模型）
      perfTierDisplay.textContent = '-';
      sortFreqDisplay.textContent = '-';
      if (mobilePerfTierDisplay) mobilePerfTierDisplay.textContent = '-';
      if (mobileSortFreqDisplay) mobileSortFreqDisplay.textContent = '-';
    }

    const measure = () => {
      this.frameCount++;
      const now = performance.now();
      const delta = now - this.lastTime;

      if (delta >= 1000) {
        this.fps = Math.round((this.frameCount * 1000) / delta);
        this.frameTime = delta / this.frameCount;
        this.frameCount = 0;
        this.lastTime = now;

        // 更新桌面端显示
        fpsDisplay.textContent = this.fps.toString();
        frameTimeDisplay.textContent = `${this.frameTime.toFixed(2)} ms`;
        
        // 更新移动端显示
        if (mobileFpsDisplay) mobileFpsDisplay.textContent = this.fps.toString();
        if (mobileFrameTimeDisplay) mobileFrameTimeDisplay.textContent = `${this.frameTime.toFixed(2)} ms`;
        
        // 更新 Splat 相关状态（支持桌面端和移动端渲染器）
        const splatCount = this.app.getSplatCount();
        splatCountDisplay.textContent = splatCount.toLocaleString();
        if (mobileSplatCountDisplay) mobileSplatCountDisplay.textContent = splatCount.toLocaleString();
        
        // 性能等级和排序频率（仅桌面端渲染器支持）
        const gsRenderer = this.app.getGSRenderer();
        if (gsRenderer) {
          const tier = gsRenderer.getPerformanceTier();
          const config = gsRenderer.getOptimizationConfig();
          const sortFreq = `1/${config.sortEveryNFrames}`;
          
          perfTierDisplay.textContent = tier;
          sortFreqDisplay.textContent = sortFreq;
          
          if (mobilePerfTierDisplay) mobilePerfTierDisplay.textContent = tier;
          if (mobileSortFreqDisplay) mobileSortFreqDisplay.textContent = sortFreq;
        } else if (this.app.isUsingMobileRenderer()) {
          // 移动端渲染器使用固定显示
          perfTierDisplay.textContent = 'mobile';
          sortFreqDisplay.textContent = '1/1';
          
          if (mobilePerfTierDisplay) mobilePerfTierDisplay.textContent = 'mobile';
          if (mobileSortFreqDisplay) mobileSortFreqDisplay.textContent = '1/1';
        }
      }

      requestAnimationFrame(measure);
    };

    requestAnimationFrame(measure);
  }

  private updateStats(): void {
    const meshCountDisplay = document.getElementById('mesh-count')!;
    const mobileMeshCountDisplay = document.getElementById('mobile-mesh-count');
    
    const meshCount = this.app.getMeshCount().toString();
    meshCountDisplay.textContent = meshCount;
    if (mobileMeshCountDisplay) mobileMeshCountDisplay.textContent = meshCount;
  }
}

// 启动 Demo
const demo = new Demo();
demo.init().catch(console.error);

// 导出到全局作用域，方便调试
(window as any).demo = demo;
