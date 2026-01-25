import { App } from '@lib';

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

    // 设置 UI 事件
    this.setupUI();
    this.setupSceneTree();
    
    // 启动渲染和性能监控
    this.app.start();
    this.startPerformanceMonitor();

    console.log('Demo 已初始化');
  }

  private setupUI(): void {
    // 文件选择按钮
    const btnLoad = document.getElementById('btn-load')!;
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    
    btnLoad.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

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

    // 距离滑块
    const distanceSlider = document.getElementById('distance') as HTMLInputElement;
    const distanceValue = document.getElementById('distance-value')!;
    distanceSlider.addEventListener('input', () => {
      const value = parseFloat(distanceSlider.value);
      distanceValue.textContent = value.toFixed(1);
      this.app.getControls().distance = value;
      this.app.getControls().update();
    });

    // FOV 滑块
    const fovSlider = document.getElementById('fov') as HTMLInputElement;
    const fovValue = document.getElementById('fov-value')!;
    fovSlider.addEventListener('input', () => {
      const value = parseInt(fovSlider.value);
      fovValue.textContent = `${value}°`;
      this.app.getCamera().fov = (value * Math.PI) / 180;
      this.app.getCamera().updateMatrix();
    });

    // 指向模型按钮
    const btnFrameModel = document.getElementById('btn-frame-model')!;
    btnFrameModel.addEventListener('click', () => {
      const success = this.app.frameCurrentModel(true);
      if (!success) {
        console.log('场景中没有模型');
      } else {
        // 更新 UI 中的距离显示
        setTimeout(() => {
          const newDistance = this.app.getControls().distance;
          distanceSlider.value = newDistance.toString();
          distanceValue.textContent = newDistance.toFixed(1);
        }, 450); // 等待动画完成后更新
      }
    });

    // 重置视角按钮
    const btnReset = document.getElementById('btn-reset')!;
    btnReset.addEventListener('click', () => {
      const controls = this.app.getControls();
      controls.distance = 5;
      controls.theta = 0;
      controls.phi = Math.PI / 4;
      controls.update();
      
      distanceSlider.value = '5';
      distanceValue.textContent = '5.0';
      
      fovSlider.value = '45';
      fovValue.textContent = '45°';
      this.app.getCamera().fov = Math.PI / 4;
      this.app.getCamera().updateMatrix();
    });

    // 同步控制器状态到 UI
    this.syncControlsToUI();
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
      
      // 计算该对象在渲染器中的实际起始索引
      // 需要考虑之前所有对象的网格数量
      let actualStartIndex = 0;
      for (let i = 0; i < objIndex; i++) {
        actualStartIndex += this.objects[i].meshCount;
      }
      
      // 从渲染器中移除所有相关网格（从后往前删除，避免索引变化问题）
      for (let i = obj.meshCount - 1; i >= 0; i--) {
        this.app.removeMeshByIndex(actualStartIndex + i);
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
    
    if (this.objects.length === 0) {
      listContainer.innerHTML = `
        <div class="empty-state">
          <div class="icon">📭</div>
          <div>场景为空</div>
          <div style="font-size: 11px; margin-top: 4px;">添加模型或几何体开始</div>
        </div>
      `;
      return;
    }

    listContainer.innerHTML = this.objects.map(obj => `
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

    // 绑定点击事件
    listContainer.querySelectorAll('.tree-item').forEach(item => {
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
    listContainer.querySelectorAll('[data-delete]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = (btn as HTMLElement).getAttribute('data-delete')!;
        this.removeObject(id);
      });
    });
  }

  private syncControlsToUI(): void {
    const controls = this.app.getControls();
    
    const distanceSlider = document.getElementById('distance') as HTMLInputElement;
    const distanceValue = document.getElementById('distance-value')!;
    distanceSlider.value = controls.distance.toString();
    distanceValue.textContent = controls.distance.toFixed(1);

    // 监听控制器变化（通过轮询）
    setInterval(() => {
      if (parseFloat(distanceSlider.value) !== controls.distance) {
        distanceSlider.value = controls.distance.toString();
        distanceValue.textContent = controls.distance.toFixed(1);
      }
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
        const arrayBuffer = await file.arrayBuffer();
        const url = URL.createObjectURL(new Blob([arrayBuffer]));
        const splatCount = await this.app.addPLY(url);
        URL.revokeObjectURL(url);
        this.addObjectToList(file.name, 'ply', splatCount);
        console.log(`已加载 PLY: ${file.name}, 包含 ${splatCount} 个 Splats`);
      } else {
        alert(`不支持的文件格式: ${ext}`);
      }
    } catch (error) {
      console.error('加载文件失败:', error);
      alert(`加载失败: ${error}`);
    }
  }

  private startPerformanceMonitor(): void {
    const fpsDisplay = document.getElementById('fps')!;
    const frameTimeDisplay = document.getElementById('frame-time')!;

    const measure = () => {
      this.frameCount++;
      const now = performance.now();
      const delta = now - this.lastTime;

      if (delta >= 1000) {
        this.fps = Math.round((this.frameCount * 1000) / delta);
        this.frameTime = delta / this.frameCount;
        this.frameCount = 0;
        this.lastTime = now;

        fpsDisplay.textContent = this.fps.toString();
        frameTimeDisplay.textContent = `${this.frameTime.toFixed(2)} ms`;
      }

      requestAnimationFrame(measure);
    };

    requestAnimationFrame(measure);
  }

  private updateStats(): void {
    const meshCountDisplay = document.getElementById('mesh-count')!;
    meshCountDisplay.textContent = this.app.getMeshCount().toString();
  }
}

// 启动 Demo
const demo = new Demo();
demo.init().catch(console.error);

// 导出到全局作用域，方便调试
(window as any).demo = demo;
