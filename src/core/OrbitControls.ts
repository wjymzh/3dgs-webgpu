import { Camera } from "./Camera";

/**
 * OrbitControls - 轨道控制器
 * 只负责鼠标/触摸输入控制相机
 */
export class OrbitControls {
  private camera: Camera;
  private canvas: HTMLCanvasElement;

  // 球坐标参数
  distance: number = 5;
  theta: number = 0; // 水平角 (绕Y轴)
  phi: number = Math.PI / 4; // 垂直角 (从Y轴向下)

  // 限制
  minDistance: number = 0.001;
  maxDistance: number = Infinity;
  minPhi: number = 0.01;
  maxPhi: number = Math.PI - 0.01;

  // 灵敏度
  rotateSpeed: number = 0.005;
  zoomSpeed: number = 0.001;
  panSpeed: number = 0.005;
  
  // 移动端触摸灵敏度
  touchZoomSpeed: number = 0.01;
  touchPanSpeed: number = 0.003;

  // 状态
  private isDragging: boolean = false;
  private lastX: number = 0;
  private lastY: number = 0;

  // 触摸手势状态
  private touchMode: 'none' | 'rotate' | 'zoom-pan' = 'none';
  private lastTouchDistance: number = 0;
  private lastTouchCenter: { x: number; y: number } = { x: 0, y: 0 };

  // 启用/禁用
  enabled: boolean = true;

  // 绑定的事件处理函数（用于移除监听器）
  private boundOnMouseDown: (e: MouseEvent) => void;
  private boundOnMouseMove: (e: MouseEvent) => void;
  private boundOnMouseUp: (e: MouseEvent) => void;
  private boundOnWheel: (e: WheelEvent) => void;
  private boundOnTouchStart: (e: TouchEvent) => void;
  private boundOnTouchMove: (e: TouchEvent) => void;
  private boundOnTouchEnd: (e: TouchEvent) => void;
  private boundOnContextMenu: (e: Event) => void;

  constructor(camera: Camera, canvas: HTMLCanvasElement) {
    this.camera = camera;
    this.canvas = canvas;

    // 绑定事件处理函数
    this.boundOnMouseDown = this.onMouseDown.bind(this);
    this.boundOnMouseMove = this.onMouseMove.bind(this);
    this.boundOnMouseUp = this.onMouseUp.bind(this);
    this.boundOnWheel = this.onWheel.bind(this);
    this.boundOnTouchStart = this.onTouchStart.bind(this);
    this.boundOnTouchMove = this.onTouchMove.bind(this);
    this.boundOnTouchEnd = this.onTouchEnd.bind(this);
    this.boundOnContextMenu = (e: Event) => e.preventDefault();

    this.setupEventListeners();
    this.update();
  }

  /**
   * 设置事件监听
   */
  private setupEventListeners(): void {
    // 鼠标事件
    this.canvas.addEventListener("mousedown", this.boundOnMouseDown);
    this.canvas.addEventListener("mousemove", this.boundOnMouseMove);
    this.canvas.addEventListener("mouseup", this.boundOnMouseUp);
    this.canvas.addEventListener("mouseleave", this.boundOnMouseUp);
    this.canvas.addEventListener("wheel", this.boundOnWheel, {
      passive: false,
    });

    // 触摸事件
    this.canvas.addEventListener("touchstart", this.boundOnTouchStart, {
      passive: false,
    });
    this.canvas.addEventListener("touchmove", this.boundOnTouchMove, {
      passive: false,
    });
    this.canvas.addEventListener("touchend", this.boundOnTouchEnd);

    // 禁用右键菜单
    this.canvas.addEventListener("contextmenu", this.boundOnContextMenu);
  }

  /**
   * 移除事件监听
   */
  private removeEventListeners(): void {
    this.canvas.removeEventListener("mousedown", this.boundOnMouseDown);
    this.canvas.removeEventListener("mousemove", this.boundOnMouseMove);
    this.canvas.removeEventListener("mouseup", this.boundOnMouseUp);
    this.canvas.removeEventListener("mouseleave", this.boundOnMouseUp);
    this.canvas.removeEventListener("wheel", this.boundOnWheel);
    this.canvas.removeEventListener("touchstart", this.boundOnTouchStart);
    this.canvas.removeEventListener("touchmove", this.boundOnTouchMove);
    this.canvas.removeEventListener("touchend", this.boundOnTouchEnd);
    this.canvas.removeEventListener("contextmenu", this.boundOnContextMenu);
  }

  /**
   * 销毁控制器
   */
  destroy(): void {
    this.removeEventListeners();
    console.log("OrbitControls: 资源已销毁");
  }

  private onMouseDown(e: MouseEvent): void {
    if (!this.enabled) return;
    this.isDragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
  }

  private onMouseMove(e: MouseEvent): void {
    if (!this.enabled || !this.isDragging) return;

    const deltaX = e.clientX - this.lastX;
    const deltaY = e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;

    // 左键旋转
    if (e.buttons === 1) {
      this.theta -= deltaX * this.rotateSpeed;
      this.phi -= deltaY * this.rotateSpeed; // 修复：向上拖动时相机向上
      this.phi = Math.max(this.minPhi, Math.min(this.maxPhi, this.phi));
    }
    // 右键平移（拖动场景模式：向上拖动场景向上移动）
    else if (e.buttons === 2) {
      const panX = -deltaX * this.panSpeed * this.distance;
      const panY = deltaY * this.panSpeed * this.distance;

      // 计算平移向量
      const sinTheta = Math.sin(this.theta);
      const cosTheta = Math.cos(this.theta);

      this.camera.target[0] += panX * cosTheta;
      this.camera.target[2] += panX * sinTheta;
      this.camera.target[1] += panY;
    }

    this.update();
  }

  private onMouseUp(): void {
    this.isDragging = false;
  }

  private onWheel(e: WheelEvent): void {
    e.preventDefault();
    if (!this.enabled) return;
    this.distance += e.deltaY * this.zoomSpeed * this.distance;
    this.distance = Math.max(
      this.minDistance,
      Math.min(this.maxDistance, this.distance),
    );
    this.update();
  }

  private onTouchStart(e: TouchEvent): void {
    e.preventDefault();
    if (!this.enabled) return;
    
    if (e.touches.length === 1) {
      // 单指：旋转模式
      this.touchMode = 'rotate';
      this.isDragging = true;
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      // 双指：缩放+平移模式
      this.touchMode = 'zoom-pan';
      this.isDragging = true;
      this.lastTouchDistance = this.getTouchDistance(e.touches);
      this.lastTouchCenter = this.getTouchCenter(e.touches);
    }
  }

  private onTouchMove(e: TouchEvent): void {
    e.preventDefault();
    if (!this.enabled || !this.isDragging) return;

    if (e.touches.length === 1 && this.touchMode === 'rotate') {
      // 单指旋转
      const deltaX = e.touches[0].clientX - this.lastX;
      const deltaY = e.touches[0].clientY - this.lastY;
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;

      this.theta -= deltaX * this.rotateSpeed;
      this.phi -= deltaY * this.rotateSpeed;
      this.phi = Math.max(this.minPhi, Math.min(this.maxPhi, this.phi));

      this.update();
    } else if (e.touches.length === 2) {
      // 双指缩放 + 平移
      const currentDistance = this.getTouchDistance(e.touches);
      const currentCenter = this.getTouchCenter(e.touches);

      // 缩放：基于双指距离变化
      if (this.lastTouchDistance > 0) {
        const scale = this.lastTouchDistance / currentDistance;
        this.distance *= Math.pow(scale, this.touchZoomSpeed * 100);
        this.distance = Math.max(
          this.minDistance,
          Math.min(this.maxDistance, this.distance)
        );
      }

      // 平移：基于双指中心点移动
      const deltaX = currentCenter.x - this.lastTouchCenter.x;
      const deltaY = currentCenter.y - this.lastTouchCenter.y;

      const panX = -deltaX * this.touchPanSpeed * this.distance;
      const panY = deltaY * this.touchPanSpeed * this.distance;

      // 计算平移向量（考虑相机朝向）
      const sinTheta = Math.sin(this.theta);
      const cosTheta = Math.cos(this.theta);

      this.camera.target[0] += panX * cosTheta;
      this.camera.target[2] += panX * sinTheta;
      this.camera.target[1] += panY;

      // 更新上一次的触摸状态
      this.lastTouchDistance = currentDistance;
      this.lastTouchCenter = currentCenter;

      this.update();
    }
  }

  private onTouchEnd(e: TouchEvent): void {
    if (e.touches.length === 0) {
      // 所有手指离开
      this.isDragging = false;
      this.touchMode = 'none';
      this.lastTouchDistance = 0;
    } else if (e.touches.length === 1) {
      // 从双指变为单指，切换到旋转模式
      this.touchMode = 'rotate';
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
    }
  }

  /**
   * 计算双指之间的距离
   */
  private getTouchDistance(touches: TouchList): number {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * 计算双指的中心点
   */
  private getTouchCenter(touches: TouchList): { x: number; y: number } {
    return {
      x: (touches[0].clientX + touches[1].clientX) / 2,
      y: (touches[0].clientY + touches[1].clientY) / 2,
    };
  }

  /**
   * 根据球坐标更新相机位置
   */
  update(): void {
    const sinPhi = Math.sin(this.phi);
    const cosPhi = Math.cos(this.phi);
    const sinTheta = Math.sin(this.theta);
    const cosTheta = Math.cos(this.theta);

    // 球坐标转笛卡尔坐标
    this.camera.position[0] =
      this.camera.target[0] + this.distance * sinPhi * sinTheta;
    this.camera.position[1] = this.camera.target[1] + this.distance * cosPhi;
    this.camera.position[2] =
      this.camera.target[2] + this.distance * sinPhi * cosTheta;

    this.camera.updateMatrix();
  }

  /**
   * 切换到标准视图
   * @param axis 轴 'X' | 'Y' | 'Z'
   * @param positive 是否正向
   * @param animate 是否动画过渡
   */
  setViewAxis(axis: string, positive: boolean, animate: boolean = true): void {
    let targetTheta = this.theta;
    let targetPhi = this.phi;

    switch (axis) {
      case "X":
        // X 轴：从右侧看（正）或从左侧看（负）
        targetTheta = positive ? Math.PI / 2 : -Math.PI / 2;
        targetPhi = Math.PI / 2;
        break;
      case "Y":
        // Y 轴：从上方看（正）或从下方看（负）
        targetPhi = positive ? 0.01 : Math.PI - 0.01;
        break;
      case "Z":
        // Z 轴：从前方看（正）或从后方看（负）
        targetTheta = positive ? 0 : Math.PI;
        targetPhi = Math.PI / 2;
        break;
    }

    if (animate) {
      this.animateToView(targetTheta, targetPhi);
    } else {
      this.theta = targetTheta;
      this.phi = targetPhi;
      this.update();
    }
  }

  /**
   * 动画过渡到目标视图
   */
  private animateToView(targetTheta: number, targetPhi: number): void {
    const startTheta = this.theta;
    const startPhi = this.phi;
    const duration = 300; // 毫秒
    const startTime = performance.now();

    // 计算最短旋转路径
    let deltaTheta = targetTheta - startTheta;
    while (deltaTheta > Math.PI) deltaTheta -= Math.PI * 2;
    while (deltaTheta < -Math.PI) deltaTheta += Math.PI * 2;

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // 使用 ease-out 缓动
      const eased = 1 - Math.pow(1 - progress, 3);

      this.theta = startTheta + deltaTheta * eased;
      this.phi = startPhi + (targetPhi - startPhi) * eased;
      this.update();

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }

  /**
   * 设置相机目标点（控制器旋转中心）
   * @param x X 坐标
   * @param y Y 坐标
   * @param z Z 坐标
   */
  setTarget(x: number, y: number, z: number): void {
    this.camera.target[0] = x;
    this.camera.target[1] = y;
    this.camera.target[2] = z;
    this.update();
  }

  /**
   * 获取当前目标点
   */
  getTarget(): [number, number, number] {
    return [
      this.camera.target[0],
      this.camera.target[1],
      this.camera.target[2],
    ];
  }

  /**
   * 根据模型参数自动调整相机位置和参数
   * @param center 模型中心点
   * @param radius 模型包围球半径
   * @param animate 是否使用动画过渡
   */
  frameModel(
    center: [number, number, number],
    radius: number,
    animate: boolean = true,
  ): void {
    // 计算合适的相机距离：确保模型完全在视野内
    // distance = radius / tan(fov/2)，加一些余量
    const fovRad = this.camera.fov;
    const halfFov = fovRad / 2;
    const marginFactor = 1.5; // 留一些边距
    const targetDistance = (radius / Math.tan(halfFov)) * marginFactor;

    // 确保距离不会太小
    const clampedDistance = Math.max(this.minDistance, targetDistance);

    // 更新相机 near/far
    const nearDistance = Math.max(0.01, clampedDistance - radius * 2);
    const farDistance = clampedDistance + radius * 3;
    this.camera.near = nearDistance;
    this.camera.far = farDistance;

    if (animate) {
      this.animateToFrame(center, clampedDistance);
    } else {
      // 直接设置
      this.camera.target[0] = center[0];
      this.camera.target[1] = center[1];
      this.camera.target[2] = center[2];
      this.distance = clampedDistance;
      this.update();
    }

    console.log(
      `OrbitControls: frameModel - center: [${center[0].toFixed(2)}, ${center[1].toFixed(2)}, ${center[2].toFixed(2)}], ` +
        `radius: ${radius.toFixed(2)}, distance: ${clampedDistance.toFixed(2)}, near: ${nearDistance.toFixed(3)}, far: ${farDistance.toFixed(2)}`,
    );
  }

  /**
   * 动画过渡到目标帧（包含目标点和距离）
   */
  private animateToFrame(
    targetCenter: [number, number, number],
    targetDistance: number,
  ): void {
    const startTarget = [
      this.camera.target[0],
      this.camera.target[1],
      this.camera.target[2],
    ];
    const startDistance = this.distance;
    const duration = 400; // 毫秒
    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // 使用 ease-out 缓动
      const eased = 1 - Math.pow(1 - progress, 3);

      // 插值目标点
      this.camera.target[0] =
        startTarget[0] + (targetCenter[0] - startTarget[0]) * eased;
      this.camera.target[1] =
        startTarget[1] + (targetCenter[1] - startTarget[1]) * eased;
      this.camera.target[2] =
        startTarget[2] + (targetCenter[2] - startTarget[2]) * eased;

      // 插值距离
      this.distance = startDistance + (targetDistance - startDistance) * eased;

      this.update();

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }
}
