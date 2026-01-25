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
  minDistance: number = 0.5;
  maxDistance: number = 100;
  minPhi: number = 0.01;
  maxPhi: number = Math.PI - 0.01;

  // 灵敏度
  rotateSpeed: number = 0.005;
  zoomSpeed: number = 0.001;
  panSpeed: number = 0.005;

  // 状态
  private isDragging: boolean = false;
  private lastX: number = 0;
  private lastY: number = 0;

  // 启用/禁用
  enabled: boolean = true;

  constructor(camera: Camera, canvas: HTMLCanvasElement) {
    this.camera = camera;
    this.canvas = canvas;
    this.setupEventListeners();
    this.update();
  }

  /**
   * 设置事件监听
   */
  private setupEventListeners(): void {
    // 鼠标事件
    this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.canvas.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.canvas.addEventListener("mouseup", this.onMouseUp.bind(this));
    this.canvas.addEventListener("mouseleave", this.onMouseUp.bind(this));
    this.canvas.addEventListener("wheel", this.onWheel.bind(this), {
      passive: false,
    });

    // 触摸事件
    this.canvas.addEventListener("touchstart", this.onTouchStart.bind(this), {
      passive: false,
    });
    this.canvas.addEventListener("touchmove", this.onTouchMove.bind(this), {
      passive: false,
    });
    this.canvas.addEventListener("touchend", this.onTouchEnd.bind(this));

    // 禁用右键菜单
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
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
      this.isDragging = true;
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
    }
  }

  private onTouchMove(e: TouchEvent): void {
    e.preventDefault();
    if (!this.enabled || !this.isDragging || e.touches.length !== 1) return;

    const deltaX = e.touches[0].clientX - this.lastX;
    const deltaY = e.touches[0].clientY - this.lastY;
    this.lastX = e.touches[0].clientX;
    this.lastY = e.touches[0].clientY;

    this.theta -= deltaX * this.rotateSpeed;
    this.phi -= deltaY * this.rotateSpeed; // 修复：触摸也要同步修改
    this.phi = Math.max(this.minPhi, Math.min(this.maxPhi, this.phi));

    this.update();
  }

  private onTouchEnd(): void {
    this.isDragging = false;
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

    // 限制距离在合理范围内
    const clampedDistance = Math.max(
      this.minDistance,
      Math.min(this.maxDistance, targetDistance),
    );

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
