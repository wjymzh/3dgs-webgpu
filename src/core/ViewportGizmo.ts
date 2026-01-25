import { Camera } from "./Camera";
import { Renderer } from "./Renderer";

/**
 * Gizmo 轴配置
 */
interface AxisConfig {
  direction: [number, number, number];
  color: [number, number, number];
  label: string;
}

/**
 * WGSL Shader - Gizmo 渲染
 */
const gizmoShaderCode = /* wgsl */ `
struct Uniforms {
  viewMatrix: mat4x4<f32>,
  projMatrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) color: vec3<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let worldPos = vec4<f32>(input.position, 1.0);
  output.position = uniforms.projMatrix * uniforms.viewMatrix * worldPos;
  output.color = input.color;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color, 1.0);
}
`;

/**
 * ViewportGizmo - 视口坐标轴指示器
 * 在画布右上角显示当前相机朝向
 */
export class ViewportGizmo {
  private renderer: Renderer;
  private camera: Camera;
  private canvas: HTMLCanvasElement;

  // 渲染资源
  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private vertexCount: number = 0;
  private indexCount: number = 0;

  // Gizmo 配置
  private size: number = 200; // Gizmo 尺寸（像素）
  private margin: number = 20; // 边距

  // Gizmo 投影矩阵
  private projMatrix: Float32Array = new Float32Array(16);
  private viewMatrix: Float32Array = new Float32Array(16);

  // 轴配置
  private axes: AxisConfig[] = [
    { direction: [1, 0, 0], color: [0.9, 0.2, 0.2], label: "X" }, // 红色 X
    { direction: [0, 1, 0], color: [0.2, 0.9, 0.2], label: "Y" }, // 绿色 Y
    { direction: [0, 0, 1], color: [0.2, 0.4, 0.9], label: "Z" }, // 蓝色 Z
  ];

  // 交互回调
  private onAxisClick?: (axis: string, positive: boolean) => void;

  constructor(renderer: Renderer, camera: Camera, canvas: HTMLCanvasElement) {
    this.renderer = renderer;
    this.camera = camera;
    this.canvas = canvas;

    this.createPipeline();
    this.createGeometry();
    this.createUniformBuffer();
    this.setupOrthoProjection();
  }

  /**
   * 设置轴点击回调
   */
  setOnAxisClick(callback: (axis: string, positive: boolean) => void): void {
    this.onAxisClick = callback;
  }

  /**
   * 创建渲染管线
   */
  private createPipeline(): void {
    const device = this.renderer.device;

    const shaderModule = device.createShaderModule({
      code: gizmoShaderCode,
    });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    // 顶点布局: position(3) + color(3)
    const vertexBufferLayout: GPUVertexBufferLayout = {
      arrayStride: 24,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
      ],
    };

    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [vertexBufferLayout],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.renderer.format }],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
      depthStencil: {
        format: this.renderer.depthFormat,
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });
  }

  /**
   * 创建 Gizmo 几何体（三个轴 + 箭头）
   */
  private createGeometry(): void {
    const vertices: number[] = [];
    const indices: number[] = [];
    let vertexOffset = 0;

    const axisLength = 0.8;
    const axisRadius = 0.04;
    const coneLength = 0.25;
    const coneRadius = 0.1;
    const segments = 12;

    for (const axis of this.axes) {
      const [dx, dy, dz] = axis.direction;
      const [r, g, b] = axis.color;

      // 创建轴的圆柱体
      const cylResult = this.createCylinder(
        [0, 0, 0],
        [dx * axisLength, dy * axisLength, dz * axisLength],
        axisRadius,
        segments,
        [r, g, b],
        vertexOffset,
      );
      vertices.push(...cylResult.vertices);
      indices.push(...cylResult.indices);
      vertexOffset += cylResult.vertexCount;

      // 创建箭头圆锥
      const coneStart: [number, number, number] = [
        dx * axisLength,
        dy * axisLength,
        dz * axisLength,
      ];
      const coneEnd: [number, number, number] = [
        dx * (axisLength + coneLength),
        dy * (axisLength + coneLength),
        dz * (axisLength + coneLength),
      ];
      const coneResult = this.createCone(
        coneStart,
        coneEnd,
        coneRadius,
        segments,
        [r, g, b],
        vertexOffset,
      );
      vertices.push(...coneResult.vertices);
      indices.push(...coneResult.indices);
      vertexOffset += coneResult.vertexCount;

      // 创建负方向的小球
      const sphereResult = this.createSphere(
        [-dx * 0.15, -dy * 0.15, -dz * 0.15],
        0.08,
        8,
        [r * 0.6, g * 0.6, b * 0.6],
        vertexOffset,
      );
      vertices.push(...sphereResult.vertices);
      indices.push(...sphereResult.indices);
      vertexOffset += sphereResult.vertexCount;
    }

    // 创建中心球
    const centerResult = this.createSphere(
      [0, 0, 0],
      0.1,
      12,
      [0.5, 0.5, 0.5],
      vertexOffset,
    );
    vertices.push(...centerResult.vertices);
    indices.push(...centerResult.indices);

    this.vertexCount = vertices.length / 6;
    this.indexCount = indices.length;

    const vertexData = new Float32Array(vertices);
    const indexData = new Uint16Array(indices);

    const device = this.renderer.device;

    this.vertexBuffer = device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vertexBuffer, 0, vertexData);

    this.indexBuffer = device.createBuffer({
      size: indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.indexBuffer, 0, indexData);
  }

  /**
   * 创建圆柱体几何
   */
  private createCylinder(
    start: [number, number, number],
    end: [number, number, number],
    radius: number,
    segments: number,
    color: [number, number, number],
    indexOffset: number,
  ): { vertices: number[]; indices: number[]; vertexCount: number } {
    const vertices: number[] = [];
    const indices: number[] = [];

    // 计算方向和长度
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const dz = end[2] - start[2];
    const length = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // 计算旋转矩阵
    const dir = [dx / length, dy / length, dz / length];
    const up = Math.abs(dir[1]) < 0.99 ? [0, 1, 0] : [1, 0, 0];
    const right = this.cross(
      up as [number, number, number],
      dir as [number, number, number],
    );
    this.normalize(right);
    const actualUp = this.cross(dir as [number, number, number], right);

    // 生成圆柱顶点
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      // 底面
      const nx0 = right[0] * cos + actualUp[0] * sin;
      const ny0 = right[1] * cos + actualUp[1] * sin;
      const nz0 = right[2] * cos + actualUp[2] * sin;
      vertices.push(
        start[0] + nx0 * radius,
        start[1] + ny0 * radius,
        start[2] + nz0 * radius,
        color[0],
        color[1],
        color[2],
      );

      // 顶面
      vertices.push(
        end[0] + nx0 * radius,
        end[1] + ny0 * radius,
        end[2] + nz0 * radius,
        color[0],
        color[1],
        color[2],
      );
    }

    // 生成索引
    for (let i = 0; i < segments; i++) {
      const i0 = indexOffset + i * 2;
      const i1 = indexOffset + i * 2 + 1;
      const i2 = indexOffset + (i + 1) * 2;
      const i3 = indexOffset + (i + 1) * 2 + 1;
      indices.push(i0, i1, i2, i2, i1, i3);
    }

    return { vertices, indices, vertexCount: (segments + 1) * 2 };
  }

  /**
   * 创建圆锥几何
   */
  private createCone(
    base: [number, number, number],
    tip: [number, number, number],
    radius: number,
    segments: number,
    color: [number, number, number],
    indexOffset: number,
  ): { vertices: number[]; indices: number[]; vertexCount: number } {
    const vertices: number[] = [];
    const indices: number[] = [];

    const dx = tip[0] - base[0];
    const dy = tip[1] - base[1];
    const dz = tip[2] - base[2];
    const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const dir = [dx / length, dy / length, dz / length];

    const up = Math.abs(dir[1]) < 0.99 ? [0, 1, 0] : [1, 0, 0];
    const right = this.cross(
      up as [number, number, number],
      dir as [number, number, number],
    );
    this.normalize(right);
    const actualUp = this.cross(dir as [number, number, number], right);

    // 尖端
    vertices.push(tip[0], tip[1], tip[2], color[0], color[1], color[2]);

    // 底面圆环
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const nx = right[0] * cos + actualUp[0] * sin;
      const ny = right[1] * cos + actualUp[1] * sin;
      const nz = right[2] * cos + actualUp[2] * sin;
      vertices.push(
        base[0] + nx * radius,
        base[1] + ny * radius,
        base[2] + nz * radius,
        color[0],
        color[1],
        color[2],
      );
    }

    // 索引（侧面）
    for (let i = 0; i < segments; i++) {
      indices.push(indexOffset, indexOffset + i + 1, indexOffset + i + 2);
    }

    // 底面中心
    const baseCenterIdx = indexOffset + segments + 2;
    vertices.push(
      base[0],
      base[1],
      base[2],
      color[0] * 0.7,
      color[1] * 0.7,
      color[2] * 0.7,
    );

    // 底面索引
    for (let i = 0; i < segments; i++) {
      indices.push(baseCenterIdx, indexOffset + i + 2, indexOffset + i + 1);
    }

    return { vertices, indices, vertexCount: segments + 3 };
  }

  /**
   * 创建球体几何
   */
  private createSphere(
    center: [number, number, number],
    radius: number,
    segments: number,
    color: [number, number, number],
    indexOffset: number,
  ): { vertices: number[]; indices: number[]; vertexCount: number } {
    const vertices: number[] = [];
    const indices: number[] = [];
    const rings = segments / 2;

    for (let ring = 0; ring <= rings; ring++) {
      const phi = (ring / rings) * Math.PI;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);

      for (let seg = 0; seg <= segments; seg++) {
        const theta = (seg / segments) * Math.PI * 2;
        const x = center[0] + radius * sinPhi * Math.cos(theta);
        const y = center[1] + radius * cosPhi;
        const z = center[2] + radius * sinPhi * Math.sin(theta);
        vertices.push(x, y, z, color[0], color[1], color[2]);
      }
    }

    for (let ring = 0; ring < rings; ring++) {
      for (let seg = 0; seg < segments; seg++) {
        const current = indexOffset + ring * (segments + 1) + seg;
        const next = current + segments + 1;
        indices.push(current, next, current + 1);
        indices.push(current + 1, next, next + 1);
      }
    }

    return { vertices, indices, vertexCount: (rings + 1) * (segments + 1) };
  }

  /**
   * 创建 uniform buffer
   */
  private createUniformBuffer(): void {
    const device = this.renderer.device;

    // viewMatrix(64) + projMatrix(64) = 128 bytes
    this.uniformBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });
  }

  /**
   * 设置正交投影矩阵
   */
  private setupOrthoProjection(): void {
    const s = 1.5; // 场景大小
    // 正交投影矩阵
    this.projMatrix[0] = 1 / s;
    this.projMatrix[5] = 1 / s;
    this.projMatrix[10] = -1 / 10;
    this.projMatrix[14] = 0;
    this.projMatrix[15] = 1;
  }

  /**
   * 更新 Gizmo 视图矩阵（从相机提取旋转部分）
   */
  private updateViewMatrix(): void {
    // 从相机视图矩阵提取旋转部分（去除平移）
    const camView = this.camera.viewMatrix;

    // 复制旋转部分
    this.viewMatrix[0] = camView[0];
    this.viewMatrix[1] = camView[1];
    this.viewMatrix[2] = camView[2];
    this.viewMatrix[3] = 0;

    this.viewMatrix[4] = camView[4];
    this.viewMatrix[5] = camView[5];
    this.viewMatrix[6] = camView[6];
    this.viewMatrix[7] = 0;

    this.viewMatrix[8] = camView[8];
    this.viewMatrix[9] = camView[9];
    this.viewMatrix[10] = camView[10];
    this.viewMatrix[11] = 0;

    // 设置固定的观察距离
    this.viewMatrix[12] = 0;
    this.viewMatrix[13] = 0;
    this.viewMatrix[14] = -3;
    this.viewMatrix[15] = 1;
  }

  /**
   * 渲染 Gizmo
   */
  render(pass: GPURenderPassEncoder): void {
    // 更新视图矩阵
    this.updateViewMatrix();

    // 计算 viewport 位置（右上角）
    const dpr = window.devicePixelRatio || 1;
    let gizmoSize = Math.floor(this.size * dpr);
    const marginX = Math.floor(this.margin * dpr);
    const marginY = Math.floor(this.margin * dpr);

    // 确保 Gizmo 不会超出 canvas 范围
    const maxSize = Math.min(
      this.canvas.width - marginX * 2,
      this.canvas.height - marginY * 2,
    );
    if (maxSize < 50) {
      // canvas 太小，跳过渲染
      return;
    }
    gizmoSize = Math.min(gizmoSize, maxSize);

    const x = Math.max(0, this.canvas.width - gizmoSize - marginX);
    const y = marginY;

    // 设置 viewport
    pass.setViewport(x, y, gizmoSize, gizmoSize, 0, 1);
    pass.setScissorRect(x, y, gizmoSize, gizmoSize);

    // 更新 uniform
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Float32Array(this.viewMatrix),
    );
    this.renderer.device.queue.writeBuffer(
      this.uniformBuffer,
      64,
      new Float32Array(this.projMatrix),
    );

    // 绘制
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, "uint16");
    pass.drawIndexed(this.indexCount);

    // 恢复全屏 viewport
    pass.setViewport(0, 0, this.canvas.width, this.canvas.height, 0, 1);
    pass.setScissorRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * 处理点击事件，检测是否点击了某个轴
   */
  handleClick(clientX: number, clientY: number): boolean {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // 计算 Gizmo 区域
    const gizmoSize = this.size;
    const marginX = this.margin;
    const marginY = this.margin;
    const gizmoLeft = rect.right - gizmoSize - marginX;
    const gizmoTop = rect.top + marginY;
    const gizmoRight = gizmoLeft + gizmoSize;
    const gizmoBottom = gizmoTop + gizmoSize;

    // 检查是否在 Gizmo 区域内
    if (
      clientX < gizmoLeft ||
      clientX > gizmoRight ||
      clientY < gizmoTop ||
      clientY > gizmoBottom
    ) {
      return false;
    }

    // 计算在 Gizmo 中的相对位置（-1 到 1）
    const relX = ((clientX - gizmoLeft) / gizmoSize) * 2 - 1;
    const relY = -(((clientY - gizmoTop) / gizmoSize) * 2 - 1);

    // 检测点击的轴
    const clickedAxis = this.detectClickedAxis(relX, relY);
    if (clickedAxis && this.onAxisClick) {
      this.onAxisClick(clickedAxis.axis, clickedAxis.positive);
      return true;
    }

    return false;
  }

  /**
   * 检测点击了哪个轴
   */
  private detectClickedAxis(
    relX: number,
    relY: number,
  ): { axis: string; positive: boolean } | null {
    // 将屏幕坐标转换为 Gizmo 空间
    // 使用视图矩阵的逆来判断
    const threshold = 0.4;

    // 计算各轴在屏幕上的投影位置
    for (const axis of this.axes) {
      const [dx, dy, dz] = axis.direction;

      // 正向轴端点
      const posX =
        this.viewMatrix[0] * dx +
        this.viewMatrix[4] * dy +
        this.viewMatrix[8] * dz;
      const posY =
        this.viewMatrix[1] * dx +
        this.viewMatrix[5] * dy +
        this.viewMatrix[9] * dz;

      // 检查正向
      const distPos = Math.sqrt(
        (relX - posX * 0.5) ** 2 + (relY - posY * 0.5) ** 2,
      );
      if (distPos < threshold) {
        return { axis: axis.label, positive: true };
      }

      // 检查负向
      const distNeg = Math.sqrt(
        (relX + posX * 0.15) ** 2 + (relY + posY * 0.15) ** 2,
      );
      if (distNeg < threshold * 0.5) {
        return { axis: axis.label, positive: false };
      }
    }

    return null;
  }

  // 向量工具函数
  private cross(
    a: [number, number, number],
    b: [number, number, number],
  ): [number, number, number] {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  private normalize(v: [number, number, number]): void {
    const len = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
    if (len > 0) {
      v[0] /= len;
      v[1] /= len;
      v[2] /= len;
    }
  }

  /**
   * 设置 Gizmo 大小
   */
  setSize(size: number): void {
    this.size = size;
  }

  /**
   * 设置边距
   */
  setMargin(margin: number): void {
    this.margin = margin;
  }
}
