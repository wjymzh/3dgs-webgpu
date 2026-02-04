/**
 * SplatTransformProxy - PLY/Splat 变换代理对象
 * 实现 TransformableObject 接口，让 TransformGizmo 可以操作 PLY 模型
 */

import type { TransformableObject, Vec3Tuple } from '../../types';
import type { IGSSplatRenderer } from '../../gs/IGSSplatRenderer';

export class SplatTransformProxy implements TransformableObject {
  position: Vec3Tuple;
  rotation: Vec3Tuple;
  scale: Vec3Tuple;

  private renderer: IGSSplatRenderer;
  private center: Vec3Tuple;

  constructor(renderer: IGSSplatRenderer, center: Vec3Tuple) {
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
