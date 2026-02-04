/**
 * MeshGroupProxy - 多 Mesh 组变换代理对象
 * 让 TransformGizmo 可以同时操作多个 Mesh
 */

import type { TransformableObject, SimpleBoundingBox, Vec3Tuple } from '../../types';
import type { Mesh } from '../../mesh/Mesh';

export class MeshGroupProxy implements TransformableObject {
  position: Vec3Tuple;
  rotation: Vec3Tuple;
  scale: Vec3Tuple;

  private meshes: Mesh[];

  constructor(meshes: Mesh[]) {
    this.meshes = meshes;

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

  setPosition(x: number, y: number, z: number): void {
    this.position = [x, y, z];
    for (const mesh of this.meshes) {
      mesh.setPosition(x, y, z);
    }
  }

  setRotation(x: number, y: number, z: number): void {
    this.rotation = [x, y, z];
    for (const mesh of this.meshes) {
      mesh.setRotation(x, y, z);
    }
  }

  setScale(x: number, y: number, z: number): void {
    this.scale = [x, y, z];
    for (const mesh of this.meshes) {
      mesh.setScale(x, y, z);
    }
  }

  /**
   * 获取组合包围盒
   */
  getBoundingBox(): SimpleBoundingBox | null {
    if (this.meshes.length === 0) return null;

    let combinedMin: Vec3Tuple | null = null;
    let combinedMax: Vec3Tuple | null = null;

    for (const mesh of this.meshes) {
      const bbox = mesh.getWorldBoundingBox();
      if (!bbox) continue;

      if (combinedMin === null || combinedMax === null) {
        combinedMin = [...bbox.min];
        combinedMax = [...bbox.max];
      } else {
        combinedMin[0] = Math.min(combinedMin[0], bbox.min[0]);
        combinedMin[1] = Math.min(combinedMin[1], bbox.min[1]);
        combinedMin[2] = Math.min(combinedMin[2], bbox.min[2]);
        combinedMax[0] = Math.max(combinedMax[0], bbox.max[0]);
        combinedMax[1] = Math.max(combinedMax[1], bbox.max[1]);
        combinedMax[2] = Math.max(combinedMax[2], bbox.max[2]);
      }
    }

    if (combinedMin === null || combinedMax === null) return null;

    return { min: combinedMin, max: combinedMax };
  }
}
