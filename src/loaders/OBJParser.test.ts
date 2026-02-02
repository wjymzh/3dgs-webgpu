/**
 * OBJParser Unit Tests
 * Tests for OBJ file parsing functionality
 */

import { describe, it, expect, vi } from 'vitest';
import { OBJParser } from './OBJParser';

describe('OBJParser', () => {
  describe('Polygon Triangulation (Requirement 1.4)', () => {
    it('should keep triangles unchanged (3 vertices)', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Triangle should produce exactly 1 triangle (3 indices)
      expect(result.objects[0].indices.length).toBe(3);
      // 3 unique vertices
      expect(result.objects[0].positions.length).toBe(9); // 3 vertices * 3 components
    });

    it('should triangulate quad (4 vertices) into 2 triangles', () => {
      const parser = new OBJParser();
      
      // Define a quad (square)
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Quad should produce 2 triangles (n-2 = 4-2 = 2)
      // 2 triangles * 3 indices = 6 indices
      expect(result.objects[0].indices.length).toBe(6);
      
      // Verify fan triangulation: (v0, v1, v2), (v0, v2, v3)
      const positions = result.objects[0].positions;
      const indices = result.objects[0].indices;
      
      // First triangle: v0, v1, v2
      // v0 = (0, 0, 0)
      const v0Idx = indices[0] * 3;
      expect(positions[v0Idx]).toBe(0);
      expect(positions[v0Idx + 1]).toBe(0);
      expect(positions[v0Idx + 2]).toBe(0);
      
      // v1 = (1, 0, 0)
      const v1Idx = indices[1] * 3;
      expect(positions[v1Idx]).toBe(1);
      expect(positions[v1Idx + 1]).toBe(0);
      expect(positions[v1Idx + 2]).toBe(0);
      
      // v2 = (1, 1, 0)
      const v2Idx = indices[2] * 3;
      expect(positions[v2Idx]).toBe(1);
      expect(positions[v2Idx + 1]).toBe(1);
      expect(positions[v2Idx + 2]).toBe(0);
      
      // Second triangle: v0, v2, v3
      // v0 again
      const v0Idx2 = indices[3] * 3;
      expect(positions[v0Idx2]).toBe(0);
      expect(positions[v0Idx2 + 1]).toBe(0);
      expect(positions[v0Idx2 + 2]).toBe(0);
      
      // v2 again
      const v2Idx2 = indices[4] * 3;
      expect(positions[v2Idx2]).toBe(1);
      expect(positions[v2Idx2 + 1]).toBe(1);
      expect(positions[v2Idx2 + 2]).toBe(0);
      
      // v3 = (0, 1, 0)
      const v3Idx = indices[5] * 3;
      expect(positions[v3Idx]).toBe(0);
      expect(positions[v3Idx + 1]).toBe(1);
      expect(positions[v3Idx + 2]).toBe(0);
    });

    it('should triangulate pentagon (5 vertices) into 3 triangles', () => {
      const parser = new OBJParser();
      
      // Define a pentagon
      const objText = `
v 0 0 0
v 1 0 0
v 1.5 0.5 0
v 0.5 1 0
v -0.5 0.5 0
f 1 2 3 4 5
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Pentagon should produce 3 triangles (n-2 = 5-2 = 3)
      // 3 triangles * 3 indices = 9 indices
      expect(result.objects[0].indices.length).toBe(9);
      
      // Verify all 5 original vertices appear in the output
      expect(result.objects[0].positions.length).toBe(15); // 5 vertices * 3 components
    });

    it('should triangulate hexagon (6 vertices) into 4 triangles', () => {
      const parser = new OBJParser();
      
      // Define a hexagon
      const objText = `
v 0 0 0
v 1 0 0
v 1.5 0.5 0
v 1 1 0
v 0 1 0
v -0.5 0.5 0
f 1 2 3 4 5 6
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Hexagon should produce 4 triangles (n-2 = 6-2 = 4)
      // 4 triangles * 3 indices = 12 indices
      expect(result.objects[0].indices.length).toBe(12);
      
      // Verify all 6 original vertices appear in the output
      expect(result.objects[0].positions.length).toBe(18); // 6 vertices * 3 components
    });

    it('should triangulate octagon (8 vertices) into 6 triangles', () => {
      const parser = new OBJParser();
      
      // Define an octagon
      const objText = `
v 0 0 0
v 1 0 0
v 1.5 0.5 0
v 1.5 1 0
v 1 1.5 0
v 0 1.5 0
v -0.5 1 0
v -0.5 0.5 0
f 1 2 3 4 5 6 7 8
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Octagon should produce 6 triangles (n-2 = 8-2 = 6)
      // 6 triangles * 3 indices = 18 indices
      expect(result.objects[0].indices.length).toBe(18);
      
      // Verify all 8 original vertices appear in the output
      expect(result.objects[0].positions.length).toBe(24); // 8 vertices * 3 components
    });

    it('should correctly apply fan triangulation pattern', () => {
      const parser = new OBJParser();
      
      // Define a quad with distinct vertices to verify fan pattern
      const objText = `
v 0 0 0
v 2 0 0
v 2 2 0
v 0 2 0
f 1 2 3 4
`;
      
      const result = parser.parse(objText);
      const positions = result.objects[0].positions;
      const indices = result.objects[0].indices;
      
      // Fan triangulation for quad [v0, v1, v2, v3]:
      // Triangle 1: (v0, v1, v2)
      // Triangle 2: (v0, v2, v3)
      
      // Helper to get position at index
      const getPos = (idx: number) => [
        positions[idx * 3],
        positions[idx * 3 + 1],
        positions[idx * 3 + 2]
      ];
      
      // Triangle 1: indices[0], indices[1], indices[2]
      const t1v0 = getPos(indices[0]);
      const t1v1 = getPos(indices[1]);
      const t1v2 = getPos(indices[2]);
      
      expect(t1v0).toEqual([0, 0, 0]); // v0
      expect(t1v1).toEqual([2, 0, 0]); // v1
      expect(t1v2).toEqual([2, 2, 0]); // v2
      
      // Triangle 2: indices[3], indices[4], indices[5]
      const t2v0 = getPos(indices[3]);
      const t2v1 = getPos(indices[4]);
      const t2v2 = getPos(indices[5]);
      
      expect(t2v0).toEqual([0, 0, 0]); // v0 (fan center)
      expect(t2v1).toEqual([2, 2, 0]); // v2
      expect(t2v2).toEqual([0, 2, 0]); // v3
    });

    it('should triangulate polygon with texture coordinates', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3 4/4
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Quad produces 2 triangles
      expect(result.objects[0].indices.length).toBe(6);
      // 4 unique vertices with UVs
      expect(result.objects[0].uvs.length).toBe(8); // 4 vertices * 2 components
    });

    it('should triangulate polygon with normals', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1 4//1
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Quad produces 2 triangles
      expect(result.objects[0].indices.length).toBe(6);
      // All vertices share the same normal
      expect(result.objects[0].normals.length).toBe(12); // 4 vertices * 3 components
      
      // All normals should be (0, 0, 1)
      for (let i = 0; i < 4; i++) {
        expect(result.objects[0].normals[i * 3]).toBe(0);
        expect(result.objects[0].normals[i * 3 + 1]).toBe(0);
        expect(result.objects[0].normals[i * 3 + 2]).toBe(1);
      }
    });

    it('should triangulate polygon with full v/vt/vn format', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0.5 0.5 0.5
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vt 0.5 0.5
vn 0 0 1
vn 0 0 1
vn 0 0 1
vn 0 0 1
vn 0 1 0
f 1/1/1 2/2/2 3/3/3 4/4/4 5/5/5
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Pentagon produces 3 triangles (5-2 = 3)
      expect(result.objects[0].indices.length).toBe(9);
      // 5 unique vertices
      expect(result.objects[0].positions.length).toBe(15); // 5 * 3
      expect(result.objects[0].uvs.length).toBe(10); // 5 * 2
      expect(result.objects[0].normals.length).toBe(15); // 5 * 3
    });

    it('should handle multiple polygon faces in same object', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 3 0 0
v 3 1 0
v 2 1 0
f 1 2 3 4
f 5 6 7 8
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Two quads = 2 * 2 triangles = 4 triangles
      // 4 triangles * 3 indices = 12 indices
      expect(result.objects[0].indices.length).toBe(12);
    });

    it('should handle mixed triangles and polygons', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 3 1 0
v 2 1 0
f 1 2 3
f 4 5 6 7
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // 1 triangle + 1 quad (2 triangles) = 3 triangles
      // 3 triangles * 3 indices = 9 indices
      expect(result.objects[0].indices.length).toBe(9);
    });
  });

  describe('Negative Index Resolution (Requirement 1.3)', () => {
    it('should resolve -1 to the last vertex', () => {
      const parser = new OBJParser();
      
      // Create OBJ with 4 vertices and a face using negative indices
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f -4 -3 -2
f -3 -2 -1
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].indices.length).toBe(6); // 2 triangles * 3 vertices
      
      // First face: -4, -3, -2 should resolve to indices 0, 1, 2
      // Second face: -3, -2, -1 should resolve to indices 1, 2, 3
      // The positions should match the original vertices
      const positions = result.objects[0].positions;
      
      // First triangle vertices (from face -4 -3 -2)
      // -4 -> vertex 0 (0, 0, 0)
      expect(positions[0]).toBe(0);
      expect(positions[1]).toBe(0);
      expect(positions[2]).toBe(0);
      
      // -3 -> vertex 1 (1, 0, 0)
      expect(positions[3]).toBe(1);
      expect(positions[4]).toBe(0);
      expect(positions[5]).toBe(0);
      
      // -2 -> vertex 2 (1, 1, 0)
      expect(positions[6]).toBe(1);
      expect(positions[7]).toBe(1);
      expect(positions[8]).toBe(0);
    });

    it('should handle mixed positive and negative indices', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
f 1 2 -1
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      const positions = result.objects[0].positions;
      
      // Vertex 1 (index 0): (0, 0, 0)
      expect(positions[0]).toBe(0);
      expect(positions[1]).toBe(0);
      expect(positions[2]).toBe(0);
      
      // Vertex 2 (index 1): (1, 0, 0)
      expect(positions[3]).toBe(1);
      expect(positions[4]).toBe(0);
      expect(positions[5]).toBe(0);
      
      // Vertex -1 (last vertex, index 2): (1, 1, 0)
      expect(positions[6]).toBe(1);
      expect(positions[7]).toBe(1);
      expect(positions[8]).toBe(0);
    });

    it('should handle negative indices with texture coordinates', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
vt 0 0
vt 1 0
vt 1 1
f -3/-3 -2/-2 -1/-1
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      const positions = result.objects[0].positions;
      const uvs = result.objects[0].uvs;
      
      // Check positions
      expect(positions[0]).toBe(0);
      expect(positions[1]).toBe(0);
      expect(positions[2]).toBe(0);
      
      // Check UVs - -3 -> uv index 0 (0, 0)
      expect(uvs[0]).toBe(0);
      expect(uvs[1]).toBe(0);
      
      // -2 -> uv index 1 (1, 0)
      expect(uvs[2]).toBe(1);
      expect(uvs[3]).toBe(0);
      
      // -1 -> uv index 2 (1, 1)
      expect(uvs[4]).toBe(1);
      expect(uvs[5]).toBe(1);
    });

    it('should handle negative indices with normals', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
vn 0 0 1
vn 0 1 0
vn 1 0 0
f -3//-3 -2//-2 -1//-1
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      const normals = result.objects[0].normals;
      
      // -3 -> normal index 0 (0, 0, 1)
      expect(normals[0]).toBe(0);
      expect(normals[1]).toBe(0);
      expect(normals[2]).toBe(1);
      
      // -2 -> normal index 1 (0, 1, 0)
      expect(normals[3]).toBe(0);
      expect(normals[4]).toBe(1);
      expect(normals[5]).toBe(0);
      
      // -1 -> normal index 2 (1, 0, 0)
      expect(normals[6]).toBe(1);
      expect(normals[7]).toBe(0);
      expect(normals[8]).toBe(0);
    });

    it('should handle negative indices with full v/vt/vn format', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
vt 0 0
vt 1 0
vt 1 1
vn 0 0 1
vn 0 1 0
vn 1 0 0
f -3/-3/-3 -2/-2/-2 -1/-1/-1
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      const positions = result.objects[0].positions;
      const uvs = result.objects[0].uvs;
      const normals = result.objects[0].normals;
      
      // Verify all data is correctly resolved
      expect(positions.length).toBe(9); // 3 vertices * 3 components
      expect(uvs.length).toBe(6); // 3 vertices * 2 components
      expect(normals.length).toBe(9); // 3 vertices * 3 components
      
      // Check last vertex (-1 resolves to index 2)
      expect(positions[6]).toBe(1);
      expect(positions[7]).toBe(1);
      expect(positions[8]).toBe(0);
      
      expect(uvs[4]).toBe(1);
      expect(uvs[5]).toBe(1);
      
      expect(normals[6]).toBe(1);
      expect(normals[7]).toBe(0);
      expect(normals[8]).toBe(0);
    });
  });

  describe('Multi-Object/Group Support (Requirement 2.4)', () => {
    it('should create separate objects for each "o" directive', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
o Object1
f 1 2 3
o Object2
f 4 5 6
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(2);
      expect(result.objects[0].name).toBe('Object1');
      expect(result.objects[1].name).toBe('Object2');
      
      // Each object should have its own triangle
      expect(result.objects[0].indices.length).toBe(3);
      expect(result.objects[1].indices.length).toBe(3);
    });

    it('should create separate objects for each "g" directive', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
g Group1
f 1 2 3
g Group2
f 4 5 6
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(2);
      expect(result.objects[0].name).toBe('Group1');
      expect(result.objects[1].name).toBe('Group2');
      
      // Each group should have its own triangle
      expect(result.objects[0].indices.length).toBe(3);
      expect(result.objects[1].indices.length).toBe(3);
    });

    it('should handle mixed "o" and "g" directives', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
v 4 0 0
v 5 0 0
v 4.5 1 0
o Object1
f 1 2 3
g Group1
f 4 5 6
o Object2
f 7 8 9
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(3);
      expect(result.objects[0].name).toBe('Object1');
      expect(result.objects[1].name).toBe('Group1');
      expect(result.objects[2].name).toBe('Object2');
    });

    it('should create default object when no o/g directive before faces', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].name).toBe('default');
    });

    it('should handle object name with spaces', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
o My Object Name
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].name).toBe('My Object Name');
    });

    it('should handle group name with spaces', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
g My Group Name
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].name).toBe('My Group Name');
    });

    it('should skip empty objects (o/g without faces)', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
o EmptyObject
o ObjectWithFaces
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      // EmptyObject should be skipped since it has no faces
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].name).toBe('ObjectWithFaces');
    });

    it('should maintain separate vertex data for each object', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 10 10 10
v 11 10 10
v 10.5 11 10
o Object1
f 1 2 3
o Object2
f 4 5 6
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(2);
      
      // Object1 should have vertices from first triangle
      const obj1Positions = result.objects[0].positions;
      expect(obj1Positions[0]).toBe(0);
      expect(obj1Positions[1]).toBe(0);
      expect(obj1Positions[2]).toBe(0);
      
      // Object2 should have vertices from second triangle
      const obj2Positions = result.objects[1].positions;
      expect(obj2Positions[0]).toBe(10);
      expect(obj2Positions[1]).toBe(10);
      expect(obj2Positions[2]).toBe(10);
    });

    it('should handle multiple faces per object', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 0 0 1
v 1 0 1
v 0.5 1 1
v 10 0 0
v 11 0 0
v 10.5 1 0
o Object1
f 1 2 3
f 4 5 6
o Object2
f 7 8 9
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(2);
      
      // Object1 should have 2 triangles (6 indices)
      expect(result.objects[0].indices.length).toBe(6);
      
      // Object2 should have 1 triangle (3 indices)
      expect(result.objects[1].indices.length).toBe(3);
    });

    it('should handle objects with texture coordinates and normals', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
vt 0 0
vt 1 0
vt 0.5 1
vn 0 0 1
vn 0 0 -1
o Object1
f 1/1/1 2/2/1 3/3/1
o Object2
f 4/1/2 5/2/2 6/3/2
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(2);
      
      // Object1 should have UVs and normals
      expect(result.objects[0].uvs.length).toBe(6); // 3 vertices * 2 components
      expect(result.objects[0].normals.length).toBe(9); // 3 vertices * 3 components
      
      // Object1 normals should all be (0, 0, 1)
      expect(result.objects[0].normals[0]).toBe(0);
      expect(result.objects[0].normals[1]).toBe(0);
      expect(result.objects[0].normals[2]).toBe(1);
      
      // Object2 normals should all be (0, 0, -1)
      expect(result.objects[1].normals[0]).toBe(0);
      expect(result.objects[1].normals[1]).toBe(0);
      expect(result.objects[1].normals[2]).toBe(-1);
    });

    it('should reset vertex deduplication for each object', () => {
      const parser = new OBJParser();
      
      // Both objects use the same vertex indices but should have separate vertex data
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
o Object1
f 1 2 3
o Object2
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(2);
      
      // Both objects should have their own vertex data
      expect(result.objects[0].positions.length).toBe(9); // 3 vertices * 3 components
      expect(result.objects[1].positions.length).toBe(9); // 3 vertices * 3 components
      
      // Both should have indices starting from 0
      expect(result.objects[0].indices).toEqual([0, 1, 2]);
      expect(result.objects[1].indices).toEqual([0, 1, 2]);
    });

    it('should handle o/g directive without name (use default)', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
o
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].name).toBe('default');
    });

    it('should correctly count k objects for k distinct o/g directives', () => {
      const parser = new OBJParser();
      
      // Create OBJ with 5 distinct objects
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
o Obj1
f 1 2 3
g Grp2
f 1 2 3
o Obj3
f 1 2 3
g Grp4
f 1 2 3
o Obj5
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      // Should have exactly 5 objects
      expect(result.objects.length).toBe(5);
      expect(result.objects[0].name).toBe('Obj1');
      expect(result.objects[1].name).toBe('Grp2');
      expect(result.objects[2].name).toBe('Obj3');
      expect(result.objects[3].name).toBe('Grp4');
      expect(result.objects[4].name).toBe('Obj5');
    });
  });
});


  describe('Error Tolerance (Requirement 1.5)', () => {
    it('should skip invalid vertex lines and continue parsing', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v invalid data here
v 1 0 0
v 0.5 1 0
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Should have 3 valid vertices (skipped the invalid one)
      // Face uses indices 1, 2, 3 which map to the 3 valid vertices
      expect(result.objects[0].indices.length).toBe(3);
    });

    it('should skip lines with incomplete vertex data', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0
v 1 0 0
v 0.5 1 0
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].indices.length).toBe(3);
    });

    it('should skip invalid texture coordinate lines', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
vt 0 0
vt invalid
vt 1 0
vt 0.5 1
f 1/1 2/2 3/3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].indices.length).toBe(3);
    });

    it('should skip invalid normal lines', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
vn 0 0 1
vn not a normal
vn 0 1 0
f 1//1 2//1 3//1
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].indices.length).toBe(3);
      // All normals should be (0, 0, 1)
      expect(result.objects[0].normals[0]).toBe(0);
      expect(result.objects[0].normals[1]).toBe(0);
      expect(result.objects[0].normals[2]).toBe(1);
    });

    it('should skip invalid face lines and continue parsing', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
f 1 2 3
f invalid face
f 4 5 6
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Should have 2 valid triangles (6 indices)
      expect(result.objects[0].indices.length).toBe(6);
    });

    it('should skip face lines with insufficient vertices', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
f 1 2 3
f 4 5
f 4 5 6
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // Should have 2 valid triangles (6 indices), skipping the face with only 2 vertices
      expect(result.objects[0].indices.length).toBe(6);
    });

    it('should handle completely malformed lines gracefully', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
@#$%^&*()
v 1 0 0
!!!invalid!!!
v 0.5 1 0
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].indices.length).toBe(3);
    });

    it('should skip lines with NaN values', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v NaN NaN NaN
v 1 0 0
v 0.5 1 0
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].indices.length).toBe(3);
    });

    it('should produce same result as parsing without invalid lines', () => {
      const parser1 = new OBJParser();
      const parser2 = new OBJParser();
      
      // OBJ with invalid lines
      const objWithInvalid = `
v 0 0 0
v invalid
v 1 0 0
garbage line
v 0.5 1 0
vt 0 0
vt bad uv
vt 1 0
vt 0.5 1
vn 0 0 1
vn bad normal
f 1/1/1 2/2/1 3/3/1
f bad face
`;
      
      // Same OBJ without invalid lines
      const objWithoutInvalid = `
v 0 0 0
v 1 0 0
v 0.5 1 0
vt 0 0
vt 1 0
vt 0.5 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
`;
      
      const result1 = parser1.parse(objWithInvalid);
      const result2 = parser2.parse(objWithoutInvalid);
      
      // Both should produce the same geometry
      expect(result1.objects.length).toBe(result2.objects.length);
      expect(result1.objects[0].positions).toEqual(result2.objects[0].positions);
      expect(result1.objects[0].normals).toEqual(result2.objects[0].normals);
      expect(result1.objects[0].uvs).toEqual(result2.objects[0].uvs);
      expect(result1.objects[0].indices).toEqual(result2.objects[0].indices);
    });

    it('should handle empty lines and comments mixed with invalid lines', () => {
      const parser = new OBJParser();
      
      const objText = `
# This is a comment
v 0 0 0

v invalid
# Another comment
v 1 0 0

v 0.5 1 0
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].indices.length).toBe(3);
    });

    it('should handle invalid lines in multi-object files', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
o Object1
f 1 2 3
f invalid
o Object2
f bad face data
f 4 5 6
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(2);
      expect(result.objects[0].name).toBe('Object1');
      expect(result.objects[1].name).toBe('Object2');
      // Each object should have 1 valid triangle
      expect(result.objects[0].indices.length).toBe(3);
      expect(result.objects[1].indices.length).toBe(3);
    });

    it('should handle faces with out-of-range indices gracefully', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
f 1 2 3
f 100 200 300
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      // The face with out-of-range indices may produce default values or be handled
      // The important thing is that parsing continues and doesn't crash
      expect(result.objects[0].indices.length).toBeGreaterThanOrEqual(3);
    });

    it('should skip unsupported directives without error', () => {
      const parser = new OBJParser();
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
s 1
l 1 2
p 1
cstype bezier
deg 3
curv 0.0 1.0 1 2 3
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].indices.length).toBe(3);
    });

    it('should log warning for unsupported directives only on first occurrence (Requirement 6.2)', () => {
      const parser = new OBJParser();
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
s 1
s 2
s off
l 1 2
l 2 3
cstype bezier
f 1 2 3
`;
      
      const result = parser.parse(objText);
      
      // Should still parse successfully
      expect(result.objects.length).toBe(1);
      expect(result.objects[0].indices.length).toBe(3);
      
      // Should warn only once per unique unsupported directive
      // 's' appears 3 times but should only warn once
      // 'l' appears 2 times but should only warn once
      // 'cstype' appears 1 time
      const sWarnings = warnSpy.mock.calls.filter(call => 
        call[0].includes("'s'")
      );
      const lWarnings = warnSpy.mock.calls.filter(call => 
        call[0].includes("'l'")
      );
      const cstypeWarnings = warnSpy.mock.calls.filter(call => 
        call[0].includes("'cstype'")
      );
      
      expect(sWarnings.length).toBe(1);
      expect(lWarnings.length).toBe(1);
      expect(cstypeWarnings.length).toBe(1);
      
      warnSpy.mockRestore();
    });

    it('should include directive name and line number in warning message (Requirement 6.2)', () => {
      const parser = new OBJParser();
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      
      const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
unknowndirective somedata
f 1 2 3
`;
      
      parser.parse(objText);
      
      // Should have warned about the unknown directive
      expect(warnSpy).toHaveBeenCalled();
      const warningMessage = warnSpy.mock.calls[0][0];
      expect(warningMessage).toContain('unknowndirective');
      expect(warningMessage).toMatch(/\d+/); // Should contain line number
      
      warnSpy.mockRestore();
    });
  });


describe('Material-Mesh Association (Requirement 4.3)', () => {
  it('should parse usemtl directive and store materialName', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
usemtl TestMaterial
f 1 2 3
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects.length).toBe(1);
    expect(result.objects[0].materialName).toBe('TestMaterial');
  });

  it('should handle usemtl with spaces in material name', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
usemtl My Material Name
f 1 2 3
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects.length).toBe(1);
    expect(result.objects[0].materialName).toBe('My Material Name');
  });

  it('should create separate objects for different materials', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
usemtl Material1
f 1 2 3
usemtl Material2
f 4 5 6
`;
    
    const result = parser.parse(objText);
    
    // Should create 2 separate objects for 2 different materials
    expect(result.objects.length).toBe(2);
    expect(result.objects[0].materialName).toBe('Material1');
    expect(result.objects[1].materialName).toBe('Material2');
    
    // Each object should have its own triangle
    expect(result.objects[0].indices.length).toBe(3);
    expect(result.objects[1].indices.length).toBe(3);
  });

  it('should apply material to subsequent faces until next usemtl', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
v 4 0 0
v 5 0 0
v 4.5 1 0
usemtl RedMaterial
f 1 2 3
f 4 5 6
usemtl BlueMaterial
f 7 8 9
`;
    
    const result = parser.parse(objText);
    
    // First object should have 2 triangles with RedMaterial
    expect(result.objects.length).toBe(2);
    expect(result.objects[0].materialName).toBe('RedMaterial');
    expect(result.objects[0].indices.length).toBe(6); // 2 triangles
    
    // Second object should have 1 triangle with BlueMaterial
    expect(result.objects[1].materialName).toBe('BlueMaterial');
    expect(result.objects[1].indices.length).toBe(3); // 1 triangle
  });

  it('should handle faces before any usemtl (null material)', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
f 1 2 3
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects.length).toBe(1);
    expect(result.objects[0].materialName).toBeNull();
  });

  it('should handle usemtl before any faces', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
usemtl PresetMaterial
f 1 2 3
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects.length).toBe(1);
    expect(result.objects[0].materialName).toBe('PresetMaterial');
  });

  it('should handle usemtl with object/group directives', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
o Object1
usemtl Material1
f 1 2 3
o Object2
usemtl Material2
f 4 5 6
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects.length).toBe(2);
    expect(result.objects[0].name).toBe('Object1');
    expect(result.objects[0].materialName).toBe('Material1');
    expect(result.objects[1].name).toBe('Object2');
    expect(result.objects[1].materialName).toBe('Material2');
  });

  it('should handle multiple usemtl in same object', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
v 2 0 0
v 3 0 0
v 2.5 1 0
v 4 0 0
v 5 0 0
v 4.5 1 0
o MyObject
usemtl Mat1
f 1 2 3
usemtl Mat2
f 4 5 6
usemtl Mat3
f 7 8 9
`;
    
    const result = parser.parse(objText);
    
    // Should create 3 separate objects (one per material change)
    expect(result.objects.length).toBe(3);
    expect(result.objects[0].materialName).toBe('Mat1');
    expect(result.objects[1].materialName).toBe('Mat2');
    expect(result.objects[2].materialName).toBe('Mat3');
  });

  it('should handle usemtl without material name (empty)', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 0.5 1 0
usemtl
f 1 2 3
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects.length).toBe(1);
    // Empty usemtl should not change material (remains null)
    expect(result.objects[0].materialName).toBeNull();
  });

  it('should preserve material association through polygon triangulation', () => {
    const parser = new OBJParser();
    
    const objText = `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
usemtl QuadMaterial
f 1 2 3 4
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects.length).toBe(1);
    expect(result.objects[0].materialName).toBe('QuadMaterial');
    // Quad should be triangulated into 2 triangles
    expect(result.objects[0].indices.length).toBe(6);
  });
});


describe('Empty File Handling (Requirement 6.1)', () => {
  it('should return empty objects array for completely empty file', () => {
    const parser = new OBJParser();
    
    const result = parser.parse('');
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with only whitespace', () => {
    const parser = new OBJParser();
    
    const objText = `
    
    
    `;
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with only comments', () => {
    const parser = new OBJParser();
    
    const objText = `
# This is a comment
# Another comment
# OBJ file with no geometry
# Created by some tool
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with vertices but no faces', () => {
    const parser = new OBJParser();
    
    const objText = `
# OBJ file with vertices but no faces
v 0 0 0
v 1 0 0
v 0.5 1 0
v 1 1 0
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with only metadata (mtllib)', () => {
    const parser = new OBJParser();
    
    const objText = `
# OBJ file with only metadata
mtllib materials.mtl
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with vertices, normals, uvs but no faces', () => {
    const parser = new OBJParser();
    
    const objText = `
# OBJ file with all vertex data but no faces
v 0 0 0
v 1 0 0
v 0.5 1 0
vt 0 0
vt 1 0
vt 0.5 1
vn 0 0 1
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with object declaration but no faces', () => {
    const parser = new OBJParser();
    
    const objText = `
# OBJ file with object but no faces
o EmptyObject
v 0 0 0
v 1 0 0
v 0.5 1 0
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with group declaration but no faces', () => {
    const parser = new OBJParser();
    
    const objText = `
# OBJ file with group but no faces
g EmptyGroup
v 0 0 0
v 1 0 0
v 0.5 1 0
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with usemtl but no faces', () => {
    const parser = new OBJParser();
    
    const objText = `
# OBJ file with material reference but no faces
mtllib materials.mtl
usemtl SomeMaterial
v 0 0 0
v 1 0 0
v 0.5 1 0
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with smoothing group but no faces', () => {
    const parser = new OBJParser();
    
    const objText = `
# OBJ file with smoothing group but no faces
s 1
v 0 0 0
v 1 0 0
v 0.5 1 0
`;
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return empty objects array for file with Windows line endings (CRLF) and no geometry', () => {
    const parser = new OBJParser();
    
    const objText = "# Comment\r\nv 0 0 0\r\nv 1 0 0\r\nv 0.5 1 0\r\n";
    
    const result = parser.parse(objText);
    
    expect(result.objects).toEqual([]);
    expect(result.objects.length).toBe(0);
  });

  it('should return objects with geometry when faces are present', () => {
    const parser = new OBJParser();
    
    const objText = `
# OBJ file with geometry
v 0 0 0
v 1 0 0
v 0.5 1 0
f 1 2 3
`;
    
    const result = parser.parse(objText);
    
    // Should NOT be empty when faces are present
    expect(result.objects.length).toBe(1);
    expect(result.objects[0].indices.length).toBe(3);
  });
});
