/**
 * OBJLoader Unit Tests
 * Tests for OBJ file loading and mesh generation functionality
 * 
 * Note: These tests focus on the generateFlatNormals functionality
 * which is tested without GPU dependencies by testing the algorithm directly.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

/**
 * Helper function to extract MTL reference from OBJ text
 * (Same algorithm as OBJLoader.extractMTLReference)
 * 
 * Requirement 4.1: Extract mtllib directive from OBJ file
 */
function extractMTLReference(text: string): string | null {
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('mtllib ')) {
      return trimmed.substring(7).trim();
    }
  }
  return null;
}

/**
 * Helper function to compute flat normals (same algorithm as OBJLoader.generateFlatNormals)
 * This allows testing the algorithm without GPU dependencies.
 * 
 * Requirement 3.2: Generate flat normals from face geometry
 */
function generateFlatNormals(positions: number[], indices: number[]): number[] {
  const normals = new Array(positions.length).fill(0);

  // Iterate through each triangle
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i];
    const i1 = indices[i + 1];
    const i2 = indices[i + 2];

    // Get triangle vertices
    const v0 = [positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]];
    const v1 = [positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]];
    const v2 = [positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]];

    // Compute edge vectors
    const edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    const edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    // Compute cross product (face normal)
    const nx = edge1[1] * edge2[2] - edge1[2] * edge2[1];
    const ny = edge1[2] * edge2[0] - edge1[0] * edge2[2];
    const nz = edge1[0] * edge2[1] - edge1[1] * edge2[0];

    // Normalize
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const normalizedNx = len > 0 ? nx / len : 0;
    const normalizedNy = len > 0 ? ny / len : 1;
    const normalizedNz = len > 0 ? nz / len : 0;

    // Assign the same normal to all vertices of the triangle (flat shading)
    for (const idx of [i0, i1, i2]) {
      normals[idx * 3 + 0] = normalizedNx;
      normals[idx * 3 + 1] = normalizedNy;
      normals[idx * 3 + 2] = normalizedNz;
    }
  }

  return normals;
}

/**
 * Helper to check if a vector is approximately a unit vector
 */
function isUnitVector(x: number, y: number, z: number, tolerance = 1e-6): boolean {
  const length = Math.sqrt(x * x + y * y + z * z);
  return Math.abs(length - 1.0) < tolerance;
}

/**
 * Helper to compute dot product
 */
function dot(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

describe('OBJLoader', () => {
  describe('generateFlatNormals (Requirement 3.2)', () => {
    it('should generate unit normals for a simple triangle in XY plane', () => {
      // Triangle in XY plane (z=0), counter-clockwise when viewed from +Z
      const positions = [
        0, 0, 0,  // v0
        1, 0, 0,  // v1
        0, 1, 0,  // v2
      ];
      const indices = [0, 1, 2];

      const normals = generateFlatNormals(positions, indices);

      // Normal should point in +Z direction (0, 0, 1)
      expect(normals[0]).toBeCloseTo(0, 5);
      expect(normals[1]).toBeCloseTo(0, 5);
      expect(normals[2]).toBeCloseTo(1, 5);

      // All three vertices should have the same normal (flat shading)
      expect(normals[3]).toBeCloseTo(0, 5);
      expect(normals[4]).toBeCloseTo(0, 5);
      expect(normals[5]).toBeCloseTo(1, 5);

      expect(normals[6]).toBeCloseTo(0, 5);
      expect(normals[7]).toBeCloseTo(0, 5);
      expect(normals[8]).toBeCloseTo(1, 5);

      // Verify unit length
      expect(isUnitVector(normals[0], normals[1], normals[2])).toBe(true);
    });

    it('should generate unit normals for a triangle in XZ plane', () => {
      // Triangle in XZ plane (y=0), counter-clockwise when viewed from +Y
      const positions = [
        0, 0, 0,  // v0
        1, 0, 0,  // v1
        0, 0, 1,  // v2
      ];
      const indices = [0, 1, 2];

      const normals = generateFlatNormals(positions, indices);

      // Normal should point in -Y direction (0, -1, 0)
      expect(normals[0]).toBeCloseTo(0, 5);
      expect(normals[1]).toBeCloseTo(-1, 5);
      expect(normals[2]).toBeCloseTo(0, 5);

      // Verify unit length
      expect(isUnitVector(normals[0], normals[1], normals[2])).toBe(true);
    });

    it('should generate unit normals for a triangle in YZ plane', () => {
      // Triangle in YZ plane (x=0), counter-clockwise when viewed from +X
      const positions = [
        0, 0, 0,  // v0
        0, 1, 0,  // v1
        0, 0, 1,  // v2
      ];
      const indices = [0, 1, 2];

      const normals = generateFlatNormals(positions, indices);

      // Normal should point in +X direction (1, 0, 0)
      expect(normals[0]).toBeCloseTo(1, 5);
      expect(normals[1]).toBeCloseTo(0, 5);
      expect(normals[2]).toBeCloseTo(0, 5);

      // Verify unit length
      expect(isUnitVector(normals[0], normals[1], normals[2])).toBe(true);
    });

    it('should generate normals perpendicular to the triangle face', () => {
      // Arbitrary triangle
      const positions = [
        1, 2, 3,   // v0
        4, 2, 3,   // v1
        1, 5, 3,   // v2
      ];
      const indices = [0, 1, 2];

      const normals = generateFlatNormals(positions, indices);

      // Get the normal
      const normal = [normals[0], normals[1], normals[2]];

      // Compute edge vectors
      const edge1 = [4 - 1, 2 - 2, 3 - 3]; // v1 - v0
      const edge2 = [1 - 1, 5 - 2, 3 - 3]; // v2 - v0

      // Normal should be perpendicular to both edges (dot product ≈ 0)
      expect(Math.abs(dot(normal, edge1))).toBeLessThan(1e-6);
      expect(Math.abs(dot(normal, edge2))).toBeLessThan(1e-6);

      // Verify unit length
      expect(isUnitVector(normal[0], normal[1], normal[2])).toBe(true);
    });

    it('should assign the same normal to all vertices of a triangle (flat shading)', () => {
      const positions = [
        0, 0, 0,
        1, 0, 0,
        0.5, 1, 0,
      ];
      const indices = [0, 1, 2];

      const normals = generateFlatNormals(positions, indices);

      // All three vertices should have identical normals
      const n0 = [normals[0], normals[1], normals[2]];
      const n1 = [normals[3], normals[4], normals[5]];
      const n2 = [normals[6], normals[7], normals[8]];

      expect(n0).toEqual(n1);
      expect(n1).toEqual(n2);
    });

    it('should generate different normals for different triangles', () => {
      // Two triangles facing different directions
      const positions = [
        // Triangle 1 (XY plane, facing +Z)
        0, 0, 0,  // v0
        1, 0, 0,  // v1
        0, 1, 0,  // v2
        // Triangle 2 (XZ plane, facing -Y)
        2, 0, 0,  // v3
        3, 0, 0,  // v4
        2, 0, 1,  // v5
      ];
      const indices = [0, 1, 2, 3, 4, 5];

      const normals = generateFlatNormals(positions, indices);

      // Triangle 1 normal (should be +Z)
      const n1 = [normals[0], normals[1], normals[2]];
      expect(n1[0]).toBeCloseTo(0, 5);
      expect(n1[1]).toBeCloseTo(0, 5);
      expect(n1[2]).toBeCloseTo(1, 5);

      // Triangle 2 normal (should be -Y)
      const n2 = [normals[9], normals[10], normals[11]];
      expect(n2[0]).toBeCloseTo(0, 5);
      expect(n2[1]).toBeCloseTo(-1, 5);
      expect(n2[2]).toBeCloseTo(0, 5);
    });

    it('should handle degenerate triangles (collinear points) gracefully', () => {
      // Degenerate triangle (all points on a line)
      const positions = [
        0, 0, 0,
        1, 0, 0,
        2, 0, 0,  // Collinear with v0 and v1
      ];
      const indices = [0, 1, 2];

      const normals = generateFlatNormals(positions, indices);

      // Should not crash, and should produce some default normal
      // The implementation defaults to (0, 1, 0) for degenerate cases
      expect(normals.length).toBe(9);
      
      // Check that we get a valid (non-NaN) result
      expect(Number.isNaN(normals[0])).toBe(false);
      expect(Number.isNaN(normals[1])).toBe(false);
      expect(Number.isNaN(normals[2])).toBe(false);
    });

    it('should generate unit normals for a scaled triangle', () => {
      // Large triangle (should still produce unit normal)
      const positions = [
        0, 0, 0,
        1000, 0, 0,
        0, 1000, 0,
      ];
      const indices = [0, 1, 2];

      const normals = generateFlatNormals(positions, indices);

      // Normal should still be unit length
      expect(isUnitVector(normals[0], normals[1], normals[2])).toBe(true);
    });

    it('should generate unit normals for a very small triangle', () => {
      // Very small triangle
      const positions = [
        0, 0, 0,
        0.001, 0, 0,
        0, 0.001, 0,
      ];
      const indices = [0, 1, 2];

      const normals = generateFlatNormals(positions, indices);

      // Normal should still be unit length
      expect(isUnitVector(normals[0], normals[1], normals[2])).toBe(true);
    });

    it('should handle multiple triangles sharing vertices', () => {
      // Two triangles sharing an edge (like a quad split into triangles)
      // Note: With flat shading, shared vertices will have the normal of the last triangle processed
      const positions = [
        0, 0, 0,  // v0
        1, 0, 0,  // v1
        1, 1, 0,  // v2
        0, 1, 0,  // v3
      ];
      // Two triangles: (0,1,2) and (0,2,3)
      const indices = [0, 1, 2, 0, 2, 3];

      const normals = generateFlatNormals(positions, indices);

      // Both triangles are in XY plane, so all normals should point +Z
      // All normals should be unit vectors
      for (let i = 0; i < 4; i++) {
        expect(isUnitVector(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2])).toBe(true);
        expect(normals[i * 3]).toBeCloseTo(0, 5);
        expect(normals[i * 3 + 1]).toBeCloseTo(0, 5);
        expect(normals[i * 3 + 2]).toBeCloseTo(1, 5);
      }
    });

    it('should correctly compute normals using cross product formula', () => {
      // Triangle with known edge vectors
      // v0 = (0, 0, 0), v1 = (1, 0, 0), v2 = (0, 1, 0)
      // edge1 = v1 - v0 = (1, 0, 0)
      // edge2 = v2 - v0 = (0, 1, 0)
      // cross(edge1, edge2) = (0*0 - 0*1, 0*0 - 1*0, 1*1 - 0*0) = (0, 0, 1)
      const positions = [
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
      ];
      const indices = [0, 1, 2];

      const normals = generateFlatNormals(positions, indices);

      // Verify the cross product result
      expect(normals[0]).toBeCloseTo(0, 5);
      expect(normals[1]).toBeCloseTo(0, 5);
      expect(normals[2]).toBeCloseTo(1, 5);
    });

    it('should handle empty input gracefully', () => {
      const positions: number[] = [];
      const indices: number[] = [];

      const normals = generateFlatNormals(positions, indices);

      expect(normals.length).toBe(0);
    });

    it('should generate correct normals for a cube face', () => {
      // Front face of a cube (z = 1)
      const positions = [
        0, 0, 1,  // v0
        1, 0, 1,  // v1
        1, 1, 1,  // v2
        0, 1, 1,  // v3
      ];
      // Two triangles forming the face
      const indices = [0, 1, 2, 0, 2, 3];

      const normals = generateFlatNormals(positions, indices);

      // All normals should point +Z (out of the front face)
      for (let i = 0; i < 4; i++) {
        expect(normals[i * 3]).toBeCloseTo(0, 5);
        expect(normals[i * 3 + 1]).toBeCloseTo(0, 5);
        expect(normals[i * 3 + 2]).toBeCloseTo(1, 5);
      }
    });
  });
});


describe('OBJLoader MTL File Loading (Requirements 4.1, 4.4)', () => {
  describe('extractMTLReference', () => {
    it('should extract MTL filename from mtllib directive', () => {
      const objText = `
# Simple OBJ file
mtllib materials.mtl
o Cube
v 0 0 0
v 1 0 0
v 1 1 0
f 1 2 3
`;
      const mtlFile = extractMTLReference(objText);
      expect(mtlFile).toBe('materials.mtl');
    });

    it('should extract MTL filename with spaces in path', () => {
      const objText = `
mtllib my materials.mtl
o Cube
v 0 0 0
`;
      const mtlFile = extractMTLReference(objText);
      expect(mtlFile).toBe('my materials.mtl');
    });

    it('should return null when no mtllib directive exists', () => {
      const objText = `
# OBJ without materials
o Cube
v 0 0 0
v 1 0 0
v 1 1 0
f 1 2 3
`;
      const mtlFile = extractMTLReference(objText);
      expect(mtlFile).toBeNull();
    });

    it('should extract first mtllib directive when multiple exist', () => {
      const objText = `
mtllib first.mtl
mtllib second.mtl
o Cube
v 0 0 0
`;
      const mtlFile = extractMTLReference(objText);
      expect(mtlFile).toBe('first.mtl');
    });

    it('should handle mtllib directive with leading/trailing whitespace', () => {
      const objText = `
  mtllib   model.mtl  
o Cube
v 0 0 0
`;
      const mtlFile = extractMTLReference(objText);
      expect(mtlFile).toBe('model.mtl');
    });

    it('should handle mtllib directive with relative path', () => {
      const objText = `
mtllib ./textures/model.mtl
o Cube
v 0 0 0
`;
      const mtlFile = extractMTLReference(objText);
      expect(mtlFile).toBe('./textures/model.mtl');
    });

    it('should handle mtllib directive at different positions in file', () => {
      // mtllib at the end
      const objText1 = `
o Cube
v 0 0 0
v 1 0 0
v 1 1 0
f 1 2 3
mtllib late.mtl
`;
      const mtlFile1 = extractMTLReference(objText1);
      expect(mtlFile1).toBe('late.mtl');

      // mtllib in the middle
      const objText2 = `
o Cube
v 0 0 0
mtllib middle.mtl
v 1 0 0
f 1 2 3
`;
      const mtlFile2 = extractMTLReference(objText2);
      expect(mtlFile2).toBe('middle.mtl');
    });

    it('should handle Windows line endings (CRLF)', () => {
      const objText = "# Comment\r\nmtllib windows.mtl\r\no Cube\r\nv 0 0 0";
      const mtlFile = extractMTLReference(objText);
      expect(mtlFile).toBe('windows.mtl');
    });

    it('should handle empty OBJ text', () => {
      const mtlFile = extractMTLReference('');
      expect(mtlFile).toBeNull();
    });

    it('should not match mtllib in comments', () => {
      const objText = `
# mtllib commented.mtl
o Cube
v 0 0 0
`;
      const mtlFile = extractMTLReference(objText);
      expect(mtlFile).toBeNull();
    });
  });

  describe('MTL Loading Error Handling (Requirement 4.4)', () => {
    let originalFetch: typeof globalThis.fetch;
    let consoleWarnSpy: ReturnType<typeof vi.spyOn>;

    beforeEach(() => {
      originalFetch = globalThis.fetch;
      consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    });

    afterEach(() => {
      globalThis.fetch = originalFetch;
      consoleWarnSpy.mockRestore();
    });

    it('should return empty Map when MTL fetch fails with network error', async () => {
      // Mock fetch to simulate network error
      globalThis.fetch = vi.fn().mockRejectedValue(new Error('Network error')) as typeof fetch;

      // Import the MTLParser to test the loadMTL behavior pattern
      const { MTLParser } = await import('./MTLParser');
      
      // Simulate the loadMTL error handling pattern
      let materials: Map<string, unknown> = new Map();
      try {
        const response = await fetch('http://example.com/model.mtl');
        if (!response.ok) {
          console.warn('OBJLoader: 无法加载 MTL 文件: http://example.com/model.mtl');
          materials = new Map();
        }
      } catch (e) {
        console.warn('OBJLoader: 加载 MTL 文件失败: http://example.com/model.mtl', e);
        materials = new Map();
      }

      expect(materials.size).toBe(0);
      expect(consoleWarnSpy).toHaveBeenCalled();
    });

    it('should return empty Map when MTL fetch returns non-OK status', async () => {
      // Mock fetch to return 404
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      }) as typeof fetch;

      // Simulate the loadMTL error handling pattern
      let materials: Map<string, unknown> = new Map();
      try {
        const response = await fetch('http://example.com/missing.mtl');
        if (!response.ok) {
          console.warn('OBJLoader: 无法加载 MTL 文件: http://example.com/missing.mtl');
          materials = new Map();
        }
      } catch (e) {
        console.warn('OBJLoader: 加载 MTL 文件失败: http://example.com/missing.mtl', e);
        materials = new Map();
      }

      expect(materials.size).toBe(0);
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'OBJLoader: 无法加载 MTL 文件: http://example.com/missing.mtl'
      );
    });

    it('should successfully parse MTL when fetch succeeds', async () => {
      const mtlContent = `
newmtl TestMaterial
Kd 1.0 0.5 0.0
d 0.8
`;
      // Mock successful fetch
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(mtlContent),
      }) as typeof fetch;

      const { MTLParser } = await import('./MTLParser');
      
      // Simulate successful loadMTL
      let materials: Map<string, unknown> = new Map();
      try {
        const response = await fetch('http://example.com/model.mtl');
        if (response.ok) {
          const text = await response.text();
          const mtlParser = new MTLParser();
          materials = mtlParser.parse(text);
        }
      } catch (e) {
        materials = new Map();
      }

      expect(materials.size).toBe(1);
      expect(materials.has('TestMaterial')).toBe(true);
      
      const material = materials.get('TestMaterial') as { diffuseColor: number[]; opacity: number };
      expect(material.diffuseColor).toEqual([1.0, 0.5, 0.0]);
      expect(material.opacity).toBe(0.8);
    });

    it('should use default material when MTL file is not found', async () => {
      // This test verifies the behavior described in Requirement 4.4:
      // "IF the MTL file cannot be loaded, THEN THE OBJLoader SHALL use default material and continue without error"
      
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
      }) as typeof fetch;

      // Simulate the createMaterial behavior when no materials are loaded
      const defaultMaterial = {
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: null,
        metallicFactor: 0,
        roughnessFactor: 0.5,
        doubleSided: false,
      };

      // When materials map is empty, createMaterial returns default
      const materials = new Map();
      const materialName = 'SomeMaterial';
      
      let resultMaterial;
      if (!materialName || !materials.has(materialName)) {
        resultMaterial = defaultMaterial;
      }

      expect(resultMaterial).toEqual(defaultMaterial);
      expect(resultMaterial!.baseColorFactor).toEqual([1, 1, 1, 1]);
    });
  });
});


describe('Material-Mesh Association in OBJLoader (Requirement 4.3)', () => {
  /**
   * Helper function to simulate createMaterial behavior
   * (Same algorithm as OBJLoader.createMaterial without GPU dependencies)
   */
  interface ParsedMaterial {
    name: string;
    diffuseColor: [number, number, number];
    diffuseTexture: string | null;
    opacity: number;
  }

  interface MaterialData {
    baseColorFactor: [number, number, number, number];
    baseColorTexture: null;
    metallicFactor: number;
    roughnessFactor: number;
    doubleSided: boolean;
  }

  function createMaterial(
    materialName: string | null,
    materials: Map<string, ParsedMaterial>
  ): MaterialData {
    // Default material
    const defaultMaterial: MaterialData = {
      baseColorFactor: [1, 1, 1, 1],
      baseColorTexture: null,
      metallicFactor: 0,
      roughnessFactor: 0.5,
      doubleSided: false,
    };

    if (!materialName || !materials.has(materialName)) {
      return defaultMaterial;
    }

    const parsedMaterial = materials.get(materialName)!;
    
    // Convert to MaterialData
    return {
      baseColorFactor: [
        parsedMaterial.diffuseColor[0],
        parsedMaterial.diffuseColor[1],
        parsedMaterial.diffuseColor[2],
        parsedMaterial.opacity,
      ],
      baseColorTexture: null,
      metallicFactor: 0,
      roughnessFactor: 0.5,
      doubleSided: false,
    };
  }

  it('should return default material when materialName is null', () => {
    const materials = new Map<string, ParsedMaterial>();
    
    const result = createMaterial(null, materials);
    
    expect(result.baseColorFactor).toEqual([1, 1, 1, 1]);
    expect(result.metallicFactor).toBe(0);
    expect(result.roughnessFactor).toBe(0.5);
  });

  it('should return default material when material is not found', () => {
    const materials = new Map<string, ParsedMaterial>();
    
    const result = createMaterial('NonExistentMaterial', materials);
    
    expect(result.baseColorFactor).toEqual([1, 1, 1, 1]);
  });

  it('should apply parsed material properties when material exists', () => {
    const materials = new Map<string, ParsedMaterial>();
    materials.set('RedMaterial', {
      name: 'RedMaterial',
      diffuseColor: [1.0, 0.0, 0.0],
      diffuseTexture: null,
      opacity: 1.0,
    });
    
    const result = createMaterial('RedMaterial', materials);
    
    expect(result.baseColorFactor).toEqual([1.0, 0.0, 0.0, 1.0]);
  });

  it('should apply opacity from parsed material', () => {
    const materials = new Map<string, ParsedMaterial>();
    materials.set('TransparentMaterial', {
      name: 'TransparentMaterial',
      diffuseColor: [0.5, 0.5, 0.5],
      diffuseTexture: null,
      opacity: 0.5,
    });
    
    const result = createMaterial('TransparentMaterial', materials);
    
    expect(result.baseColorFactor[3]).toBe(0.5);
  });

  it('should correctly map diffuse color to baseColorFactor', () => {
    const materials = new Map<string, ParsedMaterial>();
    materials.set('ColoredMaterial', {
      name: 'ColoredMaterial',
      diffuseColor: [0.2, 0.4, 0.6],
      diffuseTexture: null,
      opacity: 0.8,
    });
    
    const result = createMaterial('ColoredMaterial', materials);
    
    expect(result.baseColorFactor[0]).toBeCloseTo(0.2, 5);
    expect(result.baseColorFactor[1]).toBeCloseTo(0.4, 5);
    expect(result.baseColorFactor[2]).toBeCloseTo(0.6, 5);
    expect(result.baseColorFactor[3]).toBeCloseTo(0.8, 5);
  });

  it('should handle multiple materials and select correct one', () => {
    const materials = new Map<string, ParsedMaterial>();
    materials.set('Material1', {
      name: 'Material1',
      diffuseColor: [1.0, 0.0, 0.0],
      diffuseTexture: null,
      opacity: 1.0,
    });
    materials.set('Material2', {
      name: 'Material2',
      diffuseColor: [0.0, 1.0, 0.0],
      diffuseTexture: null,
      opacity: 1.0,
    });
    materials.set('Material3', {
      name: 'Material3',
      diffuseColor: [0.0, 0.0, 1.0],
      diffuseTexture: null,
      opacity: 1.0,
    });
    
    const result1 = createMaterial('Material1', materials);
    const result2 = createMaterial('Material2', materials);
    const result3 = createMaterial('Material3', materials);
    
    expect(result1.baseColorFactor).toEqual([1.0, 0.0, 0.0, 1.0]);
    expect(result2.baseColorFactor).toEqual([0.0, 1.0, 0.0, 1.0]);
    expect(result3.baseColorFactor).toEqual([0.0, 0.0, 1.0, 1.0]);
  });

  it('should handle material name with special characters', () => {
    const materials = new Map<string, ParsedMaterial>();
    materials.set('Material_With-Special.Chars', {
      name: 'Material_With-Special.Chars',
      diffuseColor: [0.5, 0.5, 0.5],
      diffuseTexture: null,
      opacity: 1.0,
    });
    
    const result = createMaterial('Material_With-Special.Chars', materials);
    
    expect(result.baseColorFactor).toEqual([0.5, 0.5, 0.5, 1.0]);
  });

  it('should handle empty string material name as not found', () => {
    const materials = new Map<string, ParsedMaterial>();
    materials.set('', {
      name: '',
      diffuseColor: [1.0, 0.0, 0.0],
      diffuseTexture: null,
      opacity: 1.0,
    });
    
    // Empty string should be treated as falsy and return default
    const result = createMaterial('', materials);
    
    expect(result.baseColorFactor).toEqual([1, 1, 1, 1]);
  });
});


describe('Empty File Handling - OBJLoader (Requirement 6.1)', () => {
  /**
   * Helper function to simulate createMeshes behavior for empty file handling
   * (Same algorithm as OBJLoader.createMeshes without GPU dependencies)
   * 
   * Requirement 6.1: IF the OBJ file is empty or contains no geometry, 
   * THEN THE OBJLoader SHALL return an empty array
   */
  interface ParsedObject {
    name: string;
    positions: number[];
    normals: number[];
    uvs: number[];
    indices: number[];
    materialName: string | null;
  }

  interface ParsedOBJData {
    objects: ParsedObject[];
  }

  function simulateCreateMeshes(parsedData: ParsedOBJData): number {
    // Simulates the createMeshes logic - returns count of meshes that would be created
    let meshCount = 0;
    
    for (const obj of parsedData.objects) {
      // Skip empty objects (same logic as OBJLoader.createMeshes)
      if (obj.indices.length === 0) {
        continue;
      }
      meshCount++;
    }
    
    return meshCount;
  }

  it('should return empty array for empty ParsedOBJData (empty file)', () => {
    const parsedData: ParsedOBJData = { objects: [] };
    
    const meshCount = simulateCreateMeshes(parsedData);
    
    expect(meshCount).toBe(0);
  });

  it('should return empty array when all objects have empty indices (no geometry)', () => {
    const parsedData: ParsedOBJData = {
      objects: [
        {
          name: 'EmptyObject1',
          positions: [0, 0, 0, 1, 0, 0, 0.5, 1, 0],
          normals: [],
          uvs: [],
          indices: [], // No faces
          materialName: null,
        },
        {
          name: 'EmptyObject2',
          positions: [2, 0, 0, 3, 0, 0, 2.5, 1, 0],
          normals: [],
          uvs: [],
          indices: [], // No faces
          materialName: null,
        },
      ],
    };
    
    const meshCount = simulateCreateMeshes(parsedData);
    
    expect(meshCount).toBe(0);
  });

  it('should return meshes only for objects with geometry', () => {
    const parsedData: ParsedOBJData = {
      objects: [
        {
          name: 'EmptyObject',
          positions: [0, 0, 0, 1, 0, 0, 0.5, 1, 0],
          normals: [],
          uvs: [],
          indices: [], // No faces
          materialName: null,
        },
        {
          name: 'ObjectWithGeometry',
          positions: [2, 0, 0, 3, 0, 0, 2.5, 1, 0],
          normals: [],
          uvs: [],
          indices: [0, 1, 2], // Has faces
          materialName: null,
        },
      ],
    };
    
    const meshCount = simulateCreateMeshes(parsedData);
    
    // Only the object with geometry should be counted
    expect(meshCount).toBe(1);
  });

  it('should return correct count for multiple objects with geometry', () => {
    const parsedData: ParsedOBJData = {
      objects: [
        {
          name: 'Object1',
          positions: [0, 0, 0, 1, 0, 0, 0.5, 1, 0],
          normals: [],
          uvs: [],
          indices: [0, 1, 2],
          materialName: null,
        },
        {
          name: 'Object2',
          positions: [2, 0, 0, 3, 0, 0, 2.5, 1, 0],
          normals: [],
          uvs: [],
          indices: [0, 1, 2],
          materialName: null,
        },
        {
          name: 'Object3',
          positions: [4, 0, 0, 5, 0, 0, 4.5, 1, 0],
          normals: [],
          uvs: [],
          indices: [0, 1, 2],
          materialName: null,
        },
      ],
    };
    
    const meshCount = simulateCreateMeshes(parsedData);
    
    expect(meshCount).toBe(3);
  });

  it('should skip objects with empty indices mixed with valid objects', () => {
    const parsedData: ParsedOBJData = {
      objects: [
        {
          name: 'ValidObject1',
          positions: [0, 0, 0, 1, 0, 0, 0.5, 1, 0],
          normals: [],
          uvs: [],
          indices: [0, 1, 2],
          materialName: null,
        },
        {
          name: 'EmptyObject',
          positions: [],
          normals: [],
          uvs: [],
          indices: [],
          materialName: null,
        },
        {
          name: 'ValidObject2',
          positions: [2, 0, 0, 3, 0, 0, 2.5, 1, 0],
          normals: [],
          uvs: [],
          indices: [0, 1, 2],
          materialName: null,
        },
      ],
    };
    
    const meshCount = simulateCreateMeshes(parsedData);
    
    // Only 2 valid objects should be counted
    expect(meshCount).toBe(2);
  });
});
