import { describe, it, expect } from 'vitest';
import { MTLParser, ParsedMaterial } from './MTLParser';

describe('MTLParser', () => {
  describe('basic parsing', () => {
    it('should parse empty content', () => {
      const parser = new MTLParser();
      const result = parser.parse('');
      expect(result.size).toBe(0);
    });

    it('should skip comments and empty lines', () => {
      const parser = new MTLParser();
      const mtl = `
# This is a comment
   
# Another comment
`;
      const result = parser.parse(mtl);
      expect(result.size).toBe(0);
    });

    it('should parse a single material with default values', () => {
      const parser = new MTLParser();
      const mtl = 'newmtl TestMaterial';
      const result = parser.parse(mtl);
      
      expect(result.size).toBe(1);
      expect(result.has('TestMaterial')).toBe(true);
      
      const material = result.get('TestMaterial')!;
      expect(material.name).toBe('TestMaterial');
      expect(material.diffuseColor).toEqual([1, 1, 1]);
      expect(material.diffuseTexture).toBeNull();
      expect(material.opacity).toBe(1);
    });
  });

  describe('Kd (diffuse color) parsing', () => {
    it('should parse Kd values', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl RedMaterial
Kd 1.0 0.0 0.0
`;
      const result = parser.parse(mtl);
      const material = result.get('RedMaterial')!;
      
      expect(material.diffuseColor).toEqual([1.0, 0.0, 0.0]);
    });

    it('should parse Kd with decimal values', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl GrayMaterial
Kd 0.5 0.5 0.5
`;
      const result = parser.parse(mtl);
      const material = result.get('GrayMaterial')!;
      
      expect(material.diffuseColor).toEqual([0.5, 0.5, 0.5]);
    });

    it('should handle Kd with various float formats', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl Material
Kd .25 0.75 1
`;
      const result = parser.parse(mtl);
      const material = result.get('Material')!;
      
      expect(material.diffuseColor[0]).toBeCloseTo(0.25);
      expect(material.diffuseColor[1]).toBeCloseTo(0.75);
      expect(material.diffuseColor[2]).toBeCloseTo(1);
    });
  });

  describe('map_Kd (diffuse texture) parsing', () => {
    it('should parse map_Kd texture path', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl TexturedMaterial
map_Kd texture.png
`;
      const result = parser.parse(mtl);
      const material = result.get('TexturedMaterial')!;
      
      expect(material.diffuseTexture).toBe('texture.png');
    });

    it('should parse map_Kd with path containing spaces', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl Material
map_Kd textures/my texture file.png
`;
      const result = parser.parse(mtl);
      const material = result.get('Material')!;
      
      expect(material.diffuseTexture).toBe('textures/my texture file.png');
    });

    it('should parse map_Kd with relative path', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl Material
map_Kd ../textures/diffuse.jpg
`;
      const result = parser.parse(mtl);
      const material = result.get('Material')!;
      
      expect(material.diffuseTexture).toBe('../textures/diffuse.jpg');
    });
  });

  describe('d (opacity) parsing', () => {
    it('should parse d value for full opacity', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl OpaqueMaterial
d 1.0
`;
      const result = parser.parse(mtl);
      const material = result.get('OpaqueMaterial')!;
      
      expect(material.opacity).toBe(1.0);
    });

    it('should parse d value for partial transparency', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl TransparentMaterial
d 0.5
`;
      const result = parser.parse(mtl);
      const material = result.get('TransparentMaterial')!;
      
      expect(material.opacity).toBe(0.5);
    });

    it('should parse d value for full transparency', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl FullyTransparent
d 0.0
`;
      const result = parser.parse(mtl);
      const material = result.get('FullyTransparent')!;
      
      expect(material.opacity).toBe(0.0);
    });
  });

  describe('Tr (transparency) parsing', () => {
    it('should parse Tr and convert to opacity (Tr=0 means opaque)', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl OpaqueMaterial
Tr 0.0
`;
      const result = parser.parse(mtl);
      const material = result.get('OpaqueMaterial')!;
      
      expect(material.opacity).toBe(1.0);
    });

    it('should parse Tr and convert to opacity (Tr=1 means transparent)', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl TransparentMaterial
Tr 1.0
`;
      const result = parser.parse(mtl);
      const material = result.get('TransparentMaterial')!;
      
      expect(material.opacity).toBe(0.0);
    });

    it('should parse Tr with partial transparency', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl SemiTransparent
Tr 0.3
`;
      const result = parser.parse(mtl);
      const material = result.get('SemiTransparent')!;
      
      expect(material.opacity).toBeCloseTo(0.7);
    });
  });

  describe('multiple materials', () => {
    it('should parse multiple materials', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl Material1
Kd 1.0 0.0 0.0

newmtl Material2
Kd 0.0 1.0 0.0
map_Kd green.png

newmtl Material3
Kd 0.0 0.0 1.0
d 0.5
`;
      const result = parser.parse(mtl);
      
      expect(result.size).toBe(3);
      
      const mat1 = result.get('Material1')!;
      expect(mat1.diffuseColor).toEqual([1.0, 0.0, 0.0]);
      expect(mat1.diffuseTexture).toBeNull();
      expect(mat1.opacity).toBe(1);
      
      const mat2 = result.get('Material2')!;
      expect(mat2.diffuseColor).toEqual([0.0, 1.0, 0.0]);
      expect(mat2.diffuseTexture).toBe('green.png');
      expect(mat2.opacity).toBe(1);
      
      const mat3 = result.get('Material3')!;
      expect(mat3.diffuseColor).toEqual([0.0, 0.0, 1.0]);
      expect(mat3.diffuseTexture).toBeNull();
      expect(mat3.opacity).toBe(0.5);
    });
  });

  describe('complete MTL file', () => {
    it('should parse a complete MTL file with all supported properties', () => {
      const parser = new MTLParser();
      const mtl = `
# Material Library
# Created by Test

newmtl WoodMaterial
Kd 0.6 0.4 0.2
map_Kd wood_diffuse.png
d 1.0

newmtl GlassMaterial
Kd 0.9 0.9 0.95
Tr 0.8
`;
      const result = parser.parse(mtl);
      
      expect(result.size).toBe(2);
      
      const wood = result.get('WoodMaterial')!;
      expect(wood.name).toBe('WoodMaterial');
      expect(wood.diffuseColor[0]).toBeCloseTo(0.6);
      expect(wood.diffuseColor[1]).toBeCloseTo(0.4);
      expect(wood.diffuseColor[2]).toBeCloseTo(0.2);
      expect(wood.diffuseTexture).toBe('wood_diffuse.png');
      expect(wood.opacity).toBe(1.0);
      
      const glass = result.get('GlassMaterial')!;
      expect(glass.name).toBe('GlassMaterial');
      expect(glass.diffuseColor[0]).toBeCloseTo(0.9);
      expect(glass.diffuseColor[1]).toBeCloseTo(0.9);
      expect(glass.diffuseColor[2]).toBeCloseTo(0.95);
      expect(glass.diffuseTexture).toBeNull();
      expect(glass.opacity).toBeCloseTo(0.2);
    });
  });

  describe('error handling', () => {
    it('should skip invalid Kd lines and continue parsing', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl Material
Kd invalid values
Kd 0.5 0.5 0.5
`;
      const result = parser.parse(mtl);
      const material = result.get('Material')!;
      
      // Should have the valid Kd value
      expect(material.diffuseColor).toEqual([0.5, 0.5, 0.5]);
    });

    it('should skip Kd/d/Tr before newmtl', () => {
      const parser = new MTLParser();
      const mtl = `
Kd 1.0 0.0 0.0
d 0.5
newmtl Material
Kd 0.0 1.0 0.0
`;
      const result = parser.parse(mtl);
      
      expect(result.size).toBe(1);
      const material = result.get('Material')!;
      expect(material.diffuseColor).toEqual([0.0, 1.0, 0.0]);
    });

    it('should handle material names with spaces', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl My Material Name
Kd 1.0 1.0 1.0
`;
      const result = parser.parse(mtl);
      
      expect(result.has('My Material Name')).toBe(true);
    });
  });

  describe('unsupported directives', () => {
    it('should ignore unsupported directives', () => {
      const parser = new MTLParser();
      const mtl = `
newmtl Material
Ka 0.1 0.1 0.1
Kd 0.5 0.5 0.5
Ks 1.0 1.0 1.0
Ns 100
illum 2
map_Ka ambient.png
map_Ks specular.png
bump normal.png
`;
      const result = parser.parse(mtl);
      const material = result.get('Material')!;
      
      // Only Kd should be parsed
      expect(material.diffuseColor).toEqual([0.5, 0.5, 0.5]);
      expect(material.diffuseTexture).toBeNull();
      expect(material.opacity).toBe(1);
    });
  });
});
