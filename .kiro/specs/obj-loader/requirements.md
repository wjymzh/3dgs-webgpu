# Requirements Document

## Introduction

本功能为 WebGPU 3D 渲染引擎添加 OBJ 格式模型的加载和渲染支持。OBJ 是一种广泛使用的 3D 模型格式，支持顶点位置、法线、纹理坐标和面数据。该功能将遵循现有 GLBLoader 的架构模式，与现有的 Mesh 和 MeshRenderer 系统无缝集成。

## Glossary

- **OBJLoader**: OBJ 文件加载器，负责解析 OBJ 文件并生成 Mesh 对象
- **OBJ_File**: Wavefront OBJ 格式的 3D 模型文件
- **MTL_File**: OBJ 配套的材质库文件
- **Vertex_Data**: 顶点数据，包含位置、法线、纹理坐标
- **Face_Data**: 面数据，定义三角形或多边形的顶点索引
- **Parser**: 解析器，将文本格式转换为结构化数据
- **Mesh**: 网格对象，存储 GPU 缓冲区和变换属性
- **MaterialData**: 材质数据结构，包含颜色、纹理等属性
- **LoadedMesh**: 加载后的网格数据，包含 Mesh 和 MaterialData

## Requirements

### Requirement 1: OBJ 文件解析

**User Story:** 作为开发者，我希望能够解析 OBJ 文件的文本内容，以便提取 3D 模型的几何数据。

#### Acceptance Criteria

1. WHEN the Parser receives valid OBJ text content, THE Parser SHALL extract vertex positions (v), texture coordinates (vt), and vertex normals (vn)
2. WHEN the Parser encounters face definitions (f), THE Parser SHALL parse vertex/texture/normal index combinations in formats: v, v/vt, v/vt/vn, v//vn
3. WHEN the Parser encounters negative indices in face definitions, THE Parser SHALL resolve them relative to the current vertex count
4. WHEN the Parser encounters a polygon face with more than 3 vertices, THE Parser SHALL triangulate it into multiple triangles using fan triangulation
5. IF the Parser encounters invalid or malformed lines, THEN THE Parser SHALL skip them and continue parsing
6. WHEN the Parser completes parsing, THE Parser SHALL return structured data containing positions, normals, uvs, and indices arrays

### Requirement 2: OBJ 文件加载

**User Story:** 作为开发者，我希望能够通过 URL 加载 OBJ 文件，以便在应用中使用外部 3D 模型。

#### Acceptance Criteria

1. WHEN the OBJLoader receives a valid URL, THE OBJLoader SHALL fetch the file content and parse it
2. IF the fetch operation fails, THEN THE OBJLoader SHALL throw an error with descriptive message
3. WHEN the OBJ file is successfully loaded, THE OBJLoader SHALL return an array of LoadedMesh objects
4. WHEN the OBJ file contains multiple objects (o) or groups (g), THE OBJLoader SHALL create separate Mesh instances for each

### Requirement 3: GPU 缓冲区创建

**User Story:** 作为开发者，我希望解析后的数据能够转换为 GPU 缓冲区，以便与现有渲染系统集成。

#### Acceptance Criteria

1. WHEN creating vertex buffers, THE OBJLoader SHALL use interleaved format: position(3) + normal(3) + uv(2) matching GLBLoader format
2. WHEN the OBJ file lacks normal data, THE OBJLoader SHALL generate flat normals from face geometry
3. WHEN the OBJ file lacks UV data, THE OBJLoader SHALL create Mesh with hasUV set to false
4. WHEN vertex count exceeds 65535, THE OBJLoader SHALL use uint32 index format, otherwise use uint16
5. WHEN creating buffers, THE OBJLoader SHALL compute and store bounding box information

### Requirement 4: MTL 材质文件支持

**User Story:** 作为开发者，我希望能够加载 OBJ 配套的 MTL 材质文件，以便模型能够显示正确的颜色和纹理。

#### Acceptance Criteria

1. WHEN the OBJ file references an MTL file via mtllib directive, THE OBJLoader SHALL attempt to load the MTL file from the same directory
2. WHEN parsing MTL file, THE Parser SHALL extract Kd (diffuse color), map_Kd (diffuse texture), d/Tr (transparency) properties
3. WHEN a material is referenced via usemtl directive, THE OBJLoader SHALL apply the corresponding MaterialData to subsequent faces
4. IF the MTL file cannot be loaded, THEN THE OBJLoader SHALL use default material and continue without error
5. WHEN loading diffuse textures (map_Kd), THE OBJLoader SHALL create GPUTexture and assign to MaterialData.baseColorTexture

### Requirement 5: App 类集成

**User Story:** 作为开发者，我希望能够通过 App 类的统一接口加载 OBJ 文件，以便保持 API 一致性。

#### Acceptance Criteria

1. THE App class SHALL provide an addOBJ(url) method that loads OBJ files and adds them to the scene
2. WHEN addOBJ is called, THE App SHALL use OBJLoader to load the file and add resulting meshes to MeshRenderer
3. WHEN addOBJ completes successfully, THE App SHALL return the array of loaded Mesh objects
4. IF addOBJ fails, THEN THE App SHALL throw an error with the original error message

### Requirement 6: 错误处理

**User Story:** 作为开发者，我希望加载器能够优雅地处理各种错误情况，以便应用能够稳定运行。

#### Acceptance Criteria

1. IF the OBJ file is empty or contains no geometry, THEN THE OBJLoader SHALL return an empty array
2. IF the OBJ file contains unsupported features, THEN THE OBJLoader SHALL log a warning and continue with supported features
3. WHEN texture loading fails, THE OBJLoader SHALL use null texture and log a warning
