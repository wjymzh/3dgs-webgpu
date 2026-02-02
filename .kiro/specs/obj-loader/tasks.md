# Implementation Plan: OBJ Loader

## Overview

本实现计划将 OBJ 加载器功能分解为增量式的编码任务。每个任务都建立在前一个任务的基础上，确保代码始终处于可运行状态。

## Tasks

- [x] 1. 实现 OBJ 文本解析器核心
  - [x] 1.1 创建 `src/loaders/OBJParser.ts`，定义 ParsedOBJData 和 ParsedObject 接口
    - 实现顶点池收集（v, vt, vn 指令解析）
    - 实现基本面解析（f 指令，支持四种索引格式）
    - _Requirements: 1.1, 1.2_
  
  - [ ]* 1.2 编写属性测试：顶点数据往返
    - **Property 1: Vertex Data Round-Trip**
    - **Validates: Requirements 1.1**
  
  - [ ]* 1.3 编写属性测试：面索引格式解析
    - **Property 2: Face Index Format Parsing**
    - **Validates: Requirements 1.2**

- [x] 2. 完善 OBJ 解析器高级功能
  - [x] 2.1 实现负索引解析
    - 将负索引转换为正索引（相对于当前顶点数）
    - _Requirements: 1.3_
  
  - [x] 2.2 实现多边形三角化
    - 使用扇形三角化将 n 边形转换为 (n-2) 个三角形
    - _Requirements: 1.4_
  
  - [x] 2.3 实现多对象/组支持
    - 解析 o 和 g 指令，为每个对象/组创建独立的 ParsedObject
    - _Requirements: 2.4_
  
  - [x] 2.4 实现错误容错
    - 跳过无效行，记录警告，继续解析
    - _Requirements: 1.5_
  
  - [ ]* 2.5 编写属性测试：负索引解析
    - **Property 3: Negative Index Resolution**
    - **Validates: Requirements 1.3**
  
  - [ ]* 2.6 编写属性测试：多边形三角化
    - **Property 4: Polygon Triangulation Correctness**
    - **Validates: Requirements 1.4**
  
  - [ ]* 2.7 编写属性测试：错误容错
    - **Property 5: Error Resilience**
    - **Validates: Requirements 1.5**
  
  - [ ]* 2.8 编写属性测试：多对象分离
    - **Property 6: Multi-Object Separation**
    - **Validates: Requirements 2.4**

- [x] 3. Checkpoint - 确保所有解析器测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 4. 实现 MTL 材质解析器
  - [x] 4.1 创建 `src/loaders/MTLParser.ts`
    - 定义 ParsedMaterial 接口
    - 实现 Kd（漫反射颜色）解析
    - 实现 map_Kd（漫反射纹理路径）解析
    - 实现 d/Tr（透明度）解析
    - _Requirements: 4.2_
  
  - [ ]* 4.2 编写属性测试：MTL 属性提取
    - **Property 11: MTL Property Extraction Round-Trip**
    - **Validates: Requirements 4.2**

- [x] 5. 实现 OBJLoader 主类
  - [x] 5.1 创建 `src/loaders/OBJLoader.ts`
    - 实现 load(url) 方法，获取并解析 OBJ 文件
    - 实现 parseFromText(text, baseUrl) 方法
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 5.2 实现 GPU 缓冲区创建
    - 创建交错顶点缓冲区（position + normal + uv）
    - 根据顶点数选择 uint16 或 uint32 索引格式
    - 计算并存储 bounding box
    - _Requirements: 3.1, 3.4, 3.5_
  
  - [x] 5.3 实现法线自动生成
    - 当 OBJ 缺少法线时，从面几何计算平面法线
    - _Requirements: 3.2_
  
  - [x] 5.4 实现 hasUV 标志设置
    - 根据是否有 UV 数据设置 Mesh.hasUV
    - _Requirements: 3.3_
  
  - [ ]* 5.5 编写属性测试：顶点缓冲区格式
    - **Property 7: Vertex Buffer Format Consistency**
    - **Validates: Requirements 3.1, 3.3**
  
  - [ ]* 5.6 编写属性测试：生成法线有效性
    - **Property 8: Generated Normal Validity**
    - **Validates: Requirements 3.2**
  
  - [ ]* 5.7 编写属性测试：索引格式选择
    - **Property 9: Index Format Selection**
    - **Validates: Requirements 3.4**
  
  - [ ]* 5.8 编写属性测试：包围盒正确性
    - **Property 10: Bounding Box Correctness**
    - **Validates: Requirements 3.5**

- [x] 6. Checkpoint - 确保所有加载器测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 7. 实现材质加载和关联
  - [x] 7.1 实现 MTL 文件加载
    - 解析 mtllib 指令，加载 MTL 文件
    - 处理 MTL 加载失败（使用默认材质）
    - _Requirements: 4.1, 4.4_
  
  - [x] 7.2 实现材质-网格关联
    - 解析 usemtl 指令，将材质关联到后续面
    - 创建 MaterialData 对象
    - _Requirements: 4.3_
  
  - [x] 7.3 实现纹理加载
    - 加载 map_Kd 指定的纹理文件
    - 创建 GPUTexture 并缓存
    - 处理纹理加载失败
    - _Requirements: 4.5, 6.3_
  
  - [ ]* 7.4 编写属性测试：材质-网格关联
    - **Property 12: Material-Mesh Association**
    - **Validates: Requirements 4.3**

- [x] 8. App 类集成
  - [x] 8.1 在 `src/App.ts` 中添加 addOBJ 方法
    - 创建 OBJLoader 实例
    - 加载 OBJ 文件并添加到 MeshRenderer
    - 返回加载的 Mesh 数组
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 8.2 更新 `src/index.ts` 导出
    - 导出 OBJLoader 类
    - 导出 OBJParser 和 MTLParser（可选）
    - _Requirements: 5.1_

- [x] 9. 错误处理完善
  - [x] 9.1 实现空文件处理
    - 空文件或无几何数据时返回空数组
    - _Requirements: 6.1_
  
  - [x] 9.2 实现不支持特性警告
    - 遇到不支持的指令时记录警告
    - _Requirements: 6.2_

- [x] 10. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

## Notes

- 标记 `*` 的任务为可选测试任务，可跳过以加快 MVP 开发
- 每个任务都引用具体的需求条款以确保可追溯性
- 检查点任务用于确保增量验证
- 属性测试验证通用正确性属性
- 单元测试验证特定示例和边界情况
