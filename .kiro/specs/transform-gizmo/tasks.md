# Implementation Plan: Transform Gizmo System

## Overview

This implementation plan breaks down the Transform Gizmo system into discrete, incremental coding tasks. The approach follows a bottom-up strategy: first implementing foundational math utilities, then building the gizmo components, and finally integrating with the existing rendering pipeline. Each task builds on previous work, with property-based tests placed close to implementation to catch errors early.

## Tasks

- [x] 1. Implement core math utility classes
  - [x] 1.1 Implement Vec3 class with all operations
    - Create `src/core/math/Vec3.ts`
    - Implement constructor, factory methods (fromArray, zero, one)
    - Implement basic operations (add, subtract, multiply, divide) with both new and in-place variants
    - Implement vector operations (dot, cross, length, distance)
    - Implement normalization with zero-length handling
    - Implement utility methods (clone, toArray, set)
    - _Requirements: 14.1, 14.5, 10.3_

  - [ ]\* 1.2 Write property tests for Vec3
    - **Property 1: Vector addition is commutative**
    - **Property 2: Cross product is perpendicular**
    - **Property 3: Normalization produces unit vectors**
    - **Property 4: Distance is symmetric**
    - **Property 5: Zero vector normalization is safe**
    - **Validates: Requirements 14.5, 10.3**

  - [x] 1.3 Implement Quat class with all operations
    - Create `src/core/math/Quat.ts`
    - Implement constructor and identity factory
    - Implement fromEuler (ZYX order) and toEuler conversions
    - Implement fromAxisAngle factory
    - Implement quaternion multiplication
    - Implement normalization
    - Implement slerp interpolation
    - Implement utility methods (clone)
    - _Requirements: 14.2, 14.6_

  - [ ]\* 1.4 Write property tests for Quat
    - **Property 6: Euler round trip preserves rotation**
    - **Property 7: Quaternion multiplication is associative**
    - **Property 8: Identity quaternion is neutral**
    - **Validates: Requirements 14.6**

  - [x] 1.5 Implement Mat4 class with all operations
    - Create `src/core/math/Mat4.ts`
    - Implement constructor with Float32Array storage (column-major)
    - Implement factory methods (identity, fromTranslation, fromRotation, fromScale, compose)
    - Implement matrix multiplication (both new and in-place)
    - Implement inverse with singularity check
    - Implement transpose
    - Implement decompose (extract position, rotation, scale)
    - Implement transformPoint and transformDirection
    - Implement utility methods (clone)
    - _Requirements: 14.3, 14.7_

  - [ ]\* 1.6 Write property tests for Mat4
    - **Property 9: Matrix multiplication by inverse yields identity**
    - **Property 10: Compose then decompose is round trip**
    - **Property 11: Matrix transpose is involutive**
    - **Validates: Requirements 14.7**

  - [x] 1.7 Implement Ray class with all operations
    - Create `src/core/math/Ray.ts`
    - Implement constructor with origin and direction
    - Implement fromScreenPoint factory (converts screen coords to world ray using camera)
    - Implement at() method (point at distance along ray)
    - Implement intersectPlane with parallel ray handling
    - Implement distanceToPoint
    - Implement distanceToSegment for capsule hit testing
    - Implement utility methods (clone)
    - _Requirements: 14.4, 14.8, 10.2_

  - [ ]\* 1.8 Write property tests for Ray
    - **Property 12: Ray at zero distance returns origin**
    - **Property 13: Ray-plane intersection is on plane**
    - **Property 14: Ray distance to point is non-negative**
    - **Property 15: Parallel ray and plane return null**
    - **Validates: Requirements 14.8, 10.2**

- [x] 2. Checkpoint - Ensure math utilities are working
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implement GizmoAxis class
  - [x] 3.1 Create GizmoAxis class structure
    - Create `src/core/gizmo/GizmoAxis.ts`
    - Define AxisType, GizmoMode, and GizmoAxisConfig interfaces
    - Implement constructor with config parameter
    - Add visual state properties (isHovered, isActive)
    - Add geometry buffer properties (vertexBuffer, indexBuffer, counts)
    - _Requirements: 13.2_

  - [x] 3.2 Implement geometry creation for translate mode
    - Implement createTranslateGeometry() method
    - Generate cylinder geometry (0.8 units length, 0.04 radius, 12 segments)
    - Generate cone geometry (0.25 units length, 0.1 radius, 12 segments)
    - Combine into vertex/index buffers with format [position.xyz, color.rgb]
    - Implement createGeometry() to create GPU buffers
    - _Requirements: 3.1_

  - [x] 3.3 Implement geometry creation for rotate mode
    - Implement createRotateGeometry() method
    - Generate torus geometry (major radius 1.0, minor radius 0.02, 64 segments)
    - Create vertex/index buffers with same format
    - _Requirements: 3.2_

  - [x] 3.4 Implement geometry creation for scale mode
    - Implement createScaleGeometry() method
    - Generate cylinder geometry (0.8 units length, 0.04 radius)
    - Generate cube geometry (0.15 units size) at end
    - Create vertex/index buffers
    - _Requirements: 3.3_

  - [x] 3.5 Implement hit testing for translate/scale axes
    - Implement testAxisHit() method using ray-segment distance
    - Model axis as capsule (line segment with radius)
    - Use Ray.distanceToSegment() method
    - Return true if distance < radius threshold
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ]\* 3.6 Write property test for axis hit testing
    - **Property 33: Axis hit detection uses distance threshold**
    - **Property 35: Ray-segment distance is symmetric**
    - **Validates: Requirements 7.2, 7.3**

  - [x] 3.7 Implement hit testing for rotate rings
    - Implement testRingHit() method using ray-plane intersection
    - Intersect ray with plane perpendicular to axis
    - Check if intersection distance from center is within ring radius ± width
    - Return true if within tolerance
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ]\* 3.8 Write property test for ring hit testing
    - **Property 34: Ring hit detection uses radius tolerance**
    - **Validates: Requirements 8.2, 8.3**

  - [x] 3.9 Implement color computation based on state
    - Implement getColor() method
    - Return base color \* 1.5 if isActive
    - Return base color \* 1.2 if isHovered
    - Return base color \* 0.6 otherwise
    - _Requirements: 3.5, 3.6, 3.7_

  - [ ]\* 3.10 Write property tests for visual state
    - **Property 21: Hovered axis is brighter than inactive**
    - **Property 22: Active axis is brightest**
    - **Property 23: Inactive axis is dimmed**
    - **Validates: Requirements 3.5, 3.6, 3.7**

  - [x] 3.11 Implement cleanup method
    - Implement destroy() method to release GPU buffers
    - _Requirements: 13.1_

- [x] 4. Implement TransformGizmo class structure
  - [x] 4.1 Create TransformGizmo class skeleton
    - Create `src/core/gizmo/TransformGizmo.ts`
    - Define TransformGizmoConfig interface
    - Implement constructor with renderer, camera, canvas parameters
    - Add configuration properties (baseSize, hitThreshold)
    - Add state properties (mode, targetObject, axes array)
    - Add interaction state properties (activeAxis, hoveredAxis, isDragging, dragStartPoint, dragStartTransform)
    - Add rendering resources properties (pipeline, uniformBuffer, bindGroup)
    - Add currentScale property for screen-space scaling
    - _Requirements: 13.1_

  - [x] 4.2 Implement initialization and pipeline creation
    - Implement init() method
    - Create WGSL shader code for gizmo rendering (vertex + fragment shaders)
    - Implement createPipeline() method
    - Set up vertex buffer layout [position.xyz, color.rgb]
    - Configure depth testing and alpha blending
    - Create uniform buffer for view-projection matrix and model matrix
    - Create bind group
    - _Requirements: 11.2, 11.4, 11.5_

  - [x] 4.3 Implement axis creation for all modes
    - Implement createAxes() method
    - Create three GizmoAxis instances (X, Y, Z) for current mode
    - Set colors: X=red(0.9,0.2,0.2), Y=green(0.2,0.9,0.2), Z=blue(0.2,0.4,0.9)
    - Set directions: X=(1,0,0), Y=(0,1,0), Z=(0,0,1)
    - Call createGeometry() on each axis
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.4 Implement mode management
    - Implement setMode() method
    - Clear all axes' hover and active states
    - Destroy old axes
    - Create new axes for new mode
    - Preserve target object reference
    - Implement getMode() method
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ]\* 4.5 Write property tests for mode management
    - **Property 16: Mode change clears interaction state**
    - **Property 17: Mode change preserves target**
    - **Validates: Requirements 1.3, 1.4**

  - [x] 4.6 Implement target management
    - Implement setTarget() method to set target object
    - Implement getTarget() method
    - Update gizmo position to match target position
    - _Requirements: 2.1, 2.4_

  - [ ] 4.7 Write property tests for target management
    - **Property 18: Gizmo position follows target position**
    - **Property 19: Target position changes update gizmo**
    - **Validates: Requirements 2.1, 2.2**

- [x] 5. Checkpoint - Ensure gizmo structure is working
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement screen-space scaling
  - [x] 6.1 Implement screen-space scale computation
    - Implement updateScreenSpaceScale() method
    - Compute distance from camera to gizmo position
    - Calculate scale factor: distance _ tan(fov/2) _ 0.15
    - Store in currentScale property
    - _Requirements: 3.4, 12.3_

  - [ ]\* 6.2 Write property tests for screen-space scaling
    - **Property 20: Screen-space scale is proportional to distance**
    - **Property 44: Camera movement updates gizmo scale**
    - **Validates: Requirements 3.4, 12.3, 12.4**

- [x] 7. Implement pointer event handling
  - [x] 7.1 Implement screen-to-ray conversion
    - Implement screenToRay() method
    - Use Ray.fromScreenPoint() with camera matrices
    - Convert canvas coordinates to normalized device coordinates
    - _Requirements: 12.2_

  - [ ]\* 7.2 Write property test for ray conversion
    - **Property 43: Screen-to-ray conversion is consistent**
    - **Validates: Requirements 12.2**

  - [x] 7.3 Implement hit testing
    - Implement performHitTest() method
    - Iterate through all axes
    - Call testHit() on each axis with current ray, gizmo position, and scale
    - Return first axis that registers a hit (or null)
    - _Requirements: 7.1, 7.2, 7.3, 8.1, 8.2, 8.3_

  - [x] 7.4 Implement pointer move handler
    - Implement onPointerMove() method
    - Convert pointer position to ray
    - Perform hit testing
    - Update hoveredAxis (clear previous, set new)
    - If dragging, compute and apply transformation delta
    - _Requirements: 9.1, 9.3_

  - [ ]\* 7.5 Write property tests for pointer move
    - **Property 36: Pointer move updates hover state**
    - **Property 38: Continuous drag applies cumulative transformations**
    - **Validates: Requirements 9.1, 9.3**

  - [x] 7.6 Implement pointer down handler
    - Implement onPointerDown() method
    - Convert pointer position to ray
    - Perform hit testing
    - If hit, set activeAxis, enter drag mode (isDragging = true)
    - Capture pointer on canvas
    - Store dragStartPoint and dragStartTransform
    - _Requirements: 9.2_

  - [ ]\* 7.7 Write property test for pointer down
    - **Property 26: Pointer down on axis enters drag mode**
    - **Property 37: Pointer capture enables drag mode**
    - **Validates: Requirements 4.1, 9.2**

  - [x] 7.8 Implement pointer up handler
    - Implement onPointerUp() method
    - Exit drag mode (isDragging = false)
    - Clear activeAxis
    - Release pointer capture
    - _Requirements: 9.4_

  - [ ]\* 7.9 Write property test for pointer up
    - **Property 27: Pointer up exits drag mode**
    - **Property 39: Pointer release clears drag state**
    - **Validates: Requirements 4.4, 9.4**

- [x] 8. Implement transformation computation
  - [x] 8.1 Implement translation delta computation
    - Implement computeTranslationDelta() method
    - Get active axis direction
    - Find closest point on axis to current ray
    - Compute displacement from dragStartPoint
    - Project displacement onto axis direction
    - Return displacement vector
    - Handle parallel ray/axis case (return null)
    - _Requirements: 4.2, 4.5, 10.1_

  - [ ]\* 8.2 Write property tests for translation
    - **Property 24: Translation displacement is along axis**
    - **Property 40: Parallel ray and axis handled gracefully**
    - **Validates: Requirements 4.2, 4.5, 10.1**

  - [x] 8.3 Implement rotation delta computation
    - Implement computeRotationDelta() method
    - Get active axis direction
    - Intersect current ray with plane perpendicular to axis
    - Compute vector from gizmo center to intersection
    - Project onto plane (remove axis component)
    - Compute angle from dragStartPoint using atan2
    - Handle parallel ray/plane case (return null)
    - _Requirements: 5.2, 5.5, 10.2_

  - [ ]\* 8.4 Write property tests for rotation
    - **Property 28: Rotation angle computation is correct**
    - **Property 42: Degenerate angle computation is safe**
    - **Validates: Requirements 5.2, 5.5, 10.4**

  - [x] 8.5 Implement scale delta computation
    - Implement computeScaleDelta() method
    - Similar to translation, find closest point on axis
    - Compute displacement scalar
    - Map to scale factor (1.0 + scalar)
    - Clamp to minimum 0.001
    - Return scale delta for active axis only
    - _Requirements: 6.2, 6.5, 6.6, 10.5_

  - [ ]\* 8.6 Write property tests for scale
    - **Property 30: Scale factor is clamped to minimum**
    - **Property 31: Scale computation is along axis**
    - **Validates: Requirements 6.2, 6.5, 6.6, 10.5**

- [x] 9. Implement transformation application
  - [x] 9.1 Implement translation application
    - Implement applyTranslation() method
    - Add displacement delta to target object's position
    - Call target.setPosition() with new position
    - _Requirements: 4.3_

  - [ ]\* 9.2 Write property test for translation application
    - **Property 25: Translation updates target position**
    - **Validates: Requirements 4.3**

  - [x] 9.3 Implement rotation application
    - Implement applyRotation() method
    - Add angle delta to target object's rotation on active axis
    - Call target.setRotation() with new rotation
    - _Requirements: 5.3_

  - [ ]\* 9.4 Write property test for rotation application
    - **Property 29: Rotation updates target rotation**
    - **Validates: Requirements 5.3**

  - [x] 9.5 Implement scale application
    - Implement applyScale() method
    - Multiply target object's scale by scale factor on active axis
    - Call target.setScale() with new scale
    - _Requirements: 6.3_

  - [ ]\* 9.6 Write property test for scale application
    - **Property 32: Scale updates target scale**
    - **Validates: Requirements 6.3**

- [x] 10. Checkpoint - Ensure transformations are working
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement rendering
  - [x] 11.1 Implement uniform updates
    - Implement updateUniforms() method
    - Compute model matrix from gizmo position and currentScale
    - Combine with camera view-projection matrix
    - Write to uniform buffer
    - _Requirements: 12.1_

  - [x] 11.2 Implement render method
    - Implement render() method
    - Return early if no target object
    - Update screen-space scale
    - Update uniforms
    - Set pipeline and bind group
    - For each axis, set vertex/index buffers and draw
    - _Requirements: 2.3, 11.1, 11.3_

  - [x] 11.3 Implement cleanup method
    - Implement destroy() method
    - Destroy all axes
    - Destroy GPU buffers (uniform, vertex, index)
    - _Requirements: 13.1_

- [x] 12. Integration with App class
  - [x] 12.1 Add TransformGizmo to App class
    - Import TransformGizmo in `src/App.ts`
    - Add private property for gizmo instance
    - Initialize gizmo in init() method
    - Add pointer event listeners to canvas (move, down, up)
    - Forward events to gizmo handlers
    - _Requirements: 11.1_

  - [x] 12.2 Add gizmo rendering to render loop
    - Call gizmo.render() in App.render() method
    - Render after meshes but before viewport gizmo
    - _Requirements: 11.3_

  - [x] 12.3 Add public API methods to App
    - Add getTransformGizmo() method
    - Add setGizmoMode() convenience method
    - Add setGizmoTarget() convenience method
    - _Requirements: 1.1, 2.1_

  - [ ]\* 12.4 Write integration tests
    - Test gizmo initialization in App
    - Test mode switching through App API
    - Test target setting through App API
    - Test that gizmo renders when target is set
    - **Validates: Requirements 1.1, 2.1, 11.1**

- [x] 13. Final checkpoint - Ensure complete system works
  - Ensure all tests pass, ask the user if questions arise.
  - Verify gizmo renders correctly in all three modes
  - Verify transformations apply correctly to target objects
  - Verify visual feedback (hover, active states) works
  - Verify edge cases are handled gracefully

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties across randomized inputs (minimum 100 iterations each)
- Unit tests validate specific examples and edge cases
- Math utilities are implemented first as they are foundational dependencies
- Gizmo components are built incrementally: geometry → hit testing → state management → transformations → rendering
- Integration happens last to ensure all components work together
