# Requirements Document

## Introduction

This document specifies the requirements for a production-grade Transform Gizmo system for a WebGPU-based 3D Gaussian Splatting editor. The Transform Gizmo provides interactive manipulation controls that allow users to translate, rotate, and scale 3D objects through visual handles and axes.

## Glossary

- **Transform_Gizmo**: The main interactive control system that provides visual manipulation handles for 3D objects
- **Gizmo_Axis**: An individual axis or ring component of the gizmo (X/Y/Z axis for translate/scale, or rotation ring)
- **Target_Object**: The 3D object (Mesh or Splat) currently being manipulated by the gizmo
- **Manipulation_Mode**: The current operation mode (translate, rotate, or scale)
- **Hit_Testing**: The process of determining which gizmo component the user's pointer is interacting with
- **Ray**: A mathematical representation of a line starting from a point in a direction, used for 3D picking
- **Screen_Space_Size**: A visual property where the gizmo maintains constant apparent size regardless of camera distance
- **Transformation_Delta**: The incremental change in position, rotation, or scale computed during a drag operation
- **Hover_State**: Visual feedback indicating which gizmo component is under the pointer
- **Active_State**: Visual feedback indicating which gizmo component is currently being dragged

## Requirements

### Requirement 1: Mode Management

**User Story:** As a user, I want to switch between translate, rotate, and scale modes, so that I can perform different types of transformations on my 3D objects.

#### Acceptance Criteria

1. THE Transform_Gizmo SHALL support three distinct manipulation modes: translate, rotate, and scale
2. WHEN the mode is changed, THE Transform_Gizmo SHALL update its visual representation to match the new mode
3. WHEN the mode is changed, THE Transform_Gizmo SHALL clear any active or hover states from the previous mode
4. THE Transform_Gizmo SHALL maintain the current target object when switching modes

### Requirement 2: Target Object Management

**User Story:** As a user, I want the gizmo to follow my selected object, so that I can manipulate it from its current position.

#### Acceptance Criteria

1. WHEN a target object is set, THE Transform_Gizmo SHALL position itself at the object's world position
2. WHEN the target object's position changes, THE Transform_Gizmo SHALL update its position to match
3. WHEN no target object is set, THE Transform_Gizmo SHALL not render
4. THE Transform_Gizmo SHALL support both Mesh and Splat objects as targets

### Requirement 3: Visual Representation

**User Story:** As a user, I want to see clear visual handles for each axis, so that I know which direction I'm manipulating.

#### Acceptance Criteria

1. WHEN in translate mode, THE Transform_Gizmo SHALL display three colored arrows along X (red), Y (green), and Z (blue) axes
2. WHEN in rotate mode, THE Transform_Gizmo SHALL display three colored rings around X (red), Y (green), and Z (blue) axes
3. WHEN in scale mode, THE Transform_Gizmo SHALL display three colored axes with cube handles along X (red), Y (green), and Z (blue) axes
4. THE Transform_Gizmo SHALL maintain constant screen-space size by scaling based on camera distance
5. WHEN an axis is hovered, THE Transform_Gizmo SHALL highlight that axis with increased brightness
6. WHEN an axis is active (being dragged), THE Transform_Gizmo SHALL highlight that axis with maximum brightness
7. WHEN an axis is not hovered or active, THE Transform_Gizmo SHALL render it with semi-transparent appearance

### Requirement 4: Translation Interaction

**User Story:** As a user, I want to drag an axis arrow to move my object along that axis, so that I can precisely position it.

#### Acceptance Criteria

1. WHEN the user clicks on a translate axis, THE Transform_Gizmo SHALL enter drag mode for that axis
2. WHILE dragging a translate axis, THE Transform_Gizmo SHALL compute the displacement along that axis direction
3. WHILE dragging a translate axis, THE Transform_Gizmo SHALL apply the displacement to the target object's position in real-time
4. WHEN the user releases the pointer, THE Transform_Gizmo SHALL exit drag mode and finalize the transformation
5. THE Transform_Gizmo SHALL compute displacement by projecting pointer ray intersections onto the axis direction

### Requirement 5: Rotation Interaction

**User Story:** As a user, I want to drag a rotation ring to rotate my object around that axis, so that I can orient it correctly.

#### Acceptance Criteria

1. WHEN the user clicks on a rotation ring, THE Transform_Gizmo SHALL enter drag mode for that axis
2. WHILE dragging a rotation ring, THE Transform_Gizmo SHALL compute the rotation angle around that axis
3. WHILE dragging a rotation ring, THE Transform_Gizmo SHALL apply the rotation to the target object in real-time
4. WHEN the user releases the pointer, THE Transform_Gizmo SHALL exit drag mode and finalize the transformation
5. THE Transform_Gizmo SHALL compute rotation angle by intersecting pointer rays with the plane perpendicular to the rotation axis and calculating angular difference using atan2

### Requirement 6: Scale Interaction

**User Story:** As a user, I want to drag a scale handle to resize my object along that axis, so that I can adjust its dimensions.

#### Acceptance Criteria

1. WHEN the user clicks on a scale axis, THE Transform_Gizmo SHALL enter drag mode for that axis
2. WHILE dragging a scale axis, THE Transform_Gizmo SHALL compute the scale factor along that axis direction
3. WHILE dragging a scale axis, THE Transform_Gizmo SHALL apply the scale factor to the target object in real-time
4. WHEN the user releases the pointer, THE Transform_Gizmo SHALL exit drag mode and finalize the transformation
5. THE Transform_Gizmo SHALL clamp minimum scale values to 0.001 to prevent degenerate geometry
6. THE Transform_Gizmo SHALL compute scale factor by projecting pointer ray intersections onto the axis direction

### Requirement 7: Hit Testing for Axes

**User Story:** As a user, I want the gizmo to accurately detect when I'm pointing at an axis, so that I can select the correct manipulation handle.

#### Acceptance Criteria

1. WHEN in translate or scale mode, THE Transform_Gizmo SHALL model each axis as a capsule (line segment with radius)
2. WHEN performing hit testing for translate or scale axes, THE Transform_Gizmo SHALL compute the distance from the pointer ray to the axis line segment
3. WHEN the distance from pointer ray to axis is below a threshold, THE Transform_Gizmo SHALL register a hit on that axis
4. THE Transform_Gizmo SHALL use a hit threshold appropriate for comfortable user interaction (typically 0.1 to 0.2 world units after screen-space scaling)

### Requirement 8: Hit Testing for Rotation Rings

**User Story:** As a user, I want the gizmo to accurately detect when I'm pointing at a rotation ring, so that I can select the correct rotation axis.

#### Acceptance Criteria

1. WHEN in rotate mode, THE Transform_Gizmo SHALL model each ring as a torus section on a plane perpendicular to its axis
2. WHEN performing hit testing for rotation rings, THE Transform_Gizmo SHALL compute the intersection of the pointer ray with the ring's plane
3. WHEN the intersection point's distance from the gizmo origin is within the ring radius tolerance, THE Transform_Gizmo SHALL register a hit on that ring
4. THE Transform_Gizmo SHALL use a ring width tolerance appropriate for comfortable user interaction

### Requirement 9: Pointer Event Handling

**User Story:** As a user, I want smooth and responsive interaction with the gizmo, so that manipulations feel natural and precise.

#### Acceptance Criteria

1. WHEN the pointer moves over the gizmo, THE Transform_Gizmo SHALL perform hit testing and update hover states
2. WHEN the pointer is pressed on a gizmo component, THE Transform_Gizmo SHALL capture the pointer and enter drag mode
3. WHILE the pointer is captured, THE Transform_Gizmo SHALL process pointer move events to compute transformation deltas
4. WHEN the pointer is released, THE Transform_Gizmo SHALL release capture and exit drag mode
5. THE Transform_Gizmo SHALL handle pointer events in the order: pointerdown, pointermove, pointerup

### Requirement 10: Mathematical Stability

**User Story:** As a developer, I want the gizmo's mathematical operations to be robust, so that edge cases don't cause crashes or incorrect behavior.

#### Acceptance Criteria

1. WHEN computing ray-axis distances, THE Transform_Gizmo SHALL handle parallel rays and axes gracefully
2. WHEN computing ray-plane intersections, THE Transform_Gizmo SHALL handle parallel rays and planes gracefully
3. WHEN normalizing vectors, THE Transform_Gizmo SHALL handle zero-length vectors without division by zero
4. WHEN computing angles, THE Transform_Gizmo SHALL handle degenerate cases where vectors are parallel or anti-parallel
5. THE Transform_Gizmo SHALL clamp scale values to prevent negative or near-zero scales

### Requirement 11: Rendering Integration

**User Story:** As a developer, I want the gizmo to integrate seamlessly with the existing WebGPU rendering pipeline, so that it renders correctly with other scene objects.

#### Acceptance Criteria

1. THE Transform_Gizmo SHALL use the existing WebGPU device and rendering context
2. THE Transform_Gizmo SHALL create its own render pipeline with appropriate shaders
3. THE Transform_Gizmo SHALL render after opaque geometry to ensure visibility
4. THE Transform_Gizmo SHALL use depth testing to correctly occlude behind scene geometry
5. THE Transform_Gizmo SHALL render with alpha blending for semi-transparent inactive axes

### Requirement 12: Camera Integration

**User Story:** As a user, I want the gizmo to work correctly with camera movements, so that it remains usable from any viewing angle.

#### Acceptance Criteria

1. THE Transform_Gizmo SHALL use the camera's view and projection matrices for rendering
2. THE Transform_Gizmo SHALL compute pointer rays using the camera's inverse view-projection matrix
3. THE Transform_Gizmo SHALL scale its visual size based on the distance from the camera to maintain constant screen-space size
4. WHEN the camera moves, THE Transform_Gizmo SHALL update its screen-space scaling factor

### Requirement 13: Code Quality and Architecture

**User Story:** As a developer, I want the gizmo code to be well-structured and maintainable, so that I can extend and debug it easily.

#### Acceptance Criteria

1. THE Transform_Gizmo SHALL be implemented as a TypeScript class with clear separation of concerns
2. THE Gizmo_Axis SHALL be implemented as a separate class representing individual axis components
3. THE Transform_Gizmo SHALL use descriptive variable names and include comments explaining complex mathematical operations
4. THE Transform_Gizmo SHALL not contain placeholder code or TODO comments
5. THE Transform_Gizmo SHALL follow the existing codebase conventions for matrix operations and Float32Array usage

### Requirement 14: Math Utility Types

**User Story:** As a developer, I want reusable math utility types, so that the gizmo code is clean and the utilities can be used elsewhere in the codebase.

#### Acceptance Criteria

1. THE System SHALL provide a Vec3 utility class for 3D vector operations
2. THE System SHALL provide a Quat utility class for quaternion operations
3. THE System SHALL provide a Mat4 utility class for 4x4 matrix operations
4. THE System SHALL provide a Ray utility class for ray representation and operations
5. THE Vec3 class SHALL support operations: add, subtract, multiply, divide, dot, cross, normalize, length, distance
6. THE Quat class SHALL support operations: multiply, fromEuler, toEuler, fromAxisAngle, slerp
7. THE Mat4 class SHALL support operations: multiply, inverse, transpose, decompose, compose
8. THE Ray class SHALL support operations: at (point at distance), intersectPlane, distanceToPoint, distanceToSegment
