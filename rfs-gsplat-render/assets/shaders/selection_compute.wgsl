// selection_compute.wgsl - GPU-based splat selection shader
// Similar to supersplat's intersection-shader.ts
// 
// This compute shader tests each splat against selection criteria
// and outputs 1 for selected splats, 0 for non-selected

struct SelectionParams {
    num_splats: u32,
    mode: u32,          // 0=mask, 1=rect, 2=sphere, 3=box
    use_rings: u32,     // 0=centers (point test), 1=rings (ellipse test)
    _padding: u32,
    view_projection: mat4x4<f32>,
    model_matrix: mat4x4<f32>,
    rect_params: vec4<f32>,         // (x1, y1, x2, y2) in NDC [-1, 1]
    sphere_params: vec4<f32>,       // (x, y, z, radius) in world space
    box_center: vec4<f32>,          // (x, y, z, 0) in world space
    box_half_extents: vec4<f32>,    // (x, y, z, 0) in world space
}

struct PackedVec3 {
    x: f32,
    y: f32,
    z: f32,
}

@group(0) @binding(0) var<uniform> params: SelectionParams;
@group(0) @binding(1) var<storage, read> positions: array<PackedVec3>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<PackedVec3>;     // Splat scales
@group(0) @binding(4) var<storage, read> rotations: array<vec4<f32>>;   // Splat rotations (quaternions)

// Selection mode constants (matching Rust SelectionMode enum)
const MODE_MASK: u32 = 0u;
const MODE_RECT: u32 = 1u;
const MODE_SPHERE: u32 = 2u;
const MODE_BOX: u32 = 3u;

// Build rotation matrix from quaternion
// Quaternion format: (r, x, y, z) where r is scalar part
fn quat_to_mat3(rotation: vec4<f32>) -> mat3x3<f32> {
    let r = rotation.x;  // scalar part
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;
    
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),
        vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),
        vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y))
    );
}

// Project 2D covariance to screen space and compute bounds
// Returns (center_ndc, radius_x, radius_y) for 2D ellipse in NDC space
fn project_ellipse_bounds(
    world_pos: vec3<f32>,
    scale: vec3<f32>,
    rotation: vec4<f32>,
    view_proj: mat4x4<f32>
) -> vec4<f32> {
    // Get max radius for conservative test
    let max_radius = max(max(scale.x, scale.y), scale.z);
    
    // Project center
    let clip = view_proj * vec4<f32>(world_pos, 1.0);
    if clip.w <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // Behind camera
    }
    let ndc = clip.xyz / clip.w;
    
    // Project a point at max_radius distance to estimate screen-space radius
    // Simplified: use max_radius as a screen-space extent
    let radius_clip = view_proj * vec4<f32>(world_pos + vec3<f32>(max_radius, 0.0, 0.0), 1.0);
    let radius_ndc = radius_clip.xyz / radius_clip.w;
    let screen_radius = length(radius_ndc.xy - ndc.xy);
    
    return vec4<f32>(ndc.xy, screen_radius, screen_radius);
}

// Test if a 2D ellipse in NDC space overlaps with a 2D rectangle
fn ellipse_rect_overlap(ellipse_center: vec2<f32>, ellipse_radius: vec2<f32>, rect_min: vec2<f32>, rect_max: vec2<f32>) -> bool {
    // Find closest point on rectangle to ellipse center
    let closest = clamp(ellipse_center, rect_min, rect_max);
    let delta = ellipse_center - closest;
    
    // Normalized distance (accounts for ellipse shape)
    let normalized_dist_sq = (delta.x * delta.x) / (ellipse_radius.x * ellipse_radius.x) +
                              (delta.y * delta.y) / (ellipse_radius.y * ellipse_radius.y);
    
    return normalized_dist_sq <= 1.0;
}

// Test if a 3D ellipsoid overlaps with a sphere
fn ellipsoid_sphere_overlap(
    ellipsoid_center: vec3<f32>,
    ellipsoid_scale: vec3<f32>,
    ellipsoid_rotation: vec4<f32>,
    sphere_center: vec3<f32>,
    sphere_radius: f32
) -> bool {
    // Transform sphere center to ellipsoid's local space
    let R = quat_to_mat3(ellipsoid_rotation);
    let R_inv = transpose(R); // Inverse of rotation matrix is its transpose
    let delta = sphere_center - ellipsoid_center;
    let local_delta = R_inv * delta;
    
    // In local space, ellipsoid is axis-aligned with radii = scales
    // Test if scaled distance is within sphere radius + ellipsoid extent
    let scaled_dist_sq = (local_delta.x * local_delta.x) / (ellipsoid_scale.x * ellipsoid_scale.x) +
                          (local_delta.y * local_delta.y) / (ellipsoid_scale.y * ellipsoid_scale.y) +
                          (local_delta.z * local_delta.z) / (ellipsoid_scale.z * ellipsoid_scale.z);
    
    // Conservative test: check if sphere could touch ellipsoid
    let max_extent = max(max(ellipsoid_scale.x, ellipsoid_scale.y), ellipsoid_scale.z);
    let conservative_radius = sphere_radius + max_extent;
    let dist = length(delta);
    
    if dist > conservative_radius {
        return false;
    }
    
    // More precise test: if sphere center is inside ellipsoid, definitely overlaps
    if scaled_dist_sq <= 1.0 {
        return true;
    }
    
    // Approximate: check if distance to ellipsoid surface is within sphere radius
    // For a conservative approximation, accept if within reasonable distance
    return scaled_dist_sq <= (1.0 + sphere_radius / max_extent) * (1.0 + sphere_radius / max_extent);
}

// Test if a 3D ellipsoid overlaps with an axis-aligned bounding box
fn ellipsoid_box_overlap(
    ellipsoid_center: vec3<f32>,
    ellipsoid_scale: vec3<f32>,
    ellipsoid_rotation: vec4<f32>,
    box_center: vec3<f32>,
    box_half_extents: vec3<f32>
) -> bool {
    // Conservative test: check if bounding spheres overlap
    let max_ellipsoid_radius = max(max(ellipsoid_scale.x, ellipsoid_scale.y), ellipsoid_scale.z);
    let max_box_radius = length(box_half_extents);
    let center_dist = length(ellipsoid_center - box_center);
    
    if center_dist > max_ellipsoid_radius + max_box_radius {
        return false;
    }
    
    // Transform box to ellipsoid's local space
    let R = quat_to_mat3(ellipsoid_rotation);
    let R_inv = transpose(R);
    let delta = box_center - ellipsoid_center;
    let local_delta = R_inv * delta;
    let local_half_extents = box_half_extents; // Approximate, not rotated
    
    // Find closest point on box to ellipsoid center (in ellipsoid's local space)
    let closest = clamp(local_delta, -local_half_extents, local_half_extents);
    let dist_vector = local_delta - closest;
    
    // Check if closest point is inside ellipsoid
    let scaled_dist_sq = (dist_vector.x * dist_vector.x) / (ellipsoid_scale.x * ellipsoid_scale.x) +
                          (dist_vector.y * dist_vector.y) / (ellipsoid_scale.y * ellipsoid_scale.y) +
                          (dist_vector.z * dist_vector.z) / (ellipsoid_scale.z * ellipsoid_scale.z);
    
    return scaled_dist_sq <= 1.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let splat_index = global_id.x;
    
    // Bounds check
    if (splat_index >= params.num_splats) {
        return;
    }
    
    // Read splat center position (local space)
    let pos = positions[splat_index];
    let local_pos = vec3<f32>(pos.x, pos.y, pos.z);
    
    // Transform to world space
    let world_pos = (params.model_matrix * vec4<f32>(local_pos, 1.0)).xyz;
    
    // Read splat scale and rotation (for rings mode)
    let scale_data = scales[splat_index];
    let scale = vec3<f32>(scale_data.x, scale_data.y, scale_data.z);
    let rotation = rotations[splat_index];
    
    // Check if splat is in selection area based on mode
    var is_selected = false;
    
    if (params.mode == MODE_RECT) {
        if (params.use_rings == 1u) {
            // Rings mode: test ellipse-rectangle overlap in screen space
            let ellipse_bounds = project_ellipse_bounds(world_pos, scale, rotation, params.view_projection);
            let ellipse_center = ellipse_bounds.xy;
            let ellipse_radius = ellipse_bounds.zw;
            
            // Flip Y for screen coordinates
            let ellipse_center_screen = vec2<f32>(ellipse_center.x, -ellipse_center.y);
            
            // Rectangle bounds
            let rect_min = vec2<f32>(params.rect_params.x, params.rect_params.y);
            let rect_max = vec2<f32>(params.rect_params.z, params.rect_params.w);
            
            is_selected = ellipse_rect_overlap(ellipse_center_screen, ellipse_radius, rect_min, rect_max);
        } else {
            // Centers mode: test point-rectangle overlap
            let clip = params.view_projection * vec4<f32>(world_pos, 1.0);
            let ndc = clip.xyz / clip.w;
            
            // Check if within NDC bounds (visible)
            if (all(abs(ndc) <= vec3<f32>(1.0, 1.0, 1.0))) {
                let x = ndc.x;
                let y = -ndc.y; // Flip Y for screen coordinates
                
                if (x >= params.rect_params.x && x <= params.rect_params.z &&
                    y >= params.rect_params.y && y <= params.rect_params.w) {
                    is_selected = true;
                }
            }
        }
    } else if (params.mode == MODE_SPHERE) {
        let sphere_center = params.sphere_params.xyz;
        let sphere_radius = params.sphere_params.w;
        
        if (params.use_rings == 1u) {
            // Rings mode: test ellipsoid-sphere overlap
            is_selected = ellipsoid_sphere_overlap(world_pos, scale, rotation, sphere_center, sphere_radius);
        } else {
            // Centers mode: test point-sphere overlap
            let dist = length(world_pos - sphere_center);
            is_selected = dist <= sphere_radius;
        }
    } else if (params.mode == MODE_BOX) {
        let box_center = params.box_center.xyz;
        let box_half = params.box_half_extents.xyz;
        
        if (params.use_rings == 1u) {
            // Rings mode: test ellipsoid-box overlap
            is_selected = ellipsoid_box_overlap(world_pos, scale, rotation, box_center, box_half);
        } else {
            // Centers mode: test point-box overlap
            let rel_pos = world_pos - box_center;
            is_selected = all(abs(rel_pos) <= box_half);
        }
    }
    // MODE_MASK would require a texture, not implemented yet
    
    // Write result (1 = selected, 0 = not selected)
    results[splat_index] = select(0u, 1u, is_selected);
}

