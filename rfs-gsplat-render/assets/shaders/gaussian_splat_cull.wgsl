// Gaussian Splatting Projection & Culling Compute Shader
// Ported from vk_gaussian_splatting/shaders/dist.comp.slang

#import bevy_render::view::View

// Project & Cull parameters
struct ProjectCullParams {
    point_count: u32,
    point_size: f32,
    frustum_dilation: f32,  // Frustum culling margin (matches vk_gaussian_splatting)
    _padding: u32,
}

// Transform matrices (must match vertex shader)
struct TransformUniforms {
    model_matrix: mat4x4<f32>,          // Local to world transform
    // Note: No need for model_matrix_inverse (optimized away)
}

// Packed Vec3 for memory efficiency (matches Rust struct)
struct PackedVec3 {
    x: f32,
    y: f32,
    z: f32,
}

// Bindings
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> cull_params: ProjectCullParams;
@group(0) @binding(2) var<storage, read> positions: array<PackedVec3>;
@group(0) @binding(3) var<storage, read_write> depth_keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> sorted_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> visible_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> indirect_buffer: array<atomic<u32>, 4>;
@group(0) @binding(7) var<uniform> transforms: TransformUniforms;

/// Frustum culling check (6 planes)
/// Ported from vk_gaussian_splatting/shaders/dist.comp.slang
/// WGPU/WebGPU clip space: X: [-w, +w], Y: [-w, +w], Z: [0, +w]
fn is_in_frustum(clip_pos: vec4<f32>, frustum_dilation: f32) -> bool {
    // Based on vk_gaussian_splatting implementation:
    // const float clip = 1.0f + frameInfo.frustumDilation;
    // if(abs(ndcPos.x) > clip || abs(ndcPos.y) > clip || 
    //    ndcPos.z < 0.f - frameInfo.frustumDilation || ndcPos.z > 1.0)
    
    // Note: vk_gaussian_splatting uses NDC coordinates (already divided by w)
    // We check in clip space, so we need to multiply by w
    
    let clip = (1.0 + frustum_dilation) * clip_pos.w;
    
    // Check left/right planes: X in [-clip, +clip]
    if abs(clip_pos.x) > clip {
        return false;
    }
    
    // Check top/bottom planes: Y in [-clip, +clip]
    if abs(clip_pos.y) > clip {
        return false;
    }
    
    // Check near/far planes
    let near_threshold = (0.0 - frustum_dilation) * clip_pos.w;
    if clip_pos.z < near_threshold || clip_pos.z > clip_pos.w {
        return false;
    }
    
    return true;
}

// encodes an fp32 into a uint32 that can be ordered
// Converts float to sortable uint using IEEE 754 bit manipulation
fn encode_min_max_fp32(val: f32) -> u32 {
  var bits = bitcast<u32>(val);  // HLSL: asuint(val)
  // HLSL: bits ^= (int(bits) >> 31) | 0x80000000u;
  // In WGSL, need explicit bitcasts:
  // 1. bitcast<i32>(bits) - convert u32 to i32 for arithmetic right shift
  // 2. >> 31 - arithmetic right shift (sign extension)
  // 3. bitcast<u32>(...) - convert back to u32 for XOR operation
  bits ^= bitcast<u32>(bitcast<i32>(bits) >> 31) | 0x80000000u;
  return bits;
}


/// Project & Cull Pass
/// For each point:
/// 1. Transform to clip space
/// 2. Frustum culling check
/// 3. Visible points: compute depth key
/// 4. Invisible points: not written (culled)
@compute @workgroup_size(256, 1, 1)
fn project_and_cull(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let gid = global_id.x;
    
    if gid < cull_params.point_count {
        // CRITICAL FIX: positions are in LOCAL/MODEL space, not world space!
        // Must match vertex shader coordinate transform
        let local_pos_packed = positions[gid];
        let local_pos = vec4<f32>(local_pos_packed.x, local_pos_packed.y, local_pos_packed.z, 1.0);
        
        // Transform: Local -> World -> View -> Clip (same as vertex shader)
        let world_pos = transforms.model_matrix * local_pos;
        let view_pos = view.view_from_world * world_pos;
        let clip_pos = view.clip_from_view * view_pos;
        
        // Frustum culling check
        if is_in_frustum(clip_pos, cull_params.frustum_dilation) {

            let depth = view_pos.z;  // Keep negative!

            let sortable_depth = encode_min_max_fp32(depth);
            
            // Atomically increment indirect_buffer's instance_count (index 1) to get unique index
            // indirect_buffer layout: [vertex_count(0), instance_count(1), first_vertex(2), first_instance(3)]
            let visible_idx = atomicAdd(&indirect_buffer[1], 1u);
            
            // Write to compact visible point list
            depth_keys[visible_idx] = sortable_depth;
            visible_indices[visible_idx] = gid;
        }
        
        // Invisible points are not written, saving bandwidth
    }
}
