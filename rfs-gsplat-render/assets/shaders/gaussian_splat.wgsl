#import bevy_render::view::View

const SQRT_8: f32 = 2.82842712475;
const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2: array<f32, 5> = array<f32, 5>(
    1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742
);
const SH_C3: array<f32, 7> = array<f32, 7>(
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
    -0.4570457994644658, 1.445305721320277, -0.5900435899266435
);

// SH data layout configuration
const SH_STRIDE: u32 = 45u;  // 15 coefficients × 3 channels (RGB)

// Normalized Gaussian constants (matching SuperSplat implementation)
// normExp ensures the Gaussian weight is exactly 0 at the boundary (A=1)
// This eliminates edge artifacts and provides sharper splat edges
const EXP_NEG4: f32 = 0.01831563888873418;  // exp(-4.0)
const INV_ONE_MINUS_EXP_NEG4: f32 = 1.01865736036377408;  // 1.0 / (1.0 - exp(-4.0))

//fast exp using taylor series
fn fast_exp(x: f32) -> f32 {
    // return 1.0 + x + x*x/2.0 + x*x*x/6.0 + x*x*x*x/24.0 + x*x*x*x*x/120.0;
    return exp(x);
}

//fast sigmoid using taylor series
fn fast_sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + fast_exp(-x));
}

// Normalized Gaussian function (matching SuperSplat exactly)
// SuperSplat uses normExp for sharper edges:
// At x=0 (center): returns 1.0
// At x=1 (edge): returns exactly 0.0 (not ~0.018 like standard exp(-4*x))
// This provides cleaner splat edges and better visual quality
fn norm_exp(x: f32) -> f32 {
    return (exp(-4.0 * x) - EXP_NEG4) * INV_ONE_MINUS_EXP_NEG4;
}

fn gaussian_weight(x: f32) -> f32 {
    return exp(-4.0 * x);
}
// ============================================================================
// PACK MODE SUPPORT - Data compression utilities
// ============================================================================

// Unpack two f16 values from a u32
fn unpack_half2(packed: u32) -> vec2<f32> {
    let lo = packed & 0xFFFFu;
    let hi = (packed >> 16u) & 0xFFFFu;
    // Note: unpack2x16float is a WebGPU/WGSL builtin that unpacks f16 values
    return unpack2x16float(packed);
}

// Unpack four f16 values from two u32 values  
fn unpack_uint2(packed: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(
        unpack_half2(packed.x),
        unpack_half2(packed.y)
    );
}

// Unpack 11_10_11 format (normalized direction without normalization)
// This unpacks RGB values stored as 11-bit, 10-bit, 11-bit in a single u32
// Matching C++ unpack_normal_11_10_11_uint_no_normalize EXACTLY:
//   x = bits 0-10  (low 11 bits)
//   y = bits 11-20 (mid 10 bits)
//   z = bits 21-31 (high 11 bits)
fn unpack_normal_11_10_11(packed: u32) -> vec3<f32> {
    let x = f32(packed & 0x7FFu) / 2047.0 * 2.0 - 1.0;
    let y = f32((packed >> 11u) & 0x3FFu) / 1023.0 * 2.0 - 1.0;
    let z = f32((packed >> 21u) & 0x7FFu) / 2047.0 * 2.0 - 1.0;
    return vec3<f32>(x, y, z);
}

struct PackedVec3{
    x : f32,
    y : f32,
    z : f32,
}

#ifdef PACK
// Pack mode structures (matching C++ implementation)
struct PackedVertexSH {
    sh1to3: vec4<u32>,   // SH coefficients 1-3 (packed 11_10_11 format)
    sh4to7: vec4<u32>,   // SH coefficients 4-7
    sh8to11: vec4<u32>,  // SH coefficients 8-11
    sh12to15: vec4<u32>, // SH coefficients 12-15
}

// Bindings for PACK mode (must match standard mode layout for compatibility)
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: GaussianSplatParams;
@group(0) @binding(2) var<storage, read> positions: array<PackedVec3>;
@group(0) @binding(3) var<storage, read> colors: array<vec2<u32>>;  // Packed f16 colors
@group(0) @binding(4) var<storage, read> visible_indices: array<u32>;
@group(0) @binding(5) var<storage, read> rotation_scales: array<vec4<u32>>;  // Packed rotation + scale + opacity (replaces scales & opacities)
@group(0) @binding(6) var<storage, read> _unused_pack_1: array<u32>;  // Unused in PACK mode (was opacities)
@group(0) @binding(7) var<storage, read> _unused_pack_2: array<u32>;  // Unused in PACK mode (was rotations)
@group(0) @binding(8) var<storage, read> sh_coeffs: array<PackedVertexSH>;  // Packed SH data
@group(0) @binding(9) var<uniform> transforms: TransformUniforms;
@group(0) @binding(10) var<storage, read> splat_states: array<u32>;  // Per-splat state: bit0=selected, bit1=locked, bit2=deleted

#else
// Standard mode bindings (original layout)
// NOTE: CPU pre-converts log_scales -> scales (via exp) and raw_opacities -> opacities (via sigmoid)
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: GaussianSplatParams;
@group(0) @binding(2) var<storage, read> positions: array<PackedVec3>;
@group(0) @binding(3) var<storage, read> sh_coeffs0: array<PackedVec3>;
@group(0) @binding(4) var<storage, read> visible_indices: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<PackedVec3>;      // Already exp(log_scale) from CPU
@group(0) @binding(6) var<storage, read> opacities: array<f32>;          // Already sigmoid(raw_opacity) from CPU
@group(0) @binding(7) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(9) var<uniform> transforms: TransformUniforms;
@group(0) @binding(10) var<storage, read> splat_states: array<u32>;  // Per-splat state: bit0=selected, bit1=locked, bit2=deleted
#endif

// Splat state bit flags (matches CPU-side splat_state module)
const SPLAT_STATE_SELECTED: u32 = 1u;
const SPLAT_STATE_LOCKED: u32 = 2u;
const SPLAT_STATE_DELETED: u32 = 4u;


// ============================================================================
// FETCH FUNCTIONS - Abstraction for packed/unpacked data access
// ============================================================================

#ifdef PACK
// Pack mode: fetch position from packed buffer
fn fetchPosition(index: u32) -> vec3<f32> {
    let packed = positions[index];
    return vec3<f32>(packed.x, packed.y, packed.z);
}
// Pack mode: fetch base color (SH0) from packed buffer
fn fetchColor(index: u32) -> vec3<f32> {
    let packed_color = colors[index];
    let color = unpack_uint2(packed_color);
    return color.xyz;
}

// Pack mode: fetch SH coefficients (compressed with 11_10_11 format)
fn fetchSHCoeff(index: u32, coeff_index: u32) -> vec3<f32> {
    // SH data is stored in packed format (4 coefficients per vec4u)
    let sh_data = sh_coeffs[index];
    
    // Each component of sh_data contains a packed RGB triplet
    if (coeff_index < 3u) {
        // sh1to3: coefficients 0,1,2
        return unpack_normal_11_10_11(sh_data.sh1to3[coeff_index + 1u]);
    } else if (coeff_index < 7u) {
        // sh4to7: coefficients 3,4,5,6
        return unpack_normal_11_10_11(sh_data.sh4to7[coeff_index - 3u]);
    } else if (coeff_index < 11u) {
        // sh8to11: coefficients 7,8,9,10
        return unpack_normal_11_10_11(sh_data.sh8to11[coeff_index - 7u]);
    } else if (coeff_index < 15u) {
        // sh12to15: coefficients 11,12,13,14
        return unpack_normal_11_10_11(sh_data.sh12to15[coeff_index - 11u]);
    }
    return vec3<f32>(0.0);
}

// Pack mode: fetch SH scale factor
fn fetchSHScale(index: u32) -> f32 {
    let sh_data = sh_coeffs[index];
    return bitcast<f32>(sh_data.sh1to3.x);
}

#else
// Standard mode: direct access to unpacked data
fn fetchPosition(index: u32) -> vec3<f32> {
    let packed = positions[index];
    return vec3<f32>(packed.x, packed.y, packed.z);
}

fn fetchColor(index: u32) -> vec3<f32> {
    // NOTE: Color is already pre-computed on CPU side as (dc * SH_C0 + 0.5)
    // Do NOT apply SH_C0 * x + 0.5 again here! That would cause double processing
    // and incorrect colors (too bright/washed out)
    let color = sh_coeffs0[index];
    return vec3<f32>(color.x, color.y, color.z);
}

fn fetchSHCoeff(index: u32, coeff_index: u32) -> vec3<f32> {
    let base_offset = index * SH_STRIDE + coeff_index * 3u;
    return vec3<f32>(
        sh_coeffs[base_offset],
        sh_coeffs[base_offset + 1u],
        sh_coeffs[base_offset + 2u]
    );
}

fn fetchSHScale(index: u32) -> f32 {
    return 1.0; // No scale factor in standard mode
}
#endif
// Compute normal from the smallest eigenvector of the covariance (flattest direction)
fn compute_splat_normal(scale: vec3<f32>, rotation: vec4<f32>) -> vec3<f32> {
    // CRITICAL: Quaternion format from HLSL is (r, x, y, z) where r is scalar (w in standard notation)
    let r = rotation.x;  // scalar part
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;
    
    // Build rotation matrix from quaternion (same as in covariance computation)
    let R = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),
        vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),
        vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y))
    );
    
    // The normal is the direction of the smallest scale (flattest axis)
    // Find which axis has the smallest scale
    var min_axis: u32 = 0u;
    var min_scale = scale.x;
    if scale.y < min_scale {
        min_axis = 1u;
        min_scale = scale.y;
    }
    if scale.z < min_scale {
        min_axis = 2u;
    }
    
    // Get the corresponding axis of the rotation matrix (column)
    var local_normal: vec3<f32>;
    if min_axis == 0u {
        local_normal = vec3<f32>(1.0, 0.0, 0.0);
    } else if min_axis == 1u {
        local_normal = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        local_normal = vec3<f32>(0.0, 0.0, 1.0);
    }
    
    // Transform to world space
    return normalize(R * local_normal);
}

// Computes the 3D covariance matrix from scale and rotation quaternion
// Returns the symmetric 3x3 covariance matrix
fn threedgs_compute_covariance_3d(scale: vec3<f32>, rotation: vec4<f32>) -> mat3x3<f32> {
    // Quaternion format from PLY: rotation = (w, x, y, z) where w is scalar
    // rotation.x = w (scalar part)
    // rotation.y = x
    // rotation.z = y  
    // rotation.w = z
    let r = rotation.x;  // scalar part (w)
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;
    
    // Build rotation matrix from quaternion
    // Standard quaternion to rotation matrix formula
    let R = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),
        vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),
        vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y))
    );
    
    // Build scale matrix (diagonal)
    let S = mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0),
        vec3<f32>(0.0, scale.y, 0.0),
        vec3<f32>(0.0, 0.0, scale.z)
    );
    
    // M = R * S
    let M = R * S;
    
    // Covariance = M * M^T
    return M * transpose(M);
}

// Projects 3D covariance to 2D screen space
// Based on EWA splatting (Zwicker et al. 2002)
fn threedgs_covariance_projection(
    cov3d: mat3x3<f32>,
    view_center: vec4<f32>,
    focal: vec2<f32>,
    model_view_matrix: mat4x4<f32>,
    is_ortho: bool
) -> vec3<f32> {
    // For orthographic projection, use a fixed direction (0,0,1)
    var v: vec3<f32>;
    if is_ortho {
        v = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        v = view_center.xyz;
    }
    
    let s = 1.0 / (v.z * v.z);
    
    // Jacobian matrix for projection
    let J = mat3x3<f32>(
        vec3<f32>(focal.x / v.z, 0.0, 0.0),
        vec3<f32>(0.0, focal.y / v.z, 0.0),
        vec3<f32>(-(focal.x * v.x) * s, -(focal.y * v.y) * s, 0.0)
    );
    
    // W = 3x3 part of model-view matrix
    let W = mat3x3<f32>(
        vec3<f32>(model_view_matrix[0][0], model_view_matrix[0][1], model_view_matrix[0][2]),
        vec3<f32>(model_view_matrix[1][0], model_view_matrix[1][1], model_view_matrix[1][2]),
        vec3<f32>(model_view_matrix[2][0], model_view_matrix[2][1], model_view_matrix[2][2])
    );
    
    // Transform matrix: T = J * W
    let T = J * W;
    
    // Project covariance: cov2D = T * cov3D * T^T
    let cov2d = T * cov3d * transpose(T);
    
    // Return the 3 unique elements: (xx, xy, yy)
    return vec3<f32>(cov2d[0][0], cov2d[0][1], cov2d[1][1]);
}

struct ProjectedExtentBasisResult {
    basis: vec4<f32>,
    opacity: f32
}

// ClipCorner optimization (EXACTLY matching PlayCanvas/SuperSplat)
// From PlayCanvas: clip = min(1.0, sqrt(-log(1.0 / (255.0 * alpha))) / 2.0)
// This shrinks the quad to exclude gaussian regions where alpha < 1/255
fn compute_clip_factor(alpha: f32) -> f32 {
    // Protect against very small alpha values
    // When alpha <= 1/255, the splat is invisible
    if alpha <= (1.0 / 255.0) {
        return 0.0;  // Cull this splat
    }
    // PlayCanvas formula: clip = min(1.0, sqrt(-log(1.0 / (255.0 * alpha))) / 2.0)
    // Simplify: -log(1/(255*a)) = log(255*a)
    return min(1.0, sqrt(log(255.0 * alpha)) / 2.0);
}
// This function ingests the projected 2D covariance and outputs the basis vectors of its 2D extent
// Matches PlayCanvas/SuperSplat implementation for consistent rendering quality
fn threedgs_projected_extent_basis(
    cov2d_in: vec3<f32>,
    std_dev: f32,
    splat_scale: f32,
    opacity: f32,
    viewport_size: vec2<f32>  // Added: viewport dimensions for proper size limiting
) -> ProjectedExtentBasisResult {
    var cov2d = cov2d_in;
    var alpha = opacity;
#ifdef GSPLAT_AA
    let detOrig = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
#endif
    // Add low-pass filter to avoid very small eigenvalues
    // This acts as a regularization term (matching PlayCanvas: +0.3)
    cov2d.x += 0.3;
    cov2d.z += 0.3;
#ifdef GSPLAT_AA
    // This mode is used when model is reconstructed using MipSplatting
    // https://niujinshuchong.github.io/mip-splatting/
    let detBlur = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    // apply the alpha compensation
    alpha *= sqrt(max(detOrig / detBlur, 0.0));
#endif

    // Eigenvalue decomposition of 2D covariance matrix
    // Using PlayCanvas formula: mid ± length(vec2((a-d)/2, b))
    let a = cov2d.x;  // diagonal1
    let d = cov2d.z;  // diagonal2
    let b = cov2d.y;  // offDiagonal
    
    let mid = 0.5 * (a + d);
    let radius = length(vec2<f32>((a - d) * 0.5, b));
    
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);  // PlayCanvas uses 0.1 minimum
    
    // Check if eigenvalues are valid
    if lambda2 <= 0.0 {
        return ProjectedExtentBasisResult(vec4<f32>(0.0, 0.0, 0.0, 0.0), 0.0);
    }
    
    // Use viewport-based maximum limit (matching PlayCanvas)
    // vmin = min(1024.0, min(viewport_size.x, viewport_size.y))
    let vmin = min(1024.0, min(viewport_size.x, viewport_size.y));
    
    // Compute axis lengths: l = 2.0 * min(sqrt(2.0 * lambda), vmin)
    // This is equivalent to our std_dev * sqrt(lambda) since std_dev = sqrt(8) ≈ 2.83
    // and 2.0 * sqrt(2.0 * lambda) = 2.0 * sqrt(2) * sqrt(lambda) ≈ 2.83 * sqrt(lambda)
    let l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin);
    let l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin);
    
    // CRITICAL: Early-out for gaussians smaller than 2 pixels (matching PlayCanvas)
    // This eliminates sub-pixel splats that cause "fogging" artifacts
    if l1 < 2.0 && l2 < 2.0 {
        return ProjectedExtentBasisResult(vec4<f32>(0.0, 0.0, 0.0, 0.0), 0.0);
    }
    
    // Compute eigenvector from offDiagonal and eigenvalue difference
    // diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1))
    let diag_vec = normalize(vec2<f32>(b, lambda1 - a));
    let eigenvector1 = diag_vec;
    let eigenvector2 = vec2<f32>(diag_vec.y, -diag_vec.x);
    
    // Apply splat scale and compute basis vectors
    let basis_vector1 = eigenvector1 * splat_scale * l1;
    let basis_vector2 = eigenvector2 * splat_scale * l2;
    
    return ProjectedExtentBasisResult(vec4<f32>(basis_vector1, basis_vector2), alpha);
}


const quad_verts = array<vec2<f32>, 4>(
    vec2<f32>(-1, -1),
    vec2<f32>(-1, 1),
    vec2<f32>(1, -1),
    vec2<f32>(1, 1)
);

// sRGB to linear conversion (gamma decoding)
// Matches Bevy's gamma() function in bevy_ui_render::gradient.wgsl
// https://en.wikipedia.org/wiki/SRGB
fn srgb_channel_to_linear(value: f32) -> f32 {
    if value <= 0.0 {
        return value;
    }
    if value <= 0.04045 {
        return value / 12.92;  // linear falloff in dark values
    } else {
        return pow((value + 0.055) / 1.055, 2.4);  // gamma curve
    }
}

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    return vec3(
        srgb_channel_to_linear(color.x),
        srgb_channel_to_linear(color.y),
        srgb_channel_to_linear(color.z)
    );
}

// Color space handling for render target
// 3DGS color data (DC + SH) is stored in sRGB space
// 
// RENDER TARGET MODES:
// 1. RENDER_TO_CACHE: Render to Rgba8UnormSrgb cache texture, then blit to screen
//    - Output sRGB directly (no conversion)
//    - GPU auto-converts sRGB → linear when blit shader samples the texture
// 2. RENDER_TO_HDR: Render directly to HDR screen (Rgba16Float)
//    - Must convert sRGB → linear manually (HDR expects linear input)
// 3. Neither: Render directly to LDR screen (Rgba8UnormSrgb)
//    - Output sRGB directly (GPU auto-converts linear → sRGB on write, but we're already sRGB)
fn apply_color_space_conversion(color: vec3<f32>) -> vec3<f32> {
#ifdef RENDER_TO_HDR
    // Rendering to HDR screen: convert sRGB → linear
    return srgb_to_linear(max(color, vec3<f32>(0.0)));
#else
    // Rendering to cache or LDR screen: output sRGB directly
    return max(color, vec3<f32>(0.0));
#endif
}
// Uniforms
struct GaussianSplatParams {
    point_size: f32,           // Not used for 3DGS, kept for compatibility
    surface_width: u32,
    surface_height: u32,
    point_count: u32,
    frustum_dilation: f32,     // Default: 0.2
    alpha_cull_threshold: f32, // Default: 1.0/255.0
    splat_scale: f32,          // Global splat scale multiplier, default: 1.0
    sh_degree: u32,            // Spherical harmonics degree (0-3), used in PACK mode
    select_color: vec4<f32>,   // Color for selected splats (RGB in 0-1, A = edit point size)
    unselect_color: vec4<f32>, // Color for unselected splats in selection overlay (RGB in 0-1, A unused)
    locked_color: vec4<f32>,
    tint_color: vec4<f32>,
    color_offset: vec4<f32>,
}

// Transform matrices (for model space to world space)
struct TransformUniforms {
    model_matrix: mat4x4<f32>,          // Local to world transform
    // Note: No need for model_matrix_inverse! We can derive direction transform from model_matrix
}

// Evaluate spherical harmonics for view-dependent color
// Supports both PACK and standard modes
fn eval_sh(world_view_dir: vec3<f32>, splat_index: u32) -> vec3<f32> {
    var sh_color = vec3<f32>(0.0);
    let x = world_view_dir.x;
    let y = world_view_dir.y;
    let z = world_view_dir.z;
    
#ifdef PACK
    // Pack mode: use fetch functions with SH scale and dynamic degree checking
    let sh_scale = fetchSHScale(splat_index);
    
    // Degree 1 (3 coefficients)
    #ifdef SH_DEGREE_1 
    {
        let sh1 = fetchSHCoeff(splat_index, 0u);
        let sh2 = fetchSHCoeff(splat_index, 1u);
        let sh3 = fetchSHCoeff(splat_index, 2u);
        sh_color += SH_C1 * (-sh1 * y + sh2 * z - sh3 * x);
    }
    #endif
    
    // Degree 2 (5 coefficients)
    #ifdef SH_DEGREE_2
    {
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let yz = y * z;
        let xz = x * z;
        
        let sh4 = fetchSHCoeff(splat_index, 3u);
        let sh5 = fetchSHCoeff(splat_index, 4u);
        let sh6 = fetchSHCoeff(splat_index, 5u);
        let sh7 = fetchSHCoeff(splat_index, 6u);
        let sh8 = fetchSHCoeff(splat_index, 7u);
        
        sh_color += sh4 * (SH_C2[0] * xy)
                 + sh5 * (SH_C2[1] * yz)
                 + sh6 * (SH_C2[2] * (2.0 * zz - xx - yy))
                 + sh7 * (SH_C2[3] * xz)
                 + sh8 * (SH_C2[4] * (xx - yy));
    }
    #endif
    
    // Degree 3 (7 coefficients)
    #ifdef SH_DEGREE_3
    {
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        
        let sh9  = fetchSHCoeff(splat_index, 8u);
        let sh10 = fetchSHCoeff(splat_index, 9u);
        let sh11 = fetchSHCoeff(splat_index, 10u);
        let sh12 = fetchSHCoeff(splat_index, 11u);
        let sh13 = fetchSHCoeff(splat_index, 12u);
        let sh14 = fetchSHCoeff(splat_index, 13u);
        let sh15 = fetchSHCoeff(splat_index, 14u);
        
        sh_color += SH_C3[0] * sh9  * (3.0 * xx - yy) * y
                 + SH_C3[1] * sh10 * x * y * z
                 + SH_C3[2] * sh11 * (4.0 * zz - xx - yy) * y
                 + SH_C3[3] * sh12 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                 + SH_C3[4] * sh13 * x * (4.0 * zz - xx - yy)
                 + SH_C3[5] * sh14 * (xx - yy) * z
                 + SH_C3[6] * sh15 * x * (xx - 3.0 * yy);
    }
    #endif
    // Apply clamp first, then scale (matching C++ reference: max(color, 0) * scale)
    return max(sh_color, vec3<f32>(0.0)) * sh_scale;
    
#else
    // Standard mode: use sh_coeffs with compile-time degree macros
    
    // Degree 1 (3 coefficients)
    #ifdef SH_DEGREE_1
    {
        let base1 = splat_index * SH_STRIDE;
        let sh1 = vec3<f32>(sh_coeffs[base1], sh_coeffs[base1 + 1u], sh_coeffs[base1 + 2u]);
        let sh2 = vec3<f32>(sh_coeffs[base1 + 3u], sh_coeffs[base1 + 4u], sh_coeffs[base1 + 5u]);
        let sh3 = vec3<f32>(sh_coeffs[base1 + 6u], sh_coeffs[base1 + 7u], sh_coeffs[base1 + 8u]);
        
        sh_color += SH_C1 * (-sh1 * y + sh2 * z - sh3 * x);
    }
    #endif
    
    // Degree 2 (5 coefficients)
    #ifdef SH_DEGREE_2
    {
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let yz = y * z;
        let xz = x * z;
        
        let base2 = splat_index * SH_STRIDE + 9u;
        let sh4 = vec3<f32>(sh_coeffs[base2], sh_coeffs[base2 + 1u], sh_coeffs[base2 + 2u]);
        let sh5 = vec3<f32>(sh_coeffs[base2 + 3u], sh_coeffs[base2 + 4u], sh_coeffs[base2 + 5u]);
        let sh6 = vec3<f32>(sh_coeffs[base2 + 6u], sh_coeffs[base2 + 7u], sh_coeffs[base2 + 8u]);
        let sh7 = vec3<f32>(sh_coeffs[base2 + 9u], sh_coeffs[base2 + 10u], sh_coeffs[base2 + 11u]);
        let sh8 = vec3<f32>(sh_coeffs[base2 + 12u], sh_coeffs[base2 + 13u], sh_coeffs[base2 + 14u]);
        
        sh_color += (SH_C2[0] * xy) * sh4
                 + (SH_C2[1] * yz) * sh5
                 + (SH_C2[2] * (2.0 * zz - xx - yy)) * sh6
                 + (SH_C2[3] * xz) * sh7
                 + (SH_C2[4] * (xx - yy)) * sh8;
    }
    #endif
    // Degree 3 (7 coefficients)
    #ifdef SH_DEGREE_3
    {
        let base3 = splat_index * SH_STRIDE + 24u;
        let sh9  = vec3<f32>(sh_coeffs[base3], sh_coeffs[base3 + 1u], sh_coeffs[base3 + 2u]);
        let sh10 = vec3<f32>(sh_coeffs[base3 + 3u], sh_coeffs[base3 + 4u], sh_coeffs[base3 + 5u]);
        let sh11 = vec3<f32>(sh_coeffs[base3 + 6u], sh_coeffs[base3 + 7u], sh_coeffs[base3 + 8u]);
        let sh12 = vec3<f32>(sh_coeffs[base3 + 9u], sh_coeffs[base3 + 10u], sh_coeffs[base3 + 11u]);
        let sh13 = vec3<f32>(sh_coeffs[base3 + 12u], sh_coeffs[base3 + 13u], sh_coeffs[base3 + 14u]);
        let sh14 = vec3<f32>(sh_coeffs[base3 + 15u], sh_coeffs[base3 + 16u], sh_coeffs[base3 + 17u]);
        let sh15 = vec3<f32>(sh_coeffs[base3 + 18u], sh_coeffs[base3 + 19u], sh_coeffs[base3 + 20u]);
        
        sh_color += SH_C3[0] * sh9  * (3.0 * x * x - y * y) * y
                 + SH_C3[1] * sh10 * x * y * z
                 + SH_C3[2] * sh11 * (4.0 * z * z - x * x - y * y) * y
                 + SH_C3[3] * sh12 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
                 + SH_C3[4] * sh13 * x * (4.0 * z * z - x * x - y * y)
                 + SH_C3[5] * sh14 * (x * x - y * y) * z
                 + SH_C3[6] * sh15 * x * (x * x - 3.0 * y * y);
    }
    #endif
    
    // Match HLSL: clamp SH color to non-negative values
    return max(sh_color, vec3<f32>(0.0));
#endif
}

// Depth to color gradient (matching HLSL reference using smoothstep)
// Used for VIS_DEPTH visualization mode
// Creates a blue -> green -> red gradient
fn depth_to_color(normalized_depth: f32) -> vec3<f32> {
    // Clamp depth to [0, 1]
    let d = clamp(normalized_depth, 0.0, 1.0);
    
    // Use smoothstep for smooth transitions (matching HLSL implementation)
    // Red: increases from 0.5 to 1.0 (far objects are more red)
    let r = smoothstep(0.5, 1.0, d);
    // Green: peaks at 0.5, decreases towards 0 and 1 (mid-distance objects are green)
    let g = 1.0 - abs(d - 0.5) * 2.0;
    // Blue: decreases from 0.0 to 0.5 (near objects are more blue)
    let b = 1.0 - smoothstep(0.0, 0.5, d);
    
    return vec3<f32>(r, g, b);
}

// Vertex shader output
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) frag_pos: vec2<f32>,  // UV in [-clip, clip] range ([-1,1] before clipCorner)
    @location(2) @interpolate(flat) splat_state_out: u32,  // Splat state for fragment shader
    @location(3) @interpolate(flat) splat_index_out: u32,  // Splat index for PICK_PASS
}

// Vertex shader
// Adapted from threedgs_raster.vert.slang
@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var output: VertexOutput;
    // Initialize outputs for early returns
    output.splat_state_out = 0u;
    output.splat_index_out = 0u;
    
    // Get splat index from visible_indices (after culling and sorting)
    let splat_index = visible_indices[instance_index];
    
    // Check splat state - skip deleted splats
    let splat_state = splat_states[splat_index];
    if (splat_state & SPLAT_STATE_DELETED) != 0u {
        // Emit degenerate triangle for deleted splats
        output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return output;
    }
    // Store splat state and index for fragment shader
    output.splat_state_out = splat_state;
    output.splat_index_out = splat_index;
    
    // Fetch splat data using abstraction functions (supports both PACK and standard mode)
    let position = fetchPosition(splat_index);
    let splat_center_local = vec4<f32>(position, 1.0);
#ifdef PACK
    // uint4 rotation_scale = gaussian.rotation_scale;
    // float4 scale_opacity = unpack_uint2(rotation_scale.zw);
    // float opac = scale_opacity.w;
    let rotation_scale = rotation_scales[splat_index];
    let scale_opacity = unpack_uint2(rotation_scale.zw);
    let opacity = scale_opacity.w;
    let rotation = unpack_uint2(rotation_scale.xy);
    let scale = scale_opacity.xyz;
#else
    // Standard mode: CPU already converted log_scale to scale via exp()
    // Also converted raw_opacity to opacity via sigmoid()
    let scale_data = scales[splat_index];
    let scale = vec3<f32>(scale_data.x, scale_data.y, scale_data.z);
    let opacity = opacities[splat_index];
    let rotation = rotations[splat_index];
#endif
    let quat_norm_sqr = dot(rotation, rotation);
    if quat_norm_sqr < 1e-6 {
        // Invalid covariance, emit degenerate triangle
        output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return output;
    }
    let color_0 = fetchColor(splat_index);
    var splat_color = vec4<f32>(color_0, opacity);
    let frag_pos = quad_verts[vertex_index];
    
    // Transform splat center: Local -> World -> View -> Clip
    // CRITICAL: splat_center_local is in model/local space, need model matrix first!
    let splat_center_world = transforms.model_matrix * splat_center_local;
    let view_center = view.view_from_world * splat_center_world;
    let clip_center = view.clip_from_view * view_center;
    
    // === ALPHA CULLING ===
    // Discard splats with very low opacity (below threshold)
    if splat_color.a < params.alpha_cull_threshold {
        // Emit degenerate triangle (will be culled by rasterizer)
        output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return output;
    }
    
    // === SPHERICAL HARMONICS (View-dependent color) ===
    // Calculate view direction in LOCAL/MODEL space
    // This is the most reliable method that avoids HLSL/WGSL matrix convention issues
    
    // Step 1: Direction in world space (from camera to splat)
    let world_dir = normalize(splat_center_world.xyz - view.world_position);
    
    // Step 2: Transform direction from world space to local space
    // For direction vectors, use the transpose of the 3x3 rotation part of model_matrix
    // (For orthogonal rotation matrices: inverse = transpose)
    // model_matrix transforms local -> world, so transpose transforms world -> local for directions
    let model_rot_t = mat3x3<f32>(
        vec3<f32>(transforms.model_matrix[0].x, transforms.model_matrix[1].x, transforms.model_matrix[2].x),
        vec3<f32>(transforms.model_matrix[0].y, transforms.model_matrix[1].y, transforms.model_matrix[2].y),
        vec3<f32>(transforms.model_matrix[0].z, transforms.model_matrix[1].z, transforms.model_matrix[2].z)
    );
    let view_dir_local = normalize(model_rot_t * world_dir);
    
    // Evaluate SH with local-space view direction
    // eval_sh function handles both PACK and standard modes
    let sh_color = splat_color.rgb + eval_sh(view_dir_local, splat_index);
    
    // Apply tint and offset in sRGB space (matching reference implementation)
    let final_srgb_color = sh_color * params.tint_color.xyz + params.color_offset.xyz;
    
    // Clamp alpha to valid range (matching SuperSplat: color.a = clamp(color.a, 0.0, 1.0))
    let clamped_alpha = clamp(splat_color.a * params.tint_color.w, 0.0, 1.0);
    
    // Convert color space based on render target format
    // 3DGS colors are inherently in sRGB space (DC = sh * C0 + 0.5)
    // For HDR targets (Rgba16Float): convert sRGB → linear
    // For LDR targets (Rgba8UnormSrgb): keep sRGB (GPU handles conversion)
    // let final_color = apply_color_space_conversion(final_srgb_color);
    
    splat_color = vec4<f32>(final_srgb_color, clamped_alpha);
    
    // Emit color as early as possible (for performance)
    // Note: In MipSplatting mode, opacity would be modified by extent basis calculation
    // output.color = splat_color;
    
    // === COVARIANCE PROJECTION ===
    // Compute 3D covariance from scale and rotation
    // CRITICAL: Use the HLSL-ported version with correct quaternion format
    let cov3d = threedgs_compute_covariance_3d(scale, rotation);
    
    // Compute focal length in pixels
    let focal = vec2<f32>(
        abs(view.clip_from_view[0][0]) * 0.5 * f32(params.surface_width),
        abs(view.clip_from_view[1][1]) * 0.5 * f32(params.surface_height)
    );
    
    // CRITICAL FIX: Compute model-view matrix (matching C++ implementation)
    // C++: modelViewMatrix = mul(pcRaster.modelMatrix, frameInfo.viewMatrix)
    let model_view_matrix = view.view_from_world * transforms.model_matrix;
    
    // Detect orthographic projection: proj[2][3] == 0.0 for orthographic, != 0.0 for perspective
    // In perspective: proj[2][3] is typically -1.0
    // In orthographic: proj[2][3] is 0.0
    let is_ortho = view.clip_from_view[2][3] == 0.0;
        
    // Project 3D covariance to 2D screen space (use model-view matrix)
    let cov2d = threedgs_covariance_projection(cov3d, view_center, focal, model_view_matrix, is_ortho);
    
    // === EXTENT BASIS CALCULATION ===
    // Compute the basis vectors of the extent of the projected covariance
    // We use sqrt(8) standard deviations instead of 3 to eliminate more of the 
    // splat with very low opacity.
    // CRITICAL: Pass viewport size for proper size limiting (matching PlayCanvas)
    let viewport_size = vec2<f32>(f32(params.surface_width), f32(params.surface_height));
    let projected_result = threedgs_projected_extent_basis(cov2d, SQRT_8, params.splat_scale, splat_color.a, viewport_size);
    let basis = projected_result.basis;
    // Check if covariance produced valid basis vectors
    if basis.x == 0.0 && basis.y == 0.0 {
        // Invalid covariance, emit degenerate triangle
        output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return output;
    }
    
    // === FRUSTUM EDGE CULLING (matching PlayCanvas/SuperSplat) ===
    // Cull splats that are mostly outside the frustum
    // PlayCanvas formula: if (any((abs(center.proj.xy) - vec2f(max(l1, l2)) * c) > center.proj.ww)) return false;
    // where c = center.proj.ww / viewport (converts pixels to clip space)
    // This prevents large splats at frustum edges from causing artifacts
    let max_extent_pixels = max(length(basis.xy), length(basis.zw));  // Maximum extent in pixels
    let pixel_to_clip = vec2<f32>(clip_center.w, clip_center.w) / viewport_size;  // Convert pixels to clip space
    let splat_extent_clip = vec2<f32>(max_extent_pixels, max_extent_pixels) * pixel_to_clip;  // Extent in clip space
    // Check if splat center + extent is outside frustum bounds
    if any((abs(clip_center.xy) - splat_extent_clip) > vec2<f32>(clip_center.w, clip_center.w)) {
        // Splat is mostly outside frustum, cull it
        output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return output;
    }
    
    // Update opacity with projected result (may be modified by MipSplatting antialiasing)
    splat_color.a = projected_result.opacity;
    
    // === VISUALIZATION MODE COLOR OUTPUT ===
    // Each mode calculates color differently
    
#ifdef VIS_POINT
    // Point cloud visualization: render as fixed-size points
    // Override basis vectors with small fixed size for point rendering
    // Point size is controlled by params.point_size (default ~2.0 pixels)
    let point_radius = max(params.point_size, 1.0);
    // Use isotropic (circular) basis for point rendering
    let point_basis = vec4<f32>(point_radius, 0.0, 0.0, point_radius);
    // Apply selection and lock state colors
    var point_color: vec3<f32>;
    if (splat_state & SPLAT_STATE_SELECTED) != 0u {
        // Selected: use select_color
        point_color = params.select_color.rgb;
    } else if (splat_state & SPLAT_STATE_LOCKED) != 0u {
        // Locked (not selected): use locked_color
        point_color = params.locked_color.rgb;
    } else {
        // Normal: use original splat color
        point_color = splat_color.rgb;
    }
    output.color = vec4<f32>(apply_color_space_conversion(point_color), 0.85);
    
    // Calculate NDC offset for point (simplified, no covariance)
    let pt_ndc_center = clip_center.xyz / clip_center.w;
    let pt_basis_viewport = vec2<f32>(
        1.0 / f32(params.surface_width),
        1.0 / f32(params.surface_height)
    );
    let pt_ndc_offset = (frag_pos.x * point_basis.xy + frag_pos.y * point_basis.zw) * pt_basis_viewport * 2.0;
    let pt_quad_pos = vec4<f32>(pt_ndc_center.xy + pt_ndc_offset, pt_ndc_center.z, 1.0);
    
    output.position = pt_quad_pos;
    output.frag_pos = frag_pos * SQRT_8;
    return output;
#endif

#ifdef VIS_NORMAL
    // Normal visualization: compute the splat normal and output as color
    let local_normal = compute_splat_normal(scale, rotation);
    // Transform normal from local space to world space using the rotation part of model matrix
    let model_rot = mat3x3<f32>(
        transforms.model_matrix[0].xyz,
        transforms.model_matrix[1].xyz,
        transforms.model_matrix[2].xyz
    );
    let world_normal = normalize(model_rot * local_normal);
    // Map normal from [-1,1] to [0,1] for display
    let normal_color = world_normal * 0.5 + 0.5;
    // Convert color space for HDR target (visualization colors are in sRGB)
    output.color = vec4<f32>(apply_color_space_conversion(normal_color), splat_color.a);
#endif

#ifdef VIS_DEPTH
    // Depth visualization: compute depth color directly
    let view_depth = -view_center.z; // View space depth (positive forward)
    // Normalize depth to [0, 1] range using near/far planes
    let near = 0.1;
    let far = 1000.0;  // Increased far plane for larger scenes
    let normalized_depth = clamp((view_depth - near) / (far - near), 0.0, 1.0);
    // Convert depth to heat map color
    let depth_color = depth_to_color(normalized_depth);
    // Convert color space for HDR target (visualization colors are in sRGB)
    output.color = vec4<f32>(apply_color_space_conversion(depth_color), splat_color.a);
#endif

#ifdef VIS_RINGS
    // Rings visualization: use the same color processing as default rendering
    // This ensures rings have the correct color (including SH, tint, and color space conversion)
    // The rings effect is applied in the fragment shader by modifying alpha
    var rings_color: vec3<f32>;
    if (splat_state & SPLAT_STATE_SELECTED) != 0u {
        // Selected: use select_color (already in linear space from UI)
        rings_color = params.select_color.rgb;
    } else if (splat_state & SPLAT_STATE_LOCKED) != 0u {
        // Locked (not selected): use locked_color
        rings_color = params.locked_color.rgb;
    } else {
        // Normal: use the fully processed splat color (includes SH, tint, color space conversion)
        rings_color = splat_color.rgb;
    }
    output.color = vec4<f32>(apply_color_space_conversion(rings_color), splat_color.a);
#endif

#ifdef VIS_ELLIPSOIDS
    // Ellipsoid visualization: color based on selection and lock state
    var ellipsoids_color: vec3<f32>;
    if (splat_state & SPLAT_STATE_SELECTED) != 0u {
        // Selected: use select_color
        ellipsoids_color = params.select_color.rgb;
    } else if (splat_state & SPLAT_STATE_LOCKED) != 0u {
        // Locked (not selected): use locked_color
        ellipsoids_color = params.locked_color.rgb;
    } else {
        // Normal: use original splat color
        ellipsoids_color = splat_color.rgb;
    }
    output.color = vec4<f32>(apply_color_space_conversion(ellipsoids_color), splat_color.a);
#endif

#ifdef VIS_CENTERS
    // Centers visualization: render as fixed-size circular points for selection
    // Use select_color.w (alpha) to pass edit point size from Edit Settings dialog
    // If select_color.w is 0, use default size of 2.0
    let edit_point_size = select(2.0, params.select_color.w, params.select_color.w > 0.0);
    let centers_point_radius = max(edit_point_size, 1.0);
    // Use isotropic (circular) basis for point rendering
    let centers_basis = vec4<f32>(centers_point_radius, 0.0, 0.0, centers_point_radius);
    
    // Color based on selection and lock state:
    var centers_color: vec3<f32>;
    if (splat_state & SPLAT_STATE_SELECTED) != 0u {
        // Selected: use select_color
        centers_color = params.select_color.rgb;
    } else if (splat_state & SPLAT_STATE_LOCKED) != 0u {
        // Locked (not selected): use locked_color
        centers_color = params.locked_color.rgb;
    } else {
        // Normal: use unselect_color
        centers_color = params.unselect_color.rgb;
    }
    output.color = vec4<f32>(apply_color_space_conversion(centers_color), 0.9);
    
    // Calculate NDC offset for point (simplified, no covariance)
    let centers_ndc_center = clip_center.xyz / clip_center.w;
    let centers_basis_viewport = vec2<f32>(
        1.0 / f32(params.surface_width),
        1.0 / f32(params.surface_height)
    );
    let centers_ndc_offset = (frag_pos.x * centers_basis.xy + frag_pos.y * centers_basis.zw) * centers_basis_viewport * 2.0;
    let centers_quad_pos = vec4<f32>(centers_ndc_center.xy + centers_ndc_offset, centers_ndc_center.z, 1.0);
    
    output.position = centers_quad_pos;
    output.frag_pos = frag_pos * SQRT_8;
    return output;
#endif

#ifndef VIS_POINT
#ifndef VIS_NORMAL
#ifndef VIS_DEPTH
#ifndef VIS_RINGS
#ifndef VIS_ELLIPSOIDS
#ifndef VIS_CENTERS
    // Default splat rendering: apply selection and lock state colors
    var final_splat_color: vec3<f32>;
    if (splat_state & SPLAT_STATE_SELECTED) != 0u {
        // Selected: use select_color to make selection visible even when Centers points are small/hidden
        final_splat_color = params.select_color.rgb;
    } else if (splat_state & SPLAT_STATE_LOCKED) != 0u {
        // Locked (not selected): use locked_color
        final_splat_color = params.locked_color.rgb;
    } else {
        // Normal: use original splat color
        final_splat_color = splat_color.rgb;
    }
    output.color = vec4<f32>(apply_color_space_conversion(final_splat_color), splat_color.a);
#endif
#endif
#endif
#endif
#endif
#endif
    
    // === QUAD POSITION CALCULATION ===
    // Convert clip space center to NDC
    let ndc_center = clip_center.xyz / clip_center.w;
    
    // basis_viewport: converts from pixels to NDC space
    let basis_viewport = vec2<f32>(
        1.0 / f32(params.surface_width),
        1.0 / f32(params.surface_height)
    );
    
    // ClipCorner optimization (matching PlayCanvas/SuperSplat)
    // This shrinks the quad based on opacity to exclude regions where alpha < 1/255
    // Reduces overdraw and eliminates barely-visible edge contributions
    let clip_factor = compute_clip_factor(splat_color.a);
    
    // Early exit for extremely faint splats (clip_factor = 0)
    if clip_factor <= 0.0 {
        // Degenerate triangle - GPU will cull it automatically
        output.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return output;
    }
    
    // Scale basis vectors by clip factor (shrinks the quad)
    let basis_vector1 = basis.xy * clip_factor;
    let basis_vector2 = basis.zw * clip_factor;
    
    // Calculate NDC offset for this vertex
    // inverseFocalAdjustment defaults to 1.0 (for pinhole camera)
    let inverse_focal_adjustment = 1.0;
    // basis_viewport converts from pixels to [0,1] viewport space
    // Multiply by 2.0 to convert to NDC [-1,1] space
    // NOTE: frag_pos is in [-1, 1] range, clip_factor only affects quad size (basis_vector)
    let ndc_offset = (frag_pos.x * basis_vector1 + frag_pos.y * basis_vector2) 
                     * basis_viewport * 2.0 * inverse_focal_adjustment;
    
    // Final quad vertex position
    let quad_pos = vec4<f32>(ndc_center.xy + ndc_offset, ndc_center.z, 1.0);
    
    // === OUTPUT ===
    output.position = quad_pos;
    // UV output
    // - frag_pos is in [-1, 1] range (raw quad corner coordinates)
    // - For normal rendering: clip_factor scales it down to [-clip, clip] range
    // - For VIS_RINGS: use unscaled UV so A ranges [0, 1] for proper ring detection
#ifdef VIS_RINGS
    // VIS_RINGS needs unscaled UV for correct ring size calculation
    output.frag_pos = frag_pos;
#else
    // Normal modes use clip_factor scaled UV for proper gaussian weight
    output.frag_pos = frag_pos * clip_factor;
#endif
    // Pass splat state to fragment shader for selection/locked coloring
    output.splat_state_out = splat_state;
    
    return output;
}

// Fragment shader
// Adapted from threedgs_raster.frag.slang
// This is the heart of the Gaussian Splatting rendering
@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    // === GAUSSIAN OPACITY CALCULATION (matching SuperSplat exactly) ===
    
    // A = squared distance from center, in UV space
    // UV is in [-clip, clip] range due to clipCorner, so A is in [0, clip^2] range
    let A = dot(input.frag_pos, input.frag_pos);
    
    // Discard fragments outside the unit circle
    if A > 1.0 {
        discard;
    }
    
    // Normalized Gaussian falloff (EXACTLY matching SuperSplat normExp)
    // CRITICAL FIX for "fogging" artifacts:
    // - exp(-4*A) returns ~0.018 at A=1, causing edge contributions to accumulate
    // - norm_exp returns EXACTLY 0.0 at A=1, eliminating edge fog
    // At A=0 (center): weight = 1.0
    // At A=1 (boundary): weight = exactly 0.0
    let weight = norm_exp(A);
    
    // Combine with splat opacity
    let opacity = weight * input.color.a;
    
    // Alpha threshold discard (matching SuperSplat: if (alpha < 1.0 / 255.0) discard)
    // Using hardcoded 1/255 threshold for consistent behavior (SuperSplat default)
#ifndef VIS_RINGS
    if opacity < params.alpha_cull_threshold {  // 1.0 / 255.0 ≈ 0.00392
        discard;
    }
#endif
    
    // Keep for visualization modes
    let A_normalized = A;
    
    // VIS_RINGS: Handle rings mode (matching SuperSplat exactly)
#ifdef VIS_RINGS
    // Ring visualization effect (matching SuperSplat reference implementation)
    // SuperSplat logic:
    // - Inside ring (A < 1.0 - ringSize): alpha = max(0.05, gaussianAlpha) - keeps splat visible but dim
    // - Ring edge (A >= 1.0 - ringSize): alpha = 0.6 - bright ring outline
    // This creates gaussian splats with bright ring outlines, NOT just rings on black background
    
    let ring_size = 0.04;
    
    // Apply rings effect: only for unlocked splats (SuperSplat checks texCoordIsLocked.z == 0.0)
    let is_locked = (input.splat_state_out & SPLAT_STATE_LOCKED) != 0u;
    
    // Calculate ring alpha exactly like SuperSplat
    var ring_alpha: f32 = opacity;  // Start with gaussian alpha
    
    if !is_locked && ring_size > 0.0 {
        if A_normalized < 1.0 - ring_size {
            // Inside the ring: keep gaussian alpha but ensure minimum visibility
            ring_alpha = max(0.05, opacity);
        } else {
            // Ring edge: fixed solid alpha for clear ring visibility
            ring_alpha = 0.6;
        }
    }
    
    // Alpha threshold discard
    if ring_alpha < 0.004 {
        discard;
    }
    
    // Premultiplied alpha output
    return vec4<f32>(input.color.rgb * ring_alpha, ring_alpha);
#endif
    
#ifdef OUTLINE_PASS
    // Outline pass: only output selected splats, discard unselected
    // This creates a silhouette of selected splats for edge detection
    let splat_state = input.splat_state_out;
    if (splat_state & SPLAT_STATE_SELECTED) == 0u {
        discard; // Not selected, don't render
    }
    // Output white color with full opacity for solid silhouette
    // Use full alpha to create clear boundaries for edge detection
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
#endif
    
    // === VISUALIZATION MODE OUTPUT ===

#ifdef VIS_POINT
    // Point visualization: render as solid circular point
    // Use a circular shape instead of gaussian falloff
    // A_normalized is already in [0, 1] range
    let point_dist = sqrt(A_normalized);  // Normalize to [0,1] at boundary
    if point_dist > 1.0 {
        discard;
    }
    // Smooth edge for anti-aliasing
    let point_alpha = (1.0 - smoothstep(0.8, 1.0, point_dist)) * input.color.a;
    // Premultiplied alpha output
    return vec4<f32>(input.color.rgb * point_alpha, point_alpha);
#endif
    
#ifdef VIS_ELLIPSOIDS
    // Ellipsoid visualization: solid ellipsoids with alpha = 1
    return vec4<f32>(input.color.rgb, 1.0);
#endif

#ifdef VIS_CENTERS
    // Centers visualization: render as solid circular point
    // A_normalized is already in [0, 1] range
    let centers_dist = sqrt(A_normalized);  // Normalize to [0,1] at boundary
    if centers_dist > 1.0 {
        discard;
    }
    // Smooth edge for anti-aliasing
    let centers_alpha = 1.0 - smoothstep(0.8, 1.0, centers_dist);
    // Premultiplied alpha output
    return vec4<f32>(input.color.rgb * centers_alpha, centers_alpha);
#endif

#ifdef PICK_PASS
    // PICK_PASS: Output splat index encoded as RGBA color for GPU picking
    // Encode 32-bit splat_index as 4 x 8-bit RGBA values
    let idx = input.splat_index_out;
    let r = f32((idx >> 0u) & 0xFFu) / 255.0;
    let g = f32((idx >> 8u) & 0xFFu) / 255.0;
    let b = f32((idx >> 16u) & 0xFFu) / 255.0;
    let a = f32((idx >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
#endif

#ifndef VIS_POINT
#ifndef VIS_RINGS
#ifndef VIS_ELLIPSOIDS
#ifndef VIS_CENTERS
#ifndef PICK_PASS
#ifndef OUTLINE_PASS
    // === DEFAULT OUTPUT (also used for VIS_NORMAL, VIS_DEPTH) ===
    // Color already computed in vertex shader (including selection/locked states)
    var final_color = input.color.rgb;
    
    // Note: Selection and locked colors are handled in vertex shader
    // No need to apply additional color processing here
    
    // Return PREMULTIPLIED alpha output (matching PlayCanvas/SuperSplat)
    // This works with blend mode: src=ONE, dst=ONE_MINUS_SRC_ALPHA
    // IMPORTANT: cache_blit.wgsl expects premultiplied alpha!
    return vec4<f32>(final_color * opacity, opacity);
#endif
#endif
#endif
#endif
#endif
#endif
}
