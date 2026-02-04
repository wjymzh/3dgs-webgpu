// sqrt(8) = 2.828... used for extent calculation
const SQRT_8: f32 = 2.82842712475;

// Computes the 3D covariance matrix from scale and rotation quaternion
// Returns the 6 unique elements of the symmetric 3x3 matrix
fn threedgs_compute_covariance_3d(scale: vec3<f32>, rotation: vec4<f32>) -> mat3x3<f32> {
    // CRITICAL: Quaternion format from HLSL is (r, x, y, z) where r is scalar (w in standard notation)
    // rotation.x = r (scalar/w)
    // rotation.y = x
    // rotation.z = y  
    // rotation.w = z
    let r = rotation.x;  // scalar part
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;
    
    // Build rotation matrix from quaternion
    // HLSL row-major construction (by rows):
    // Row 0: [1-2(yy+zz), 2(xy-rz), 2(xz+ry)]
    // Row 1: [2(xy+rz), 1-2(xx+zz), 2(yz-rx)]
    // Row 2: [2(xz-ry), 2(yz+rx), 1-2(xx+yy)]
    //
    // WGSL column-major: construct by columns (transpose of rows)
    let R = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),  // Column 0
        vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),  // Column 1
        vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y))   // Column 2
    );
    
    // Build scale matrix (diagonal)
    // WGSL column-major
    let S = mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0),  // Column 0
        vec3<f32>(0.0, scale.y, 0.0),  // Column 1
        vec3<f32>(0.0, 0.0, scale.z)   // Column 2
    );
    
    // M = R * S (rotation followed by scale, matching HLSL mul(R, S))
    // In WGSL column-major: R * S means apply S first, then R
    let M = R * S;
    
    // Covariance is M * M^T
    // Compute efficiently by calculating upper triangle elements
    return M * transpose(M);
}

// Projects 3D covariance to 2D screen space
// Based on EWA splatting (Zwicker et al. 2002)
// Supports both perspective and orthographic projection
fn threedgs_covariance_projection(
    cov3d: mat3x3<f32>,
    view_center: vec4<f32>,
    focal: vec2<f32>,
    model_view_matrix: mat4x4<f32>,
    is_ortho: bool
) -> vec3<f32> {
    // For orthographic projection, use a fixed direction (0,0,1) instead of actual view center
    // This is because orthographic projection has parallel projection lines
    var v: vec3<f32>;
    if is_ortho {
        v = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        v = view_center.xyz;
    }
    
    let s = 1.0 / (v.z * v.z);
    
    // Jacobian matrix for projection
    // WGSL column-major
    let J = mat3x3<f32>(
        vec3<f32>(focal.x / v.z, 0.0, 0.0),
        vec3<f32>(0.0, focal.y / v.z, 0.0),
        vec3<f32>(-(focal.x * v.x) * s, -(focal.y * v.y) * s, 0.0)
    );
    
    // WGSL needs: W = mat3x3(column0, column1, column2)
    // So we transpose the HLSL construction
    let W = mat3x3<f32>(
        vec3<f32>(model_view_matrix[0][0], model_view_matrix[0][1], model_view_matrix[0][2]),  // Column 0
        vec3<f32>(model_view_matrix[1][0], model_view_matrix[1][1], model_view_matrix[1][2]),  // Column 1
        vec3<f32>(model_view_matrix[2][0], model_view_matrix[2][1], model_view_matrix[2][2])   // Column 2
    );
    
    // Transform matrix: T = J * W (matching HLSL mul(J, W))
    let T = J * W;
    
    // Project covariance: cov2D = T * cov3D * T^T
    let cov2d = T * cov3d * transpose(T);
    
    // Return the 3 unique elements: (xx, xy, yy)
    return vec3<f32>(cov2d[0][0], cov2d[0][1], cov2d[1][1]);
}

struct ProjectedExtentBasisResult {
    basis_vector1: vec2<f32>,
    basis_vector2: vec2<f32>,
    opacity: f32
}
// This function ingests the projected 2D covariance and outputs the basis vectors of its 2D extent
// Matches PlayCanvas/SuperSplat implementation for consistent rendering quality
fn threedgs_projected_extent_basis(
    cov2d_in: vec3<f32>,
    std_dev: f32,
    splat_scale: f32,
    opacity: f32,
    viewport_size: vec2<f32>  // Added: viewport dimensions for proper size limiting
) -> vec4<f32> {
    var cov2d = cov2d_in;
    // Add low-pass filter to avoid very small eigenvalues
    // This acts as a regularization term (matching PlayCanvas: +0.3)
    cov2d.x += 0.3;
    cov2d.z += 0.3;

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
        return vec4<f32>(0.0);
    }
    
    // Use viewport-based maximum limit (matching PlayCanvas)
    // vmin = min(1024.0, min(viewport_size.x, viewport_size.y))
    let vmin = min(1024.0, min(viewport_size.x, viewport_size.y));
    
    // Compute axis lengths: l = 2.0 * min(sqrt(2.0 * lambda), vmin)
    let l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin);
    let l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin);
    
    // CRITICAL: Early-out for gaussians smaller than 2 pixels (matching PlayCanvas)
    // This eliminates sub-pixel splats that cause "fogging" artifacts
    if l1 < 2.0 && l2 < 2.0 {
        return vec4<f32>(0.0);
    }
    
    // Compute eigenvector from offDiagonal and eigenvalue difference
    let diag_vec = normalize(vec2<f32>(b, lambda1 - a));
    let eigenvector1 = diag_vec;
    let eigenvector2 = vec2<f32>(diag_vec.y, -diag_vec.x);
    
    // Apply splat scale and compute basis vectors
    let basis_vector1 = eigenvector1 * splat_scale * l1;
    let basis_vector2 = eigenvector2 * splat_scale * l2;
    
    return vec4<f32>(basis_vector1, basis_vector2);
}

