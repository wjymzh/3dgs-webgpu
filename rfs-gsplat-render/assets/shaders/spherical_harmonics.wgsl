// Spherical Harmonics constants and utilities for 3D Gaussian Splatting
// Float32 uncompressed version

// SH basis function constants
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

// NOTE: The eval_sh function must be defined in the importing shader (e.g., gaussian_splat.wgsl)
// because WGSL does not allow passing storage buffer pointers as function parameters.
// The function should directly access the global sh_buffer variable.
