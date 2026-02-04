// Fullscreen triangle shader for blitting cached 3DGS render result
// Uses premultiplied alpha blending to composite cached splats onto the scene
//
// BLIT ARCHITECTURE:
// - Cache texture uses Rgba8Unorm format (stores sRGB colors from gaussian_splat.wgsl directly)
// - Rgba8Unorm does NOT auto-convert, so we sample sRGB values directly
// - This shader manually converts sRGB → linear before output
// - This shader outputs LINEAR color to the final render target (HDR or LDR)
// - If final target is Rgba8UnormSrgb (LDR), GPU auto-converts linear → sRGB on write
// - If final target is Rgba16Float (HDR), linear values are used directly for post-processing
//
// Why Rgba8Unorm instead of Rgba8UnormSrgb for cache?
// - gaussian_splat.wgsl outputs sRGB colors directly
// - Rgba8UnormSrgb would apply linear→sRGB on write (double-gamma = too bright!)
// - Rgba8Unorm stores sRGB values as-is, we convert here during blit

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle - no vertex buffer needed
// Covers the entire screen with a single triangle
@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    
    // Generate fullscreen triangle vertices
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    
    // Convert from [-1, 1] to [0, 1] UV coordinates
    // Note: Y is flipped for texture sampling
    output.uv = vec2<f32>(
        (x + 1.0) * 0.5,
        (1.0 - y) * 0.5
    );
    
    return output;
}

@group(0) @binding(0)
var cache_texture: texture_2d<f32>;
@group(0) @binding(1)
var cache_sampler: sampler;

// sRGB to linear conversion (matches Bevy's implementation)
// Single channel conversion
fn srgb_to_linear_channel(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        return srgb / 12.92;
    } else {
        return pow((srgb + 0.055) / 1.055, 2.4);
    }
}

// RGB conversion (alpha is already linear)
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear_channel(srgb.x),
        srgb_to_linear_channel(srgb.y),
        srgb_to_linear_channel(srgb.z)
    );
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the cached render result
    // Cache texture is Rgba8Unorm: values are stored as-is (sRGB values)
    let cached_color = textureSample(cache_texture, cache_sampler, input.uv);
    
    // The cached color is in sRGB space with premultiplied alpha
    // We need to convert RGB from sRGB to linear for correct output
    // Note: For premultiplied alpha, we convert the pre-multiplied RGB values directly
    let linear_rgb = srgb_to_linear(cached_color.rgb);
    
    // Output LINEAR color to final render target for correct compositing
    return vec4<f32>(linear_rgb, cached_color.a);
}

