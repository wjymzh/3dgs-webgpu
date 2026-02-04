// Outline shader for selected Gaussian splats
// Based on super-splat's outline implementation
// 
// This shader performs edge detection on the selection texture:
// - Renders selected splats to a texture
// - Detects edges by checking neighboring pixels
// - Draws outline where solid pixels meet transparent pixels

struct OutlineParams {
    color: vec4<f32>,          // Outline color (RGBA)
    alpha_cutoff: f32,         // Alpha threshold for edge detection
    kernel_size: i32,          // Edge detection kernel size (1-3)
    _padding: vec2<f32>,       // Padding for alignment
}

@group(0) @binding(0)
var outline_texture: texture_2d<f32>;

@group(0) @binding(1)
var outline_sampler: sampler;

@group(0) @binding(2)
var<uniform> params: OutlineParams;

// Vertex shader - fullscreen quad
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate fullscreen triangle
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// Fragment shader - edge detection and outline rendering
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_size = textureDimensions(outline_texture);
    let texel = vec2<i32>(in.position.xy);
    
    // Get center pixel alpha
    let center_alpha = textureLoad(outline_texture, texel, 0).a;
    
    // Skip if center pixel is solid (above alpha cutoff)
    // We want to draw outline on transparent pixels that are near solid pixels
    if (center_alpha > params.alpha_cutoff) {
        discard;
    }
    
    // Check if this transparent pixel has a solid neighbor
    let kernel_size = params.kernel_size;
    
    for (var x = -kernel_size; x <= kernel_size; x++) {
        for (var y = -kernel_size; y <= kernel_size; y++) {
            // Skip center pixel and orthogonal neighbors (only check diagonals for performance)
            if (x == 0 || y == 0) {
                continue;
            }
            
            // Sample neighbor pixel
            let neighbor_pos = texel + vec2<i32>(x, y);
            
            // Check bounds
            if (neighbor_pos.x >= 0 && neighbor_pos.x < i32(texture_size.x) &&
                neighbor_pos.y >= 0 && neighbor_pos.y < i32(texture_size.y)) {
                
                let neighbor_alpha = textureLoad(outline_texture, neighbor_pos, 0).a;
                
                // If neighbor is solid, draw outline at this transparent pixel
                if (neighbor_alpha > params.alpha_cutoff) {
                    return params.color;
                }
            }
        }
    }
    
    // No solid neighbor found, discard
    discard;
}

