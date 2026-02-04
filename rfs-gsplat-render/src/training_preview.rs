// Training Preview for Gaussian Splatting
// Handles uploading CPU-side rendered images from training backend to GPU
// and blitting them to the render target for real-time training visualization
//
// This module is backend-agnostic and works with any training backend
// (CUDA, Burn, etc.) that provides RGBA8 image buffers in CPU memory.
//
// Data flow:
// 1. Training backend renders 3DGS to RGBA8 image buffer (CPU memory, sRGB colors)
// 2. This module uploads the buffer to a Rgba8Unorm GPU texture (stores sRGB as-is)
// 3. Blit shader (cache_blit.wgsl) samples the texture, converts sRGB→linear,
//    and outputs linear color to final render target (HDR or LDR)
//
// NOTE: Uses Rgba8Unorm (not Rgba8UnormSrgb) to match 3DGS cache texture format.
// This allows sharing the same blit shader for both training preview and 3DGS rendering.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{
    Texture, TextureView, TextureDescriptor, TextureUsages, TextureDimension,
    TextureFormat, Extent3d, TextureViewDescriptor,
    BindGroup, BindGroupLayout, BindGroupLayoutEntries, BindGroupEntries,
    SamplerDescriptor, FilterMode, AddressMode,
    CachedRenderPipelineId, RenderPipelineDescriptor, PipelineCache,
    VertexState, FragmentState, PrimitiveState, PrimitiveTopology,
    MultisampleState, ColorTargetState, BlendState, BlendComponent, BlendFactor, BlendOperation,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::asset::AssetServer;
use std::sync::Arc;

/// Training preview image data shared between main world and render world
/// This is set by the main app when new preview data arrives from training backend
#[derive(Resource, Clone, Default)]
pub struct TrainingPreviewImageData {
    /// RGBA8 pixel data from training backend (sRGB color space)
    pub pixels: Option<Arc<Vec<u8>>>,
    /// Image dimensions
    pub width: u32,
    pub height: u32,
    /// Generation counter to detect updates
    pub generation: u64,
    /// Whether preview rendering is enabled
    pub enabled: bool,
}

impl ExtractResource for TrainingPreviewImageData {
    type Source = Self;
    
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// GPU-side resources for training preview rendering
/// Only exists in render world
#[derive(Resource)]
pub struct TrainingPreviewRenderTarget {
    /// RGBA8Unorm texture for the preview image
    pub texture: Option<Texture>,
    /// Texture view for sampling
    pub view: Option<TextureView>,
    /// Sampler for texture filtering
    pub sampler: Option<bevy::render::render_resource::Sampler>,
    /// Bind group for the blit shader
    pub bind_group: Option<BindGroup>,
    /// Current texture dimensions
    pub width: u32,
    pub height: u32,
    /// Last generation that was uploaded (to avoid redundant uploads)
    pub last_generation: u64,
}

impl Default for TrainingPreviewRenderTarget {
    fn default() -> Self {
        Self {
            texture: None,
            view: None,
            sampler: None,
            bind_group: None,
            width: 0,
            height: 0,
            last_generation: 0,
        }
    }
}

impl TrainingPreviewRenderTarget {
    /// Create or recreate the texture if dimensions changed
    pub fn ensure_texture(
        &mut self,
        render_device: &RenderDevice,
        width: u32,
        height: u32,
        bind_group_layout: Option<&BindGroupLayout>,
    ) {
        // Check if we need to recreate
        if self.texture.is_some() && self.width == width && self.height == height {
            return;
        }
        
        if width == 0 || height == 0 {
            return;
        }
        
        // Create RGBA8Unorm texture (sRGB data from training)
        // Note: We use Rgba8Unorm (NOT Rgba8UnormSrgb) because:
        // - Training backend outputs sRGB color bytes
        // - CPU upload stores bytes directly without conversion
        // - Blit shader (cache_blit.wgsl) does sRGB → linear conversion
        // This matches the 3DGS cache texture format for shader reuse
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("training_preview_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm, // Store sRGB bytes as-is, blit shader converts
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let view = texture.create_view(&TextureViewDescriptor::default());
        
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("training_preview_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create bind group if layout is provided
        let bind_group = bind_group_layout.map(|layout| {
            render_device.create_bind_group(
                Some("training_preview_bind_group"),
                layout,
                &BindGroupEntries::sequential((
                    &view,
                    &sampler,
                )),
            )
        });
        
        self.texture = Some(texture);
        self.view = Some(view);
        self.sampler = Some(sampler);
        self.bind_group = bind_group;
        self.width = width;
        self.height = height;
        
        info!("🎨 Created training preview texture: {}x{}", width, height);
    }
    
    /// Upload new image data to the texture
    pub fn upload_image(
        &mut self,
        render_queue: &RenderQueue,
        pixels: &[u8],
        width: u32,
        height: u32,
        generation: u64,
    ) {
        // Skip if already uploaded this generation
        if generation == self.last_generation {
            return;
        }
        
        let Some(ref texture) = self.texture else {
            return;
        };
        
        // Validate dimensions match
        if self.width != width || self.height != height {
            warn!("Training preview: dimension mismatch, texture={}x{}, image={}x{}",
                  self.width, self.height, width, height);
            return;
        }
        
        // Calculate expected size (RGBA = 4 bytes per pixel)
        let expected_size = (width * height * 4) as usize;
        if pixels.len() != expected_size {
            warn!("Training preview: size mismatch, expected {} bytes, got {}",
                  expected_size, pixels.len());
            return;
        }
        
        // Upload to GPU
        render_queue.write_texture(
            texture.as_image_copy(),
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        
        self.last_generation = generation;
    }
    
    /// Check if the preview texture is ready to use
    pub fn is_ready(&self) -> bool {
        self.texture.is_some() && self.view.is_some() && self.bind_group.is_some()
    }
}

/// Pipeline for blitting training preview to the render target
#[derive(Resource)]
pub struct TrainingPreviewBlitPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shader: Handle<Shader>,
    pub pipeline_id: Option<CachedRenderPipelineId>,
}

impl FromWorld for TrainingPreviewBlitPipeline {
    fn from_world(world: &mut World) -> Self {
        use bevy::render::render_resource::binding_types;
        use bevy::render::render_resource::ShaderStages;
        
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();
        
        // Create bind group layout for preview texture + sampler
        let bind_group_layout = render_device.create_bind_group_layout(
            Some("training_preview_blit_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // @binding(0): Preview texture (sRGB)
                    binding_types::texture_2d(wgpu::TextureSampleType::Float { filterable: true }),
                    // @binding(1): Sampler
                    binding_types::sampler(wgpu::SamplerBindingType::Filtering),
                ),
            ),
        );
        
        // UNIFIED BLIT SHADER: Use cache_blit.wgsl for both 3DGS cache and training preview
        // Both sources output premultiplied alpha format, so same shader works for both
        let shader = load_embedded_asset!(asset_server, "../assets/shaders/cache_blit.wgsl");
        
        Self {
            bind_group_layout,
            shader,
            pipeline_id: None,
        }
    }
}

impl TrainingPreviewBlitPipeline {
    /// Get or create the blit pipeline for the given format
    pub fn get_pipeline(
        &mut self,
        pipeline_cache: &PipelineCache,
        hdr: bool,
        msaa_samples: u32,
    ) -> Option<CachedRenderPipelineId> {
        // If we already have a pipeline ID, check if it's actually compiled
        if let Some(id) = self.pipeline_id {
            if pipeline_cache.get_render_pipeline(id).is_some() {
                // Pipeline is ready
                return self.pipeline_id;
            }
            // Pipeline exists but not ready yet - keep waiting
            // Don't reset because it might still be compiling
            return self.pipeline_id;
        }
        
        // Choose format based on HDR setting
        // Use ViewTarget constants for consistency with Bevy's render pipeline
        let format = if hdr {
            bevy::render::view::ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::Rgba8UnormSrgb
        };
        
        info!("🔧 Creating training preview blit pipeline: hdr={}, msaa={}, format={:?}", 
              hdr, msaa_samples, format);
        
        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("training_preview_blit_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("vertex".into()),
                buffers: vec![], // Fullscreen triangle, no vertex buffer
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, // No depth testing for blit
            multisample: MultisampleState {
                count: msaa_samples,
                ..Default::default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    // Premultiplied alpha blending
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One, // Premultiplied: use ONE
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: bevy::render::render_resource::ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: true,
        });
        
        self.pipeline_id = Some(pipeline_id);
        Some(pipeline_id)
    }
}

/// Plugin for training preview rendering
pub struct TrainingPreviewPlugin;

impl Plugin for TrainingPreviewPlugin {
    fn build(&self, app: &mut App) {
        // Initialize the main world resource for preview data
        app.init_resource::<TrainingPreviewImageData>();
        
        // Extract resource to render world
        app.add_plugins(ExtractResourcePlugin::<TrainingPreviewImageData>::default());
    }
    
    fn finish(&self, app: &mut App) {
        // Initialize render world resources
        let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
            return;
        };
        
        render_app
            .init_resource::<TrainingPreviewRenderTarget>()
            .init_resource::<TrainingPreviewBlitPipeline>();
        
        // Add systems for preparing and uploading preview data
        use bevy::render::Render;
        use bevy::render::RenderSystems;
        
        render_app.add_systems(
            Render,
            (prepare_training_preview_texture, prepare_training_preview_pipeline)
                .in_set(RenderSystems::Prepare),
        );
    }
}

/// Prepare training preview texture and upload new data
fn prepare_training_preview_texture(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    preview_data: Res<TrainingPreviewImageData>,
    mut preview_target: ResMut<TrainingPreviewRenderTarget>,
    blit_pipeline: Res<TrainingPreviewBlitPipeline>,
) {
    // Skip if preview is disabled
    if !preview_data.enabled {
        return;
    }
    
    // Skip if no data
    let Some(ref pixels) = preview_data.pixels else {
        return;
    };
    
    let width = preview_data.width;
    let height = preview_data.height;
    
    if width == 0 || height == 0 {
        return;
    }
    
    // Ensure texture exists with correct dimensions
    preview_target.ensure_texture(
        &render_device,
        width,
        height,
        Some(&blit_pipeline.bind_group_layout),
    );
    
    // Upload image data
    preview_target.upload_image(
        &render_queue,
        pixels,
        width,
        height,
        preview_data.generation,
    );
}

/// Prepare the blit pipeline
fn prepare_training_preview_pipeline(
    mut blit_pipeline: ResMut<TrainingPreviewBlitPipeline>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(&bevy::render::view::ExtractedView, &bevy::render::view::Msaa)>,
) {
    // Note: We don't check shader loading here because:
    // 1. Assets<Shader> doesn't exist in render world
    // 2. PipelineCache handles shader loading internally - pipeline stays in queue until shader is ready
    
    // Use simpler query without ViewTarget
    let Some((view, msaa)) = views.iter().next() else {
        // Fallback: try to create pipeline with defaults if no view available
        if blit_pipeline.pipeline_id.is_none() {
            blit_pipeline.get_pipeline(&pipeline_cache, false, 1);
        }
        return;
    };
    
    let msaa_samples = msaa.samples();
    blit_pipeline.get_pipeline(&pipeline_cache, view.hdr, msaa_samples);
}

/// Get the bind group and pipeline for blitting the preview
/// Called from the main render node when it needs to composite the preview
pub fn get_training_preview_blit_resources(
    preview_target: &TrainingPreviewRenderTarget,
    pipeline: &TrainingPreviewBlitPipeline,
    pipeline_cache: &PipelineCache,
) -> Option<(BindGroup, bevy::render::render_resource::RenderPipeline)> {
    if !preview_target.is_ready() {
        return None;
    }
    
    let bind_group = preview_target.bind_group.clone()?;
    let pipeline_id = pipeline.pipeline_id?;
    let render_pipeline = pipeline_cache.get_render_pipeline(pipeline_id)?.clone();
    
    Some((bind_group, render_pipeline))
}
