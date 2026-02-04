// Outline rendering for selected Gaussian splats
// Inspired by super-splat project's outline system
//
// Implementation:
// 1. Render selected splats to a separate outline texture (handled in GaussianSplatNode)
// 2. Apply edge detection shader to create outline
// 3. Composite outline onto main render target
//
// Usage:
// - Set RenderingConfig.show_outline = true on GaussianSplats entity
// - Configure outline appearance with OutlineConfig component

use bevy::{
    asset::embedded_asset,
    core_pipeline::core_3d::{
        graph::{Core3d, Node3d},
    },
    prelude::*,
    render::{
        render_graph::{RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner},
        render_resource::*,
        render_resource::binding_types::{texture_2d, sampler, uniform_buffer},
        renderer::RenderDevice,
        view::{ExtractedView, ViewTarget},
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};

/// Configuration for outline rendering on selected splats
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct OutlineConfig {
    /// Enable outline rendering (default: false)
    pub enabled: bool,
    
    /// Outline color (RGBA, default: white with 50% alpha)
    pub color: Vec4,
    
    /// Alpha cutoff threshold for edge detection (default: 0.4)
    /// Lower values = thicker outlines
    /// Set to 0.0 for rings mode
    pub alpha_cutoff: f32,
    
    /// Edge detection kernel size (default: 2)
    /// Range: 1-3, larger values = thicker outlines
    pub kernel_size: i32,
}

impl Default for OutlineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            color: Vec4::new(1.0, 1.0, 0.0, 1.0), // Bright yellow, full opacity
            alpha_cutoff: 0.4, // Threshold to detect splat boundaries
            kernel_size: 2, // Kernel size for edge detection
        }
    }
}

impl OutlineConfig {
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }
    
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
    
    pub fn with_color(mut self, color: Vec4) -> Self {
        self.color = color;
        self
    }
    
    pub fn with_alpha_cutoff(mut self, alpha_cutoff: f32) -> Self {
        self.alpha_cutoff = alpha_cutoff;
        self
    }
    
    pub fn with_kernel_size(mut self, kernel_size: i32) -> Self {
        self.kernel_size = kernel_size.clamp(1, 3);
        self
    }
}

/// GPU uniform for outline shader
#[derive(Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct OutlineParams {
    color: Vec4,
    alpha_cutoff: f32,
    kernel_size: i32,
    _padding: [f32; 2],
}

impl From<&OutlineConfig> for OutlineParams {
    fn from(config: &OutlineConfig) -> Self {
        Self {
            color: config.color,
            alpha_cutoff: config.alpha_cutoff,
            kernel_size: config.kernel_size,
            _padding: [0.0; 2],
        }
    }
}

/// Extracted outline config (render world)
#[derive(Resource)]
pub struct ExtractedOutlineConfig {
    pub(crate) enabled: bool,
    pub(crate) params: OutlineParams,
}

// Note: We don't use ExtractComponent trait anymore since we're extracting as a Resource

/// Outline render target resource
#[derive(Resource)]
pub struct OutlineRenderTarget {
    pub texture: Texture,
    pub view: TextureView,
    pub depth_texture: Texture,
    pub depth_view: TextureView,
    pub size: Extent3d,
}

/// Outline pipeline resource
#[derive(Resource)]
pub struct OutlinePipeline {
    pub bind_group_layout: BindGroupLayout,
    pub sampler: Sampler,
    pub shader: Handle<Shader>,
}

/// Cached pipeline ID
#[derive(Resource)]
pub struct OutlinePipelineId(pub CachedRenderPipelineId);

/// Render label for outline node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct OutlineLabel;

/// Prepare outline render target
fn prepare_outline_render_target(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    views: Query<(&ExtractedView, &Msaa)>,
    existing_target: Option<Res<OutlineRenderTarget>>,
) {
    let Some((view, _msaa)) = views.iter().next() else {
        return;
    };

    let size = Extent3d {
        width: view.viewport.z as u32,
        height: view.viewport.w as u32,
        depth_or_array_layers: 1,
    };

    // Only recreate if size changed
    if let Some(target) = existing_target {
        if target.size == size {
            return;
        }
    }

    // Create outline texture with HDR format to match Gaussian Splat pipeline
    // NOTE: We use sample_count = 1 (no MSAA) for the outline texture because:
    // 1. It's read by the outline shader as texture_2d (not texture_multisampled_2d)
    // 2. Edge detection works better on non-MSAA textures
    // The Gaussian Splat pipeline will handle MSAA, we just need compatible format
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some("outline_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1, // No MSAA for outline texture (read by shader as texture_2d)
        dimension: TextureDimension::D2,
        format: ViewTarget::TEXTURE_FORMAT_HDR, // Use HDR format (Rgba16Float)
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());

    // Create depth texture (also no MSAA to match color attachment)
    let depth_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("outline_depth_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1, // No MSAA to match color attachment
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

    commands.insert_resource(OutlineRenderTarget {
        texture,
        view,
        depth_texture,
        depth_view,
        size,
    });
}

/// Prepare outline pipeline
fn prepare_outline_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    outline_pipeline: Res<OutlinePipeline>,
    views: Query<&Msaa>,
    existing_id: Option<Res<OutlinePipelineId>>,
) {
    if existing_id.is_some() {
        return; // Already prepared
    }

    // Get MSAA settings from the first view (typically the main camera)
    let msaa_samples = views.iter().next().map(|m| m.samples()).unwrap_or(1);

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("outline_pipeline".into()),
        layout: vec![outline_pipeline.bind_group_layout.clone()],
        push_constant_ranges: vec![],
        vertex: VertexState {
            shader: outline_pipeline.shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vertex".into()),
            buffers: vec![],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: MultisampleState {
            count: msaa_samples,  // Match the MSAA setting of the ViewTarget
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(FragmentState {
            shader: outline_pipeline.shader.clone(),
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                // Use HDR format to match the ViewTarget format (Rgba16Float)
                // Our camera is configured with HDR enabled
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        // Set to false for render pipelines (only relevant for compute shaders)
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(OutlinePipelineId(pipeline_id));
}

/// Outline rendering node
#[derive(Default)]
pub struct OutlineNode;

impl ViewNode for OutlineNode {
    type ViewQuery = (&'static ExtractedView, &'static ViewTarget);

    fn run<'w>(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext<'w>,
        (view, target): bevy::ecs::query::QueryItem<'w, 'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        // Check if outline config resource exists
        let Some(config) = world.get_resource::<ExtractedOutlineConfig>() else {
            debug!("OutlineNode: No ExtractedOutlineConfig found");
            return Ok(());
        };
        
        // Skip if outline is disabled
        if !config.enabled {
            return Ok(());
        }

        // Get resources
        let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
            warn!("OutlineNode: PipelineCache not found");
            return Ok(());
        };
        let Some(outline_pipeline) = world.get_resource::<OutlinePipeline>() else {
            warn!("OutlineNode: OutlinePipeline not found");
            return Ok(());
        };
        let Some(outline_target) = world.get_resource::<OutlineRenderTarget>() else {
            warn!("OutlineNode: OutlineRenderTarget not found");
            return Ok(());
        };
        let Some(pipeline_id) = world.get_resource::<OutlinePipelineId>() else {
            warn!("OutlineNode: OutlinePipelineId not found");
            return Ok(());
        };
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
            warn!("OutlineNode: Failed to get render pipeline from cache");
            return Ok(());
        };

        // Create params buffer
        let render_device = render_context.render_device();
        let params_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("outline_params_buffer"),
            contents: bytemuck::cast_slice(&[config.params]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group = render_device.create_bind_group(
            "outline_bind_group",
            &outline_pipeline.bind_group_layout,
            &BindGroupEntries::sequential((
                &outline_target.view,
                &outline_pipeline.sampler,
                params_buffer.as_entire_buffer_binding(),
            )),
        );

        // Render outline pass
        let mut color_attachment = target.get_color_attachment();
        color_attachment.ops = Operations {
            load: LoadOp::Load,
            store: StoreOp::Store,
        };

        // Calculate safe viewport dimensions to prevent scissor rect out of bounds
        let target_main_texture = target.main_texture_view();
        let target_size = target_main_texture.texture().size();
        
        let viewport_x = view.viewport.x;
        let viewport_y = view.viewport.y;
        let viewport_width = view.viewport.z.min(target_size.width.saturating_sub(viewport_x));
        let viewport_height = view.viewport.w.min(target_size.height.saturating_sub(viewport_y));
        
        // Skip rendering if viewport is invalid (e.g., window minimized)
        if viewport_width == 0 || viewport_height == 0 {
            return Ok(());
        }

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("outline_pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Set viewport to ensure scissor rect is valid
        render_pass.set_viewport(
            viewport_x as f32,
            viewport_y as f32,
            viewport_width as f32,
            viewport_height as f32,
            0.0,
            1.0,
        );

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Fullscreen triangle

        Ok(())
    }
}

/// Extract outline configs
/// Note: OutlineConfig should be on Camera entities, not on regular entities
fn extract_outline_configs(
    mut commands: Commands,
    configs: Extract<Query<&OutlineConfig, With<Camera>>>,
) {
    // Extract outline config from the first camera (should only be one per view)
    if let Some(config) = configs.iter().next() {
        commands.insert_resource(ExtractedOutlineConfig {
            enabled: config.enabled,
            params: config.into(),
        });
    }
}

/// Outline plugin
pub struct OutlinePlugin;

impl Plugin for OutlinePlugin {
    fn build(&self, app: &mut App) {
        // Embed outline shader
        embedded_asset!(app, "../assets/shaders/outline.wgsl");
        
        // Register component
        app.register_type::<OutlineConfig>();
        
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        
            render_app
                .add_systems(ExtractSchedule, extract_outline_configs)
                .add_systems(
                    Render,
                    prepare_outline_render_target.in_set(RenderSystems::Prepare),
                )
                .add_systems(
                    Render,
                    prepare_outline_pipeline.in_set(RenderSystems::Prepare),
                )
            .add_render_graph_node::<ViewNodeRunner<OutlineNode>>(Core3d, OutlineLabel)
            .add_render_graph_edges(
                Core3d,
                // Render outline after main pass post-processing, before upscaling
                (Node3d::EndMainPassPostProcessing, OutlineLabel, Node3d::Upscaling),
            );
    }

    fn finish(&self, app: &mut App) {
        // Get asset server before borrowing render_app mutably
        let shader = app.world().resource::<AssetServer>()
            .load("embedded://rfs_gsplat_render/../assets/shaders/outline.wgsl");
        
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_device = render_app.world().resource::<RenderDevice>();

        // Create bind group layout
        let bind_group_layout = render_device.create_bind_group_layout(
            "outline_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    uniform_buffer::<OutlineParams>(false),
                ),
            ),
        );

        // Create sampler
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("outline_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        render_app.insert_resource(OutlinePipeline {
            bind_group_layout,
            sampler,
            shader,
        });
    }
}
