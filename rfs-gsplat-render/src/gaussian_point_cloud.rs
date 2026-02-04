use bevy::{
    asset::load_embedded_asset,
    core_pipeline::core_3d::{
        graph::{Core3d, Node3d},
        CORE_3D_DEPTH_FORMAT,
    },
    prelude::*,
    render::{
        extract_component::ExtractComponent,
        extract_resource::ExtractResourcePlugin,
        render_graph::{RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner},
        render_resource::{
            binding_types::{storage_buffer_read_only_sized, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferInitDescriptor, BufferUsages,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, CommandEncoderDescriptor,
            CompareFunction, ComputePassDescriptor, ComputePipelineDescriptor, DepthBiasState,
            DepthStencilState, FragmentState, LoadOp, MultisampleState, Operations, PipelineCache,
            PrimitiveState, PrimitiveTopology, RenderPassDepthStencilAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, ShaderType,
            SpecializedComputePipeline, SpecializedComputePipelines,
            SpecializedRenderPipeline, SpecializedRenderPipelines, StencilFaceState, StencilState,
            StoreOp, TextureFormat, VertexState,
        },
        renderer::{RenderDevice, RenderQueue},
        view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};

use crate::gaussian_splats::{GaussianSplats, sigmoid, PackModeConfig, SplatSelectionState};
use crate::radix_sort::{
    RadixSortPlugin, RadixSortPipelines, RadixSortBuffers, RadixSortBindGroups,
    create_radix_sort_buffers, execute_radix_sort,
};
use crate::temporal_coherence::{
    TemporalCoherenceCache, TemporalCoherenceConfig, TemporalCoherenceStats, 
    GaussianSplatRenderCache, should_skip_sorting,
};

// Parallel processing with rayon (native only)
#[cfg(feature = "rayon")]
use rayon::prelude::*;


/// Splat visualization mode for different debug/rendering outputs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Reflect)]
pub enum SplatVisMode {
    /// Normal splat rendering with colors
    #[default]
    Splat,
    /// Point cloud rendering (single pixel per splat)
    Point,
    /// Rings visualization (concentric rings on splats)
    Rings,
    /// Centers mode for splat selection editing (blue points)
    Centers,
    /// Pick mode for GPU-based splat selection (outputs splat ID as color)
    Pick,
    /// Outline mode (render only selected splats for outline detection)
    Outline,
}

/// Packed Vec3 for GPU storage (matches WGSL struct PackedVec3)
/// Saves memory compared to Vec4 (12 bytes vs 16 bytes)
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct PackedVec3 {
    x: f32,
    y: f32,
    z: f32,
}

/// Packed vertex SH data (matches C++ PackedVertexSH)
/// Stores 15 SH coefficients in compressed 11_10_11 format
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct PackedVertexSH {
    sh1to3: [u32; 4],   // SH coefficients 1-3 + scale factor
    sh4to7: [u32; 4],   // SH coefficients 4-7
    sh8to11: [u32; 4],  // SH coefficients 8-11
    sh12to15: [u32; 4], // SH coefficients 12-15
}

/// Pack a normal/direction vector in 11_10_11 format (32 bits total)
/// Matching C++ pack_unit_direction_11_10_11 EXACTLY:
///   x = bits 0-10  (low 11 bits)
///   y = bits 11-20 (mid 10 bits)
///   z = bits 21-31 (high 11 bits)
fn pack_normal_11_10_11(v: Vec3) -> u32 {
    let x = ((v.x + 1.0) * 0.5 * 2047.0).clamp(0.0, 2047.0) as u32;
    let y = ((v.y + 1.0) * 0.5 * 1023.0).clamp(0.0, 1023.0) as u32;
    let z = ((v.z + 1.0) * 0.5 * 2047.0).clamp(0.0, 2047.0) as u32;
    (z << 21) | (y << 11) | x
}

/// Pack two f32 values into f16 format and store in u32
fn pack_half2(a: f32, b: f32) -> u32 {
    use half::f16;
    let a_u16 = f16::from_f32(a).to_bits();
    let b_u16 = f16::from_f32(b).to_bits();
    (a_u16 as u32) | ((b_u16 as u32) << 16)
}

/// Gaussian Point Cloud rendering parameters (GPU uniform)
#[derive(Component, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
#[repr(C)]
pub struct GaussianSplatParams {
    pub point_size: f32,           // Legacy, not used in 3DGS
    pub surface_width: u32,
    pub surface_height: u32,
    pub point_count: u32,
    pub frustum_dilation: f32,     // Default: 0.2
    pub alpha_cull_threshold: f32, // Default: 0.005
    pub splat_scale: f32,          // Global splat scale, default: 1.0
    pub sh_degree: u32,            // Spherical harmonics degree (0-3), used in PACK mode
    pub select_color: Vec4,        // Color for selected splats (RGB in 0-1, A = edit point size)
    pub unselect_color: Vec4,      // Color for unselected splats in overlay (RGB in 0-1, A unused)
    pub locked_color: Vec4,
    pub tint_color: Vec4,
    pub color_offset: Vec4,
}

/// Point size configuration component (user-configurable)
/// Attach to GaussianSplats entity to control the size of rendered points
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct PointSizeConfig {
    /// Point size in pixels (default: 1.0)
    pub size: f32,
}

impl Default for PointSizeConfig {
    fn default() -> Self {
        Self { size: 1.0 }
    }
}

/// Culling configuration component (user-configurable)
/// Attach to GaussianSplats entity to control culling behavior
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct CullingConfig {
    /// Frustum culling margin (default: 0.0)
    /// 0.0 = tight culling, >0 = looser culling (keeps more points near frustum edges)
    pub frustum_dilation: f32,
    /// Alpha threshold for culling (default: 0.0)
    /// Points with alpha below this threshold will be culled
    pub alpha_cull_threshold: f32,
}

impl Default for CullingConfig {
    fn default() -> Self {
        Self {
            frustum_dilation: 0.2,
            alpha_cull_threshold: 0.005,
        }
    }
}

/// Rendering configuration component (user-configurable)
/// Attach to GaussianSplats entity to control rendering features
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct RenderingConfig {
    /// Enable anti-aliasing (default: true)
    /// Controls GSPLAT_AA shader variant
    pub antialias: bool,
    /// Spherical harmonics degree (0-3, default: 3)
    /// Controls SH_DEGREE shader variant
    /// Higher degrees provide more accurate view-dependent lighting but cost more
    pub sh_band: u32,
    /// Global splat scale multiplier (default: 1.0)
    /// Values > 1.0 make splats larger, < 1.0 make them smaller
    pub splat_scale: f32,
    /// Point size in pixels (default: 1.0)
    pub point_size: f32,
    /// Frustum culling margin (default: 0.2)
    /// 0.0 = tight culling, >0 = looser culling (keeps more points near frustum edges)
    pub frustum_dilation: f32,
    /// Alpha threshold for culling (default: 0.005)
    /// Points with alpha below this threshold will be culled
    pub alpha_cull_threshold: f32,
    pub transparency: f32,
    pub brightness: f32,
    pub white_point: f32,
    pub black_point: f32,
    pub albedo_color: Vec3,
    /// Visualization mode (default: Splat)
    /// Controls how splats are rendered (normal, depth, rings, etc.)
    pub vis_mode: SplatVisMode,
    /// Show selection overlay (blue center points) on top of splat rendering
    /// When true, renders both splats and centers in two passes
    pub show_selection_overlay: bool,
    /// Visualization mode for selection overlay (Centers or Rings)
    /// Only used when show_selection_overlay is true
    pub overlay_vis_mode: Option<SplatVisMode>,
    /// Enable outline rendering for selected splats (default: false)
    /// When true, renders selected splats to outline texture for edge detection
    pub show_outline: bool,
    /// DEPRECATED: This field is no longer used for controlling shader color space conversion.
    /// Color space conversion is now automatically determined by the render target format (HDR vs LDR).
    /// - HDR targets (Rgba16Float): always convert sRGB → linear
    /// - LDR targets (Rgba8UnormSrgb): keep sRGB (GPU handles conversion)
    /// This field is kept for backward compatibility but has no effect on rendering.
    pub use_tonemapping: bool,
}

/// Convert sRGB color component (0-1 range) to linear space
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert Vec4 color from sRGB (0-1 range) to linear space (0-1 range)
/// W component (alpha or other data) is not gamma corrected
fn srgb_vec4_to_linear(v: Vec4) -> Vec4 {
    Vec4::new(
        srgb_to_linear(v.x),
        srgb_to_linear(v.y),
        srgb_to_linear(v.z),
        v.w, // Alpha or point size is not gamma corrected
    )
}

#[derive(Component, Clone, Copy, Debug)]
pub struct SplatEditingColorConfig {
    /// Select color (RGB in 0-1 normalized range, W = edit point size in pixels)
    pub select_color: Vec4,
    /// Unselect color for overlay (RGB in 0-1 normalized range, W unused)
    pub unselect_color: Vec4,
    /// Locked color (RGB in 0-1 normalized range, W = intensity)
    pub locked_color: Vec4,
}

impl Default for SplatEditingColorConfig {
    fn default() -> Self {
        Self {
            // Default select color: bright yellow in 0-1 normalized range
            // W component is edit point size in pixels (not normalized)
            select_color: Vec4::new(1.0, 1.0, 0.0, 3.0),
            // Default unselect color: deep blue in 0-1 normalized range
            unselect_color: Vec4::new(0.0, 0.0, 1.0, 0.0),
            // Default locked color: dark gray in 0-1 normalized range
            // W component is intensity (0.5 = 50%)
            locked_color: Vec4::new(0.3, 0.3, 0.3, 0.5),
        }
    }
}

impl Default for RenderingConfig {
    fn default() -> Self {
        Self {
            antialias: false,  // Default OFF: only enable when model is trained with MipSplatting (antialiased flag)
            sh_band: 3,
            splat_scale: 1.0,
            point_size: 1.0,
            frustum_dilation: 0.2,
            alpha_cull_threshold: 0.005,
            transparency: 1.0,
            brightness: 0.0,
            white_point: 1.0,
            black_point: 0.0,
            albedo_color: Vec3::new(1.0, 1.0, 1.0),
            vis_mode: SplatVisMode::Splat,
            show_selection_overlay: false,
            overlay_vis_mode: None, // Defaults to Centers when show_selection_overlay is true
            show_outline: false,
            use_tonemapping: true, // DEPRECATED: no longer used, conversion is automatic based on target format
        }
    }
}

impl ExtractComponent for GaussianSplats {
    type QueryData = &'static Self;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(item: &Self) -> Option<Self> {
        Some(item.clone())
    }
}

impl ExtractComponent for SplatEditingColorConfig {
    type QueryData = &'static Self;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(item: &Self) -> Option<Self> {
        Some(*item)
    }
}


/// System to update temporal coherence cache based on camera movement and data updates
fn update_temporal_coherence_cache(
    mut cache: ResMut<TemporalCoherenceCache>,
    mut stats: ResMut<TemporalCoherenceStats>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    config_query: Query<&TemporalCoherenceConfig>,
    // Detect if any splat data was updated this frame
    entities_with_update: Query<(), With<BuffersNeedUpdate>>,
    // Detect if any splat transform changed this frame
    entities_with_changed_transform: Query<(), (With<GaussianSplats>, Changed<GlobalTransform>)>,
) {
    // Check if any data was updated this frame OR any transform changed
    // Transform changes require re-sorting because splat order depends on view direction
    let data_updated = !entities_with_update.is_empty();
    let transform_changed = !entities_with_changed_transform.is_empty();
    cache.data_updated_this_frame = data_updated || transform_changed;
    
    // If data was updated or transform changed, reset render skip count
    if cache.data_updated_this_frame {
        cache.render_skip_count = 0;
    }
    
    // Get first camera
    let Some(camera_transform) = cameras.iter().next() else {
        return;
    };
    
    // Get config (use default if not found)
    let config = config_query.iter().next().copied().unwrap_or_default();
    
    // Extract camera info
    let view_matrix = camera_transform.to_matrix();
    let camera_pos = view_matrix.w_axis.truncate();
    let camera_dir = -view_matrix.z_axis.truncate().normalize();
    let camera_up = view_matrix.y_axis.truncate().normalize();
    
    // Update cache
    let skip_sorting = should_skip_sorting(
        &mut cache,
        &config,
        camera_pos,
        camera_dir,
        camera_up,
    );
    
    // Update stats
    stats.update(skip_sorting, cache.skip_count);
}

/// Gaussian Point Cloud rendering plugin
pub struct GaussianPointCloudPlugin;

impl Plugin for GaussianPointCloudPlugin {
    fn build(&self, app: &mut App) {
        // Register embedded shaders before anything else
        // This must be done in the main app, not render app
        crate::EmbeddedShadersPlugin.build(app);
        
        // Initialize GPU picker resources in main app
        app.init_resource::<PickRequest>();
        app.init_resource::<PickResult>();
        app.init_resource::<PickPendingReadback>();
        
        // Add system to poll pick results and apply to selection
        app.add_systems(Update, (poll_pick_results, apply_pick_results).chain());
        
        // Initialize temporal coherence resources (main world)
        app.init_resource::<TemporalCoherenceCache>();
        app.init_resource::<TemporalCoherenceStats>();
        
        // Add system to update temporal coherence cache
        app.add_systems(PostUpdate, update_temporal_coherence_cache);
        
        app.add_plugins((
            // ExtractComponentPlugin::<GaussianSplats>::default(), 
            RadixSortPlugin,  // Add radix sort plugin
            // Extract temporal coherence resources to render world
            ExtractResourcePlugin::<TemporalCoherenceCache>::default(),
            ExtractResourcePlugin::<TemporalCoherenceStats>::default(),
            // Note: GaussianSplatRenderCache is initialized directly in render world (not extracted)
            // Note: BuffersNeedUpdate are handled manually in extract_gaussian_splats
            // because render world uses different entity IDs than main world
        ));
        
        // Selection state synchronization uses hash comparison in Extract
        // No need for dirty flag clearing system

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<GaussianSplatPipeline>>()
            .init_resource::<SpecializedComputePipelines<GaussianSplatCullPipeline>>()
            // No need for init_resource, use Option<Res<T>> for automatic detection
            // Extract systems must be added to ExtractSchedule to properly access MainWorld
            .add_systems(
                ExtractSchedule,
                (extract_gaussian_splats, extract_pick_request),
            )
            .add_systems(
                Render,
                (prepare_gaussian_splat_pipelines, prepare_gaussian_splat_cull_pipelines, prepare_pick_render_target)
                    .in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                (prepare_render_cache, prepare_blit_pipeline)
                    .in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                (prepare_gaussian_splat_buffers)
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Render,
                (update_gaussian_splat_buffer_contents)
                    .in_set(RenderSystems::PrepareResources)
                    .after(prepare_gaussian_splat_buffers),
            )
            .add_systems(
                Render,
                (update_gaussian_uniforms, upload_selection_state_to_gpu)
                    .in_set(RenderSystems::PrepareResources)
                    .after(update_gaussian_splat_buffer_contents),
            )
            .add_systems(
                Render,
                (prepare_gaussian_splat_bind_groups, prepare_gaussian_splat_cull_bind_groups, prepare_radix_sort_bind_groups)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                execute_pick_readback.in_set(RenderSystems::Cleanup),
            )
            .add_render_graph_node::<ViewNodeRunner<GaussianSplatNode>>(Core3d, GaussianSplatLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::EndMainPass, GaussianSplatLabel, Node3d::StartMainPassPostProcessing),
                // (Node3d::StartMainPass, GaussianSplatLabel, Node3d::MainOpaquePass),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<GaussianSplatPipeline>()
            .init_resource::<GaussianSplatCullPipeline>()
            .init_resource::<GaussianSplatRenderCache>()
            .init_resource::<CacheBlitPipeline>();
            // RadixSortPipelines is initialized by RadixSortPlugin
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct GaussianSplatLabel;

/// Manual extraction of GaussianSplats (Performance optimization: only extract NEW entities)
/// 
/// **Key Performance Optimization**:
/// - ExtractComponentPlugin clones entire GaussianSplats every frame (~23MB for 100K points)
/// - This causes 4 FPS performance bottleneck
/// - Manual extraction only extracts GaussianSplats data once per entity
/// - But configuration components (small data) are extracted every frame to support live editing
/// Track last extracted selection state per entity (hash of states)
#[derive(Default)]
struct LastExtractedSelection(std::collections::HashMap<Entity, u64>);

fn extract_gaussian_splats(
    mut commands: Commands,
    mut extracted_entities: Local<std::collections::HashMap<Entity, Entity>>,
    mut last_selection: Local<LastExtractedSelection>,
    // Query all entities with GaussianSplats (both PLY and training entities use the same path now)
    // Also check for BuffersNeedUpdate to trigger data re-extraction for training
    // Use InheritedVisibility which correctly handles parent-child visibility relationships
    main_world_splats: Extract<Query<(Entity, &GaussianSplats, &GlobalTransform, Option<&RenderingConfig>, Option<&PackModeConfig>, Option<&SplatEditingColorConfig>, Option<&InheritedVisibility>, Option<&SplatSelectionState>, Option<&BuffersNeedUpdate>)>>,
) {
    // Collect all current main world entities (only visible ones for rendering)
    let current_entities: std::collections::HashSet<Entity> = main_world_splats
        .iter()
        .filter(|(_, _, _, _, _, _, inherited_visibility, _, _)| {
            // Check InheritedVisibility which respects parent-child relationships
            // InheritedVisibility is updated by Bevy's visibility propagation system
            inherited_visibility.map(|v| v.get()).unwrap_or(true)
        })
        .map(|(entity, _, _, _, _, _, _, _, _)| entity)
        .collect();
    
    // Clean up render entities for main entities that no longer exist
    let mut entities_to_remove = Vec::new();
    for (&main_entity, &render_entity) in extracted_entities.iter() {
        if !current_entities.contains(&main_entity) {
            // Main world entity was deleted - despawn render world entity
            commands.entity(render_entity).despawn();
            entities_to_remove.push(main_entity);
            println!("🗑️ Cleaning up deleted GaussianSplats entity: {:?}", main_entity);
        }
    }
    
    // Remove deleted entities from the mapping
    for entity in entities_to_remove {
        extracted_entities.remove(&entity);
    }
    
    // Performance critical: Only extract GaussianSplats data once, but update configs every frame
    // EXCEPTION: When BuffersNeedUpdate is present, re-extract GaussianSplats data (for training)
    for (main_entity, splats, global_transform, rendering_config, pack_mode_config, splat_editing_color_config, inherited_visibility, selection_state, needs_update) in main_world_splats.iter() {
        // Skip entities that are hidden (check InheritedVisibility which respects parent-child relationships)
        let is_visible = inherited_visibility.map(|v| v.get()).unwrap_or(true);
        if !is_visible {
            // If entity was previously extracted but is now hidden, it was already removed above
            continue;
        }
        
        // Check if this entity needs data update (training scenario)
        let needs_data_update = needs_update.is_some();
        
        // Check if this entity has already been extracted
        if let Some(&render_entity) = extracted_entities.get(&main_entity) {
            // Entity already exists in render world - update config components
            let mut entity_commands = commands.entity(render_entity);
            
            let pack_config = pack_mode_config.copied().unwrap_or_default();
            
            // If BuffersNeedUpdate is present, also update GaussianSplats data
            if needs_data_update {
                // Re-extract GaussianSplats data for training updates
                entity_commands.insert((
                    splats.clone(),  // Clone updated data
                    *global_transform,
                    rendering_config.copied().unwrap_or_default(),
                    pack_config,
                    BuffersNeedUpdate,  // Forward the marker to render world
                ));
            } else {
                entity_commands.insert((
                    *global_transform,  // Update transform (16 floats, ~64 bytes)
                    rendering_config.copied().unwrap_or_default(),
                    pack_config,
                ));
            }
            
            // Also insert SplatEditingColorConfig if present
            if let Some(config) = splat_editing_color_config {
                entity_commands.insert(*config);
            }
            
            // Extract selection state
            if let Some(sel_state) = selection_state {
                // Compute hash of selection state
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                sel_state.num_selected.hash(&mut hasher);
                sel_state.num_locked.hash(&mut hasher);
                sel_state.num_deleted.hash(&mut hasher);
                // Hash ALL states for accurate change detection (not just samples)
                for &s in &sel_state.states {
                    s.hash(&mut hasher);
                }
                let current_hash = hasher.finish();
                
                let last_hash = last_selection.0.get(&main_entity).copied().unwrap_or(0);
                
                // Extract if dirty flag is set OR hash changed
                if sel_state.dirty || current_hash != last_hash {
                    // Selection changed, extract it
                    let states_u32: Vec<u32> = sel_state.states.iter().map(|&s| s as u32).collect();
                    entity_commands.insert(ExtractedSelectionState {
                        states: states_u32,
                        dirty: true,
                    });
                    last_selection.0.insert(main_entity, current_hash);
                }
            }
        } else {
            // New entity - extract everything
            println!("🔄 Extracting GaussianSplats entity (one-time): {} points", splats.len());
            
            let pack_config = pack_mode_config.copied().unwrap_or_default();
            
            // Spawn new entity in render world with cloned GaussianSplats data
            let mut entity_commands = commands.spawn((
                splats.clone(),  // Clone only once per entity!
                *global_transform,
                rendering_config.copied().unwrap_or_default(),
                pack_config,
            ));
            
            // Also insert SplatEditingColorConfig if present
            if let Some(config) = splat_editing_color_config {
                entity_commands.insert(*config);
            }
            
            // Extract initial selection state if present
            if let Some(sel_state) = selection_state {
                let states_u32: Vec<u32> = sel_state.states.iter().map(|&s| s as u32).collect();
                entity_commands.insert(ExtractedSelectionState {
                    states: states_u32,
                    dirty: true,
                });
            }
            
            let render_entity = entity_commands.id();
            
            // Map main world entity to render world entity
            extracted_entities.insert(main_entity, render_entity);
        }
    }
}

/// Transform uniform for model matrix
/// OPTIMIZED: Only stores model_matrix (64 bytes instead of 128 bytes)
/// Direction transforms are computed from model_matrix transpose in shader
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
#[repr(C)]
struct TransformUniforms {
    pub model_matrix: Mat4,  // Local to world transform (64 bytes)
    // Removed model_matrix_inverse: Saves 64 bytes!
    // Direction vectors can be transformed using transpose of 3x3 rotation part
}

/// GPU buffer resources (per-entity Component, not global Resource)
/// Each GaussianSplats entity has its own set of GPU buffers
#[derive(Component)]
/// GPU buffer reference with optional offset (for external buffers from Burn)
#[derive(Clone)]
pub struct GpuBufferWithOffset {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: Option<u64>,  // None means entire buffer from offset
}

impl GpuBufferWithOffset {
    /// Create from a simple buffer (offset = 0, size = entire buffer)
    pub fn from_buffer(buffer: Buffer) -> Self {
        Self { buffer, offset: 0, size: None }
    }
    
    /// Get the binding resource for bind group creation
    pub fn as_binding(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: self.size.and_then(std::num::NonZeroU64::new),
        })
    }
}

#[derive(Component)]
pub struct GaussianSplatGpuBuffers {
    pub position_buffer: GpuBufferWithOffset,
    pub color_buffer: GpuBufferWithOffset,       // sh_coeffs0 / DC color
    pub scale_buffer: GpuBufferWithOffset,       // Scale data (log_scales converted to actual scales)
    pub opacity_buffer: GpuBufferWithOffset,     // Opacity data
    pub rotation_buffer: GpuBufferWithOffset,    // Rotation quaternions
    pub sh_buffer: GpuBufferWithOffset,          // SH coefficients (float32, 45 floats per splat)
    pub uniform_buffer: Buffer,
    pub transform_buffer: Buffer,    // Transform matrix (model matrix)
    pub point_count: u32,            // Current point count for this entity
    pub buffer_capacity: u32,        // Buffer capacity (pre-allocated size for training)
    // GPU sorting related buffers
    pub depth_keys: Buffer,          // Depth keys (compact, visible points only)
    pub sorted_indices: Buffer,      // Sorted indices (values for radix sort)
    pub visible_indices: Buffer,     // Compact index list of visible points (sorted)
    pub cull_params: Buffer,         // Project & Cull parameters
    pub indirect_buffer: Buffer,     // Indirect draw buffer
    pub radix_sort_buffers: RadixSortBuffers,  // Radix sort temp buffers
    
    // PACK mode buffers (optional, only created when PackModeConfig is enabled)
    pub rotation_scales_packed: Option<Buffer>,  // U32x4 packed: rotation + scale + opacity (f16)
    pub colors_packed: Option<Buffer>,           // U32x2 packed: RGB colors (f16)
    pub sh_packed: Option<Buffer>,               // Packed SH coefficients (f16 format)
    
    // Selection state buffer - one u32 per splat (for GPU atomics support)
    // Bits: 0=selected, 1=locked, 2=deleted
    pub state_buffer: Buffer,
}

// Spherical harmonics data is now stored as simple float32 arrays
// Layout: 45 floats per splat (15 coefficients × 3 channels RGB)
// [sh1_r, sh1_g, sh1_b, sh2_r, sh2_g, sh2_b, ..., sh15_r, sh15_g, sh15_b]

/// Extracted selection state data (render world component)
/// Contains the state data that needs to be written to GPU buffer
#[derive(Component)]
pub struct ExtractedSelectionState {
    /// Selection states to upload to GPU (u32 per splat)
    pub states: Vec<u32>,
    /// Whether this data needs to be uploaded
    pub dirty: bool,
}

/// Bind group (per-entity Component)
#[derive(Component)]
pub struct GaussianSplatBindGroup(pub BindGroup);

/// Pipeline ID
#[derive(Component)]
pub struct GaussianSplatPipelineId(pub CachedRenderPipelineId);

/// Overlay pipeline ID for Centers mode (VIS_CENTERS)
#[derive(Component)]
pub struct GaussianSplatOverlayCentersPipelineId(pub CachedRenderPipelineId);

/// Overlay pipeline ID for Rings mode (VIS_RINGS)
#[derive(Component)]
pub struct GaussianSplatOverlayRingsPipelineId(pub CachedRenderPipelineId);

/// Pick pipeline ID (PICK_PASS shader variant)
#[derive(Component)]
pub struct GaussianSplatPickPipelineId(pub CachedRenderPipelineId);

/// Outline pipeline ID for rendering selected splats to outline texture (MSAA 1x)
#[derive(Component)]
pub struct GaussianSplatOutlinePipelineId(pub CachedRenderPipelineId);

// ============================================================================
// GPU Picker System
// ============================================================================

/// Resource to request a pick operation
/// Set this resource to trigger a pick pass on the next frame
#[derive(Resource, Default)]
pub struct PickRequest {
    /// Whether a pick operation is requested
    pub active: bool,
    /// Screen-space rectangle to pick (in pixels)
    pub rect: Option<PickRect>,
    /// Selection operation mode
    pub op: PickOp,
    /// Entity to pick from (if None, picks from all splat entities)
    pub target_entity: Option<Entity>,
}

/// Pick rectangle in screen space (pixels)
#[derive(Clone, Copy, Debug, Default)]
pub struct PickRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Pick operation mode (matches supersplat's pickMode)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PickOp {
    /// Add to current selection
    Add,
    /// Remove from current selection
    Remove,
    /// Replace current selection
    #[default]
    Set,
}

/// Resource to store pick results
/// After a pick operation completes, results are stored here
#[derive(Resource, Default)]
pub struct PickResult {
    /// Whether results are ready
    pub ready: bool,
    /// Picked splat indices (deduplicated)
    pub splat_indices: Vec<u32>,
    /// The operation that was performed
    pub op: PickOp,
    /// Target entity for the pick
    pub target_entity: Option<Entity>,
}

/// Pick render target and readback buffers (render world resource)
#[derive(Resource)]
pub struct PickRenderTarget {
    /// Off-screen texture for pick rendering (RGBA8Unorm)
    pub texture: bevy::render::render_resource::Texture,
    pub view: bevy::render::render_resource::TextureView,
    /// Staging buffer for CPU readback
    pub staging_buffer: Buffer,
    /// Depth texture for pick pass
    pub depth_texture: bevy::render::render_resource::Texture,
    pub depth_view: bevy::render::render_resource::TextureView,
    pub width: u32,
    pub height: u32,
    /// Whether a pick operation is in progress
    pub pick_active: bool,
    /// The rect being picked
    pub pick_rect: Option<PickRect>,
    /// The operation mode
    pub pick_op: PickOp,
    /// Target entity
    pub target_entity: Option<Entity>,
}

/// Pending pick data for main world (shared between main and render worlds)
#[derive(Resource, Default)]
pub struct PickPendingReadback {
    /// Shared data for async readback
    pub data: std::sync::Arc<std::sync::Mutex<PickReadbackData>>,
}

/// Data structure for pick readback
#[derive(Default)]
pub struct PickReadbackData {
    /// Whether results are ready
    pub ready: bool,
    /// Raw pixel data (RGBA8)
    pub pixels: Vec<u8>,
    /// Pick rect info
    pub rect: Option<PickRect>,
    /// Operation mode
    pub op: PickOp,
    /// Target entity
    pub target_entity: Option<Entity>,
}

/// Culling compute pipeline ID
#[derive(Component)]
pub struct GaussianSplatCullPipelineId(pub bevy::render::render_resource::CachedComputePipelineId);

/// Project & Cull parameters
#[derive(Component, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
#[repr(C)]
struct ProjectCullParams {
    point_count: u32,
    point_size: f32,
    frustum_dilation: f32,  // Matches CullingConfig
    _padding: u32,
}

/// Resource to track last used vis_mode for change detection
#[derive(Resource, Default)]
pub struct LastVisMode(pub SplatVisMode);

/// Prepare render pipelines (PER-ENTITY)
fn prepare_gaussian_splat_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<GaussianSplatPipeline>>,
    pipeline: Res<GaussianSplatPipeline>,
    // FILTER: Only process 3D cameras, exclude 2D UI cameras
    views: Query<(Entity, &ExtractedView, &Msaa), Without<Camera2d>>,
    gaussian_splats: Query<(Entity, Option<&RenderingConfig>, Option<&PackModeConfig>), With<GaussianSplats>>,
    // Detect rendering config changes to trigger pipeline recompilation
    changed_rendering: Query<(), Changed<RenderingConfig>>,
    changed_pack_config: Query<(), Changed<PackModeConfig>>,
    // Track entities that already have pipeline IDs
    entities_with_pipeline: Query<Entity, With<GaussianSplatPipelineId>>,
    // Track last vis_mode to detect changes
    mut last_vis_mode: Local<SplatVisMode>,
) {
    // Only prepare pipeline when there are Gaussian point clouds
    if gaussian_splats.is_empty() {
        // Clean up pipeline IDs from entities when no splats exist
        for entity in &entities_with_pipeline {
            commands.entity(entity).remove::<GaussianSplatPipelineId>();
        }
        return;
    }

    // Get global rendering config (for vis_mode detection)
    let global_rendering_config = gaussian_splats.iter()
        .filter_map(|(_, config, _)| config.copied())
        .next()
        .unwrap_or_default();

    // Check if vis_mode changed (more reliable than Changed<RenderingConfig>)
    let vis_mode_changed = *last_vis_mode != global_rendering_config.vis_mode;
    if vis_mode_changed {
        debug!("🎨 Vis mode changed: {:?} -> {:?}", *last_vis_mode, global_rendering_config.vis_mode);
        *last_vis_mode = global_rendering_config.vis_mode;
    }

    // Check if we need to update all pipelines (config changed)
    let config_changed = !changed_rendering.is_empty() || !changed_pack_config.is_empty() || vis_mode_changed;
    
    // Get the first view's properties (for HDR and MSAA)
    // All views are assumed to have same HDR/MSAA settings
    let Some((_, view, msaa)) = views.iter().next() else {
        return;  // No views, nothing to render
    };
    
    // ✅ PER-ENTITY PIPELINE: Each entity gets its own pipeline based on its PackModeConfig
    for (entity, rendering_config_opt, pack_config_opt) in gaussian_splats.iter() {
        // Skip if entity already has pipeline and config hasn't changed
        let has_pipeline = entities_with_pipeline.contains(entity);
        if !config_changed && has_pipeline {
            continue;  // Skip - no changes needed
        }
        
        let rendering_config = rendering_config_opt.copied().unwrap_or_default();
        let pack_mode = pack_config_opt.map_or(false, |c| c.enabled);
        
        let key = GaussianSplatPipelineKey { 
            hdr: view.hdr,
            msaa_samples: msaa.samples(),
            enable_aa: rendering_config.antialias,
            sh_degree: rendering_config.sh_band.min(3), // Clamp to 0-3
            pack_mode, // ✅ Use THIS entity's pack_mode!
            vis_mode: rendering_config.vis_mode,
            use_tonemapping: rendering_config.use_tonemapping,
        };
        
        // Only log on initial creation, not on every config change
        if !has_pipeline {
            debug!("🔧 Entity {:?}: Specializing pipeline with pack={}, aa={}, sh={}, vis={:?}", 
                entity, key.pack_mode, key.enable_aa, key.sh_degree, key.vis_mode);
        }
        
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            key,
        );

        commands.entity(entity).insert(GaussianSplatPipelineId(pipeline_id));
        
        // Create overlay pipeline for Centers mode (VIS_CENTERS)
        let overlay_centers_key = GaussianSplatPipelineKey { 
            hdr: view.hdr,
            msaa_samples: msaa.samples(),
            enable_aa: false, // No AA for overlay points
            sh_degree: 0, // SH not needed for overlay
            pack_mode, // ✅ Use THIS entity's pack_mode!
            vis_mode: SplatVisMode::Centers,
            use_tonemapping: rendering_config.use_tonemapping,
        };
        
        let overlay_centers_pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            overlay_centers_key,
        );
        
        commands.entity(entity).insert(GaussianSplatOverlayCentersPipelineId(overlay_centers_pipeline_id));
        
        // Create overlay pipeline for Rings mode (VIS_RINGS)
        let overlay_rings_key = GaussianSplatPipelineKey {
            hdr: view.hdr,
            msaa_samples: msaa.samples(),
            enable_aa: false, // No AA for overlay
            sh_degree: 0, // SH not needed for overlay
            pack_mode, // ✅ Use THIS entity's pack_mode!
            vis_mode: SplatVisMode::Rings,
            use_tonemapping: rendering_config.use_tonemapping,
        };
        
        let overlay_rings_pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            overlay_rings_key,
        );
        
        commands.entity(entity).insert(GaussianSplatOverlayRingsPipelineId(overlay_rings_pipeline_id));
        
        // Create pick pipeline (PICK_PASS) for GPU selection
        let pick_key = GaussianSplatPipelineKey { 
            hdr: false, // Pick always uses RGBA8
            msaa_samples: 1, // No MSAA for pick
            enable_aa: false, // No AA for pick
            sh_degree: 0, // SH not needed for pick
            pack_mode, // ✅ Use THIS entity's pack_mode!
            vis_mode: SplatVisMode::Pick,
            use_tonemapping: false, // Pick pass doesn't need tonemapping
        };
        
        let pick_pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            pick_key,
        );
        
        commands.entity(entity).insert(GaussianSplatPickPipelineId(pick_pipeline_id));
        
        // Create outline pipeline for rendering selected splats to outline texture
        // Use Outline mode which adds OUTLINE_PASS shader define to filter only selected splats
        let outline_key = GaussianSplatPipelineKey { 
            hdr: view.hdr, // Match main pass HDR setting
            msaa_samples: 1, // No MSAA for outline (outline texture is non-MSAA)
            enable_aa: false, // No AA for outline
            sh_degree: 0, // SH not needed for outline
            pack_mode, // ✅ Use THIS entity's pack_mode!
            vis_mode: SplatVisMode::Outline, // Outline mode: only renders selected splats
            use_tonemapping: false, // Outline pass doesn't need tonemapping
        };
        
        let outline_pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            outline_key,
        );
        
        commands.entity(entity).insert(GaussianSplatOutlinePipelineId(outline_pipeline_id));
    }
}

/// Prepare culling compute pipeline
fn prepare_gaussian_splat_cull_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedComputePipelines<GaussianSplatCullPipeline>>,
    cull_pipeline: Res<GaussianSplatCullPipeline>,
    // FILTER: Only process 3D cameras, exclude 2D UI cameras
    views: Query<Entity, (With<ExtractedView>, Without<Camera2d>, Without<GaussianSplatCullPipelineId>)>,
    views_with_cull_pipeline: Query<Entity, With<GaussianSplatCullPipelineId>>,
    gaussian_splats: Query<Entity, With<GaussianSplats>>,
) {
    // Only prepare pipeline for views when there are Gaussian point clouds
    if gaussian_splats.is_empty() {
        // Clean up cull pipeline IDs from views when no splats exist
        for view_entity in &views_with_cull_pipeline {
            commands.entity(view_entity).remove::<GaussianSplatCullPipelineId>();
        }
        return;
    }

    // Only prepare pipeline for views without pipeline ID (avoid recreating every frame)
    for view_entity in &views {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &cull_pipeline,
            GaussianSplatCullPipelineKey {},
        );
        
        commands.entity(view_entity).insert(GaussianSplatCullPipelineId(pipeline_id));
    }
}

/// Marker component to indicate that buffer contents need to be updated
/// but not recreated (for training scenarios)
/// When this component is present, the extract system will re-extract GaussianSplats data
/// and the GPU buffers will be updated using write_buffer (fast path).
#[derive(Component, Clone)]
pub struct BuffersNeedUpdate;

impl ExtractComponent for BuffersNeedUpdate {
    type QueryData = &'static Self;
    type QueryFilter = ();
    type Out = Self;
    
    fn extract_component(_item: &Self) -> Option<Self> {
        Some(BuffersNeedUpdate)
    }
}

/// Marker component to indicate this is a training entity
/// 
/// Training entities can skip the entire render pass when camera is static,
/// using a cached render result. This saves GPU resources for training.
/// 
/// Normal PLY entities only skip the sort pass when camera is static,
/// but still render every frame (for consistent quality).
#[derive(Component, Clone, Default)]
pub struct TrainingMode;

impl ExtractComponent for TrainingMode {
    type QueryData = &'static Self;
    type QueryFilter = ();
    type Out = Self;
    
    fn extract_component(_item: &Self) -> Option<Self> {
        Some(TrainingMode)
    }
}

/// Prepare GPU buffers for each entity independently
fn prepare_gaussian_splat_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    views: Query<&ExtractedView, Without<Camera2d>>,
    // Query entities that have GaussianSplats but no GPU buffers yet
    new_entities: Query<(Entity, &GaussianSplats, &GlobalTransform, Option<&RenderingConfig>, Option<&PackModeConfig>, Option<&SplatEditingColorConfig>), Without<GaussianSplatGpuBuffers>>,
) {
    const SH_C0: f32 = 0.28209479;
    
    // Get viewport size from first view
    let Some(view) = views.iter().next() else {
        return;
    };
    let surface_width = view.viewport.z as u32;
    let surface_height = view.viewport.w as u32;
    
    // Process new entities and entities that need rebuild
    let entities_to_process = new_entities.iter();
    
    for (entity, splats, global_transform, rendering_config, pack_mode_config, splat_editing_color_config) in entities_to_process {
        if splats.is_empty() {
            continue;
        }
        
        let point_count = splats.len() as u32;
        // Use capacity for buffer allocation to support training data growth
        let buffer_capacity = splats.capacity().max(splats.len()) as u32;
        
        // Get configuration
        let config = rendering_config.copied().unwrap_or_default();
        let point_size = config.point_size;
        let frustum_dilation = config.frustum_dilation;
        let alpha_cull_threshold = config.alpha_cull_threshold;
        let splat_scale = config.splat_scale;
        let use_pack_mode = pack_mode_config.map_or(false, |c| c.enabled);
        
        debug!("🔧 Creating GPU buffers for entity {:?}: {} splats, capacity {} (PACK mode: {})", entity, point_count, buffer_capacity, use_pack_mode);
        
        // Collect positions
        let positions: Vec<PackedVec3> = splats.means.iter()
            .map(|pos| PackedVec3 { x: pos.x, y: pos.y, z: pos.z })
            .collect();
        
        // Collect colors (from DC coefficient of SH)
        // CRITICAL: Do NOT clamp colors - HDR values can exceed [0,1] range
        // Match C++ reference: color = dc * SH_C0 + 0.5 (no clamp)
        let colors: Vec<PackedVec3> = splats.sh_coeffs.iter()
            .map(|sh| {
                let dc = &sh[0];
                let r = (dc.x * SH_C0 + 0.5).max(0.0);
                let g = (dc.y * SH_C0 + 0.5).max(0.0);
                let b = (dc.z * SH_C0 + 0.5).max(0.0);
                PackedVec3 { x: r, y: g, z: b }
            })
            .collect();
        
        // Collect scales (convert from log-space)
        // Clamp exp(scale) to reasonable range to prevent rendering artifacts
        // SuperSplat clamps log_scale to [-20, 20], but we also clamp the exp result
        // to avoid extremely large splats that cause "floater" artifacts
        const MAX_SCALE: f32 = 100.0;  // Maximum scale in world units
        let scales: Vec<PackedVec3> = splats.log_scales.iter()
            .map(|log_scale| {
                let scale = Vec3::new(
                    log_scale.x.exp().min(MAX_SCALE),
                    log_scale.y.exp().min(MAX_SCALE),
                    log_scale.z.exp().min(MAX_SCALE)
                );
                PackedVec3 { x: scale.x, y: scale.y, z: scale.z }
            })
            .collect();
        //collect opacities
        let opacities: Vec<f32> = splats.raw_opacities.iter()
            .map(|&raw_opacity| sigmoid(raw_opacity))
            .collect();
        // Collect rotations
        let rotations: Vec<Vec4> = splats.rotations.clone();
        
        // Collect SH data (15 coefficients × 3 channels = 45 floats per splat)
        let mut sh_data = Vec::with_capacity(point_count as usize * 45);
        for coeffs in &splats.sh_coeffs {
            let mut all_coeffs = coeffs.clone();
            // Pad to 16 coefficients if needed
            while all_coeffs.len() < 16 {
                all_coeffs.push(Vec3::ZERO);
            }
            // Store coefficients 1-15 (skip DC coefficient at index 0)
            for coeff in &all_coeffs[1..16] {
                sh_data.push(coeff.x);
                sh_data.push(coeff.y);
                sh_data.push(coeff.z);
            }
        }
        
        // Create GPU buffers - position buffer always needed
        // Use buffer_capacity for allocation to support training data growth
        let position_buffer = {
            let mut padded_positions = positions.clone();
            padded_positions.resize(buffer_capacity as usize, PackedVec3 { x: 0.0, y: 0.0, z: 0.0 });
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("gaussian_splat_position_buffer"),
                contents: bytemuck::cast_slice(&padded_positions),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })
        };

        // In PACK mode, only create minimal dummy buffers for bind group compatibility
        // In standard mode, create full-size buffers with actual data
        let (color_buffer, scale_buffer, opacity_buffer, rotation_buffer, sh_buffer) = if use_pack_mode {
            println!("  💾 Creating minimal dummy buffers for PACK mode (saving GPU memory)");
            
            // Create minimal dummy buffers (1 element each) for bind group slots
            let dummy_color: Vec<PackedVec3> = vec![PackedVec3 { x: 0.0, y: 0.0, z: 0.0 }];
            let dummy_scale: Vec<PackedVec3> = vec![PackedVec3 { x: 1.0, y: 1.0, z: 1.0 }];
            let dummy_opacity: Vec<f32> = vec![1.0];
            let dummy_rotation: Vec<Vec4> = vec![Vec4::new(0.0, 0.0, 0.0, 1.0)];
            let dummy_sh: Vec<f32> = vec![0.0];
            
            (
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_dummy_color"),
                    contents: bytemuck::cast_slice(&dummy_color),
                    usage: BufferUsages::STORAGE,
                }),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_dummy_scale"),
                    contents: bytemuck::cast_slice(&dummy_scale),
                    usage: BufferUsages::STORAGE,
                }),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_dummy_opacity"),
                    contents: bytemuck::cast_slice(&dummy_opacity),
                    usage: BufferUsages::STORAGE,
                }),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_dummy_rotation"),
                    contents: bytemuck::cast_slice(&dummy_rotation),
                    usage: BufferUsages::STORAGE,
                }),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_dummy_sh"),
                    contents: bytemuck::cast_slice(&dummy_sh),
                    usage: BufferUsages::STORAGE,
                }),
            )
        } else {
            // Standard mode: create full-size buffers with actual data
            (
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_splat_color_buffer"),
                    contents: bytemuck::cast_slice(&colors),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                }),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_splat_scale_buffer"),
                    contents: bytemuck::cast_slice(&scales),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                }),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_splat_opacity_buffer"),
                    contents: bytemuck::cast_slice(&opacities),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                }),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_splat_rotation_buffer"),
                    contents: bytemuck::cast_slice(&rotations),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                }),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_splat_sh_buffer"),
                    contents: bytemuck::cast_slice(&sh_data),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                }),
            )
        };

        // Create PACK mode buffers if enabled
        let (rotation_scales_packed, colors_packed, sh_packed) = if use_pack_mode {
            println!("  📦 Creating PACK mode buffers (58% memory saving)");
            
            // 1. Pack rotation + scale + opacity into vec4<u32> array
            // Use buffer_capacity for allocation to support training data growth
            let mut rotation_scales_data = Vec::with_capacity(buffer_capacity as usize * 4);
            for i in 0..point_count as usize {
                let rot = &rotations[i];
                let scale = &scales[i];
                let opacity = opacities[i];
                
                // Pack rotation (4 f32 -> 2 u32, each u32 contains 2 f16)
                let rot_xy = pack_half2(rot.x, rot.y);
                let rot_zw = pack_half2(rot.z, rot.w);
                
                // Pack scale + opacity (4 f32 -> 2 u32, each u32 contains 2 f16)
                let scale_xy = pack_half2(scale.x, scale.y);
                let scale_z_opacity = pack_half2(scale.z, opacity);
                
                rotation_scales_data.push(rot_xy);
                rotation_scales_data.push(rot_zw);
                rotation_scales_data.push(scale_xy);
                rotation_scales_data.push(scale_z_opacity);
            }
            
            // Pad to buffer_capacity for training growth
            rotation_scales_data.resize(buffer_capacity as usize * 4, 0u32);
            let rotation_scales_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("gaussian_rotation_scales_packed"),
                contents: bytemuck::cast_slice(&rotation_scales_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            
            // 2. Pack colors into vec2<u32> array (RGB f32 -> 2 u32, f16 format)
            let mut colors_packed_data = Vec::with_capacity(buffer_capacity as usize * 2);
            for color in &colors {
                let rg = pack_half2(color.x, color.y);
                let b_pad = pack_half2(color.z, 0.0);
                colors_packed_data.push(rg);
                colors_packed_data.push(b_pad);
            }
            // Pad to buffer_capacity for training growth
            colors_packed_data.resize(buffer_capacity as usize * 2, 0u32);
            
            let colors_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("gaussian_colors_packed"),
                contents: bytemuck::cast_slice(&colors_packed_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            
            // 3. Pack SH coefficients into PackedVertexSH format (matching C++ and shader)
            // struct PackedVertexSH {
            //     sh1to3: vec4<u32>,   // x=scale, yzw=sh[0,1,2] (packed 11_10_11)
            //     sh4to7: vec4<u32>,   // xyzw=sh[3,4,5,6] (packed 11_10_11)
            //     sh8to11: vec4<u32>,  // xyzw=sh[7,8,9,10] (packed 11_10_11)
            //     sh12to15: vec4<u32>, // xyzw=sh[11,12,13,14] (packed 11_10_11)
            // }
            
            // Helper function to pack a Vec3 into 11_10_11 format
            // Matching C++ pack_unit_direction_11_10_11 EXACTLY:
            //   x = bits 0-10  (low 11 bits)
            //   y = bits 11-20 (mid 10 bits)
            //   z = bits 21-31 (high 11 bits)
            fn pack_11_10_11(v: Vec3) -> u32 {
                // Convert from [-1, 1] to [0, 1] range
                let x = ((v.x * 0.5 + 0.5).clamp(0.0, 1.0) * 2047.0) as u32;
                let y = ((v.y * 0.5 + 0.5).clamp(0.0, 1.0) * 1023.0) as u32;
                let z = ((v.z * 0.5 + 0.5).clamp(0.0, 1.0) * 2047.0) as u32;
                // Pack: x in bits 0-10, y in bits 11-20, z in bits 21-31
                (x & 0x7FF) | ((y & 0x3FF) << 11) | ((z & 0x7FF) << 21)
            }
            
            let mut sh_packed_structs: Vec<[u32; 16]> = Vec::with_capacity(buffer_capacity as usize);
            
            for splat_idx in 0..point_count as usize {
                let base = splat_idx * 45; // 15 coefficients × 3 channels
                
                // Get all 15 SH coefficients as Vec3
                let mut sh_coeffs_arr = [Vec3::ZERO; 15];
                for coeff_idx in 0..15 {
                    let offset = base + coeff_idx * 3;
                    if offset + 2 < sh_data.len() {
                        sh_coeffs_arr[coeff_idx] = Vec3::new(
                            sh_data[offset],
                            sh_data[offset + 1],
                            sh_data[offset + 2],
                        );
                    }
                }
                
                // Find max absolute value for scale (normalize to [-1, 1])
                let mut max_abs = 0.0f32;
                for coeff in &sh_coeffs_arr {
                    max_abs = max_abs.max(coeff.x.abs()).max(coeff.y.abs()).max(coeff.z.abs());
                }
                let sh_scale = if max_abs > 0.0 { max_abs } else { 1.0 };
                let inv_scale = 1.0 / sh_scale;
                
                // Normalize coefficients
                let normalized: Vec<Vec3> = sh_coeffs_arr.iter()
                    .map(|c| *c * inv_scale)
                    .collect();
                
                // Pack into PackedVertexSH structure (4 vec4<u32> = 16 u32)
                let mut packed = [0u32; 16];
                
                // sh1to3: x=scale (as f32 bits), yzw=sh[0,1,2]
                packed[0] = sh_scale.to_bits();
                packed[1] = pack_11_10_11(normalized[0]);
                packed[2] = pack_11_10_11(normalized[1]);
                packed[3] = pack_11_10_11(normalized[2]);
                
                // sh4to7: xyzw=sh[3,4,5,6]
                packed[4] = pack_11_10_11(normalized[3]);
                packed[5] = pack_11_10_11(normalized[4]);
                packed[6] = pack_11_10_11(normalized[5]);
                packed[7] = pack_11_10_11(normalized[6]);
                
                // sh8to11: xyzw=sh[7,8,9,10]
                packed[8] = pack_11_10_11(normalized[7]);
                packed[9] = pack_11_10_11(normalized[8]);
                packed[10] = pack_11_10_11(normalized[9]);
                packed[11] = pack_11_10_11(normalized[10]);
                
                // sh12to15: xyzw=sh[11,12,13,14]
                packed[12] = pack_11_10_11(normalized[11]);
                packed[13] = pack_11_10_11(normalized[12]);
                packed[14] = pack_11_10_11(normalized[13]);
                packed[15] = pack_11_10_11(normalized[14]);
                
                sh_packed_structs.push(packed);
            }
            // Pad to buffer_capacity for training growth
            sh_packed_structs.resize(buffer_capacity as usize, [0u32; 16]);
            
            let sh_buffer_packed = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("gaussian_sh_packed"),
                contents: bytemuck::cast_slice(&sh_packed_structs),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            
            (Some(rotation_scales_buffer), Some(colors_buffer), Some(sh_buffer_packed))
        } else {
            (None, None, None)
        };

        // Create uniform buffer
        let sh_degree = splats.sh_degree(); // Get SH degree from splat data
        let offset = -config.black_point + config.brightness;
        let scale = 1.0 as f32 / (config.white_point - config.black_point);
        let transparency = config.transparency;
        // Get editing color config from entity or use default
        let default_editing_colors = SplatEditingColorConfig::default();
        let editing_colors = splat_editing_color_config.copied().unwrap_or(default_editing_colors);
        
        // Colors are already in 0-1 normalized range, use directly
        let select_color =   srgb_vec4_to_linear(editing_colors.select_color);
        let unselect_color = srgb_vec4_to_linear(editing_colors.unselect_color);
        let locked_color = srgb_vec4_to_linear(editing_colors.locked_color);
        let tint_color = Vec4::new(config.albedo_color.x * scale, config.albedo_color.y * scale, config.albedo_color.z * scale, transparency);
        let color_offset = Vec4::new(offset, offset, offset,1.0);
                
        // let model_ref = global_transform.to_matrix();
        let uniforms = GaussianSplatParams {
            point_size,
            surface_width,
            surface_height,
            point_count,
            frustum_dilation,
            alpha_cull_threshold,
            splat_scale,
            sh_degree,
            select_color,
            unselect_color,
            locked_color,
            tint_color,
            color_offset,
        };
        
        let uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_splat_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        // Create transform uniform buffer
        let model_matrix = global_transform.to_matrix();
        let transforms = TransformUniforms { model_matrix };
        let transform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_splat_transforms"),
            contents: bytemuck::bytes_of(&transforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Create sorting-related buffers (use buffer_capacity for training growth)
        let max_points = buffer_capacity as usize;
        
        let depth_keys_data = vec![0u32; max_points];
        let depth_keys = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_depth_keys"),
            contents: bytemuck::cast_slice(&depth_keys_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        let sorted_indices_data = vec![0u32; max_points];
        let sorted_indices = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_sorted_indices"),
            contents: bytemuck::cast_slice(&sorted_indices_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let visible_indices_data: Vec<u32> = (0..max_points as u32).collect();
        let visible_indices = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_visible_indices"),
            contents: bytemuck::cast_slice(&visible_indices_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Indirect draw buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct DrawIndirectCommand {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        }
        let indirect_cmd = DrawIndirectCommand {
            vertex_count: 4,
            instance_count: point_count,
            first_vertex: 0,
            first_instance: 0,
        };
        let indirect_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_indirect_buffer"),
            contents: bytemuck::bytes_of(&indirect_cmd),
            usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });

        // Cull params buffer
        let cull_params_data = ProjectCullParams {
            point_count,
            point_size,
            frustum_dilation,
            _padding: 0,
        };
        let cull_params = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_cull_params"),
            contents: bytemuck::bytes_of(&cull_params_data),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        // Create radix sort buffers
        let radix_sort_buffers = create_radix_sort_buffers(&render_device, max_points);
        
        // Create state buffer - one u32 per splat, initialized to 0 (normal state)
        // Using u32 instead of u8 for better GPU alignment and atomics support
        // Use buffer_capacity for training growth
        let state_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_splat_state_buffer"),
            contents: bytemuck::cast_slice(&vec![0u32; buffer_capacity as usize]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        // Insert GaussianSplatGpuBuffers as component to this entity
        // For PLY-loaded entities, use GpuBufferWithOffset::from_buffer (offset = 0)
        commands.entity(entity).insert(GaussianSplatGpuBuffers {
            position_buffer: GpuBufferWithOffset::from_buffer(position_buffer),
            color_buffer: GpuBufferWithOffset::from_buffer(color_buffer),
            scale_buffer: GpuBufferWithOffset::from_buffer(scale_buffer),
            opacity_buffer: GpuBufferWithOffset::from_buffer(opacity_buffer),
            rotation_buffer: GpuBufferWithOffset::from_buffer(rotation_buffer),
            sh_buffer: GpuBufferWithOffset::from_buffer(sh_buffer),
            uniform_buffer,
            transform_buffer,
            point_count,
            buffer_capacity,
            depth_keys,
            sorted_indices,
            visible_indices,
            cull_params,
            indirect_buffer,
            radix_sort_buffers,
            rotation_scales_packed,
            colors_packed,
            sh_packed,
            state_buffer,
        });
        
        println!("✅ GPU buffers created for entity {:?}", entity);
    }
}

/// Update GPU buffer contents when BuffersNeedUpdate is present (training fast path)
/// 
/// This system uses write_buffer to update existing GPU buffers without recreating them.
fn update_gaussian_splat_buffer_contents(
    mut commands: Commands,
    render_queue: Res<RenderQueue>,
    // Query entities that have BuffersNeedUpdate and existing GPU buffers
    // Use &mut to update point_count when splat count changes
    mut entities_need_update: Query<(Entity, &GaussianSplats, &mut GaussianSplatGpuBuffers, Option<&PackModeConfig>), With<BuffersNeedUpdate>>,
) {
    const SH_C0: f32 = 0.28209479;
    
    for (entity, splats, mut gpu_buffers, pack_mode_config) in entities_need_update.iter_mut() {
        if splats.is_empty() {
            // Remove marker even if empty
            commands.entity(entity).remove::<BuffersNeedUpdate>();
            continue;
        }
        
        let point_count = splats.len();
        let use_pack_mode = pack_mode_config.map_or(false, |c| c.enabled);
        
        // CRITICAL: Check if point_count exceeds buffer_capacity
        // If so, we need to recreate buffers with larger capacity
        // Remove GaussianSplatGpuBuffers to trigger prepare_gaussian_splat_buffers to recreate them
        if point_count as u32 > gpu_buffers.buffer_capacity {
            println!("⚠️ Training buffer capacity exceeded: {} > {}, recreating buffers for entity {:?}", 
                     point_count, gpu_buffers.buffer_capacity, entity);
            // Remove old GPU buffers - prepare system will create new ones with updated capacity
            commands.entity(entity).remove::<GaussianSplatGpuBuffers>();
            // Keep BuffersNeedUpdate marker so next frame will process this entity
            // Actually, prepare will handle it since GaussianSplatGpuBuffers is removed
            commands.entity(entity).remove::<BuffersNeedUpdate>();
            continue;
        }
        
        // Update the point_count in GaussianSplatGpuBuffers and related GPU buffers
        // CRITICAL: Also update cull_params and indirect_buffer to reflect new point count!
        if gpu_buffers.point_count != point_count as u32 {
            gpu_buffers.point_count = point_count as u32;
            
            // Update cull_params buffer with new point_count
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct CullParamsUpdate {
                point_count: u32,
                point_size: f32,
                frustum_dilation: f32,
                _padding: u32,
            }
            let cull_params_data = CullParamsUpdate {
                point_count: point_count as u32,
                point_size: 1.0, // Default value, will be overwritten in update_gaussian_uniforms
                frustum_dilation: 0.2, // Default value
                _padding: 0,
            };
            render_queue.write_buffer(
                &gpu_buffers.cull_params,
                0,
                bytemuck::bytes_of(&cull_params_data),
            );
            
            // Update indirect_buffer with new instance_count
            // DrawIndirectCommand: vertex_count (4), instance_count, first_vertex (0), first_instance (0)
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct DrawIndirectCommand {
                vertex_count: u32,
                instance_count: u32,
                first_vertex: u32,
                first_instance: u32,
            }
            let indirect_cmd = DrawIndirectCommand {
                vertex_count: 4,
                instance_count: point_count as u32,
                first_vertex: 0,
                first_instance: 0,
            };
            render_queue.write_buffer(
                &gpu_buffers.indirect_buffer,
                0,
                bytemuck::bytes_of(&indirect_cmd),
            );
        }
        
        // Helper function for SH packing
        #[inline(always)]
        fn pack_11_10_11(v: Vec3) -> u32 {
            let x = ((v.x * 0.5 + 0.5).clamp(0.0, 1.0) * 2047.0) as u32;
            let y = ((v.y * 0.5 + 0.5).clamp(0.0, 1.0) * 1023.0) as u32;
            let z = ((v.z * 0.5 + 0.5).clamp(0.0, 1.0) * 2047.0) as u32;
            (x & 0x7FF) | ((y & 0x3FF) << 11) | ((z & 0x7FF) << 21)
        }
        
        // Helper: pack SH coefficients (16 u32 output)
        #[inline(always)]
        fn pack_sh_coeffs_inline(sh_coeffs: &[Vec3]) -> [u32; 16] {
            let mut all_coeffs: [Vec3; 16] = [Vec3::ZERO; 16];
            let copy_len = sh_coeffs.len().min(16);
            all_coeffs[..copy_len].copy_from_slice(&sh_coeffs[..copy_len]);
            
            // Find max absolute value for normalization (coefficients 1..16)
            let mut max_abs = 0.0f32;
            for coeff in &all_coeffs[1..16] {
                max_abs = max_abs.max(coeff.x.abs()).max(coeff.y.abs()).max(coeff.z.abs());
            }
            let sh_scale = if max_abs > 0.0 { max_abs } else { 1.0 };
            let inv_scale = 1.0 / sh_scale;
            
            let mut sh_packed = [0u32; 16];
            sh_packed[0] = sh_scale.to_bits();
            for (j, coeff) in all_coeffs[1..16].iter().enumerate() {
                sh_packed[j + 1] = pack_11_10_11(*coeff * inv_scale);
            }
            sh_packed
        }
        
        // =============================================================================
        // HIGH-PERFORMANCE SINGLE-PASS PARALLEL UPDATE
        // =============================================================================
        // For million-scale gaussians, we use:
        // 1. Pre-allocated arrays (avoid dynamic allocation during iteration)
        // 2. Single parallel pass (one traversal of source data)
        // 3. Direct index-based writes (each thread writes to unique indices)
        // 4. Cache-friendly access patterns
        
        if use_pack_mode {
            // Pack mode: Single parallel pass for all attributes
            #[cfg(feature = "rayon")]
            let (positions, rotation_scales, colors_packed, sh_packed) = {
                // Pre-allocate all buffers
                let positions = vec![PackedVec3 { x: 0.0, y: 0.0, z: 0.0 }; point_count];
                let rotation_scales = vec![[0u32; 4]; point_count];
                let colors_packed = vec![[0u32; 2]; point_count];
                let sh_packed = vec![[0u32; 16]; point_count];
                
                // Use par_chunks for better cache locality on large datasets
                // Each chunk processes a contiguous range of splats
                const CHUNK_SIZE: usize = 4096; // Tuned for L2 cache
                
                let chunks: Vec<_> = (0..point_count).step_by(CHUNK_SIZE)
                    .map(|start| {
                        let end = (start + CHUNK_SIZE).min(point_count);
                        (start, end)
                    })
                    .collect();
                
                chunks.into_par_iter().for_each(|(start, end)| {
                    for i in start..end {
                        // Position
                        let pos = &splats.means[i];
                        
                        // Rotation + Scale + Opacity
                        let log_scale = &splats.log_scales[i];
                        let scale_x = log_scale.x.exp();
                        let scale_y = log_scale.y.exp();
                        let scale_z = log_scale.z.exp();
                        let opacity = sigmoid(splats.raw_opacities[i]);
                        let rotation = splats.rotations[i];
                        
                        // Color from SH DC
                        let sh_coeffs = &splats.sh_coeffs[i];
                        let dc = &sh_coeffs[0];
                        let r = dc.x * SH_C0 + 0.5;
                        let g = dc.y * SH_C0 + 0.5;
                        let b = dc.z * SH_C0 + 0.5;
                        
                        // Write to pre-allocated arrays (safe: each index is unique)
                        // SAFETY: Each iteration writes to a unique index, no data races
                        unsafe {
                            let positions_ptr = positions.as_ptr() as *mut PackedVec3;
                            let rotation_scales_ptr = rotation_scales.as_ptr() as *mut [u32; 4];
                            let colors_ptr = colors_packed.as_ptr() as *mut [u32; 2];
                            let sh_ptr = sh_packed.as_ptr() as *mut [u32; 16];
                            
                            *positions_ptr.add(i) = PackedVec3 { x: pos.x, y: pos.y, z: pos.z };
                            *rotation_scales_ptr.add(i) = [
                                pack_half2(rotation.x, rotation.y),
                                pack_half2(rotation.z, rotation.w),
                                pack_half2(scale_x, scale_y),
                                pack_half2(scale_z, opacity),
                            ];
                            *colors_ptr.add(i) = [pack_half2(r, g), pack_half2(b, 0.0)];
                            *sh_ptr.add(i) = pack_sh_coeffs_inline(sh_coeffs);
                        }
                    }
                });
                
                (positions, rotation_scales, colors_packed, sh_packed)
            };
            
            #[cfg(not(feature = "rayon"))]
            let (positions, rotation_scales, colors_packed, sh_packed) = {
                let mut positions = vec![PackedVec3 { x: 0.0, y: 0.0, z: 0.0 }; point_count];
                let mut rotation_scales = vec![[0u32; 4]; point_count];
                let mut colors_packed = vec![[0u32; 2]; point_count];
                let mut sh_packed = vec![[0u32; 16]; point_count];
                
                for i in 0..point_count {
                    let pos = &splats.means[i];
                    positions[i] = PackedVec3 { x: pos.x, y: pos.y, z: pos.z };
                    
                    let log_scale = &splats.log_scales[i];
                    let scale_x = log_scale.x.exp();
                    let scale_y = log_scale.y.exp();
                    let scale_z = log_scale.z.exp();
                    let opacity = sigmoid(splats.raw_opacities[i]);
                    let rotation = splats.rotations[i];
                    
                    rotation_scales[i] = [
                        pack_half2(rotation.x, rotation.y),
                        pack_half2(rotation.z, rotation.w),
                        pack_half2(scale_x, scale_y),
                        pack_half2(scale_z, opacity),
                    ];
                    
                    let sh_coeffs = &splats.sh_coeffs[i];
                    let dc = &sh_coeffs[0];
                    let r = (dc.x * SH_C0 + 0.5).max(0.0);
                    let g = (dc.y * SH_C0 + 0.5).max(0.0);
                    let b = (dc.z * SH_C0 + 0.5).max(0.0);
                    colors_packed[i] = [pack_half2(r, g), pack_half2(b, 0.0)];
                    
                    sh_packed[i] = pack_sh_coeffs_inline(sh_coeffs);
                }
                
                (positions, rotation_scales, colors_packed, sh_packed)
            };
            
            // Write all buffers to GPU
            render_queue.write_buffer(
                &gpu_buffers.position_buffer.buffer,
                gpu_buffers.position_buffer.offset,
                bytemuck::cast_slice(&positions),
            );
            
            if let Some(ref packed_buffer) = gpu_buffers.rotation_scales_packed {
                render_queue.write_buffer(packed_buffer, 0, bytemuck::cast_slice(&rotation_scales));
            }
            
            if let Some(ref packed_buffer) = gpu_buffers.colors_packed {
                render_queue.write_buffer(packed_buffer, 0, bytemuck::cast_slice(&colors_packed));
            }
            
            if let Some(ref packed_buffer) = gpu_buffers.sh_packed {
                render_queue.write_buffer(packed_buffer, 0, bytemuck::cast_slice(&sh_packed));
            }
        } else {
            // Standard mode: Single parallel pass for all attributes
            #[cfg(feature = "rayon")]
            let (positions, colors, scales, opacities, rotations, sh_data) = {
                let positions = vec![PackedVec3 { x: 0.0, y: 0.0, z: 0.0 }; point_count];
                let colors = vec![PackedVec3 { x: 0.0, y: 0.0, z: 0.0 }; point_count];
                let scales = vec![PackedVec3 { x: 0.0, y: 0.0, z: 0.0 }; point_count];
                let opacities = vec![0.0f32; point_count];
                let rotations = vec![Vec4::ZERO; point_count];
                let sh_data = vec![0.0f32; point_count * 45];
                
                const CHUNK_SIZE: usize = 4096;
                
                let chunks: Vec<_> = (0..point_count).step_by(CHUNK_SIZE)
                    .map(|start| {
                        let end = (start + CHUNK_SIZE).min(point_count);
                        (start, end)
                    })
                    .collect();
                
                chunks.into_par_iter().for_each(|(start, end)| {
                    for i in start..end {
                        let pos = &splats.means[i];
                        let log_scale = &splats.log_scales[i];
                        let sh_coeffs = &splats.sh_coeffs[i];
                        let dc = &sh_coeffs[0];
                        
                        // SAFETY: Each iteration writes to unique indices
                        unsafe {
                            let pos_ptr = positions.as_ptr() as *mut PackedVec3;
                            let col_ptr = colors.as_ptr() as *mut PackedVec3;
                            let scale_ptr = scales.as_ptr() as *mut PackedVec3;
                            let opac_ptr = opacities.as_ptr() as *mut f32;
                            let rot_ptr = rotations.as_ptr() as *mut Vec4;
                            let sh_ptr = sh_data.as_ptr() as *mut f32;
                            
                            *pos_ptr.add(i) = PackedVec3 { x: pos.x, y: pos.y, z: pos.z };
                            *col_ptr.add(i) = PackedVec3 {
                                x: (dc.x * SH_C0 + 0.5).max(0.0),
                                y: (dc.y * SH_C0 + 0.5).max(0.0),
                                z: (dc.z * SH_C0 + 0.5).max(0.0),
                            };
                            *scale_ptr.add(i) = PackedVec3 {
                                x: log_scale.x.exp(),
                                y: log_scale.y.exp(),
                                z: log_scale.z.exp(),
                            };
                            *opac_ptr.add(i) = sigmoid(splats.raw_opacities[i]);
                            *rot_ptr.add(i) = splats.rotations[i];
                            
                            // SH data: 15 coefficients × 3 channels
                            let sh_base = i * 45;
                            for j in 1..16 {
                                let coeff = if j < sh_coeffs.len() { sh_coeffs[j] } else { Vec3::ZERO };
                                let idx = sh_base + (j - 1) * 3;
                                *sh_ptr.add(idx) = coeff.x;
                                *sh_ptr.add(idx + 1) = coeff.y;
                                *sh_ptr.add(idx + 2) = coeff.z;
                            }
                        }
                    }
                });
                
                (positions, colors, scales, opacities, rotations, sh_data)
            };
            
            #[cfg(not(feature = "rayon"))]
            let (positions, colors, scales, opacities, rotations, sh_data) = {
                let mut positions = vec![PackedVec3 { x: 0.0, y: 0.0, z: 0.0 }; point_count];
                let mut colors = vec![PackedVec3 { x: 0.0, y: 0.0, z: 0.0 }; point_count];
                let mut scales = vec![PackedVec3 { x: 0.0, y: 0.0, z: 0.0 }; point_count];
                let mut opacities = vec![0.0f32; point_count];
                let mut rotations = vec![Vec4::ZERO; point_count];
                let mut sh_data = vec![0.0f32; point_count * 45];
                
                for i in 0..point_count {
                    let pos = &splats.means[i];
                    positions[i] = PackedVec3 { x: pos.x, y: pos.y, z: pos.z };
                    
                    let sh_coeffs = &splats.sh_coeffs[i];
                    let dc = &sh_coeffs[0];
                    colors[i] = PackedVec3 {
                        x: dc.x * SH_C0 + 0.5,
                        y: dc.y * SH_C0 + 0.5,
                        z: dc.z * SH_C0 + 0.5,
                    };
                    
                    let log_scale = &splats.log_scales[i];
                    scales[i] = PackedVec3 {
                        x: log_scale.x.exp(),
                        y: log_scale.y.exp(),
                        z: log_scale.z.exp(),
                    };
                    
                    opacities[i] = sigmoid(splats.raw_opacities[i]);
                    rotations[i] = splats.rotations[i];
                    
                    // SH data
                    let sh_base = i * 45;
                    for j in 1..16 {
                        let coeff = if j < sh_coeffs.len() { sh_coeffs[j] } else { Vec3::ZERO };
                        let idx = sh_base + (j - 1) * 3;
                        sh_data[idx] = coeff.x;
                        sh_data[idx + 1] = coeff.y;
                        sh_data[idx + 2] = coeff.z;
                    }
                }
                
                (positions, colors, scales, opacities, rotations, sh_data)
            };
            
            // Write all buffers to GPU
            render_queue.write_buffer(
                &gpu_buffers.position_buffer.buffer,
                gpu_buffers.position_buffer.offset,
                bytemuck::cast_slice(&positions),
            );
            
            render_queue.write_buffer(
                &gpu_buffers.color_buffer.buffer,
                gpu_buffers.color_buffer.offset,
                bytemuck::cast_slice(&colors),
            );
            
            render_queue.write_buffer(
                &gpu_buffers.scale_buffer.buffer,
                gpu_buffers.scale_buffer.offset,
                bytemuck::cast_slice(&scales),
            );
            
            render_queue.write_buffer(
                &gpu_buffers.opacity_buffer.buffer,
                gpu_buffers.opacity_buffer.offset,
                bytemuck::cast_slice(&opacities),
            );
            
            render_queue.write_buffer(
                &gpu_buffers.rotation_buffer.buffer,
                gpu_buffers.rotation_buffer.offset,
                bytemuck::cast_slice(&rotations),
            );
            
            render_queue.write_buffer(
                &gpu_buffers.sh_buffer.buffer,
                gpu_buffers.sh_buffer.offset,
                bytemuck::cast_slice(&sh_data),
            );
        }
        
        // Remove the update marker
        commands.entity(entity).remove::<BuffersNeedUpdate>();
    }
}

/// Update Gaussian uniform data for each entity independently (per-entity optimization)
/// 
/// This system uses change detection and only updates uniforms when:
/// 1. RenderingConfig changed (user adjusted rendering parameters)
/// 2. Viewport size changed (window resized)
/// 3. Transform changed (entity moved/rotated/scaled)
fn update_gaussian_uniforms(
    render_queue: Res<RenderQueue>,
    views: Query<&ExtractedView, Without<Camera2d>>,
    // Query entities with buffers and detect changes
    mut entities_with_buffers: Query<(
        Entity,
        &GaussianSplats,
        &GlobalTransform,
        &GaussianSplatGpuBuffers,
        Option<&RenderingConfig>,
        Option<&SplatEditingColorConfig>,
    )>,
    // Change detection queries
    changed_rendering: Query<Entity, Changed<RenderingConfig>>,
    changed_transform: Query<Entity, Changed<GlobalTransform>>,
    changed_splats: Query<Entity, Changed<GaussianSplats>>,
    changed_views: Query<(), (Changed<ExtractedView>, Without<Camera2d>)>,
    changed_editing_color: Query<Entity, Changed<SplatEditingColorConfig>>,
    changed_gpu_buffers: Query<Entity, Changed<GaussianSplatGpuBuffers>>,
) {
    // Get viewport size
    let Some(view) = views.iter().next() else {
        return;
    };
    let surface_width = view.viewport.z as u32;
    let surface_height = view.viewport.w as u32;
    let view_changed = !changed_views.is_empty();
    
    // Update each entity independently
    for (entity, splats, global_transform, buffers, rendering_config, splat_editing_color_config) in entities_with_buffers.iter_mut() {
        // Check if this entity needs update
        let rendering_changed = changed_rendering.contains(entity);
        let transform_changed = changed_transform.contains(entity);
        let splats_changed = changed_splats.contains(entity);
        let editing_color_changed = changed_editing_color.contains(entity);
        let gpu_buffers_changed = changed_gpu_buffers.contains(entity);
        
        // Skip if no changes for this entity
        if !rendering_changed && !transform_changed && !splats_changed && !view_changed 
           && !editing_color_changed && !gpu_buffers_changed {
            continue;
        }
        
        // Get point count from GaussianSplats
        let actual_point_count = {
            splats.len() as u32
        };
        
        
        // Skip entities with no splats
        if actual_point_count == 0 {
            continue;
        }
        
        // Get configuration
        let config = rendering_config.copied().unwrap_or_default();
        let point_size = config.point_size;
        let frustum_dilation = config.frustum_dilation;
        let alpha_cull_threshold = config.alpha_cull_threshold;
        let splat_scale = config.splat_scale;
        
        // Update transform buffer if transform changed
        if transform_changed {
            let model_matrix = global_transform.to_matrix();
            let transforms = TransformUniforms { model_matrix };
            render_queue.write_buffer(
                &buffers.transform_buffer,
                0,
                bytemuck::bytes_of(&transforms),
            );
        }
        
        // Update uniform buffer
        let sh_degree = splats.sh_degree(); // Get SH degree from splat data
        let default_editing_colors = SplatEditingColorConfig::default();
        let editing_colors = splat_editing_color_config.copied().unwrap_or(default_editing_colors);
        
        // Convert colors from sRGB (0-1) to linear space (0-1) for proper HDR rendering
        let select_color = editing_colors.select_color;
        let unselect_color = editing_colors.unselect_color;
        let locked_color = editing_colors.locked_color;
        
        let scale = 1.0 as f32 / (config.white_point - config.black_point);
        let tint_color = Vec4::new(config.albedo_color.x * scale, config.albedo_color.y * scale, config.albedo_color.z * scale, config.transparency);
        let offset = -config.black_point + config.brightness;
        let color_offset = Vec4::new(offset, offset, offset,1.0);
        let uniforms = GaussianSplatParams {
            point_size,
            surface_width,
            surface_height,
            point_count: actual_point_count,  // Use actual count from ExternalGpuBuffers if training
            frustum_dilation,
            alpha_cull_threshold,
            splat_scale,
            sh_degree,
            select_color,
            unselect_color,
            locked_color,
            tint_color,
            color_offset,
        };
        render_queue.write_buffer(
            &buffers.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        
        // Update cull parameters
        let cull_params = ProjectCullParams {
            point_count: actual_point_count,  // Use actual count from ExternalGpuBuffers if training
            point_size,
            frustum_dilation,
            _padding: 0,
        };
        
        render_queue.write_buffer(
            &buffers.cull_params,
            0,
            bytemuck::bytes_of(&cull_params),
        );
    }
}

/// Upload selection state to GPU buffer when changed
/// Uses Bevy's change detection (Changed<ExtractedSelectionState>)
fn upload_selection_state_to_gpu(
    render_queue: Res<RenderQueue>,
    entities_with_selection: Query<(
        Entity,
        &GaussianSplatGpuBuffers,
        &ExtractedSelectionState,
    ), Changed<ExtractedSelectionState>>,
) {
    for (entity, buffers, selection_state) in entities_with_selection.iter() {
        // Validate buffer sizes match
        if selection_state.states.len() != buffers.point_count as usize {
            // Only log once per entity to avoid spam
            use std::sync::OnceLock;
            use std::sync::Mutex;
            static WARNED_ENTITIES: OnceLock<Mutex<std::collections::HashSet<Entity>>> = OnceLock::new();
            let warned = WARNED_ENTITIES.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
            let mut warned_set = warned.lock().unwrap();
            if !warned_set.contains(&entity) {
                debug!(
                    "Selection state size mismatch: {} vs {} points for entity {:?} (this is normal during training, will sync soon)",
                    selection_state.states.len(),
                    buffers.point_count,
                    entity
                );
                warned_set.insert(entity);
            }
            continue;
        }
        
        // Write selection states to GPU buffer
        render_queue.write_buffer(
            &buffers.state_buffer,
            0,
            bytemuck::cast_slice(&selection_state.states),
        );
    }
}

/// Prepare render bind groups for each entity (per-entity Component)
fn prepare_gaussian_splat_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<GaussianSplatPipeline>,
    view_uniforms: Res<ViewUniforms>,
    pipeline_cache: Res<PipelineCache>,
    // Query entities that have buffers but no bind group yet
    // Training entities are processed the same way as normal entities (non-PACK mode)
    entities_without_bind_group: Query<(Entity, &GaussianSplatGpuBuffers, &GaussianSplatPipelineId), Without<GaussianSplatBindGroup>>,
) {
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };
    
    for (entity, buffers, pipeline_id) in entities_without_bind_group.iter() {
        // ✅ Check if pipeline is ready before creating bind group
        if pipeline_cache.get_render_pipeline(pipeline_id.0).is_none() {
            // Use trace! to avoid spamming logs during pipeline initialization
            trace!("⏳ Entity {:?}: Pipeline not ready yet, skipping bind group creation", entity);
            continue; // Skip this entity, will try again next frame
        }
        
        // Check if PACK mode buffers are available
        // Training entities always use non-PACK mode (their packed buffers are None)
        let use_pack_mode = buffers.rotation_scales_packed.is_some() 
            && buffers.colors_packed.is_some() 
            && buffers.sh_packed.is_some();
        
        let bind_group = if use_pack_mode {
            render_device.create_bind_group(
                Some("gaussian_splat_bind_group_pack"),
                &pipeline.bind_group_layout,
                &BindGroupEntries::sequential((
                    view_binding.clone(),                                           // @binding(0): View
                    buffers.uniform_buffer.as_entire_binding(),                     // @binding(1): Uniforms
                    buffers.position_buffer.as_binding(),                           // @binding(2): Positions - with offset!
                    buffers.colors_packed.as_ref().unwrap().as_entire_binding(),    // @binding(3): Colors (packed)
                    buffers.visible_indices.as_entire_binding(),                    // @binding(4): Visible indices
                    buffers.rotation_scales_packed.as_ref().unwrap().as_entire_binding(), // @binding(5): Rotation+Scale+Opacity (packed)
                    buffers.opacity_buffer.as_binding(),                            // @binding(6): Dummy (unused in PACK) - with offset!
                    buffers.rotation_buffer.as_binding(),                           // @binding(7): Dummy (unused in PACK) - with offset!
                    buffers.sh_packed.as_ref().unwrap().as_entire_binding(),        // @binding(8): SH buffer (packed)
                    buffers.transform_buffer.as_entire_binding(),                   // @binding(9): Transform uniforms
                    buffers.state_buffer.as_entire_binding(),                       // @binding(10): Splat state buffer
                )),
            )
        } else {
            // Standard (non-PACK) mode - used by both normal PLY entities and training entities
            // CRITICAL: Use as_binding() for external buffers to respect offset!
            render_device.create_bind_group(
                Some("gaussian_splat_bind_group_standard"),
                &pipeline.bind_group_layout,
                &BindGroupEntries::sequential((
                    view_binding.clone(),                          // @binding(0): View uniform
                    buffers.uniform_buffer.as_entire_binding(),    // @binding(1): Gaussian uniforms
                    buffers.position_buffer.as_binding(),          // @binding(2): Positions (STORAGE) - with offset!
                    buffers.color_buffer.as_binding(),             // @binding(3): sh_coeffs0/colors (STORAGE) - with offset!
                    buffers.visible_indices.as_entire_binding(),   // @binding(4): Visible indices (STORAGE)
                    buffers.scale_buffer.as_binding(),             // @binding(5): log_scales (STORAGE) - with offset!
                    buffers.opacity_buffer.as_binding(),           // @binding(6): raw_opacities (STORAGE) - with offset!
                    buffers.rotation_buffer.as_binding(),          // @binding(7): Rotations (STORAGE) - with offset!
                    buffers.sh_buffer.as_binding(),                // @binding(8): SH coeffs (STORAGE) - with offset!
                    buffers.transform_buffer.as_entire_binding(),  // @binding(9): Transform uniforms
                    buffers.state_buffer.as_entire_binding(),      // @binding(10): Splat states (STORAGE)
                )),
            )
        };

        commands.entity(entity).insert(GaussianSplatBindGroup(bind_group));
        debug!("✅ Render bind group created for entity {:?} (PACK: {})", entity, use_pack_mode);
    }
}

/// Culling bind group (for Project & Cull) - per-entity Component
#[derive(Component)]
pub struct GaussianSplatCullBindGroup(pub BindGroup);

/// Prepare radix sort bind groups for each entity (per-entity Component)
fn prepare_radix_sort_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipelines: Res<RadixSortPipelines>,
    // Query entities that have buffers but no radix sort bind groups yet
    entities_without_sort_bg: Query<(Entity, &GaussianSplatGpuBuffers), Without<RadixSortBindGroups>>,
) {
    use crate::radix_sort::SortParams;
    
    for (entity, buffers) in entities_without_sort_bg.iter() {
        // CRITICAL: Use indirect_buffer for dynamic element count!
        // indirect_buffer layout: [vertex_count, instance_count, first_vertex, first_instance]
        // Shader reads indirect_buffer[1] (instance_count) as the dynamic visible count from cull shader
        // This fixes the flickering issue where we were sorting all points instead of just visible ones
        
        // Create bind groups for 4 passes
        let mut upsweep_bind_groups = Vec::new();
        let mut spine_bind_groups = Vec::new();
        let mut downsweep_bind_groups = Vec::new();
        
        for pass in 0..4 {
            let bit_shift = pass * 8;
            
            // Create params buffer
            let params = SortParams {
                max_element_count: buffers.point_count,
                bit_shift,
                pass_index: pass as u32,
                _padding: 0,
            };
            let params_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(&format!("radix_params_p{}", pass)),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            });
            
            // Ping-pong: even passes use original buffers as input, odd passes use temp buffers
            let (keys_in, keys_out, values_in, values_out) = if pass % 2 == 0 {
                (&buffers.depth_keys, &buffers.radix_sort_buffers.keys_temp,
                 &buffers.visible_indices, &buffers.radix_sort_buffers.values_temp)
            } else {
                (&buffers.radix_sort_buffers.keys_temp, &buffers.depth_keys,
                 &buffers.radix_sort_buffers.values_temp, &buffers.visible_indices)
            };
            
            // Upsweep bind group - uses indirect_buffer for dynamic element count
            let upsweep_bg = render_device.create_bind_group(
                None,
                &pipelines.upsweep_bind_group_layout,
                &BindGroupEntries::sequential((
                    params_buffer.as_entire_binding(),
                    buffers.indirect_buffer.as_entire_binding(),  // Dynamic count from cull shader
                    keys_in.as_entire_binding(),
                    buffers.radix_sort_buffers.global_histogram.as_entire_binding(),
                    buffers.radix_sort_buffers.partition_histogram.as_entire_binding(),
                )),
            );
            upsweep_bind_groups.push(upsweep_bg);
            
            // Spine bind group - uses indirect_buffer for dynamic element count
            let spine_bg = render_device.create_bind_group(
                None,
                &pipelines.spine_bind_group_layout,
                &BindGroupEntries::sequential((
                    buffers.indirect_buffer.as_entire_binding(),  // Dynamic count from cull shader
                    buffers.radix_sort_buffers.global_histogram.as_entire_binding(),
                    buffers.radix_sort_buffers.partition_histogram.as_entire_binding(),
                    params_buffer.as_entire_binding(),
                )),
            );
            spine_bind_groups.push(spine_bg);
            
            // Downsweep bind group - uses indirect_buffer for dynamic element count
            let downsweep_bg = render_device.create_bind_group(
                None,
                &pipelines.downsweep_bind_group_layout,
                &BindGroupEntries::sequential((
                    params_buffer.as_entire_binding(),
                    buffers.indirect_buffer.as_entire_binding(),  // Dynamic count from cull shader
                    buffers.radix_sort_buffers.global_histogram.as_entire_binding(),
                    buffers.radix_sort_buffers.partition_histogram.as_entire_binding(),
                    keys_in.as_entire_binding(),
                    values_in.as_entire_binding(),
                    keys_out.as_entire_binding(),
                    values_out.as_entire_binding(),
                )),
            );
            downsweep_bind_groups.push(downsweep_bg);
        }
        
        commands.entity(entity).insert(RadixSortBindGroups {
            upsweep_bind_groups,
            spine_bind_groups,
            downsweep_bind_groups,
        });
        
        println!("✅ Radix sort bind groups created for entity {:?}", entity);
    }
}

/// Prepare culling bind groups for each entity (per-entity Component)
fn prepare_gaussian_splat_cull_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    cull_pipeline: Res<GaussianSplatCullPipeline>,
    view_uniforms: Res<ViewUniforms>,
    // Query entities that have buffers but no cull bind group yet
    entities_without_cull_bg: Query<(Entity, &GaussianSplatGpuBuffers), Without<GaussianSplatCullBindGroup>>,
) {
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    for (entity, buffers) in entities_without_cull_bg.iter() {
        // Create bind group for Project & Cull
        // CRITICAL: Use as_binding() for external buffers to respect offset!
        let bind_group = render_device.create_bind_group(
            Some("gaussian_splat_cull_bind_group"),
            &cull_pipeline.bind_group_layout,
            &BindGroupEntries::sequential((
                view_binding.clone(),                        // @binding(0): View uniform
                buffers.cull_params.as_entire_binding(),     // @binding(1): ProjectCullParams
                buffers.position_buffer.as_binding(),        // @binding(2): Positions - with offset!
                buffers.depth_keys.as_entire_binding(),      // @binding(3): Depth keys
                buffers.sorted_indices.as_entire_binding(),  // @binding(4): Sorted indices (for radix sort)
                buffers.visible_indices.as_entire_binding(), // @binding(5): Visible indices
                buffers.indirect_buffer.as_entire_binding(), // @binding(6): Indirect buffer
                buffers.transform_buffer.as_entire_binding(), // @binding(7): Transform uniforms
            )),
        );

        commands.entity(entity).insert(GaussianSplatCullBindGroup(bind_group));
        debug!("✅ Cull bind group created for entity {:?}", entity);
    }
}

/// Render pipeline
#[derive(Resource)]
pub struct GaussianSplatPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shader: Handle<Shader>,
}

impl FromWorld for GaussianSplatPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            Some("gaussian_splat_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    // @binding(0): View uniform
                    uniform_buffer::<ViewUniform>(true),
                    // @binding(1): Gaussian splat uniforms
                    uniform_buffer::<GaussianSplatParams>(false),
                    // @binding(2): Position buffer
                    storage_buffer_read_only_sized(false, None),
                    // @binding(3): Color buffer
                    storage_buffer_read_only_sized(false, None),
                    // @binding(4): Visible indices buffer (compact, sorted)
                    storage_buffer_read_only_sized(false, None),
                    // @binding(5): Scale buffer
                    storage_buffer_read_only_sized(false, None),
                    // @binding(6): Opacity buffer
                    storage_buffer_read_only_sized(false, None),
                    // @binding(7): Rotation buffer
                    storage_buffer_read_only_sized(false, None),
                    // @binding(8): SH data buffer (packed higher-order SH coefficients)
                    storage_buffer_read_only_sized(false, None),
                    // @binding(9): Transform uniforms (model matrix)
                    uniform_buffer::<TransformUniforms>(false),
                    // @binding(10): Splat state buffer (per-splat selection/lock/delete state)
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );

        // Load embedded shader using Bevy's recommended method
        let shader = load_embedded_asset!(asset_server, "../assets/shaders/gaussian_splat.wgsl");

        Self {
            bind_group_layout,
            shader,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct GaussianSplatPipelineKey {
    pub hdr: bool,
    pub msaa_samples: u32,
    pub enable_aa: bool,        // GSPLAT_AA: Enable anti-aliasing
    pub sh_degree: u32,         // SH_DEGREE: Spherical harmonics degree (0-3)
    pub pack_mode: bool,        // PACK: Enable compressed data format
    pub vis_mode: SplatVisMode, // Visualization mode (normal, depth, rings, etc.)
    pub use_tonemapping: bool,  // DEPRECATED: Kept for compatibility, actual conversion controlled by key.hdr
}

impl SpecializedRenderPipeline for GaussianSplatPipeline {
    type Key = GaussianSplatPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // Build shader defs based on key
        let mut shader_defs = vec![];
        
        // On WASM platform, enable fast exp approximation (PlayCanvas fastExp)
        // This provides ~3x performance improvement for exp() calls on WebGPU
        #[cfg(target_arch = "wasm32")]
        shader_defs.push("WASM_PLATFORM".into());
        
        // Enable PACK mode if requested
        if key.pack_mode {
            shader_defs.push("PACK".into());
        }
        
        // Anti-aliasing variant
        if key.enable_aa {
            shader_defs.push("GSPLAT_AA".into());
        }
        
        // Visualization mode variants
        let vis_def = match key.vis_mode {
            SplatVisMode::Splat => None, // Default, no special define needed
            SplatVisMode::Point => Some("VIS_POINT"),
            SplatVisMode::Rings => Some("VIS_RINGS"),
            SplatVisMode::Centers => Some("VIS_CENTERS"), // Centers mode: splat + blue center point overlay
            SplatVisMode::Pick => Some("PICK_PASS"), // Pick mode: output splat ID as RGBA color
            SplatVisMode::Outline => Some("OUTLINE_PASS"), // Outline mode: only render selected splats
        };
        
        if let Some(def) = vis_def {
            shader_defs.push(def.into());
            println!("📝 Adding shader def: {} for vis_mode {:?}", def, key.vis_mode);
        }
        
        // Spherical harmonics degree variant (0-3)
        // SH calculation is CUMULATIVE: degree 3 needs coefficients from 1, 2, AND 3
        // So we define all macros up to the requested degree
        let sh_degree = key.sh_degree.min(3); // Clamp to 0-3
        for d in 1..=sh_degree {
            shader_defs.push(format!("SH_DEGREE_{}", d).into());
        }
        
        // RENDER TARGET FORMAT SELECTION:
        // - Main splat rendering (Splat, Point): renders to cache texture (Rgba8Unorm)
        //   Then blit to screen. Shader outputs sRGB colors directly (no conversion).
        //   Cache uses Rgba8Unorm (NOT Rgba8UnormSrgb) to avoid double-gamma!
        //   Blit shader handles sRGB→linear conversion.
        //   Cache render pass has NO depth attachment (splats are radix-sorted).
        // - Overlay passes (Centers, Rings, Outline) and other modes: render directly to screen
        //   Must use screen format (HDR or LDR) based on key.hdr
        //   Shader must convert sRGB → linear when rendering to HDR target
        let render_to_cache = matches!(key.vis_mode, 
            SplatVisMode::Splat | SplatVisMode::Point);
        
        if render_to_cache {
            // Rendering to cache (Rgba8Unorm): no color conversion needed
            // sRGB values stored directly; blit shader converts sRGB→linear
            shader_defs.push("RENDER_TO_CACHE".into());
        } else if key.hdr {
            // Rendering to HDR screen: need sRGB → linear conversion
            shader_defs.push("RENDER_TO_HDR".into());
        }
        // Rendering to LDR screen (Rgba8UnormSrgb): no conversion needed (same as cache)
        
        let target_format = if render_to_cache {
            // Main splat rendering: use Rgba8Unorm for cache texture
            // NOT Rgba8UnormSrgb - that would cause double-gamma (too bright)!
            // Blit shader samples sRGB and converts to linear for output
            TextureFormat::Rgba8Unorm
        } else {
            // Overlay passes render directly to screen: use screen format
            if key.hdr {
                ViewTarget::TEXTURE_FORMAT_HDR
            } else {
                TextureFormat::Rgba8UnormSrgb
            }
        };
        
        // Depth stencil configuration:
        // - Cache rendering: NO depth (cache is sample_count=1, splats are radix-sorted)
        // - Screen rendering: USE depth for proper occlusion with scene
        let depth_stencil = if render_to_cache {
            None // No depth for cache (avoids MSAA mismatch, splats are pre-sorted)
        } else {
            Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: false,
                // Outline pass: disable depth test by using Always compare function
                // Normal pass: enable depth testing for proper occlusion
                depth_compare: if key.vis_mode == SplatVisMode::Outline {
                    CompareFunction::Always  // Always pass = no depth test
                } else {
                    CompareFunction::GreaterEqual  // Reverse-Z: Greater = closer
                },
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            })
        };
        
        RenderPipelineDescriptor {
            label: Some("gaussian_splat_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                entry_point: Some("vertex".into()),
                buffers: vec![],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                front_face: wgpu::FrontFace::Cw,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                ..default()
            },
            // Depth configuration: None for cache rendering, Some for screen rendering
            depth_stencil,
            multisample: MultisampleState {
                // Cache rendering uses sample_count=1 (no MSAA), screen uses view's MSAA
                count: if render_to_cache { 1 } else { key.msaa_samples },
                ..Default::default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: target_format,
                    // PREMULTIPLIED ALPHA blending (matching PlayCanvas/SuperSplat):
                    // Shader outputs: vec4(color * alpha, alpha)
                    // Blend: src.rgb * 1 + dst.rgb * (1 - src.a)
                    // IMPORTANT: cache_blit.wgsl expects premultiplied alpha!
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,  // Premultiplied: use ONE
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

/// Culling compute pipeline
#[derive(Resource)]
pub struct GaussianSplatCullPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shader: Handle<Shader>,
}

impl FromWorld for GaussianSplatCullPipeline {
    fn from_world(world: &mut World) -> Self {
        use bevy::render::render_resource::{binding_types, ShaderStages};

        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();
        
        // Create bind group layout
        let bind_group_layout = render_device.create_bind_group_layout(
            Some("gaussian_splat_cull_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // @binding(0): View uniform
                    uniform_buffer::<ViewUniform>(true),
                    // @binding(1): ProjectCullParams uniform
                    uniform_buffer::<ProjectCullParams>(false),
                    // @binding(2): Position buffer (readonly)
                    storage_buffer_read_only_sized(false, None),
                    // @binding(3): Depth keys buffer (read/write, compact)
                    binding_types::storage_buffer_sized(false, None),
                    // @binding(4): Sorted indices buffer (read/write, for radix sort)
                    binding_types::storage_buffer_sized(false, None),
                    // @binding(5): Visible indices buffer (read/write, compact visible point indices)
                    binding_types::storage_buffer_sized(false, None),
                    // @binding(6): Indirect buffer (read/write, atomic)
                    binding_types::storage_buffer_sized(false, None),
                    // @binding(7): Transform uniforms (model matrix + inverse)
                    uniform_buffer::<TransformUniforms>(false),
                ),
            ),
        );
        
        // Load embedded shader using Bevy's recommended method
        let shader = load_embedded_asset!(asset_server, "../assets/shaders/gaussian_splat_cull.wgsl");
        
        Self {
            bind_group_layout,
            shader,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct GaussianSplatCullPipelineKey {}

impl SpecializedComputePipeline for GaussianSplatCullPipeline {
    type Key = GaussianSplatCullPipelineKey;

    fn specialize(&self, _key: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("gaussian_splat_cull_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: self.shader.clone(),
            shader_defs: vec![],
            entry_point: Some("project_and_cull".into()),
            zero_initialize_workgroup_memory: true,
        }
    }
}

/// Pipeline for blitting cached render result to screen
/// Used when camera is static and no data updates - skips full 3DGS render
#[derive(Resource)]
pub struct CacheBlitPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shader: Handle<Shader>,
    pub pipeline_id: Option<CachedRenderPipelineId>,
}

impl FromWorld for CacheBlitPipeline {
    fn from_world(world: &mut World) -> Self {
        use bevy::render::render_resource::{binding_types, ShaderStages};
        
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();
        
        // Create bind group layout for cache texture + sampler
        let bind_group_layout = render_device.create_bind_group_layout(
            Some("cache_blit_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // @binding(0): Cache texture
                    binding_types::texture_2d(wgpu::TextureSampleType::Float { filterable: true }),
                    // @binding(1): Sampler
                    binding_types::sampler(wgpu::SamplerBindingType::Filtering),
                ),
            ),
        );
        
        // Load the blit shader
        let shader = load_embedded_asset!(asset_server, "../assets/shaders/cache_blit.wgsl");
        
        Self {
            bind_group_layout,
            shader,
            pipeline_id: None,
        }
    }
}

impl CacheBlitPipeline {
    /// Get or create the blit pipeline for the given format
    pub fn get_pipeline(
        &mut self,
        pipeline_cache: &PipelineCache,
        hdr: bool,
        msaa_samples: u32,
    ) -> Option<CachedRenderPipelineId> {
        if self.pipeline_id.is_some() {
            return self.pipeline_id;
        }
        
        let format = if hdr {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::Rgba8UnormSrgb
        };
        
        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("cache_blit_pipeline".into()),
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
                    // Use premultiplied alpha blending (same as 3DGS)
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One, // Premultiplied alpha
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: false,
        });
        
        self.pipeline_id = Some(pipeline_id);
        self.pipeline_id
    }
}

/// Render node
#[derive(Default)]
pub struct GaussianSplatNode;

impl ViewNode for GaussianSplatNode {
    type ViewQuery = (
        &'static ExtractedView,
        &'static ViewTarget,
        &'static ViewDepthTexture,
        Option<&'static GaussianSplatPipelineId>,
        &'static ViewUniformOffset,
        Option<&'static GaussianSplatCullPipelineId>,
        Option<&'static GaussianSplatOverlayCentersPipelineId>,
        Option<&'static GaussianSplatOverlayRingsPipelineId>,
        Option<&'static GaussianSplatOutlinePipelineId>,
    );

    fn run<'w>(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext<'w>,
        (view, target, depth, _deprecated_pipeline_id, view_uniform_offset, cull_pipeline_id, _overlay_centers_pipeline_id, _overlay_rings_pipeline_id, _outline_pipeline_id): bevy::ecs::query::QueryItem<
            'w,
            'w,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        // ✅ Per-entity pipeline: No longer use view's pipeline ID
        // Each entity will provide its own pipeline ID
        
        let pipeline_cache = world.resource::<PipelineCache>();
        
        // Collect all entities to render (with their own pipeline IDs)
        let mut entities_to_render = Vec::new();
        
        for archetype in world.archetypes().iter() {
            if !archetype.contains(world.components().component_id::<GaussianSplatGpuBuffers>().unwrap()) {
                continue;
            }
            if !archetype.contains(world.components().component_id::<GaussianSplatBindGroup>().unwrap()) {
                continue;
            }
            if !archetype.contains(world.components().component_id::<GaussianSplatCullBindGroup>().unwrap()) {
                continue;
            }
            if !archetype.contains(world.components().component_id::<RadixSortBindGroups>().unwrap()) {
                continue;
            }
            
            // 遍历这个archetype中的所有实体
            for entity in archetype.entities() {
                let entity_id = entity.id();
                if let Ok(entity_ref) = world.get_entity(entity_id) {
                    if let (Some(buffers), Some(bind_group), Some(cull_bind_group), Some(radix_sort_bind_groups), Some(pipeline_id)) = (
                        entity_ref.get::<GaussianSplatGpuBuffers>(),
                        entity_ref.get::<GaussianSplatBindGroup>(),
                        entity_ref.get::<GaussianSplatCullBindGroup>(),
                        entity_ref.get::<RadixSortBindGroups>(),
                        entity_ref.get::<GaussianSplatPipelineId>(), // ✅ Get entity's own pipeline ID!
                    ) {
                        // Get rendering config options
                        let (show_selection_overlay, overlay_vis_mode, show_outline) = entity_ref
                            .get::<RenderingConfig>()
                            .map(|c| (c.show_selection_overlay, c.overlay_vis_mode, c.show_outline))
                            .unwrap_or((false, None, false));
                        
                        // Get entity's own overlay pipeline IDs
                        let overlay_centers_pipeline_id = entity_ref.get::<GaussianSplatOverlayCentersPipelineId>().map(|p| p.0);
                        let overlay_rings_pipeline_id = entity_ref.get::<GaussianSplatOverlayRingsPipelineId>().map(|p| p.0);
                        let outline_pipeline_id_entity = entity_ref.get::<GaussianSplatOutlinePipelineId>().map(|p| p.0);
                        
                        // Check if this entity is in training mode (allows full render skip)
                        let is_training = entity_ref.get::<TrainingMode>().is_some();
                        
                        entities_to_render.push((
                            entity_id,
                            pipeline_id.0,                     // ✅ Entity's own pipeline ID!
                            buffers.point_count,
                            buffers.indirect_buffer.clone(),
                            buffers.radix_sort_buffers.global_histogram.clone(),
                            buffers.radix_sort_buffers.partition_histogram.clone(),
                            buffers.radix_sort_buffers.num_partitions,
                            bind_group.0.clone(),
                            cull_bind_group.0.clone(),
                            radix_sort_bind_groups.clone(),
                            buffers.depth_keys.clone(),        // For clearing before cull
                            buffers.visible_indices.clone(),   // For clearing before cull
                            buffers.radix_sort_buffers.keys_temp.clone(),    // For clearing before sort
                            buffers.radix_sort_buffers.values_temp.clone(),  // For clearing before sort
                            show_selection_overlay,            // For second pass overlay
                            overlay_vis_mode,                  // Which overlay mode (Centers or Rings)
                            show_outline,                      // For outline rendering
                            overlay_centers_pipeline_id,       // ✅ Entity's own overlay pipeline IDs!
                            overlay_rings_pipeline_id,
                            outline_pipeline_id_entity,
                            is_training,                       // Whether this is a training entity
                        ));
                    }
                }
            }
        }
        
        // === TRAINING PREVIEW: Check if we have a training-rendered preview image ===
        // This check MUST happen before the "no entities" early return, because during
        // training mode the GaussianSplats entity may not have data yet, but we still
        // want to display the CUDA-rendered preview.
        // NOTE: Use get_resource() instead of resource() for WASM compatibility
        // These resources are only initialized when TrainingPreviewPlugin is added (native/training mode)
        let training_preview_data = world.get_resource::<crate::training_preview::TrainingPreviewImageData>();
        let training_preview_target = world.get_resource::<crate::training_preview::TrainingPreviewRenderTarget>();
        let training_preview_pipeline = world.get_resource::<crate::training_preview::TrainingPreviewBlitPipeline>();
        
        // Check if we have a valid training preview to use
        // Only proceed if all training preview resources exist and are enabled
        let use_training_preview = training_preview_data
            .map(|data| data.enabled)
            .unwrap_or(false) 
            && training_preview_target
                .map(|target| target.is_ready())
                .unwrap_or(false);
        
        // Get training preview blit resources if available
        let training_preview_bind_group = if use_training_preview {
            training_preview_target.and_then(|t| t.bind_group.clone())
        } else {
            None
        };
        let training_preview_pipeline_ready = if use_training_preview {
            training_preview_pipeline
                .and_then(|p| p.pipeline_id)
                .and_then(|id| pipeline_cache.get_render_pipeline(id))
                .cloned()
        } else {
            None
        };
        
        // === FASTEST PATH: Use training-rendered preview image ===
        // If training backend provides a pre-rendered image, just blit it to screen.
        // This works even when no GaussianSplats entities exist (e.g., during training startup).
        if let (Some(preview_bg), Some(preview_pipe)) = (training_preview_bind_group.clone(), training_preview_pipeline_ready.clone()) {
            // Training preview blit path - no need to log every frame
            
            // Get target size before moving into closure
            let target_main_texture = target.main_texture_view();
            let target_size = target_main_texture.texture().size();
            
            // Get color attachment for final output (screen)
            let mut color_attachment = target.get_color_attachment();
            color_attachment.ops = Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            };
            
            // Clamp viewport to target bounds
            let viewport_x = view.viewport.x;
            let viewport_y = view.viewport.y;
            let viewport_width = view.viewport.z.min(target_size.width.saturating_sub(viewport_x));
            let viewport_height = view.viewport.w.min(target_size.height.saturating_sub(viewport_y));
            
            // Additional safety: ensure y + h doesn't exceed target height
            let safe_height = viewport_height.min(target_size.height.saturating_sub(viewport_y));
            
            // Only render if viewport is valid
            if viewport_width > 0 && safe_height > 0 {
                render_context.add_command_buffer_generation_task(move |render_device| {
                    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("training_preview_blit"),
                    });
                    
                    // Blit training preview to screen (alpha blending)
                    {
                        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                            label: Some("training_preview_blit_pass"),
                            color_attachments: &[Some(color_attachment)],
                            depth_stencil_attachment: None, // Blit doesn't need depth testing
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        
                        render_pass.set_viewport(
                            viewport_x as f32,
                            viewport_y as f32,
                            viewport_width as f32,
                            safe_height as f32,
                            0.0,
                            1.0,
                        );
                        
                        render_pass.set_pipeline(&preview_pipe);
                        render_pass.set_bind_group(0, &preview_bg, &[]);
                        render_pass.draw(0..3, 0..1); // Fullscreen triangle
                    }
                    
                    encoder.finish()
                });
            }
            
            // Skip all 3DGS rendering - we're using training preview!
            return Ok(());
        }
        
        // Early return if no entities to render (and no training preview)
        if entities_to_render.is_empty() {
            return Ok(());
        }
        
        // Check if ALL entities are training mode (only then can we skip entire render)
        // If there are any non-training (normal PLY) entities, we must render every frame
        let all_training_mode = entities_to_render.iter().all(|(.., is_training)| *is_training);
        
        // === TEMPORAL COHERENCE: Use pre-computed skip decision from main world ===
        // 
        // IMPORTANT: We use the pre-computed `sorting_skipped` flag from the main world's
        // `update_temporal_coherence_cache` system instead of recomputing here.
        // 
        // Previously, we recomputed `pos_delta = camera_pos.distance(temporal_cache.last_camera_pos)`
        // in the render node, but this was WRONG because `last_camera_pos` was already updated
        // to the CURRENT frame's position by `should_skip_sorting()` in the main world system.
        // This caused `pos_delta=0` even when the camera moved, leading to incorrect cache usage
        // and "ghost frames" / residual artifacts.
        //
        // Now we rely on the main world's decision which correctly compares:
        // last_frame_camera_pos vs current_frame_camera_pos BEFORE updating last_camera_pos.
        let temporal_cache = world.resource::<TemporalCoherenceCache>();
        
        // CRITICAL: Check data_updated_this_frame to ensure training updates are rendered!
        // If data was updated, never skip sorting even if camera was static
        let skip_sorting = !temporal_cache.data_updated_this_frame && temporal_cache.sorting_skipped;
        
        // === RENDER CACHE + BLIT ARCHITECTURE ===
        // Strategy: ALL 3DGS entities render to intermediate Rgba8Unorm cache texture,
        // then blit to final render target. This approach:
        // - Shader outputs sRGB colors directly (no manual conversion)
        // - Rgba8Unorm stores sRGB values without conversion (avoids double-gamma)
        // - Blit shader converts sRGB→linear, then outputs to final HDR/LDR render target
        // 
        // Cache skip optimization (when camera static + no data update):
        // - Training entities: skip entire render, just blit from cache
        // - Normal PLY entities: skip sorting only (still render every frame for quality)
        let render_cache = world.resource::<GaussianSplatRenderCache>();
        let blit_pipeline = world.resource::<CacheBlitPipeline>();
        
        // Check if we can use cached render result (skip entire 3DGS render)
        // Only training entities can skip render entirely; normal PLY must render every frame
        let can_use_cache = all_training_mode && skip_sorting && render_cache.can_use();
        
        // Get cache resources for rendering/blitting (ALL entities use cache+blit path now)
        let cache_bind_group = render_cache.bind_group.clone();
        let blit_pipeline_id = blit_pipeline.pipeline_id;
        let blit_pipeline_ready = blit_pipeline_id
            .and_then(|id| pipeline_cache.get_render_pipeline(id))
            .cloned();
        
        // For rendering to cache, we need access to the cache's texture view
        // This is tricky because TextureView has lifetime requirements
        // We'll use a workaround by storing the texture itself and creating view in the closure
        
        let radix_sort_pipelines = world.resource::<RadixSortPipelines>();
        let radix_sort_pipelines_cloned = radix_sort_pipelines.clone();
        
        let cull_pipeline_opt = cull_pipeline_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id.0))
            .cloned();
        
        // ✅ Per-entity pipeline: overlay and outline pipelines are now per-entity
        // No longer need to get them from view entity
        
        let view_uniform_offset = view_uniform_offset.offset;
        let depth_view = depth.view().clone();
        
        // Get color attachment for final output (screen)
        let mut color_attachment = target.get_color_attachment();
        color_attachment.ops = Operations {
            load: LoadOp::Load,
            store: StoreOp::Store,
        };
        
        // === FAST PATH: Use cached render result ===
        if can_use_cache {
            if let (Some(cache_bg), Some(blit_pipe)) = (cache_bind_group.clone(), blit_pipeline_ready.clone()) {
                // Get target size before moving into closure
                let target_main_texture = target.main_texture_view();
                let target_size = target_main_texture.texture().size();
                
                // Clamp viewport to target bounds
                let viewport_x = view.viewport.x;
                let viewport_y = view.viewport.y;
                let viewport_width = view.viewport.z.min(target_size.width.saturating_sub(viewport_x));
                let viewport_height = view.viewport.w.min(target_size.height.saturating_sub(viewport_y));
                
                // Additional safety: ensure y + h doesn't exceed target height
                let safe_height = viewport_height.min(target_size.height.saturating_sub(viewport_y));
                
                // Only render if viewport is valid
                if viewport_width > 0 && safe_height > 0 {
                    render_context.add_command_buffer_generation_task(move |render_device| {
                        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("gaussian_splat_cache_blit"),
                        });
                        
                        // Blit cached 3DGS result to screen (alpha blending)
                        // NOTE: No depth attachment needed for blit - we're just copying the cached texture
                        {
                            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                                label: Some("gaussian_splat_blit_pass"),
                                color_attachments: &[Some(color_attachment)],
                                depth_stencil_attachment: None, // Blit doesn't need depth testing
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            
                            render_pass.set_viewport(
                                viewport_x as f32,
                                viewport_y as f32,
                                viewport_width as f32,
                                safe_height as f32,
                                0.0,
                                1.0,
                            );
                            
                            render_pass.set_pipeline(&blit_pipe);
                            render_pass.set_bind_group(0, &cache_bg, &[]);
                            render_pass.draw(0..3, 0..1); // Fullscreen triangle
                        }
                        
                        encoder.finish()
                    });
                }
                
                // Skip the full render - we're using cache!
                return Ok(());
            }
        }
        
        // Get cache texture view for render target (ALL entities use cache+blit path)
        // wgpu::TextureView uses Arc internally, so clone is cheap
        let cache_texture_view = render_cache.view.as_ref().map(|v| v.clone());

        // Calculate safe viewport dimensions BEFORE moving into closure
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

        render_context.add_command_buffer_generation_task(move |render_device| {
            let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("gaussian_splat_encoder"),
            });

            // Process each entity independently
            for (entity, _entity_pipeline_id, point_count, indirect_buffer, global_hist, part_hist, num_partitions, 
                 _bind_group, cull_bind_group, radix_sort_bind_groups, depth_keys, visible_indices,
                 keys_temp, values_temp, _show_selection_overlay, _overlay_vis_mode, _show_outline,
                 _overlay_centers_pid, _overlay_rings_pid, _outline_pid, _is_training) in entities_to_render.iter() {
                
                // 1. Run Project & Cull compute shader for this entity
                // Skip if temporal coherence says we can reuse last frame's sorting
                let cull_executed = if !skip_sorting {
                    if let Some(ref cull_pipeline) = cull_pipeline_opt {
                        // CRITICAL: Clear ALL sort-related buffers before cull (like diverse's clear_points.hlsl)
                        // This prevents stale data from previous frames causing flickering
                        encoder.clear_buffer(depth_keys, 0, None);
                        encoder.clear_buffer(visible_indices, 0, None);
                        encoder.clear_buffer(keys_temp, 0, None);    // Ping-pong temp buffer
                        encoder.clear_buffer(values_temp, 0, None);  // Ping-pong temp buffer
                        
                        // Clear indirect_buffer's instance_count (offset 4)
                        encoder.clear_buffer(indirect_buffer, 4, Some(4_u64));

                        // Run project_and_cull compute shader
                        {
                            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                                label: Some(&format!("gaussian_cull_{:?}", entity)),
                                timestamp_writes: None,
                            });

                            compute_pass.set_pipeline(cull_pipeline);
                            compute_pass.set_bind_group(0, cull_bind_group, &[view_uniform_offset]);

                            // 256 threads per workgroup
                            let workgroup_count = (*point_count + 255) / 256;
                            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    // Temporal coherence: reuse last frame's culling/sorting results
                    false
                };
                
                // 2. Run radix sort for this entity (only if culling was executed)
                if cull_executed {
                    // Clear histogram buffers before sorting
                    // global_histogram: 4 * 256 slots (one section per pass)
                    // partition_histogram: num_partitions * 256 slots (reused each pass)
                    encoder.clear_buffer(global_hist, 0, None);
                    encoder.clear_buffer(part_hist, 0, None);  // CRITICAL: Must clear partition histogram too!
                    
                    // Execute radix sort with proper memory barriers
                    execute_radix_sort(
                        &mut encoder,
                        &pipeline_cache,
                        &radix_sort_pipelines_cloned,
                        radix_sort_bind_groups,
                        *num_partitions,
                    );
                }
            }

            // 3. Render all entities to CACHE TEXTURE (transparent background)
            // This allows us to reuse the cached result when camera is static
            // NOTE: No depth attachment because cache texture is sample_count=1 (for texture sampling)
            // and depth texture may have different MSAA settings. Splat order is correct
            // because they are sorted by radix sort before rendering.
            if let Some(ref cache_view) = cache_texture_view {
                let cache_attachment = bevy::render::render_resource::RenderPassColorAttachment {
                    view: cache_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                };
                let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("gaussian_splat_to_cache"),
                    color_attachments: &[Some(cache_attachment)],
                    // NO depth attachment: cache is sample_count=1, splats are radix-sorted
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                // IMPORTANT: Render to cache from (0,0), NOT from viewport offset!
                // Cache texture stores the viewport content, so we render from origin.
                // The viewport offset is only used when blitting to the final screen.
                render_pass.set_viewport(
                    0.0, // Start at origin for cache
                    0.0,
                    viewport_width as f32,
                    viewport_height as f32,
                    0.0,
                    1.0,
                );
                
                // ✅ Per-entity pipeline: Each entity uses its own pipeline!
                // Draw each entity with its own specialized pipeline
                for (_entity, entity_pipeline_id, _point_count, indirect_buffer, _global_hist, _part_hist, _num_partitions,
                     bind_group, _cull_bind_group, _radix_sort_bind_groups, _depth_keys, _visible_indices,
                     _keys_temp, _values_temp, _show_selection_overlay, _overlay_vis_mode, _show_outline, 
                     _overlay_centers_pid, _overlay_rings_pid, _outline_pid, _is_training) in entities_to_render.iter() {
                    
                    // Get entity's pipeline
                    if let Some(entity_pipeline) = pipeline_cache.get_render_pipeline(*entity_pipeline_id) {
                        render_pass.set_pipeline(entity_pipeline);
                        render_pass.set_bind_group(0, bind_group, &[view_uniform_offset]);
                        
                        // Use indirect draw (instance_count determined by GPU)
                        render_pass.draw_indirect(indirect_buffer, 0);
                    }
                }
            }
            
            // 3.5. Blit cache texture to screen (alpha blending)
            // This composites the 3DGS render result onto the scene
            // NOTE: If cache is not available, skip rendering this frame (pipeline format mismatch)
            if let (Some(ref cache_bg), Some(ref blit_pipe)) = (&cache_bind_group, &blit_pipeline_ready) {
                let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("gaussian_splat_blit_to_screen"),
                    color_attachments: &[Some(color_attachment.clone())],
                    depth_stencil_attachment: None, // No depth test for blit
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                render_pass.set_viewport(
                    viewport_x as f32,
                    viewport_y as f32,
                    viewport_width as f32,
                    viewport_height as f32,
                    0.0,
                    1.0,
                );
                
                render_pass.set_pipeline(blit_pipe);
                render_pass.set_bind_group(0, cache_bg, &[]);
                render_pass.draw(0..3, 0..1); // Fullscreen triangle
            }
            // NOTE: Fallback path removed - if cache is not available, skip splat rendering this frame
            // Cache should be created by prepare_render_cache system before this node runs
            // If not available, next frame will have it ready
            
            // 4. Second pass: Render selection overlay for entities with show_selection_overlay=true
            // Select pipeline based on overlay_vis_mode (Centers or Rings)
            let has_overlay = entities_to_render.iter().any(|e| e.14); // e.14 is show_selection_overlay
            if has_overlay {
                let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("gaussian_splat_overlay_pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                render_pass.set_viewport(
                    viewport_x as f32,
                    viewport_y as f32,
                    viewport_width as f32,
                    viewport_height as f32,
                    0.0,
                    1.0,
                );
                
                // Draw only entities with selection overlay enabled
                for (_entity, _entity_pipeline_id, _point_count, indirect_buffer, _global_hist, _part_hist, _num_partitions,
                     bind_group, _cull_bind_group, _radix_sort_bind_groups, _depth_keys, _visible_indices,
                     _keys_temp, _values_temp, show_selection_overlay, overlay_vis_mode, _show_outline,
                     overlay_centers_pid, overlay_rings_pid, _outline_pid, _is_training) in entities_to_render.iter() {
                    
                    if *show_selection_overlay {
                        // ✅ Use entity's own overlay pipeline based on overlay_vis_mode
                        let overlay_pipeline_id = match overlay_vis_mode {
                            Some(SplatVisMode::Rings) => overlay_rings_pid,
                            _ => overlay_centers_pid, // Default to Centers
                        };
                        
                        if let Some(pid) = overlay_pipeline_id {
                            if let Some(pipeline) = pipeline_cache.get_render_pipeline(*pid) {
                                render_pass.set_pipeline(pipeline);
                                render_pass.set_bind_group(0, bind_group, &[view_uniform_offset]);
                                render_pass.draw_indirect(indirect_buffer, 0);
                            }
                        }
                    }
                }
            }
            
            // 4.5. Outline pass: Render selected splats to outline texture
            // This pass renders only selected splats to a separate texture for edge detection
            // We use the normal Splat pipeline which will render splats with selection color
            // The fragment shader will discard unselected splats for outline
            let has_outline = entities_to_render.iter().any(|e| e.16); // e.16 is show_outline (moved due to new fields)
            if has_outline {
                // Get outline render target
                if let Some(outline_target) = world.get_resource::<crate::outline::OutlineRenderTarget>() {
                    let mut outline_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                        label: Some("gaussian_splat_outline_pass"),
                        color_attachments: &[Some(bevy::render::render_resource::RenderPassColorAttachment {
                            view: &outline_target.view,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: &outline_target.depth_view,
                            depth_ops: Some(Operations {
                                load: LoadOp::Clear(1.0),
                                store: StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    
                    outline_pass.set_viewport(
                        viewport_x as f32,
                        viewport_y as f32,
                        viewport_width as f32,
                        viewport_height as f32,
                        0.0,
                        1.0,
                    );
                    
                    // ✅ Per-entity outline pipeline
                    // Draw only entities with outline enabled
                    for (_entity, _entity_pipeline_id, _point_count, indirect_buffer, _global_hist, _part_hist, _num_partitions,
                         bind_group, _cull_bind_group, _radix_sort_bind_groups, _depth_keys, _visible_indices,
                         _keys_temp, _values_temp, _show_selection_overlay, _overlay_vis_mode, show_outline,
                         _overlay_centers_pid, _overlay_rings_pid, outline_pid, _is_training) in entities_to_render.iter() {
                        
                        if *show_outline {
                            if let Some(pid) = outline_pid {
                                if let Some(outline_pipeline) = pipeline_cache.get_render_pipeline(*pid) {
                                    outline_pass.set_pipeline(outline_pipeline);
                                    outline_pass.set_bind_group(0, bind_group, &[view_uniform_offset]);
                                    outline_pass.draw_indirect(indirect_buffer, 0);
                                }
                            }
                        }
                    }
                } else {
                    warn!("Outline pass: OutlineRenderTarget not found");
                }
            }

            // 5. Pick pass (if active)
            if let Some(pick_target) = world.get_resource::<PickRenderTarget>() {
                if pick_target.pick_active {
                    // ✅ Per-entity pick pipeline
                    // Render to pick target
                    {
                        let mut pick_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                            label: Some("gaussian_splat_pick_pass"),
                            color_attachments: &[Some(bevy::render::render_resource::RenderPassColorAttachment {
                                view: &pick_target.view,
                                resolve_target: None,
                                ops: Operations {
                                    load: LoadOp::Clear(wgpu::Color::BLACK),
                                    store: StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                                view: &pick_target.depth_view,
                                depth_ops: Some(Operations {
                                    load: LoadOp::Clear(1.0),
                                    store: StoreOp::Discard,
                                }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        
                        pick_pass.set_viewport(
                            0.0, 0.0,
                            pick_target.width as f32,
                            pick_target.height as f32,
                            0.0, 1.0,
                        );
                        
                        // Draw all entities to pick buffer (each with its own pick pipeline)
                        for (entity, _entity_pipeline_id, _point_count, indirect_buffer, _global_hist, _part_hist, _num_partitions,
                             bind_group, _cull_bind_group, _radix_sort_bind_groups, _depth_keys, _visible_indices,
                             _keys_temp, _values_temp, _show_selection_overlay, _overlay_vis_mode, _show_outline,
                             _overlay_centers_pid, _overlay_rings_pid, _outline_pid, _is_training) in entities_to_render.iter() {
                            
                            // Get entity's pick pipeline
                            if let Ok(entity_ref) = world.get_entity(*entity) {
                                if let Some(pick_id) = entity_ref.get::<GaussianSplatPickPipelineId>() {
                                    if let Some(pick_pipeline) = pipeline_cache.get_render_pipeline(pick_id.0) {
                                        pick_pass.set_pipeline(pick_pipeline);
                                        pick_pass.set_bind_group(0, bind_group, &[view_uniform_offset]);
                                        pick_pass.draw_indirect(indirect_buffer, 0);
                                    }
                                }
                            }
                        }
                    }
                    
                    // Copy pick rect to staging buffer for readback
                    if let Some(rect) = pick_target.pick_rect {
                        let bytes_per_row = pick_target.width * 4; // RGBA8 = 4 bytes
                        let copy_size = bevy::render::render_resource::Extent3d {
                            width: rect.width.min(pick_target.width - rect.x),
                            height: rect.height.min(pick_target.height - rect.y),
                            depth_or_array_layers: 1,
                        };
                        
                        encoder.copy_texture_to_buffer(
                            bevy::render::render_resource::TexelCopyTextureInfo {
                                texture: &pick_target.texture,
                                mip_level: 0,
                                origin: bevy::render::render_resource::Origin3d {
                                    x: rect.x,
                                    y: rect.y,
                                    z: 0,
                                },
                                aspect: bevy::render::render_resource::TextureAspect::All,
                            },
                            bevy::render::render_resource::TexelCopyBufferInfo {
                                buffer: &pick_target.staging_buffer,
                                layout: bevy::render::render_resource::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(bytes_per_row),
                                    rows_per_image: Some(pick_target.height),
                                },
                            },
                            copy_size,
                        );
                    }
                }
            }

            encoder.finish()
        });

        Ok(())
    }
}


// ============================================================================
// PICK PASS IMPLEMENTATION
// ============================================================================

/// Extracted pick request for render world
#[derive(Resource, Default)]
struct ExtractedPickRequest {
    active: bool,
    rect: Option<PickRect>,
    op: PickOp,
    target_entity: Option<Entity>,
    pending_data: Option<std::sync::Arc<std::sync::Mutex<PickReadbackData>>>,
}

/// Extract pick request from main world to render world
fn extract_pick_request(
    mut commands: Commands,
    pick_request: Extract<Res<PickRequest>>,
    pending_readback: Extract<Res<PickPendingReadback>>,
) {
    if pick_request.active {
        commands.insert_resource(ExtractedPickRequest {
            active: true,
            rect: pick_request.rect,
            op: pick_request.op,
            target_entity: pick_request.target_entity,
            pending_data: Some(pending_readback.data.clone()),
        });
    } else {
        commands.insert_resource(ExtractedPickRequest::default());
    }
}

/// Prepare render cache texture for caching 3DGS render results
/// Called each frame to ensure cache texture matches viewport size
fn prepare_render_cache(
    render_device: Res<RenderDevice>,
    mut render_cache: ResMut<GaussianSplatRenderCache>,
    blit_pipeline: Res<CacheBlitPipeline>,
    views: Query<(&ExtractedView, &ViewTarget)>,
    temporal_cache: Res<TemporalCoherenceCache>,
) {
    let Some((view, target)) = views.iter().next() else {
        return;
    };
    
    // Use VIEWPORT size for cache texture, not the full render target size!
    // The cache stores the viewport content, and blit outputs to the viewport position.
    // This avoids wasting memory and ensures correct UV mapping.
    let target_size = target.main_texture_view().texture().size();
    let width = view.viewport.z.min(target_size.width.saturating_sub(view.viewport.x));
    let height = view.viewport.w.min(target_size.height.saturating_sub(view.viewport.y));
    
    // Skip if window is minimized or has zero size
    if width == 0 || height == 0 {
        return;
    }
    
    // BLIT ARCHITECTURE: Cache texture uses Rgba8Unorm format (NOT Rgba8UnormSrgb!)
    // 
    // Why Rgba8Unorm instead of Rgba8UnormSrgb:
    // - gaussian_splat.wgsl outputs sRGB colors directly (no conversion)
    // - Rgba8Unorm stores values WITHOUT conversion (correct for sRGB data)
    // - Blit shader does sRGB→linear conversion manually when sampling
    // 
    // If we used Rgba8UnormSrgb:
    // - GPU would assume input is linear, apply linear→sRGB on write
    // - This causes double-gamma: sRGB → (GPU thinks linear) → sRGB = too bright!
    let format = TextureFormat::Rgba8Unorm;
    
    // Check if viewport size changed (invalidates cache)
    let size_changed = render_cache.width != width || render_cache.height != height;
    
    // Create or update cache texture (always sample_count=1 for texture sampling)
    render_cache.ensure_texture(
        &render_device,
        width,
        height,
        format,
        Some(&blit_pipeline.bind_group_layout),
    );
    
    // Invalidate cache if data was updated or viewport size changed
    if temporal_cache.data_updated_this_frame || size_changed {
        render_cache.invalidate();
    } else if !render_cache.valid && temporal_cache.frame_count > 0 {
        // Mark cache as valid after rendering to it
        // The cache is rendered to when we don't skip (first frame or after data update)
        // Once rendered, it stays valid until next data update or resize
        render_cache.mark_valid();
    }
}

/// Prepare blit pipeline for cache-to-screen blitting
fn prepare_blit_pipeline(
    mut blit_pipeline: ResMut<CacheBlitPipeline>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(&ExtractedView, &Msaa, &ViewTarget)>,
) {
    let Some((view, msaa, target)) = views.iter().next() else {
        return;
    };
    
    // CRITICAL: Check actual render target size, not viewport
    let target_size = target.main_texture_view().texture().size();
    if target_size.width == 0 || target_size.height == 0 {
        return;
    }
    
    let msaa_samples = msaa.samples();
    blit_pipeline.get_pipeline(&pipeline_cache, view.hdr, msaa_samples);
}

/// Prepare pick render target (render world system)
fn prepare_pick_render_target(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    views: Query<&ExtractedView>,
    pick_target: Option<Res<PickRenderTarget>>,
    extracted_request: Res<ExtractedPickRequest>,
) {
    // Only create if we have an active pick request
    if !extracted_request.active {
        return;
    }
    
    let Some(view) = views.iter().next() else {
        return;
    };
    
    let width = view.viewport.z;
    let height = view.viewport.w;
    
    // Check if we need to recreate the target
    let needs_recreate = match &pick_target {
        Some(target) => target.width != width || target.height != height,
        None => true,
    };
    
    if needs_recreate {
        use bevy::render::render_resource::{TextureDescriptor, TextureUsages, TextureDimension, TextureFormat, Extent3d};
        
        // Create pick texture (RGBA8Unorm for splat index encoding)
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("pick_render_target"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        
        let view_tex = texture.create_view(&Default::default());
        
        // Create depth texture for pick pass
        let depth_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("pick_depth_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: CORE_3D_DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        
        let depth_view = depth_texture.create_view(&Default::default());
        
        // Create staging buffer for readback
        // We read the entire pick rect area
        let buffer_size = (width * height * 4) as u64; // RGBA8 = 4 bytes per pixel
        let staging_buffer = render_device.create_buffer(&bevy::render::render_resource::BufferDescriptor {
            label: Some("pick_staging_buffer"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        commands.insert_resource(PickRenderTarget {
            texture,
            view: view_tex,
            staging_buffer,
            depth_texture,
            depth_view,
            width,
            height,
            pick_active: extracted_request.active,
            pick_rect: extracted_request.rect,
            pick_op: extracted_request.op,
            target_entity: extracted_request.target_entity,
        });
    } else if let Some(target) = pick_target {
        // Update pick state by re-inserting with new values
        // (Can't mutate Res directly)
        commands.insert_resource(PickRenderTarget {
            texture: target.texture.clone(),
            view: target.view.clone(),
            staging_buffer: target.staging_buffer.clone(),
            depth_texture: target.depth_texture.clone(),
            depth_view: target.depth_view.clone(),
            width: target.width,
            height: target.height,
            pick_active: extracted_request.active,
            pick_rect: extracted_request.rect,
            pick_op: extracted_request.op,
            target_entity: extracted_request.target_entity,
        });
    }
}

/// Poll pick results from render world (main world system)
fn poll_pick_results(
    mut pick_result: ResMut<PickResult>,
    pending: Res<PickPendingReadback>,
) {
    // Check if results are ready
    if let Ok(mut data) = pending.data.try_lock() {
        if data.ready {
            // Decode splat indices from pixel data
            let mut splat_indices = std::collections::HashSet::new();
            
            for chunk in data.pixels.chunks_exact(4) {
                // Decode RGBA8 to u32 splat index
                let r = chunk[0] as u32;
                let g = chunk[1] as u32;
                let b = chunk[2] as u32;
                let a = chunk[3] as u32;
                
                let index = r | (g << 8) | (b << 16) | (a << 24);
                
                // Skip background (index 0 or very high values)
                if index > 0 && index < 0xFFFFFFFF {
                    splat_indices.insert(index);
                }
            }
            
            pick_result.ready = true;
            pick_result.splat_indices = splat_indices.into_iter().collect();
            pick_result.op = data.op;
            pick_result.target_entity = data.target_entity;
            
            // Clear pending data
            data.ready = false;
            data.pixels.clear();
        }
    }
}

/// Execute pick buffer readback (render world system)
fn execute_pick_readback(
    render_device: Res<RenderDevice>,
    _render_queue: Res<RenderQueue>,
    pick_target: Option<Res<PickRenderTarget>>,
    extracted_request: Res<ExtractedPickRequest>,
) {
    let Some(pick_target) = pick_target else {
        return;
    };
    
    if !pick_target.pick_active {
        return;
    }
    
    let pending_arc = match &extracted_request.pending_data {
        Some(arc) => arc.clone(),
        None => return,
    };
    
    let Some(rect) = pick_target.pick_rect else {
        return;
    };
    
    // Map the staging buffer and read pixels
    let buffer_slice = pick_target.staging_buffer.slice(..);
    let op = pick_target.pick_op;
    let target_entity = pick_target.target_entity;
    let rect_width = rect.width.min(pick_target.width - rect.x);
    let rect_height = rect.height.min(pick_target.height - rect.y);
    let full_width = pick_target.width;
    
    buffer_slice.map_async(bevy::render::render_resource::MapMode::Read, |_result| {
        // Callback - we handle data synchronously after poll
    });
    
    // Submit and wait for GPU
    let wgpu_device = render_device.wgpu_device();
    let _ = wgpu_device.poll(wgpu::PollType::Wait);
    
    // Read the mapped data
    let data = buffer_slice.get_mapped_range();
    let pixels: Vec<u8> = data.to_vec();
    drop(data);
    pick_target.staging_buffer.unmap();
    
    // Extract only the pick rect pixels
    let mut rect_pixels = Vec::with_capacity((rect_width * rect_height * 4) as usize);
    for y in 0..rect_height {
        let row_start = (y * full_width * 4) as usize;
        let row_end = row_start + (rect_width * 4) as usize;
        if row_end <= pixels.len() {
            rect_pixels.extend_from_slice(&pixels[row_start..row_end]);
        }
    }
    
    // Store results in pending data
    if let Ok(mut pending_lock) = pending_arc.lock() {
        pending_lock.pixels = rect_pixels;
        pending_lock.rect = Some(rect);
        pending_lock.op = op;
        pending_lock.target_entity = target_entity;
        pending_lock.ready = true;
    };
}

/// Apply pick results to splat selection state (main world system)
fn apply_pick_results(
    mut pick_result: ResMut<PickResult>,
    mut pick_request: ResMut<PickRequest>,
    mut splat_query: Query<(Entity, &mut SplatSelectionState)>,
) {
    if !pick_result.ready {
        return;
    }
    
    // Find target entity or use first available
    for (entity, mut state) in splat_query.iter_mut() {
        if let Some(target) = pick_result.target_entity {
            if entity != target {
                continue;
            }
        }
        
        // Apply selection based on operation mode
        // Filter valid indices
        let valid_indices: Vec<u32> = pick_result.splat_indices
            .iter()
            .filter(|&&idx| (idx as usize) < state.states.len())
            .copied()
            .collect();
        
        match pick_result.op {
            PickOp::Set => {
                state.deselect_all();
                state.select(&valid_indices);
            }
            PickOp::Add => {
                state.select(&valid_indices);
            }
            PickOp::Remove => {
                state.deselect(&valid_indices);
            }
        }
        
        state.recount();
        state.dirty = true;
        break;
    }
    
    // Clear results and request
    pick_result.ready = false;
    pick_result.splat_indices.clear();
    pick_request.active = false;
}
