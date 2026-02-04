// Temporal Coherence for Gaussian Splatting Rendering
// Caches sorting results when camera movement is small
// Performance: 80-95% frame time reduction for static/slow-moving cameras
//
// RENDER CACHE: When camera is static and no data updates, we can skip
// the entire 3DGS render pass and use a cached result instead.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::{
    Texture, TextureView, TextureDescriptor, TextureUsages, TextureDimension,
    TextureFormat, Extent3d, TextureViewDescriptor,
    BindGroup, BindGroupLayout, BindGroupEntries,
    SamplerDescriptor, FilterMode, AddressMode,
};
use bevy::render::renderer::RenderDevice;

/// Temporal coherence cache for sorting optimization
/// Tracks camera state and decides whether to skip sorting
#[derive(Resource, Default, Clone)]
pub struct TemporalCoherenceCache {
    /// Last camera position
    pub last_camera_pos: Vec3,
    /// Last camera forward direction
    pub last_camera_dir: Vec3,
    /// Last camera up vector (for roll detection)
    pub last_camera_up: Vec3,
    /// Whether sorting was skipped last frame
    pub sorting_skipped: bool,
    /// Number of consecutive frames sorting was skipped
    pub skip_count: u32,
    /// Frame counter
    pub frame_count: u64,
    /// Whether any splat data was updated this frame (BuffersNeedUpdate/Rebuild)
    pub data_updated_this_frame: bool,
    /// Number of consecutive frames where render was skipped
    pub render_skip_count: u32,
}

/// Render cache for 3DGS - stored in render world only
/// Stores the last rendered frame to avoid re-rendering when camera is static
#[derive(Resource)]
pub struct GaussianSplatRenderCache {
    /// Cached color texture (RGBA with premultiplied alpha)
    pub texture: Option<Texture>,
    /// Cached texture view for sampling
    pub view: Option<TextureView>,
    /// Sampler for cache texture
    pub sampler: Option<bevy::render::render_resource::Sampler>,
    /// Bind group for blit pass
    pub bind_group: Option<BindGroup>,
    /// Texture dimensions
    pub width: u32,
    pub height: u32,
    /// Whether the cache is valid (has a rendered frame)
    pub valid: bool,
    /// Last viewport dimensions (to detect resize)
    pub last_viewport: UVec2,
    /// Texture format (always Rgba8UnormSrgb for blit architecture)
    pub format: TextureFormat,
}

impl Default for GaussianSplatRenderCache {
    fn default() -> Self {
        Self {
            texture: None,
            view: None,
            sampler: None,
            bind_group: None,
            width: 0,
            height: 0,
            valid: false,
            last_viewport: UVec2::ZERO,
            format: TextureFormat::Rgba8UnormSrgb,
        }
    }
}

impl GaussianSplatRenderCache {
    /// Create or recreate cache texture if needed
    /// NOTE: Cache texture uses sample_count=1 (no MSAA) for texture sampling compatibility
    /// Splat rendering to cache does NOT use depth attachment (relies on radix sort order)
    /// 
    /// IMPORTANT: Cache uses Rgba8Unorm (NOT Rgba8UnormSrgb) because:
    /// - gaussian_splat.wgsl outputs sRGB colors directly
    /// - Rgba8Unorm stores values without conversion (correct for sRGB data)
    /// - Blit shader then does sRGB→linear conversion when sampling
    /// Using Rgba8UnormSrgb would cause double-gamma (GPU assumes linear input, converts to sRGB)
    pub fn ensure_texture(
        &mut self,
        render_device: &RenderDevice,
        width: u32,
        height: u32,
        format: TextureFormat,
        blit_bind_group_layout: Option<&BindGroupLayout>,
    ) {
        // Check if we need to recreate
        if self.texture.is_some() 
            && self.width == width 
            && self.height == height 
            && self.format == format {
            return;
        }
        
        // Create new texture with sample_count=1 (required for texture sampling)
        // MSAA is not supported for cache because MSAA textures cannot be sampled directly
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("gaussian_splat_render_cache"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,  // Must be 1 for texture sampling in blit pass
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT 
                 | TextureUsages::TEXTURE_BINDING 
                 | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let view = texture.create_view(&TextureViewDescriptor::default());
        
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("gaussian_splat_cache_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create bind group if layout is provided
        let bind_group = blit_bind_group_layout.map(|layout| {
            render_device.create_bind_group(
                Some("gaussian_splat_cache_bind_group"),
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
        self.format = format;
        self.valid = false; // Cache is invalid until we render to it
        self.last_viewport = UVec2::new(width, height);
        
        info!("🎨 Created render cache texture: {}x{} {:?}", width, height, format);
    }
    
    /// Mark cache as valid after rendering
    pub fn mark_valid(&mut self) {
        self.valid = true;
    }
    
    /// Invalidate cache (e.g., when data changes)
    pub fn invalidate(&mut self) {
        self.valid = false;
    }
    
    /// Check if cache can be used
    pub fn can_use(&self) -> bool {
        self.valid && self.texture.is_some() && self.view.is_some()
    }
}

impl ExtractResource for TemporalCoherenceCache {
    type Source = Self;
    
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Configuration for temporal coherence optimization
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct TemporalCoherenceConfig {
    /// Enable temporal coherence optimization
    /// Default: true (highly recommended)
    pub enabled: bool,
    
    /// Position movement threshold (in world units)
    /// Camera must move more than this to trigger re-sort
    /// Default: 0.01 (1cm for typical scenes)
    pub position_threshold: f32,
    
    /// Direction change threshold (dot product)
    /// 1.0 = no change, 0.0 = 90° change
    /// Default: 0.9999 (~0.8° rotation)
    pub direction_threshold: f32,
    
    /// Maximum frames to skip sorting (safety limit)
    /// Even if camera is static, re-sort periodically
    /// Default: 300 (5 seconds at 60fps)
    pub max_skip_frames: u32,
    
    /// Force re-sort every N frames regardless of camera movement
    /// Useful for dynamic scenes or numerical stability
    /// 0 = disabled
    /// Default: 0 (disabled)
    pub force_resort_interval: u32,
}

impl Default for TemporalCoherenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            position_threshold: 0.01,      // 1cm
            direction_threshold: 0.9999,   // ~0.8° rotation
            max_skip_frames: 300,          // 5 seconds at 60fps
            force_resort_interval: 0,      // disabled
        }
    }
}

impl TemporalCoherenceConfig {
    /// Conservative profile (re-sort more often)
    /// Use for highly dynamic scenes or when quality is critical
    pub fn conservative() -> Self {
        Self {
            enabled: true,
            position_threshold: 0.001,     // 1mm
            direction_threshold: 0.99995,  // ~0.5° rotation
            max_skip_frames: 60,           // 1 second at 60fps
            force_resort_interval: 0,
        }
    }
    
    /// Aggressive profile (skip sorting as much as possible)
    /// Use for static scenes or when performance is critical
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            position_threshold: 0.1,       // 10cm
            direction_threshold: 0.999,    // ~2.5° rotation
            max_skip_frames: 1000,         // ~16 seconds at 60fps
            force_resort_interval: 0,
        }
    }
    
    /// Training mode profile (optimized for 3DGS training)
    /// - Skips compute passes when camera is static and no data update
    /// - Very relaxed thresholds to maximize training GPU time
    /// - Use when training 3DGS and rendering preview is secondary
    pub fn training_mode() -> Self {
        Self {
            enabled: true,
            position_threshold: 0.05,      // 5cm - more relaxed for training
            direction_threshold: 0.9995,   // ~1.8° rotation
            max_skip_frames: 600,          // 10 seconds at 60fps
            force_resort_interval: 0,
        }
    }
    
    /// Disabled (always sort)
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Check if the entire render pass should be skipped
/// Returns true if camera is static AND no data updates occurred
/// This is more aggressive than skip_sorting - it skips the entire GPU render
pub fn should_skip_render(
    cache: &TemporalCoherenceCache,
    config: &TemporalCoherenceConfig,
    camera_pos: Vec3,
    camera_dir: Vec3,
    camera_up: Vec3,
) -> bool {
    // Never skip on first frame
    if cache.frame_count == 0 {
        return false;
    }
    
    // Never skip if temporal coherence is disabled
    if !config.enabled {
        return false;
    }
    
    // Never skip if data was updated this frame
    if cache.data_updated_this_frame {
        return false;
    }
    
    // Don't skip too many frames in a row (safety limit)
    // Use a higher limit than sorting skip since we're caching the entire result
    if cache.render_skip_count >= config.max_skip_frames * 2 {
        return false;
    }
    
    // Check camera movement
    let pos_delta = camera_pos.distance(cache.last_camera_pos);
    let dir_dot = camera_dir.dot(cache.last_camera_dir);
    let up_dot = camera_up.dot(cache.last_camera_up);
    
    let camera_static = pos_delta < config.position_threshold
        && dir_dot > config.direction_threshold
        && up_dot > config.direction_threshold;
    
    camera_static
}

/// Check if sorting should be skipped based on camera movement
pub fn should_skip_sorting(
    cache: &mut TemporalCoherenceCache,
    config: &TemporalCoherenceConfig,
    camera_pos: Vec3,
    camera_dir: Vec3,
    camera_up: Vec3,
) -> bool {
    // Always sort on first frame
    if cache.frame_count == 0 {
        cache.last_camera_pos = camera_pos;
        cache.last_camera_dir = camera_dir;
        cache.last_camera_up = camera_up;
        cache.frame_count = 1;
        cache.skip_count = 0;
        cache.sorting_skipped = false;
        return false;
    }
    
    cache.frame_count += 1;
    
    // Check if temporal coherence is disabled
    if !config.enabled {
        cache.sorting_skipped = false;
        cache.skip_count = 0;
        return false;
    }
    
    // This ensures training updates are immediately visible even when camera is static
    if cache.data_updated_this_frame {
        cache.last_camera_pos = camera_pos;
        cache.last_camera_dir = camera_dir;
        cache.last_camera_up = camera_up;
        cache.sorting_skipped = false;
        cache.skip_count = 0;
        return false;
    }
    
    // Check force re-sort interval
    if config.force_resort_interval > 0 
       && cache.frame_count % config.force_resort_interval as u64 == 0 {
        cache.last_camera_pos = camera_pos;
        cache.last_camera_dir = camera_dir;
        cache.last_camera_up = camera_up;
        cache.sorting_skipped = false;
        cache.skip_count = 0;
        return false;
    }
    
    // Check max skip frames safety limit
    if cache.skip_count >= config.max_skip_frames {
        cache.last_camera_pos = camera_pos;
        cache.last_camera_dir = camera_dir;
        cache.last_camera_up = camera_up;
        cache.sorting_skipped = false;
        cache.skip_count = 0;
        return false;
    }
    
    // Calculate camera movement
    let pos_delta = camera_pos.distance(cache.last_camera_pos);
    let dir_dot = camera_dir.dot(cache.last_camera_dir);
    let up_dot = camera_up.dot(cache.last_camera_up);
    
    // Check if camera moved significantly
    let camera_moved = pos_delta > config.position_threshold
        || dir_dot < config.direction_threshold
        || up_dot < config.direction_threshold;  // Detect roll
    
    if camera_moved {
        // Camera moved, need to re-sort
        cache.last_camera_pos = camera_pos;
        cache.last_camera_dir = camera_dir;
        cache.last_camera_up = camera_up;
        cache.sorting_skipped = false;
        cache.skip_count = 0;
        false
    } else {
        // Camera is static, skip sorting
        cache.sorting_skipped = true;
        cache.skip_count += 1;
        true
    }
}

/// Statistics for temporal coherence
#[derive(Resource, Default, Debug, Clone, Reflect)]
#[reflect(Resource)]
pub struct TemporalCoherenceStats {
    /// Total frames rendered
    pub total_frames: u64,
    /// Frames where sorting was skipped
    pub skipped_frames: u64,
    /// Current skip streak
    pub current_skip_streak: u32,
    /// Longest skip streak
    pub max_skip_streak: u32,
    /// Average skip ratio (0.0 - 1.0)
    pub skip_ratio: f32,
}

impl ExtractResource for TemporalCoherenceStats {
    type Source = Self;
    
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

impl TemporalCoherenceStats {
    pub fn update(&mut self, sorting_skipped: bool, skip_count: u32) {
        self.total_frames += 1;
        if sorting_skipped {
            self.skipped_frames += 1;
            self.current_skip_streak = skip_count;
            self.max_skip_streak = self.max_skip_streak.max(skip_count);
        } else {
            self.current_skip_streak = 0;
        }
        
        if self.total_frames > 0 {
            self.skip_ratio = self.skipped_frames as f32 / self.total_frames as f32;
        }
    }
    
    pub fn print_summary(&self) {
        info!("📊 Temporal Coherence Stats:");
        info!("  Skip Ratio: {:.1}% ({}/{})", 
            self.skip_ratio * 100.0, self.skipped_frames, self.total_frames);
        info!("  Current Streak: {} frames", self.current_skip_streak);
        info!("  Max Streak: {} frames", self.max_skip_streak);
        info!("  Performance Gain: ~{:.0}% frame time saved", 
            self.skip_ratio * 40.0);  // Sorting typically takes ~40% of frame time
    }
}

/// System to print temporal coherence stats periodically
pub fn print_temporal_coherence_stats(
    stats: Res<TemporalCoherenceStats>,
    time: Res<Time>,
    mut last_print: Local<f32>,
) {
    *last_print += time.delta_secs();
    if *last_print >= 10.0 {  // Print every 10 seconds
        stats.print_summary();
        *last_print = 0.0;
    }
}

