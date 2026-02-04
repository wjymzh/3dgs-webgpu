// 3D Gaussian Splatting Rendering for Bevy
// Ported from brush project with adaptations for Bevy 0.17

#![allow(dead_code)]

pub mod gaussian_splats;
pub mod gaussian_point_cloud;
pub mod radix_sort;
pub mod splat_state;
pub mod gpu_picker;
pub mod temporal_coherence;
pub mod outline;
pub mod training_preview;

// Native-only modules (require tinygsplat_io with compression libraries)
#[cfg(feature = "native")]
pub mod loader;

// WebAssembly viewer support (optional feature)
#[cfg(feature = "wasm-viewer")]
pub mod wasm_viewer;

// Old version modules (temporarily disabled, has compilation errors)
// pub mod render_node;
// pub mod point_cloud;

// Re-exports - core types always available
pub use gaussian_splats::{GaussianSplats, create_test_splats, PackModeConfig, inverse_sigmoid, sigmoid, SplatSelectionState};
pub use gaussian_point_cloud::{GaussianPointCloudPlugin, GaussianSplatParams, PointSizeConfig, CullingConfig, RenderingConfig, SplatVisMode, SplatEditingColorConfig, BuffersNeedUpdate, TrainingMode};
pub use splat_state::{SelectionOp, SelectionMode, RectParams, SphereParams, BoxParams, state_bits};
pub use gpu_picker::{GpuPickerPlugin, PickerRequest, PickerResult};
pub use temporal_coherence::{TemporalCoherenceCache, TemporalCoherenceConfig, TemporalCoherenceStats, GaussianSplatRenderCache, should_skip_sorting, print_temporal_coherence_stats};
pub use outline::{OutlineConfig, OutlinePlugin};
pub use training_preview::{TrainingPreviewPlugin, TrainingPreviewImageData, TrainingPreviewRenderTarget, TrainingPreviewBlitPipeline, get_training_preview_blit_resources};

// Native-only re-exports (file I/O functions)
#[cfg(feature = "native")]
pub use loader::{
    load_ply_file, load_splat_file, load_gaussian_file, load_spz_file, load_compress_ply_file, load_sog_file,
    save_splat_file, save_ply_file, save_compress_ply_file, save_spz_file, save_sog_file, save_sog_to_memory,
};

// Embed shaders into the binary to protect IP and simplify deployment
use bevy::asset::embedded_asset;

pub struct EmbeddedShadersPlugin;

impl bevy::app::Plugin for EmbeddedShadersPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        // Embed all shader files into the binary using embedded_asset!
        // This is the same approach used by Bevy's skybox and core pipeline
        // embedded_asset!(app, "../assets/shaders/threedgs_covariance.wgsl");
        // embedded_asset!(app, "../assets/shaders/spherical_harmonics.wgsl");
        // embedded_asset!(app, "../assets/shaders/gaussian_common.wgsl");
        embedded_asset!(app, "../assets/shaders/radix_sort.wgsl");
        embedded_asset!(app, "../assets/shaders/gaussian_splat.wgsl");
        embedded_asset!(app, "../assets/shaders/gaussian_splat_cull.wgsl");
        embedded_asset!(app, "../assets/shaders/selection_compute.wgsl");
        embedded_asset!(app, "../assets/shaders/outline.wgsl");
        // UNIFIED BLIT SHADER: cache_blit.wgsl is used for both 3DGS cache and training preview
        embedded_asset!(app, "../assets/shaders/cache_blit.wgsl");
        // Note: training_preview_blit.wgsl is deprecated - cache_blit.wgsl is used instead
    }
}
