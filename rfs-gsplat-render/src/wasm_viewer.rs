//! WebAssembly Viewer Module
//! 
//! This module provides a Bevy-based WebGL2 viewer for Gaussian Splats
//! that can be compiled to WebAssembly and run in browsers.
//!
//! Build with: `wasm-pack build --target web --features wasm-viewer --no-default-features`

#![cfg(all(feature = "wasm-viewer", target_arch = "wasm32"))]

use wasm_bindgen::prelude::*;
use bevy::prelude::*;
use glam::{Vec3, Vec4};

use crate::gaussian_splats::GaussianSplats;
use crate::gaussian_point_cloud::{GaussianPointCloudPlugin, RenderingConfig};

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init_wasm() {
    console_error_panic_hook::set_once();
}

/// Orbit camera controller component
#[derive(Component)]
pub struct OrbitCameraController {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub sensitivity: f32,
}

impl Default for OrbitCameraController {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.0,
            sensitivity: 0.005,
        }
    }
}

impl OrbitCameraController {
    pub fn position(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }
}

/// Splat data container for WebAssembly (used to pass data from JS)
#[wasm_bindgen]
pub struct SplatData {
    positions: Vec<f32>,
    colors: Vec<f32>,
    scales: Vec<f32>,
    rotations: Vec<f32>,
    count: u32,
}

#[wasm_bindgen]
impl SplatData {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        SplatData {
            positions: Vec::new(),
            colors: Vec::new(),
            scales: Vec::new(),
            rotations: Vec::new(),
            count: 0,
        }
    }
    
    /// Load from simple binary format
    /// Format: [count: u32][positions: f32 * 3 * count][colors: f32 * 4 * count][scales: f32 * 3 * count][rotations: f32 * 4 * count]
    pub fn load_from_binary(data: &[u8]) -> Result<SplatData, JsValue> {
        if data.len() < 4 {
            return Err("Data too short".into());
        }
        
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let expected_size = 4 + count * (3 + 4 + 3 + 4) * 4;
        
        if data.len() < expected_size {
            return Err(format!("Data size mismatch: expected {}, got {}", expected_size, data.len()).into());
        }
        
        let float_data: &[f32] = bytemuck::cast_slice(&data[4..]);
        
        let mut offset = 0;
        let positions: Vec<f32> = float_data[offset..offset + count * 3].to_vec();
        offset += count * 3;
        
        let colors: Vec<f32> = float_data[offset..offset + count * 4].to_vec();
        offset += count * 4;
        
        let scales: Vec<f32> = float_data[offset..offset + count * 3].to_vec();
        offset += count * 3;
        
        let rotations: Vec<f32> = float_data[offset..offset + count * 4].to_vec();
        
        Ok(SplatData {
            positions,
            colors,
            scales,
            rotations,
            count: count as u32,
        })
    }
    
    /// Load from SOG (Splat Optimized Gaussian) format
    /// SOG is a compressed format using WebP images for efficient web loading
    pub fn load_from_sog(data: &[u8]) -> Result<SplatData, JsValue> {
        // SOG is a ZIP file, load it using tinygsplat_io
        let splats = tinygsplat_io::load_sog_from_memory(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to load SOG: {}", e)))?;
        
        let count = splats.num_splats();
        info!("Loaded {} splats from SOG format", count);
        
        // Convert to SplatData format
        let mut positions = Vec::with_capacity(count * 3);
        let mut colors = Vec::with_capacity(count * 4);
        let mut scales = Vec::with_capacity(count * 3);
        let mut rotations = Vec::with_capacity(count * 4);
        
        for i in 0..count {
            // Position
            positions.push(splats.means[i].x);
            positions.push(splats.means[i].y);
            positions.push(splats.means[i].z);
            
            // Color from SH0
            let sh0 = if !splats.sh_coeffs[i].is_empty() {
                splats.sh_coeffs[i][0]
            } else {
                Vec3::ZERO
            };
            let sh_c0 = 0.28209479177387814_f32;
            let r = (sh0.x * sh_c0 + 0.5).clamp(0.0, 1.0);
            let g = (sh0.y * sh_c0 + 0.5).clamp(0.0, 1.0);
            let b = (sh0.z * sh_c0 + 0.5).clamp(0.0, 1.0);
            
            // Opacity (sigmoid of raw)
            let raw_opacity = splats.raw_opacities[i];
            let opacity = 1.0 / (1.0 + (-raw_opacity).exp());
            
            colors.push(r);
            colors.push(g);
            colors.push(b);
            colors.push(opacity);
            
            // Scale (exp of log_scale)
            scales.push(splats.log_scales[i].x.exp());
            scales.push(splats.log_scales[i].y.exp());
            scales.push(splats.log_scales[i].z.exp());
            
            // Rotation quaternion
            rotations.push(splats.rotations[i].x);
            rotations.push(splats.rotations[i].y);
            rotations.push(splats.rotations[i].z);
            rotations.push(splats.rotations[i].w);
        }
        
        Ok(SplatData {
            positions,
            colors,
            scales,
            rotations,
            count: count as u32,
        })
    }
    
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> u32 {
        self.count
    }
}

impl SplatData {
    /// Convert to GaussianSplats for Bevy rendering (internal, not exposed to JS)
    pub fn to_gaussian_splats(&self) -> GaussianSplats {
        let count = self.count as usize;
        
        let mut means = Vec::with_capacity(count);
        let mut sh_coeffs_all: Vec<Vec<Vec3>> = Vec::with_capacity(count);
        let mut log_scales = Vec::with_capacity(count);
        let mut rotations = Vec::with_capacity(count);
        let mut raw_opacities = Vec::with_capacity(count);
        
        for i in 0..count {
            // Position
            means.push(Vec3::new(
                self.positions[i * 3],
                self.positions[i * 3 + 1],
                self.positions[i * 3 + 2],
            ));
            
            // Color -> SH0 coefficient (reverse of export)
            let r = self.colors[i * 4];
            let g = self.colors[i * 4 + 1];
            let b = self.colors[i * 4 + 2];
            let a = self.colors[i * 4 + 3];
            
            let sh_c0 = 0.28209479177387814_f32;
            let sh0_r = (r - 0.5) / sh_c0;
            let sh0_g = (g - 0.5) / sh_c0;
            let sh0_b = (b - 0.5) / sh_c0;
            
            // SH coefficients for this splat: 15 Vec3 coefficients
            // First one is SH0, rest are zeros for higher order
            let mut sh_for_splat: Vec<Vec3> = Vec::with_capacity(15);
            sh_for_splat.push(Vec3::new(sh0_r, sh0_g, sh0_b)); // SH0
            for _ in 1..15 {
                sh_for_splat.push(Vec3::ZERO); // Higher order SH coefficients
            }
            sh_coeffs_all.push(sh_for_splat);
            
            // Scale (convert from linear to log)
            let sx = self.scales[i * 3].max(0.0001);
            let sy = self.scales[i * 3 + 1].max(0.0001);
            let sz = self.scales[i * 3 + 2].max(0.0001);
            log_scales.push(Vec3::new(sx.ln(), sy.ln(), sz.ln()));
            
            // Rotation (quaternion)
            rotations.push(Vec4::new(
                self.rotations[i * 4],
                self.rotations[i * 4 + 1],
                self.rotations[i * 4 + 2],
                self.rotations[i * 4 + 3],
            ));
            
            // Opacity (convert from sigmoid to raw)
            let opacity = a.clamp(0.001, 0.999);
            let raw_opacity = (opacity / (1.0 - opacity)).ln();
            raw_opacities.push(raw_opacity);
        }
        
        GaussianSplats::new(means, rotations, log_scales, sh_coeffs_all, raw_opacities)
    }
}

impl Default for SplatData {
    fn default() -> Self {
        Self::new()
    }
}

/// WebViewer - Bevy-based Gaussian Splat viewer for browsers
#[wasm_bindgen]
pub struct WebViewer {
    // App will be started separately
}

#[wasm_bindgen]
impl WebViewer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        WebViewer {}
    }
    
    /// Start the Bevy app with the given splat data
    pub fn start(&self, splat_data: SplatData) {
        let gaussian_splats = splat_data.to_gaussian_splats();
        
        App::new()
            .add_plugins(DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    canvas: Some("#canvas".into()),
                    fit_canvas_to_parent: true,
                    prevent_default_event_handling: true,
                    ..default()
                }),
                ..default()
            }))
            .add_plugins(GaussianPointCloudPlugin)
            .add_plugins(crate::EmbeddedShadersPlugin)
            .insert_resource(SplatsToSpawn(Some(gaussian_splats)))
            .add_systems(Startup, setup_scene)
            .add_systems(Update, (orbit_camera_system, spawn_splats_system))
            .run();
    }
}

impl Default for WebViewer {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource to hold splats that need to be spawned
#[derive(Resource, Default)]
struct SplatsToSpawn(Option<GaussianSplats>);

/// Setup the 3D scene
fn setup_scene(mut commands: Commands) {
    // Camera with orbit controller
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCameraController::default(),
    ));
    
    // Ambient light
    commands.spawn((
        AmbientLight {
            color: Color::WHITE,
            brightness: 300.0,
            ..default()
        },
    ));
}

/// Spawn splats when ready
fn spawn_splats_system(
    mut commands: Commands,
    mut splats_resource: ResMut<SplatsToSpawn>,
) {
    if let Some(splats) = splats_resource.0.take() {
        info!("Spawning {} Gaussian splats", splats.num_splats());
        commands.spawn((
            splats,
            Transform::default(),
            GlobalTransform::default(),
            RenderingConfig::default(),
        ));
    }
}

/// Orbit camera control system
fn orbit_camera_system(
    mut query: Query<(&mut Transform, &mut OrbitCameraController)>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<bevy::input::mouse::MouseMotion>,
    mut scroll_events: MessageReader<bevy::input::mouse::MouseWheel>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    let (mut transform, mut controller) = match query.iter_mut().next() {
        Some(q) => q,
        None => return,
    };
    
    // Reset on R key
    if keyboard.just_pressed(KeyCode::KeyR) {
        *controller = OrbitCameraController::default();
    }
    
    // Rotation (left mouse)
    if mouse_input.pressed(MouseButton::Left) {
        for event in mouse_motion.read() {
            controller.yaw -= event.delta.x * controller.sensitivity;
            controller.pitch -= event.delta.y * controller.sensitivity;
            controller.pitch = controller.pitch.clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );
        }
    }
    
    // Pan (right mouse)
    if mouse_input.pressed(MouseButton::Right) {
        for event in mouse_motion.read() {
            let forward = (controller.target - controller.position()).normalize();
            let right = forward.cross(Vec3::Y).normalize();
            let up = right.cross(forward).normalize();
            
            let pan_speed = controller.distance * 0.002;
            controller.target -= right * event.delta.x * pan_speed;
            controller.target += up * event.delta.y * pan_speed;
        }
    }
    
    // Zoom (scroll wheel)
    for event in scroll_events.read() {
        controller.distance *= 1.0 - event.y * 0.1;
        controller.distance = controller.distance.clamp(0.1, 100.0);
    }
    
    // Update camera transform
    let position = controller.position();
    *transform = Transform::from_translation(position).looking_at(controller.target, Vec3::Y);
}

/// Get library version
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
