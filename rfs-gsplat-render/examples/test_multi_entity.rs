//! Test multi-entity Gaussian Splatting rendering
//!
//! This example demonstrates that multiple GaussianSplats entities can be rendered simultaneously
//!
//! Usage:
//! ```bash
//! cargo run --example test_multi_entity
//! ```

use bevy::prelude::*;
use rfs_gsplat_render::{
    gaussian_point_cloud::GaussianPointCloudPlugin,
    gaussian_splats::GaussianSplats,
    RenderingConfig,
};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Multi-Entity Gaussian Splatting Test".to_string(),
                    resolution: (1280, 720).into(),
                    ..default()
                }),
                ..default()
            }),
            GaussianPointCloudPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (
            rotate_camera,
            keyboard_input,
        ))
        .run();
}

fn setup(mut commands: Commands) {
    println!("🚀 Testing Multi-Entity Gaussian Splatting Rendering");
    println!("========================================");
    println!("This test spawns 3 separate GaussianSplats entities:");
    println!("  1. Red cube   at (-2, 0, 0)");
    println!("  2. Green cube at ( 0, 0, 0)");
    println!("  3. Blue cube  at ( 2, 0, 0)");
    println!("========================================\n");

    // Spawn camera
    let camera_distance = 8.0;
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, camera_distance * 0.3, camera_distance)
            .looking_at(Vec3::ZERO, Vec3::Y),
        CameraController {
            distance: camera_distance,
            rotation_speed: 0.3,
            yaw: 0.0,
            pitch: 0.2,
        },
    ));

    // Create three separate entities with different positions and colors
    println!("📦 Creating Entity 1: Red Cube (10,000 splats)");
    let splats1 = create_colored_test_splats(10_000, Vec3::new(1.0, 0.0, 0.0));
    commands.spawn((
        Name::new("Red Cube"),
        splats1,
        Transform::from_translation(Vec3::new(-2.5, 0.0, 0.0)),
        GlobalTransform::default(),
        Visibility::default(),
        RenderingConfig::default(),
    ));

    println!("📦 Creating Entity 2: Green Cube (10,000 splats)");
    let splats2 = create_colored_test_splats(10_000, Vec3::new(0.0, 1.0, 0.0));
    commands.spawn((
        Name::new("Green Cube"),
        splats2,
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        GlobalTransform::default(),
        Visibility::default(),
        RenderingConfig::default(),
    ));

    println!("📦 Creating Entity 3: Blue Cube (10,000 splats)");
    let splats3 = create_colored_test_splats(10_000, Vec3::new(0.0, 0.0, 1.0));
    commands.spawn((
        Name::new("Blue Cube"),
        splats3,
        Transform::from_translation(Vec3::new(2.5, 0.0, 0.0)),
        GlobalTransform::default(),
        Visibility::default(),
        RenderingConfig::default(),
    ));

    println!("✅ All entities created! Total: 30,000 splats\n");
    println!("=== Camera Controls ===");
    println!("  Arrow Keys  - Rotate camera");
    println!("  ESC         - Exit");
}

/// Create test splats with a specific color
fn create_colored_test_splats(count: usize, base_color: Vec3) -> GaussianSplats {
    use rfs_gsplat_render::gaussian_splats::inverse_sigmoid;
    
    let mut means = Vec::with_capacity(count);
    let mut rotations = Vec::with_capacity(count);
    let mut log_scales = Vec::with_capacity(count);
    let mut sh_coeffs = Vec::with_capacity(count);
    let mut raw_opacities = Vec::with_capacity(count);
    
    // Create a dense 3D grid of splats
    let grid_size = (count as f32).cbrt() as usize;
    let spacing = 0.08; // Spacing between points
    let offset = -(grid_size as f32 * spacing) / 2.0;
    
    for i in 0..count {
        let x = (i % grid_size) as f32 * spacing + offset;
        let y = ((i / grid_size) % grid_size) as f32 * spacing + offset;
        let z = (i / (grid_size * grid_size)) as f32 * spacing + offset;
        
        means.push(Vec3::new(x, y, z));
        
        // Simple rotation
        let angle = (x + y + z) * 0.5;
        let axis = Vec3::new(1.0, 1.0, 1.0).normalize();
        let s = (angle * 0.5).sin();
        let c = (angle * 0.5).cos();
        rotations.push(Vec4::new(c, axis.x * s, axis.y * s, axis.z * s));
        
        // Small scale: e^(-4.5) ≈ 0.011
        log_scales.push(Vec3::new(-4.5, -4.5, -4.5));
        
        // Use the base color with slight variation based on position
        let color_variation = Vec3::new(
            (x - offset) / (grid_size as f32 * spacing) * 0.3,
            (y - offset) / (grid_size as f32 * spacing) * 0.3,
            (z - offset) / (grid_size as f32 * spacing) * 0.3,
        );
        let final_color = (base_color * 0.7 + color_variation).clamp(Vec3::ZERO, Vec3::ONE);
        sh_coeffs.push(vec![final_color]);
        
        raw_opacities.push(inverse_sigmoid(0.9)); // High opacity
    }
    
    GaussianSplats::new(means, rotations, log_scales, sh_coeffs, raw_opacities)
}

// Camera controller component
#[derive(Component)]
struct CameraController {
    distance: f32,
    rotation_speed: f32,
    yaw: f32,
    pitch: f32,
}

// Rotate camera system
fn rotate_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_q: Query<(&mut Transform, &mut CameraController)>,
) {
    let delta = time.delta_secs();
    
    for (mut transform, mut controller) in &mut camera_q {
        let mut changed = false;
        
        // Manual rotation
        let rotation_speed = 2.0;
        if keyboard.pressed(KeyCode::ArrowLeft) {
            controller.yaw += rotation_speed * delta;
            changed = true;
        }
        if keyboard.pressed(KeyCode::ArrowRight) {
            controller.yaw -= rotation_speed * delta;
            changed = true;
        }
        if keyboard.pressed(KeyCode::ArrowUp) {
            controller.pitch = (controller.pitch + rotation_speed * delta).min(std::f32::consts::FRAC_PI_2 - 0.1);
            changed = true;
        }
        if keyboard.pressed(KeyCode::ArrowDown) {
            controller.pitch = (controller.pitch - rotation_speed * delta).max(-std::f32::consts::FRAC_PI_2 + 0.1);
            changed = true;
        }
        
        // Update camera position if changed
        if changed {
            let horizontal_dist = controller.distance * controller.pitch.cos();
            transform.translation.x = horizontal_dist * controller.yaw.sin();
            transform.translation.z = horizontal_dist * controller.yaw.cos();
            transform.translation.y = controller.distance * controller.pitch.sin();
            transform.look_at(Vec3::ZERO, Vec3::Y);
        }
    }
}

// Handle keyboard input
fn keyboard_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut exit: MessageWriter<bevy::app::AppExit>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        println!("Exiting...");
        exit.write(bevy::app::AppExit::Success);
    }
}

