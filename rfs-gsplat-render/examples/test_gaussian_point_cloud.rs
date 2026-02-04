//! Test Gaussian Point Cloud rendering plugin
//!
//! This example demonstrates how to use GaussianPointCloudPlugin to render 3D Gaussian Splatting data
//!
//! Usage:
//! ```bash
//! cargo run --example test_gaussian_point_cloud
//! ```

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures::FutureExt;
use rfs_gsplat_render::{
    gaussian_point_cloud::GaussianPointCloudPlugin,
    gaussian_splats::{create_test_splats, GaussianSplats},
    loader::load_ply_file,
    RenderingConfig,
};
use std::path::PathBuf;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Gaussian Point Cloud Rendering Test".to_string(),
                    resolution: (1080, 720).into(),
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
            adjust_point_size, 
            adjust_culling_params, 
            check_loading_task,
            update_loading_ui,
        ))
        .run();
}

// Loading state components
#[derive(Component)]
struct LoadingTask(Task<Result<GaussianSplats, String>>);

#[derive(Component)]
struct LoadingUI;

#[derive(Resource)]
struct LoadingState {
    start_time: std::time::Instant,
}

fn setup(mut commands: Commands) {
    println!("🚀 Starting Gaussian Splatting Viewer...");
    println!("⏳ Window will appear immediately, data loading in background...");

    // Spawn camera FIRST - window will show immediately
    let default_distance = 5.0;
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(default_distance, default_distance * 0.3, default_distance)
            .looking_at(Vec3::ZERO, Vec3::Y),
        CameraController {
            distance: default_distance,
            height: 0.0,
            rotation_speed: 0.3,
            auto_rotate: true,
            yaw: 0.0,
            pitch: 0.15,
        },
    ));

    // Spawn loading UI text
    commands.spawn((
        Text::new("⏳ Loading Gaussian Splats...\nPlease wait..."),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(20.0),
            top: Val::Px(20.0),
            ..default()
        },
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 0.0)),
        LoadingUI,
    ));

    // Start async loading task
    let ply_path = PathBuf::from(r"D:\ScanVideo\bike\flowers_1.ply");
    let thread_pool = AsyncComputeTaskPool::get();
    
    let task = thread_pool.spawn(async move {
        println!("🔄 Background: Starting PLY file loading...");
        let start = std::time::Instant::now();
        
        // Try to load PLY file
        let result = load_ply_file(&ply_path);
        
        match result {
            Ok(splats) => {
                let elapsed = start.elapsed();
                println!("✅ Background: Successfully loaded {} points in {:.2}s", 
                         splats.len(), elapsed.as_secs_f32());
                Ok(splats)
            }
            Err(e) => {
                println!("⚠️  Background: Failed to load PLY: {}", e);
                println!("🔄 Background: Generating test data instead...");
                let splats = create_test_splats(100_000);
                println!("✅ Background: Created {} test points", splats.len());
                Ok(splats)
            }
        }
    });

    commands.spawn(LoadingTask(task));
    commands.insert_resource(LoadingState {
        start_time: std::time::Instant::now(),
    });

    println!("✅ Window initialized! Loading data in background...");
    println!("\n=== Camera Controls ===");
    println!("  SPACE       - Pause/resume auto rotation");
    println!("  Arrow Keys  - Rotate camera manually");
    println!("  W/S or ↑/↓  - Move closer/farther");
    println!("  A/D or ←/→  - Rotate left/right");
    println!("  Q/E         - Move up/down");
    println!("  Mouse Wheel - Zoom in/out");
    println!("\n=== Render Controls ===");
    println!("  +/- or =/−  - Adjust point size");
    println!("  R           - Reset point size");
    println!("  1/2         - Adjust frustum culling");
    println!("  3/4         - Adjust alpha threshold");
    println!("  ESC         - Exit");
}

// Check if loading task is complete
fn check_loading_task(
    mut commands: Commands,
    mut loading_query: Query<(Entity, &mut LoadingTask)>,
    mut camera_query: Query<&mut CameraController>,
    loading_state: Option<Res<LoadingState>>,
) {
    for (entity, mut task) in &mut loading_query {
        // Check if task is ready (non-blocking using now_or_never)
        if let Some(result) = (&mut task.0).now_or_never() {
            // Task completed!
            match result {
                Ok(splats) => {
                    if let Some(ref state) = loading_state {
                        let total_time = state.start_time.elapsed();
                        println!("✅ Total loading time: {:.2}s", total_time.as_secs_f32());
                    }
                    
                    // Analyze scene and update camera
                    let scene_center = splats.center();
                    let scene_size = splats.size();
                    let camera_distance = splats.suggested_camera_distance();
                    
                    println!("📊 Scene Info:");
                    println!("  - Point count: {}", splats.len());
                    println!("  - Center: {:?}", scene_center);
                    println!("  - Size: {:?}", scene_size);
                    println!("  - Camera Distance: {:.2}", camera_distance);
                    
                    // Update camera to proper distance
                    for mut controller in &mut camera_query {
                        controller.distance = camera_distance;
                    }
                    
                    // Spawn the Gaussian splats entity
                    commands.spawn((
                        splats,
                        RenderingConfig {
                            point_size: 1.0,
                            frustum_dilation: 0.2,
                            alpha_cull_threshold: 1.0/255.0,
                            ..default()
                        },
                        Transform::from_translation(-scene_center),
                        GlobalTransform::default(),
                        Visibility::default(),
                    ));
                    
                    println!("🎉 Scene loaded! GPU resources will be created in next frames.");
                }
                Err(e) => {
                    println!("❌ Loading failed: {}", e);
                }
            }
            
            // Remove loading task
            commands.entity(entity).despawn();
            commands.remove_resource::<LoadingState>();
        }
    }
}

// Update loading UI
fn update_loading_ui(
    mut commands: Commands,
    loading_query: Query<&LoadingTask>,
    ui_query: Query<Entity, With<LoadingUI>>,
    loading_state: Option<Res<LoadingState>>,
    time: Res<Time>,
) {
    let is_loading = !loading_query.is_empty();
    
    if !is_loading {
        // Remove loading UI when done
        for entity in &ui_query {
            commands.entity(entity).despawn();
        }
    } else if let Some(state) = loading_state {
        // Update loading text with elapsed time
        let elapsed = state.start_time.elapsed().as_secs_f32();
        let dots = ".".repeat(((time.elapsed_secs() * 2.0) as usize % 4) + 1);
        
        for entity in &ui_query {
            commands.entity(entity).try_insert(
                Text::new(format!(
                    "⏳ Loading Gaussian Splats{}\nElapsed: {:.1}s\n\nWindow is ready, data loading in background...",
                    dots, elapsed
                ))
            );
        }
    }
}

// Camera controller component
#[derive(Component)]
struct CameraController {
    distance: f32,
    height: f32,
    rotation_speed: f32,
    auto_rotate: bool,
    // Manual control state
    yaw: f32,    // Horizontal angle (radians)
    pitch: f32,  // Vertical angle (radians)
}

// Rotate camera around center (auto + manual control)
fn rotate_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_q: Query<(&mut Transform, &mut CameraController)>,
) {
    let delta = time.delta_secs();
    
    for (mut transform, mut controller) in &mut camera_q {
        let mut changed = false;
        
        // Auto rotation
        if controller.auto_rotate {
            controller.yaw += controller.rotation_speed * delta;
            changed = true;
        }
        
        // Manual rotation
        let rotation_speed = 2.0;
        if keyboard.pressed(KeyCode::ArrowLeft) || keyboard.pressed(KeyCode::KeyA) {
            controller.yaw += rotation_speed * delta;
            changed = true;
        }
        if keyboard.pressed(KeyCode::ArrowRight) || keyboard.pressed(KeyCode::KeyD) {
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
        
        // Distance control (zoom)
        let zoom_speed = 5.0;
        if keyboard.pressed(KeyCode::KeyW) {
            controller.distance = (controller.distance - zoom_speed * delta).max(0.1);
            changed = true;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            controller.distance += zoom_speed * delta;
            changed = true;
        }
        
        // Height control
        let height_speed = 3.0;
        if keyboard.pressed(KeyCode::KeyQ) {
            controller.height += height_speed * delta;
            changed = true;
        }
        if keyboard.pressed(KeyCode::KeyE) {
            controller.height -= height_speed * delta;
            changed = true;
        }
        
        // Update camera position if changed
        if changed {
            // Calculate position using spherical coordinates
            let horizontal_dist = controller.distance * controller.pitch.cos();
            transform.translation.x = horizontal_dist * controller.yaw.sin();
            transform.translation.z = horizontal_dist * controller.yaw.cos();
            transform.translation.y = controller.distance * controller.pitch.sin() + controller.height;
            transform.look_at(Vec3::ZERO, Vec3::Y);
        }
    }
}

// Handle keyboard input
fn keyboard_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_q: Query<&mut CameraController>,
    mut exit: MessageWriter<bevy::app::AppExit>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        for mut controller in &mut camera_q {
            controller.auto_rotate = !controller.auto_rotate;
            if controller.auto_rotate {
                println!("✓ Auto rotation enabled");
            } else {
                println!("✓ Auto rotation paused - use Arrow keys or WASD to control camera");
            }
        }
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        println!("Exiting program...");
        exit.write(bevy::app::AppExit::Success);
    }
}

// Adjust point size based on keyboard input
fn adjust_point_size(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut point_clouds: Query<&mut RenderingConfig>,
) {
    let mut size_changed = false;
    let mut new_size = 0.0;
    
    for mut config in &mut point_clouds {
        // Increase point size
        if keyboard.just_pressed(KeyCode::Equal) || keyboard.just_pressed(KeyCode::NumpadAdd) {
            config.point_size = (config.point_size + 0.5).min(50.0);
            size_changed = true;
            new_size = config.point_size;
        }
        
        // Decrease point size
        if keyboard.just_pressed(KeyCode::Minus) || keyboard.just_pressed(KeyCode::NumpadSubtract) {
            config.point_size = (config.point_size - 0.5).max(0.1);
            size_changed = true;
            new_size = config.point_size;
        }
        
        // Reset to default
        if keyboard.just_pressed(KeyCode::KeyR) {
            config.point_size = 1.0;
            size_changed = true;
            new_size = config.point_size;
        }
    }
    
    if size_changed {
        println!("Point size adjusted to: {:.1} pixels", new_size);
    }
}

// Adjust culling parameters based on keyboard input
fn adjust_culling_params(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut point_clouds: Query<&mut RenderingConfig>,
) {
    for mut config in &mut point_clouds {
        let delta = time.delta_secs();
        let mut changed = false;

        // Adjust frustum_dilation
        if keyboard.pressed(KeyCode::Digit1) {
            config.frustum_dilation = (config.frustum_dilation + delta * 0.5).min(1.0);
            changed = true;
        }
        if keyboard.pressed(KeyCode::Digit2) {
            config.frustum_dilation = (config.frustum_dilation - delta * 0.5).max(0.0);
            changed = true;
        }

        // Adjust alpha_cull_threshold
        if keyboard.pressed(KeyCode::Digit3) {
            config.alpha_cull_threshold = (config.alpha_cull_threshold + delta * 0.05).min(1.0);
            changed = true;
        }
        if keyboard.pressed(KeyCode::Digit4) {
            config.alpha_cull_threshold = (config.alpha_cull_threshold - delta * 0.05).max(0.0);
            changed = true;
        }

        if changed {
            println!("Culling params - frustum_dilation: {:.3}, alpha_threshold: {:.3}", 
                     config.frustum_dilation, config.alpha_cull_threshold);
        }
    }
}

