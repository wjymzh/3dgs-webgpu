// Temporal Coherence Performance Demo
// This example demonstrates the performance benefits of temporal coherence
// 
// Controls:
// - WASD: Move camera
// - Mouse: Look around
// - Space: Toggle temporal coherence on/off
// - 1/2/3: Switch between Default/Conservative/Aggressive profiles

use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use rfs_gsplat_render::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            FrameTimeDiagnosticsPlugin,
            LogDiagnosticsPlugin::default(),
            GaussianPointCloudPlugin,
        ))
        .init_resource::<DemoState>()
        .add_systems(Startup, setup)
        .add_systems(Update, (
            toggle_temporal_coherence,
            switch_profile,
            update_ui,
            print_temporal_coherence_stats,
        ))
        .run();
}

#[derive(Resource, Default)]
struct DemoState {
    temporal_enabled: bool,
    current_profile: usize,
}

fn setup(
    mut commands: Commands,
    mut demo_state: ResMut<DemoState>,
) {
    // Create test splats (100K points for quick demo, increase for stress test)
    let splats = create_test_splats(100_000);
    
    // Spawn Gaussian Splats with Temporal Coherence enabled
    demo_state.temporal_enabled = true;
    demo_state.current_profile = 0;  // Default
    
    commands.spawn((
        splats,
        RenderingConfig::default(),
        TemporalCoherenceConfig::default(),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Visibility::default(),
    ));
    
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    
    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.5, 0.0)),
    ));
    
    // UI
    commands.spawn((
        Text::new("Temporal Coherence Demo\n\nSpace: Toggle\n1/2/3: Profiles\n\nStatus: Enabled (Default)"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        TextColor(Color::WHITE),
        TextFont {
            font_size: 20.0,
            ..default()
        },
    ));
}

fn toggle_temporal_coherence(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut demo_state: ResMut<DemoState>,
    mut query: Query<&mut TemporalCoherenceConfig>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        demo_state.temporal_enabled = !demo_state.temporal_enabled;
        
        for mut config in &mut query {
            config.enabled = demo_state.temporal_enabled;
        }
        
        info!("Temporal Coherence: {}", if demo_state.temporal_enabled { "ENABLED" } else { "DISABLED" });
    }
}

fn switch_profile(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut demo_state: ResMut<DemoState>,
    mut query: Query<&mut TemporalCoherenceConfig>,
) {
    let new_profile = if keyboard.just_pressed(KeyCode::Digit1) {
        Some(0)  // Default
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        Some(1)  // Conservative
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        Some(2)  // Aggressive
    } else {
        None
    };
    
    if let Some(profile) = new_profile {
        demo_state.current_profile = profile;
        
        for mut config in &mut query {
            *config = match profile {
                0 => TemporalCoherenceConfig::default(),
                1 => TemporalCoherenceConfig::conservative(),
                2 => TemporalCoherenceConfig::aggressive(),
                _ => TemporalCoherenceConfig::default(),
            };
        }
        
        let profile_name = match profile {
            0 => "Default",
            1 => "Conservative",
            2 => "Aggressive",
            _ => "Unknown",
        };
        info!("Switched to {} profile", profile_name);
    }
}

fn update_ui(
    demo_state: Res<DemoState>,
    stats: Res<TemporalCoherenceStats>,
    mut query: Query<&mut Text>,
) {
    for mut text in &mut query {
        let profile_name = match demo_state.current_profile {
            0 => "Default",
            1 => "Conservative",
            2 => "Aggressive",
            _ => "Unknown",
        };
        
        let status = if demo_state.temporal_enabled {
            format!("Enabled ({})", profile_name)
        } else {
            "DISABLED".to_string()
        };
        
        **text = format!(
            "Temporal Coherence Demo\n\n\
             Space: Toggle\n\
             1/2/3: Profiles\n\n\
             Status: {}\n\
             Skip Ratio: {:.1}%\n\
             Current Streak: {} frames\n\
             Max Streak: {} frames",
            status,
            stats.skip_ratio * 100.0,
            stats.current_skip_streak,
            stats.max_skip_streak,
        );
    }
}

