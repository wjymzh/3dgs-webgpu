// PACK模式使用示例
//
// 这个示例展示如何启用和使用PACK模式（压缩模式）来减少GPU内存使用
//
// 运行: cargo run --example pack_mode_example

use bevy::prelude::*;
use rfs_gsplat_render::{GaussianSplats, PackModeConfig, GaussianSplatPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GaussianSplatPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (toggle_pack_mode, display_memory_info))
        .run();
}

fn setup(mut commands: Commands) {
    // 创建相机
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // 示例1: 使用默认配置（PACK模式默认启用）
    // 这会使用压缩的数据格式，节省约46%的GPU内存
    let splats_default = create_example_splats(1000);
    commands.spawn((
        splats_default,
        PackModeConfig::default(),  // 🔥 默认就是PACK模式！
        Transform::from_xyz(-2.0, 0.0, 0.0),
    ));

    // 示例2: 显式启用PACK模式（与默认值相同）
    let splats_packed = create_example_splats(1000);
    commands.spawn((
        splats_packed,
        PackModeConfig::enabled(),  // 显式启用（实际与default相同）
        Transform::from_xyz(2.0, 0.0, 0.0),
    ));

    // 示例3: 禁用PACK模式使用标准模式
    let splats_standard = create_example_splats(1000);
    commands.spawn((
        splats_standard,
        PackModeConfig::disabled(),  // 显式禁用，使用标准模式
        Transform::from_xyz(0.0, 2.0, 0.0),
    ));

    println!("\n🎮 PACK模式示例已启动!");
    println!("📦 左侧: PACK模式（默认）");
    println!("📦 右侧: PACK模式（显式启用）");
    println!("📄 上方: 标准模式（禁用PACK）");
    println!("\n⚠️  注意: PACK模式现在是默认启用的！");
    println!("\n⌨️  按 'P' 切换所有entities的PACK模式");
    println!("⌨️  按 'I' 显示内存信息\n");
}

/// 创建示例Gaussian Splats数据
fn create_example_splats(count: usize) -> GaussianSplats {
    // 这里应该从文件加载或生成真实的splats数据
    // 为了示例简化，创建一些随机数据
    
    use glam::{Vec3, Vec4};
    
    let means: Vec<Vec3> = (0..count)
        .map(|i| {
            let angle = (i as f32 / count as f32) * std::f32::consts::TAU;
            Vec3::new(angle.cos() * 0.5, angle.sin() * 0.5, 0.0)
        })
        .collect();
    
    let rotations: Vec<Vec4> = vec![Vec4::new(0.0, 0.0, 0.0, 1.0); count];
    let log_scales: Vec<Vec3> = vec![Vec3::new(-2.0, -2.0, -2.0); count];
    let raw_opacities: Vec<f32> = vec![2.0; count];
    
    // SH coefficients (degree 0, DC component only)
    let sh_coeffs: Vec<Vec<Vec3>> = vec![vec![Vec3::new(0.5, 0.3, 0.8)]; count];
    
    GaussianSplats::new(means, rotations, log_scales, raw_opacities, sh_coeffs)
}

/// 按P键切换PACK模式
fn toggle_pack_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut PackModeConfig>,
) {
    if keyboard.just_pressed(KeyCode::KeyP) {
        let mut count_enabled = 0;
        let mut count_disabled = 0;
        
        for mut config in query.iter_mut() {
            config.enabled = !config.enabled;
            if config.enabled {
                count_enabled += 1;
            } else {
                count_disabled += 1;
            }
        }
        
        println!("\n🔄 切换PACK模式:");
        println!("  📦 PACK启用: {} entities", count_enabled);
        println!("  📄 PACK禁用: {} entities", count_disabled);
        println!("  ⚠️  注意: 需要重新spawn entity才能看到内存变化\n");
    }
}

/// 按I键显示内存信息
fn display_memory_info(
    keyboard: Res<ButtonInput<KeyCode>>,
    query: Query<(Entity, &GaussianSplats, Option<&PackModeConfig>)>,
) {
    if keyboard.just_pressed(KeyCode::KeyI) {
        println!("\n📊 GPU内存使用估算:");
        
        for (entity, splats, pack_config) in query.iter() {
            let count = splats.len();
            let is_packed = pack_config.map_or(false, |c| c.enabled);
            
            // 估算内存使用（简化计算）
            let position_mb = (count * 12) as f32 / 1024.0 / 1024.0;
            let color_mb = if is_packed {
                (count * 6) as f32 / 1024.0 / 1024.0
            } else {
                (count * 12) as f32 / 1024.0 / 1024.0
            };
            let rot_scale_mb = if is_packed {
                (count * 16) as f32 / 1024.0 / 1024.0
            } else {
                (count * 28) as f32 / 1024.0 / 1024.0
            };
            let sh_mb = if is_packed {
                (count * 90) as f32 / 1024.0 / 1024.0
            } else {
                (count * 180) as f32 / 1024.0 / 1024.0
            };
            let total_mb = position_mb + color_mb + rot_scale_mb + sh_mb;
            
            println!("\n  Entity {:?}: {} splats", entity, count);
            println!("    模式: {}", if is_packed { "📦 PACK" } else { "📄 标准" });
            println!("    位置:     {:.2} MB", position_mb);
            println!("    颜色:     {:.2} MB", color_mb);
            println!("    旋转缩放: {:.2} MB", rot_scale_mb);
            println!("    球谐系数: {:.2} MB", sh_mb);
            println!("    总计:     {:.2} MB", total_mb);
        }
        println!();
    }
}

