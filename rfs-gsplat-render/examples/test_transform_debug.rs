//! 调试3DGS变换矩阵对齐问题的测试工具
//! 
//! 此工具会:
//! 1. 显示3DGS实体的Transform信息
//! 2. 显示相机的View矩阵
//! 3. 显示GPU buffer中的model_matrix
//! 4. 对比包围盒和3DGS渲染的坐标

use bevy::prelude::*;
use rfs_gsplat_render::{
    gaussian_point_cloud::{GaussianPointCloudPlugin, GaussianSplatGpuBuffers},
    gaussian_splats::{create_test_splats, GaussianSplats},
};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "3DGS Transform Debug".to_string(),
                    resolution: (800, 600).into(),
                    ..default()
                }),
                ..default()
            }),
            GaussianPointCloudPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (
            debug_transforms,
            keyboard_input,
        ))
        .run();
}

fn setup(mut commands: Commands) {
    println!("🔍 3DGS Transform Debug Tool");
    println!("按 SPACE 键打印调试信息");
    
    // 创建相机
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, 3.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        Name::new("Debug Camera"),
    ));
    
    // 创建测试3DGS (100个点,足够小可以快速加载)
    let splats = create_test_splats(100);
    let scene_center = splats.center();
    
    println!("📊 场景信息:");
    println!("  点数: {}", splats.len());
    println!("  中心: {:?}", scene_center);
    
    // 生成实体,应用偏移变换
    commands.spawn((
        splats,
        Transform::from_translation(-scene_center),
        GlobalTransform::default(),
        Visibility::default(),
        Name::new("Test Splats"),
    ));
}

fn debug_transforms(
    keyboard: Res<ButtonInput<KeyCode>>,
    camera_query: Query<(&Transform, &GlobalTransform, &Camera), With<Camera3d>>,
    splat_query: Query<(&Transform, &GlobalTransform), With<GaussianSplats>>,
    gpu_buffers: Option<Res<GaussianSplatGpuBuffers>>,
) {
    if !keyboard.just_pressed(KeyCode::Space) {
        return;
    }
    
    println!("\n" + &"=".repeat(60));
    println!("🔍 Transform Debug Info");
    println!("=".repeat(60));
    
    // 1. 相机信息
    if let Some((cam_transform, cam_global, _camera)) = camera_query.iter().next() {
        println!("\n📷 相机:");
        println!("  Local Transform:");
        println!("    translation: {:?}", cam_transform.translation);
        println!("    rotation: {:?}", cam_transform.rotation);
        
        let view_matrix = cam_global.to_matrix().inverse();
        println!("  View Matrix (from GlobalTransform.inverse()):");
        print_matrix("    ", &view_matrix);
    }
    
    // 2. 3DGS实体Transform信息
    if let Some((transform, global_transform)) = splat_query.iter().next() {
        println!("\n🎨 3DGS实体:");
        println!("  Local Transform:");
        println!("    translation: {:?}", transform.translation);
        println!("    rotation: {:?}", transform.rotation);
        println!("    scale: {:?}", transform.scale);
        
        println!("  GlobalTransform Matrix:");
        let global_matrix = global_transform.to_matrix();
        print_matrix("    ", &global_matrix);
        
        // 3. GPU Buffer中的model_matrix (如果已创建)
        if let Some(buffers) = gpu_buffers.as_ref() {
            println!("\n💾 GPU Buffer信息:");
            println!("  Point count: {}", buffers.point_count);
            println!("  ⚠️  注意:无法直接读取GPU buffer中的model_matrix");
            println!("     但它应该等于上面的GlobalTransform Matrix");
        } else {
            println!("\n💾 GPU Buffer: 尚未创建");
        }
        
        // 4. 测试点变换
        println!("\n🧪 测试点变换:");
        let test_point_local = Vec3::new(0.0, 0.0, 0.0);
        let test_point_world = global_matrix.transform_point3(test_point_local);
        println!("  本地坐标 {:?} -> 世界坐标 {:?}", test_point_local, test_point_world);
        
        if let Some((_, cam_global, _)) = camera_query.iter().next() {
            let view_matrix = cam_global.to_matrix().inverse();
            let test_point_view = view_matrix.transform_point3(test_point_world);
            println!("  世界坐标 {:?} -> 视图坐标 {:?}", test_point_world, test_point_view);
        }
    }
    
    println!("\n" + &"=".repeat(60));
}

fn print_matrix(indent: &str, mat: &Mat4) {
    for row in 0..4 {
        print!("{}", indent);
        for col in 0..4 {
            print!("{:9.4} ", mat.row(row)[col]);
        }
        println!();
    }
}

fn keyboard_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut exit: MessageWriter<bevy::app::AppExit>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        exit.write(bevy::app::AppExit::Success);
    }
}

