//! 可视化验证3DGS和包围盒对齐的测试工具
//! 
//! 此示例会:
//! 1. 加载真实的 PLY 文件
//! 2. 显示包围盒
//! 3. 在包围盒的8个角绘制参考点
//! 4. 验证3DGS渲染是否与包围盒对齐

use bevy::prelude::*;
use rfs_gsplat_render::{
    gaussian_point_cloud::GaussianPointCloudPlugin,
    loader::load_ply_file,
};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "3DGS对齐测试 - 包围盒应该完全包含点云".to_string(),
                    resolution: (1200, 800).into(),
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
            render_bounding_box,
            render_corner_markers,
        ))
        .run();
}

/// 简单的包围盒组件
#[derive(Component)]
struct TestBoundingBox {
    min: Vec3,
    max: Vec3,
}

/// 相机控制器
#[derive(Component)]
struct CameraController {
    angle: f32,
    distance: f32,
    speed: f32,
    center: Vec3,
}

fn setup(mut commands: Commands) {
    println!("🔍 3DGS对齐测试 - 加载真实PLY文件");
    println!("说明:");
    println!("  - 绿色包围盒应该完全包含所有3DGS点");
    println!("  - 红色球体标记包围盒的8个角");
    println!("  - 如果点云超出包围盒或有明显偏移,说明存在对齐问题");
    println!("\n控制:");
    println!("  SPACE - 暂停/继续旋转");
    println!("  ESC   - 退出");
    
    // 加载 PLY 文件
    let ply_path = r"D:\Models\lego1\3dgs.ply";
    println!("\n📂 加载 PLY 文件: {}", ply_path);
    
    let splats = match load_ply_file(ply_path) {
        Ok(s) => {
            println!("✅ 成功加载 {} 个高斯点", s.means.len());
            s
        }
        Err(e) => {
            eprintln!("❌ 加载 PLY 文件失败: {}", e);
            eprintln!("   请确保文件存在: {}", ply_path);
            return;
        }
    };
    
    // 计算实际包围盒 (基于所有点的位置)
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    
    for mean in &splats.means {
        min = min.min(*mean);
        max = max.max(*mean);
    }
    
    println!("\n📊 计算得到的包围盒:");
    println!("  min: {:?}", min);
    println!("  max: {:?}", max);
    println!("  size: {:?}", max - min);
    println!("  center: {:?}", (min + max) / 2.0);
    
    // 根据包围盒大小设置相机距离
    let bbox_size = (max - min).length();
    let camera_distance = bbox_size * 1.5; // 相机距离为包围盒对角线的1.5倍
    let center = (min + max) / 2.0;
    
    println!("  camera_distance: {}", camera_distance);
    
    // 创建相机
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(camera_distance, camera_distance * 0.5, camera_distance))
            .looking_at(center, Vec3::Y),
        CameraController {
            angle: 0.0,
            distance: camera_distance,
            speed: 0.3,
            center,
        },
    ));
    
    // 创建3DGS实体 (无变换,世界空间原点)
    let splat_entity = commands.spawn((
        splats,
        Transform::default(), // 无变换
        GlobalTransform::default(),
        Visibility::default(),
        Name::new("Lego Splats"),
        TestBoundingBox {
            min,
            max,
        },
    )).id();
    
    println!("✅ 实体ID: {:?}", splat_entity);
    println!("\n🎯 观察包围盒（绿色线框）是否完全包含所有渲染的高斯点");
}

fn rotate_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_query: Query<(&mut Transform, &mut CameraController)>,
) {
    for (mut transform, mut controller) in &mut camera_query {
        if keyboard.just_pressed(KeyCode::Space) {
            controller.speed = if controller.speed > 0.0 { 0.0 } else { 0.3 };
            println!("旋转: {}", if controller.speed > 0.0 { "开" } else { "关" });
        }
        
        controller.angle += controller.speed * time.delta_secs();
        
        // 围绕包围盒中心旋转
        let x = controller.center.x + controller.distance * controller.angle.cos();
        let z = controller.center.z + controller.distance * controller.angle.sin();
        let y = controller.center.y + controller.distance * 0.5;
        
        transform.translation = Vec3::new(x, y, z);
        transform.look_at(controller.center, Vec3::Y);
    }
}

fn render_bounding_box(
    bbox_query: Query<&TestBoundingBox>,
    mut gizmos: Gizmos,
) {
    for bbox in &bbox_query {
        let min = bbox.min;
        let max = bbox.max;
        let color = Color::srgb(0.0, 1.0, 0.0); // 绿色
        
        // 绘制12条边
        // 底面
        gizmos.line(Vec3::new(min.x, min.y, min.z), Vec3::new(max.x, min.y, min.z), color);
        gizmos.line(Vec3::new(max.x, min.y, min.z), Vec3::new(max.x, min.y, max.z), color);
        gizmos.line(Vec3::new(max.x, min.y, max.z), Vec3::new(min.x, min.y, max.z), color);
        gizmos.line(Vec3::new(min.x, min.y, max.z), Vec3::new(min.x, min.y, min.z), color);
        
        // 顶面
        gizmos.line(Vec3::new(min.x, max.y, min.z), Vec3::new(max.x, max.y, min.z), color);
        gizmos.line(Vec3::new(max.x, max.y, min.z), Vec3::new(max.x, max.y, max.z), color);
        gizmos.line(Vec3::new(max.x, max.y, max.z), Vec3::new(min.x, max.y, max.z), color);
        gizmos.line(Vec3::new(min.x, max.y, max.z), Vec3::new(min.x, max.y, min.z), color);
        
        // 垂直边
        gizmos.line(Vec3::new(min.x, min.y, min.z), Vec3::new(min.x, max.y, min.z), color);
        gizmos.line(Vec3::new(max.x, min.y, min.z), Vec3::new(max.x, max.y, min.z), color);
        gizmos.line(Vec3::new(max.x, min.y, max.z), Vec3::new(max.x, max.y, max.z), color);
        gizmos.line(Vec3::new(min.x, min.y, max.z), Vec3::new(min.x, max.y, max.z), color);
    }
}

fn render_corner_markers(
    bbox_query: Query<&TestBoundingBox>,
    mut gizmos: Gizmos,
) {
    for bbox in &bbox_query {
        let corners = [
            Vec3::new(bbox.min.x, bbox.min.y, bbox.min.z),
            Vec3::new(bbox.max.x, bbox.min.y, bbox.min.z),
            Vec3::new(bbox.min.x, bbox.max.y, bbox.min.z),
            Vec3::new(bbox.max.x, bbox.max.y, bbox.min.z),
            Vec3::new(bbox.min.x, bbox.min.y, bbox.max.z),
            Vec3::new(bbox.max.x, bbox.min.y, bbox.max.z),
            Vec3::new(bbox.min.x, bbox.max.y, bbox.max.z),
            Vec3::new(bbox.max.x, bbox.max.y, bbox.max.z),
        ];
        
        // 在每个角绘制小球体作为标记
        let color = Color::srgb(1.0, 0.0, 0.0); // 红色
        for corner in corners {
            gizmos.sphere(Isometry3d::from_translation(corner), 0.05, color);
        }
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

