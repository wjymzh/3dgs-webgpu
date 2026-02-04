// GPU Radix Sort Validation Example
// Validates GPU sort against CPU reference implementation
// Uses per-workgroup architecture for correctness and performance

use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        Render, RenderApp, RenderSystems,
    },
};
use rand::Rng;
use rfs_gsplat_render::radix_sort::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

const TEST_SIZE: usize = 10_000_000;  // 10M elements

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set(bevy::window::WindowPlugin {
            primary_window: Some(Window {
                title: "GPU Radix Sort Validation (Running...)".to_string(),
                resolution: (400, 100).into(),
                visible: false,  // Hidden window
                ..default()
            }),
            ..default()
        }),
        RadixSortPlugin,
    ));
    
    println!("🧪 GPU Radix Sort Validation Test (Per-Workgroup Architecture)\n");
    println!("Test size: {} elements", TEST_SIZE);
    
    // Add test system to render app
    let render_app = app.sub_app_mut(RenderApp);
    render_app.add_systems(Render, run_validation_test.in_set(RenderSystems::Render));
    
    // Run the app (will exit after test completes)
    app.run();
}

#[derive(Resource)]
struct TestExecuted;

fn run_validation_test(world: &mut World) {
    // Check if already executed
    if world.contains_resource::<TestExecuted>() {
        return;
    }
    
    // Check if pipelines are ready first
    {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipelines = world.resource::<RadixSortPipelines>();
        
        if pipeline_cache.get_compute_pipeline(pipelines.upsweep_pipeline).is_none() ||
           pipeline_cache.get_compute_pipeline(pipelines.spine_pipeline).is_none() ||
           pipeline_cache.get_compute_pipeline(pipelines.downsweep_pipeline).is_none() {
            // Pipelines not ready, will try again next frame
            return;
        }
    }
    
    // Mark as executed so we don't run again
    world.insert_resource(TestExecuted);
    
    // Now clone what we need
    let render_device = world.resource::<RenderDevice>().clone();
    let render_queue = world.resource::<RenderQueue>().clone();
    let pipeline_cache = world.resource::<PipelineCache>();
    let pipelines = world.resource::<RadixSortPipelines>().clone();
    
    println!("\nInitializing GPU resources...");
    println!("Pipelines ready. Starting validation test...\n");
    
    // Generate random test data
    let mut rng = rand::thread_rng();
    let original_keys: Vec<u32> = (0..TEST_SIZE)
        .map(|_| rng.gen())
        .collect();
    let original_values: Vec<u32> = (0..TEST_SIZE as u32).collect();
    
    println!("Original keys (first 10): {:?}", &original_keys[..10]);
    
    // Create GPU buffers with COPY_SRC for readback
    let keys_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("validation_keys"),
        contents: bytemuck::cast_slice(&original_keys),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });
    
    let values_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("validation_values"),
        contents: bytemuck::cast_slice(&original_values),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });
    
    // Create element count buffer
    let element_count_data = vec![TEST_SIZE as u32];
    let element_count_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("validation_element_count"),
        contents: bytemuck::cast_slice(&element_count_data),
        usage: BufferUsages::STORAGE,
    });
    
    // Create radix sort buffers
    let sort_buffers = create_radix_sort_buffers(&render_device, TEST_SIZE);
    let num_partitions = sort_buffers.num_partitions;
    
    println!("Number of partitions: {}", num_partitions);
    
    // Create bind groups for 3-pass architecture
    let mut upsweep_bind_groups = Vec::new();
    let mut spine_bind_groups = Vec::new();
    let mut downsweep_bind_groups = Vec::new();
    
    for pass in 0..4 {
        let bit_shift = pass * 8;
        
        // Create params buffer
        let params = SortParams {
            max_element_count: TEST_SIZE as u32,
            bit_shift,
            pass_index: pass as u32,
            _padding: 0,
        };
        let params_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(&format!("params_pass_{}", pass)),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        
        // Determine ping-pong buffers
        let (keys_in, keys_out, values_in, values_out) = if pass % 2 == 0 {
            (&keys_buffer, &sort_buffers.keys_temp, &values_buffer, &sort_buffers.values_temp)
        } else {
            (&sort_buffers.keys_temp, &keys_buffer, &sort_buffers.values_temp, &values_buffer)
        };
        
        // Upsweep bind group
        let upsweep_bg = render_device.create_bind_group(
            None,
            &pipelines.upsweep_bind_group_layout,
            &BindGroupEntries::sequential((
                params_buffer.as_entire_binding(),
                element_count_buffer.as_entire_binding(),
                keys_in.as_entire_binding(),
                sort_buffers.global_histogram.as_entire_binding(),
                sort_buffers.partition_histogram.as_entire_binding(),
            )),
        );
        upsweep_bind_groups.push(upsweep_bg);
        
        // Spine bind group
        let spine_bg = render_device.create_bind_group(
            None,
            &pipelines.spine_bind_group_layout,
            &BindGroupEntries::sequential((
                element_count_buffer.as_entire_binding(),
                sort_buffers.global_histogram.as_entire_binding(),
                sort_buffers.partition_histogram.as_entire_binding(),
                params_buffer.as_entire_binding(),
            )),
        );
        spine_bind_groups.push(spine_bg);
        
        // Downsweep bind group
        let downsweep_bg = render_device.create_bind_group(
            None,
            &pipelines.downsweep_bind_group_layout,
            &BindGroupEntries::sequential((
                params_buffer.as_entire_binding(),
                element_count_buffer.as_entire_binding(),
                sort_buffers.global_histogram.as_entire_binding(),
                sort_buffers.partition_histogram.as_entire_binding(),
                keys_in.as_entire_binding(),
                values_in.as_entire_binding(),
                keys_out.as_entire_binding(),
                values_out.as_entire_binding(),
            )),
        );
        downsweep_bind_groups.push(downsweep_bg);
    }
    
    let bind_groups = RadixSortBindGroups {
        upsweep_bind_groups,
        spine_bind_groups,
        downsweep_bind_groups,
    };
    
    // Execute GPU sort
    println!("Executing GPU radix sort...");
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("radix_sort_validation"),
    });
    
    // Clear histogram buffers before first pass
    // Global histogram: 4 passes * 256 bins = 1024 elements
    // Partition histogram: num_partitions * 256 bins
    // Note: We clear partition_histogram before first pass, but upsweep will overwrite it each pass
    encoder.clear_buffer(&sort_buffers.global_histogram, 0, None);
    encoder.clear_buffer(&sort_buffers.partition_histogram, 0, None);
    
    // Clear temp buffers (they will be used as output in first pass)
    encoder.clear_buffer(&sort_buffers.keys_temp, 0, None);
    encoder.clear_buffer(&sort_buffers.values_temp, 0, None);
    
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("radix_sort_pass"),
            timestamp_writes: None,
        });
        
        execute_radix_sort(
            &mut compute_pass,
            &pipeline_cache,
            &pipelines,
            &bind_groups,
            num_partitions,
        );
    }
    
    // Create readback buffer
    let readback_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("readback_buffer"),
        size: (TEST_SIZE * 4) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    
    // Determine final result buffer (after 4 passes, pass 3 is odd, so result is in keys_buffer)
    // Pass 0: keys_buffer -> temp (even)
    // Pass 1: temp -> keys_buffer (odd)
    // Pass 2: keys_buffer -> temp (even)
    // Pass 3: temp -> keys_buffer (odd)
    // So final result is in keys_buffer
    let final_keys_buffer = &keys_buffer;
    
    // Copy final sorted keys to readback buffer
    encoder.copy_buffer_to_buffer(
        final_keys_buffer,
        0,
        &readback_buffer,
        0,
        (TEST_SIZE * 4) as u64,
    );
    
    render_queue.submit(std::iter::once(encoder.finish()));
    
    // CPU reference sort
    println!("Running CPU reference sort...");
    let mut cpu_keys = original_keys.clone();
    let mut cpu_values = original_values.clone();
    cpu_radix_sort(&mut cpu_keys, &mut cpu_values);
    
    println!("CPU sorted (first 10): {:?}", &cpu_keys[..10]);
    
    // Read back GPU results (synchronous for testing)
    println!("Reading back GPU results...");
    let buffer_slice = readback_buffer.slice(..);
    
    // Use atomic flag to track mapping completion
    let mapping_done = Arc::new(AtomicBool::new(false));
    let mapping_done_clone = mapping_done.clone();
    
    buffer_slice.map_async(MapMode::Read, move |result| {
        if result.is_ok() {
            mapping_done_clone.store(true, Ordering::Release);
        }
    });
    
    // Poll device until mapping completes
    let wgpu_device = render_device.wgpu_device();
    let mut poll_count = 0;
    let timeout = std::time::Duration::from_secs(10);
    let start = std::time::Instant::now();
    
    loop {
        let _ = wgpu_device.poll(wgpu::PollType::Wait);
        poll_count += 1;
        
        if mapping_done.load(Ordering::Acquire) {
            println!("Buffer mapped successfully (poll count: {})", poll_count);
            break;
        }
        
        if start.elapsed() > timeout {
            println!("✗ Timeout waiting for buffer mapping!");
            std::process::exit(1);
        }
        
        if poll_count > 10000 {
            println!("✗ Too many poll attempts!");
            std::process::exit(1);
        }
        
        // Avoid busy waiting
        if poll_count % 100 == 0 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    
    // Read mapped data
    let data = buffer_slice.get_mapped_range();
    let gpu_keys: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback_buffer.unmap();
    
    println!("GPU sorted (first 10): {:?}", &gpu_keys[..10]);
    
    // Validate results
    println!("\n=== Validation Results ===");
    let mut passed = true;
    let mut first_error_idx = None;
    
    for i in 0..TEST_SIZE {
        if gpu_keys[i] != cpu_keys[i] {
            if first_error_idx.is_none() {
                first_error_idx = Some(i);
            }
            passed = false;
        }
    }
    
    if passed {
        println!("✓ GPU Radix Sort PASSED!");
        println!("All {} elements sorted correctly.", TEST_SIZE);
        println!("\n🎉 Test completed successfully!");
        std::process::exit(0);
    } else {
        println!("✗ GPU Radix Sort FAILED!");
        if let Some(idx) = first_error_idx {
            println!("First error at index {}", idx);
            println!("  Expected (CPU): {}", cpu_keys[idx]);
            println!("  Got (GPU): {}", gpu_keys[idx]);
            
            // Show context
            let start_idx = idx.saturating_sub(5);
            let end_idx = (idx + 5).min(TEST_SIZE);
            println!("\nContext (indices {}..{}):", start_idx, end_idx);
            println!("  CPU: {:?}", &cpu_keys[start_idx..end_idx]);
            println!("  GPU: {:?}", &gpu_keys[start_idx..end_idx]);
        }
        std::process::exit(1);
    }
}

// CPU reference radix sort implementation
fn cpu_radix_sort(keys: &mut [u32], values: &mut [u32]) {
    let n = keys.len();
    let mut keys_temp = vec![0u32; n];
    let mut values_temp = vec![0u32; n];
    
    // 4 passes for 32-bit keys (8 bits each)
    for pass in 0..4 {
        let shift = pass * 8;
        let mut histogram = vec![0usize; 256];
        
        // Build histogram
        for &key in keys.iter() {
            let digit = ((key >> shift) & 0xFF) as usize;
            histogram[digit] += 1;
        }
        
        // Exclusive prefix sum
        let mut sum = 0;
        for count in histogram.iter_mut() {
            let temp = *count;
            *count = sum;
            sum += temp;
        }
        
        // Reorder
        for i in 0..n {
            let key = keys[i];
            let value = values[i];
            let digit = ((key >> shift) & 0xFF) as usize;
            let pos = histogram[digit];
            histogram[digit] += 1;
            keys_temp[pos] = key;
            values_temp[pos] = value;
        }
        
        // Swap buffers by copying back
        keys.copy_from_slice(&keys_temp);
        values.copy_from_slice(&values_temp);
    }
}
