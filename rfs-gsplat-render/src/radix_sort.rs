// GPU Radix Sort Plugin for Bevy
// 3-Pass Architecture (Upsweep-Spine-Downsweep) inspired by vrdx
// Pass 1: Upsweep - Build local histograms and accumulate to global
// Pass 2: Spine - Prefix sum on partition and global histograms
// Pass 3: Downsweep - Scatter elements using computed offsets

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{binding_types::*, *},
        renderer::RenderDevice,
        RenderApp,
    },
};
use std::borrow::Cow;

const RADIX: usize = 256;
const BLOCK_SIZE: usize = 256 * 4;  // 1024

pub struct RadixSortPlugin;

impl Plugin for RadixSortPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<RadixSortConfig>::default());
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<RadixSortPipelines>();
    }
}

#[derive(Resource, Clone, ExtractResource)]
pub struct RadixSortConfig {
    pub enabled: bool,
}

impl Default for RadixSortConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Resource, Clone)]
pub struct RadixSortPipelines {
    pub upsweep_pipeline: CachedComputePipelineId,
    pub spine_pipeline: CachedComputePipelineId,
    pub downsweep_pipeline: CachedComputePipelineId,
    
    pub upsweep_bind_group_layout: BindGroupLayout,
    pub spine_bind_group_layout: BindGroupLayout,
    pub downsweep_bind_group_layout: BindGroupLayout,
    
    pub shader: Handle<Shader>,
}

impl FromWorld for RadixSortPipelines {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        
        // Load embedded shader using Bevy's recommended method
        let shader = load_embedded_asset!(asset_server, "../assets/shaders/radix_sort.wgsl");
        
        // Upsweep layout: params, element_count, keys_in, global_histogram, partition_histogram
        let upsweep_bind_group_layout = render_device.create_bind_group_layout(
            Some("upsweep_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<SortParams>(false),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        
        // Spine layout: element_count, global_histogram, partition_histogram, params
        let spine_bind_group_layout = render_device.create_bind_group_layout(
            Some("spine_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                    uniform_buffer::<SortParams>(false),
                ),
            ),
        );
        
        // Downsweep layout: params, element_count, global_histogram, partition_histogram,
        //                   keys_in, values_in, keys_out, values_out
        let downsweep_bind_group_layout = render_device.create_bind_group_layout(
            Some("downsweep_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<SortParams>(false),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        
        let upsweep_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("radix_sort_upsweep")),
            layout: vec![upsweep_bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("upsweep")),
            zero_initialize_workgroup_memory: false,
        });
        
        let spine_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("radix_sort_spine")),
            layout: vec![spine_bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("spine")),
            zero_initialize_workgroup_memory: false,
        });
        
        let downsweep_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("radix_sort_downsweep")),
            layout: vec![downsweep_bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("downsweep")),
            zero_initialize_workgroup_memory: false,
        });
        
        Self {
            upsweep_pipeline,
            spine_pipeline,
            downsweep_pipeline,
            upsweep_bind_group_layout,
            spine_bind_group_layout,
            downsweep_bind_group_layout,
            shader,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
pub struct SortParams {
    pub max_element_count: u32,
    pub bit_shift: u32,
    pub pass_index: u32,  // Pass index (0-3)
    pub _padding: u32,
}

#[derive(Clone, Resource)]
pub struct RadixSortBuffers {
    pub global_histogram: Buffer,
    pub partition_histogram: Buffer,
    pub keys_temp: Buffer,
    pub values_temp: Buffer,
    pub num_partitions: u32,
}

pub fn create_radix_sort_buffers(
    render_device: &RenderDevice,
    max_elements: usize,
) -> RadixSortBuffers {
    let num_partitions = ((max_elements + BLOCK_SIZE - 1) / BLOCK_SIZE) as u32;
    
    // Global histogram: 4 passes * RADIX bins (1024 total)
    // Each pass has its own 256-bin histogram
    let global_histogram_data = vec![0u32; RADIX * 4];
    let global_histogram = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("radix_sort_global_histogram"),
        contents: bytemuck::cast_slice(&global_histogram_data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    
    // Partition histogram: RADIX counts per partition
    let partition_histogram_data = vec![0u32; RADIX * num_partitions as usize];
    let partition_histogram = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("radix_sort_partition_histogram"),
        contents: bytemuck::cast_slice(&partition_histogram_data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    
    // Temp buffers for ping-pong
    let keys_temp_data = vec![0u32; max_elements];
    let keys_temp = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("radix_sort_keys_temp"),
        contents: bytemuck::cast_slice(&keys_temp_data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });
    
    let values_temp_data = vec![0u32; max_elements];
    let values_temp = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("radix_sort_values_temp"),
        contents: bytemuck::cast_slice(&values_temp_data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });
    
    RadixSortBuffers {
        global_histogram,
        partition_histogram,
        keys_temp,
        values_temp,
        num_partitions,
    }
}

#[derive(Clone, Resource, Component)]
pub struct RadixSortBindGroups {
    pub upsweep_bind_groups: Vec<BindGroup>,
    pub spine_bind_groups: Vec<BindGroup>,
    pub downsweep_bind_groups: Vec<BindGroup>,
}

/// Execute radix sort with proper memory barriers between stages.
/// 
/// Execute radix sort with proper memory barriers between stages.
/// 
/// CRITICAL: Each stage (upsweep, spine, downsweep) runs in a separate compute pass
/// to ensure proper memory synchronization.
/// 
/// # Arguments
/// * `encoder` - Command encoder to record commands
/// * `pipeline_cache` - Pipeline cache to get compute pipelines
/// * `pipelines` - Radix sort pipeline resources
/// * `bind_groups` - Pre-created bind groups for all 4 passes
/// * `num_partitions` - Number of partitions (ceil(element_count / BLOCK_SIZE))
pub fn execute_radix_sort(
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    pipelines: &RadixSortPipelines,
    bind_groups: &RadixSortBindGroups,
    num_partitions: u32,
) {
    // Execute 4 radix sort passes (8-bit increments, total 32 bits)
    for pass_idx in 0..4usize {
        // Upsweep: build histograms (separate compute pass for memory barrier)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("radix_upsweep_p{}", pass_idx)),
                timestamp_writes: None,
            });
            if let Some(upsweep_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.upsweep_pipeline) {
                compute_pass.set_pipeline(upsweep_pipeline);
                compute_pass.set_bind_group(0, &bind_groups.upsweep_bind_groups[pass_idx], &[]);
                compute_pass.dispatch_workgroups(num_partitions, 1, 1);
            }
        } // End compute pass = implicit memory barrier
        
        // Spine/Scan: prefix sum (separate compute pass for memory barrier)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("radix_spine_p{}", pass_idx)),
                timestamp_writes: None,
            });
            if let Some(spine_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.spine_pipeline) {
                compute_pass.set_pipeline(spine_pipeline);
                compute_pass.set_bind_group(0, &bind_groups.spine_bind_groups[pass_idx], &[]);
                compute_pass.dispatch_workgroups(RADIX as u32, 1, 1);
            }
        } // End compute pass = implicit memory barrier
        
        // Downsweep: scatter (separate compute pass for memory barrier)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("radix_downsweep_p{}", pass_idx)),
                timestamp_writes: None,
            });
            if let Some(downsweep_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.downsweep_pipeline) {
                compute_pass.set_pipeline(downsweep_pipeline);
                compute_pass.set_bind_group(0, &bind_groups.downsweep_bind_groups[pass_idx], &[]);
                compute_pass.dispatch_workgroups(num_partitions, 1, 1);
            }
        } // End compute pass = implicit memory barrier
    }
}

#[derive(Resource)]
pub struct RadixSortRequest {
    pub keys_buffer: Buffer,
    pub values_buffer: Buffer,
    pub indirect_buffer: Buffer,
    pub max_element_count: u32,
    pub enabled: bool,
}
