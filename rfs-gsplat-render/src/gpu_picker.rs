// gpu_picker.rs - GPU-based splat selection system
// Similar to supersplat's data-processor.ts intersect() function
//
// Architecture:
// 1. Main world: PickerRequest resource triggers selection
// 2. Render world: Compute shader tests each splat against selection criteria
// 3. Results are copied to staging buffer and mapped for CPU readback
// 4. Results are sent back to main world via PickerResult resource

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::{
        render_resource::{
            binding_types::{storage_buffer_read_only_sized, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            PipelineCache, ShaderStages, ShaderType, SpecializedComputePipeline,
            SpecializedComputePipelines, MapMode,
        },
        renderer::{RenderDevice, RenderQueue},
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};
use std::sync::{Arc, Mutex};

use crate::gaussian_point_cloud::GaussianSplatGpuBuffers;
use crate::gaussian_splats::SplatSelectionState;
use crate::splat_state::{BoxParams, RectParams, SelectionMode, SelectionOp, SphereParams};

/// GPU Picker plugin - handles GPU-based splat selection
pub struct GpuPickerPlugin;

impl Plugin for GpuPickerPlugin {
    fn build(&self, app: &mut App) {
        // Main world resources
        app.init_resource::<PickerRequest>();
        app.init_resource::<PickerResult>();
        app.init_resource::<PickerPendingReadback>();

        // System to apply picker results to splat state
        app.add_systems(Update, (
            setup_picker_pending,
            poll_picker_readback, 
            apply_picker_results,
        ).chain());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedComputePipelines<SelectionComputePipeline>>()
            .init_resource::<ExtractedPickerRequest>()
            .init_resource::<RenderPickerState>()
            .add_systems(ExtractSchedule, extract_picker_request)
            .add_systems(
                Render,
                prepare_selection_pipeline.in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                prepare_selection_resources.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Render,
                prepare_selection_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                execute_selection_compute.in_set(RenderSystems::Render),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<SelectionComputePipeline>();
    }
}

/// Request a GPU selection operation (main world resource)
#[derive(Resource, Default)]
pub struct PickerRequest {
    /// Whether a selection is requested
    pub active: bool,
    /// Target entity to select from
    pub target_entity: Option<Entity>,
    /// Selection operation
    pub op: SelectionOp,
    /// Selection mode
    pub mode: SelectionMode,
    /// Use rings mode (ellipse test) instead of centers mode (point test)
    pub use_rings: bool,
    /// Rectangle parameters (for Rect mode) - normalized 0-1
    pub rect: Option<RectParams>,
    /// Sphere parameters (for Sphere mode) - world space
    pub sphere: Option<SphereParams>,
    /// Box parameters (for Box mode) - world space
    pub box_params: Option<BoxParams>,
    /// View-projection matrix (needed for screen-space selection)
    pub view_projection: Mat4,
    /// Model matrix of the target entity
    pub model_matrix: Mat4,
    /// Number of splats in the target entity
    pub num_splats: u32,
}

impl PickerRequest {
    /// Create a rectangle selection request
    pub fn rect(
        entity: Entity,
        op: SelectionOp,
        rect: RectParams,
        view_projection: Mat4,
        model_matrix: Mat4,
        num_splats: u32,
        use_rings: bool,
    ) -> Self {
        Self {
            active: true,
            target_entity: Some(entity),
            op,
            mode: SelectionMode::Rect,
            use_rings,
            rect: Some(rect),
            sphere: None,
            box_params: None,
            view_projection,
            model_matrix,
            num_splats,
        }
    }

    /// Create a sphere selection request
    pub fn sphere(
        entity: Entity,
        op: SelectionOp,
        center: Vec3,
        radius: f32,
        model_matrix: Mat4,
        num_splats: u32,
        use_rings: bool,
    ) -> Self {
        Self {
            active: true,
            target_entity: Some(entity),
            op,
            mode: SelectionMode::Sphere,
            use_rings,
            rect: None,
            sphere: Some(SphereParams { center, radius }),
            box_params: None,
            view_projection: Mat4::IDENTITY,
            model_matrix,
            num_splats,
        }
    }

    /// Create a box selection request
    pub fn box_select(
        entity: Entity,
        op: SelectionOp,
        center: Vec3,
        half_extents: Vec3,
        model_matrix: Mat4,
        num_splats: u32,
        use_rings: bool,
    ) -> Self {
        Self {
            active: true,
            target_entity: Some(entity),
            op,
            mode: SelectionMode::Box,
            use_rings,
            rect: None,
            sphere: None,
            box_params: Some(BoxParams {
                center,
                half_extents,
            }),
            view_projection: Mat4::IDENTITY,
            model_matrix,
            num_splats,
        }
    }

    /// Clear the request after processing
    pub fn clear(&mut self) {
        self.active = false;
        self.target_entity = None;
        self.rect = None;
        self.sphere = None;
        self.box_params = None;
    }
}

/// Result of GPU selection operation (main world resource)
#[derive(Resource, Default)]
pub struct PickerResult {
    /// Whether results are ready to be read
    pub ready: bool,
    /// Selection operation that was performed
    pub op: SelectionOp,
    /// Target entity
    pub target_entity: Option<Entity>,
    /// Selection results (one u8 per splat: 0 = not selected, 1 = selected)
    pub results: Vec<u8>,
    /// Whether the results have been applied
    pub applied: bool,
}

/// Pending GPU readback state (main world resource)
/// Uses Arc<Mutex<>> to share data between render thread and main thread
#[derive(Resource, Default)]
pub struct PickerPendingReadback {
    /// Shared state for async readback
    pub pending: Option<Arc<Mutex<PendingReadbackData>>>,
}

/// Data for pending readback
pub struct PendingReadbackData {
    pub op: SelectionOp,
    pub target_entity: Option<Entity>,
    pub num_splats: u32,
    pub ready: bool,
    pub data: Vec<u8>,
}

/// Extracted picker request (render world)
#[derive(Resource, Default)]
pub struct ExtractedPickerRequest {
    pub active: bool,
    pub target_entity: Option<Entity>,
    pub use_rings: bool,
    pub op: SelectionOp,
    pub mode: SelectionMode,
    pub rect: Option<RectParams>,
    pub sphere: Option<SphereParams>,
    pub box_params: Option<BoxParams>,
    pub view_projection: Mat4,
    pub model_matrix: Mat4,
    pub num_splats: u32,
}

/// Render world state for picker
#[derive(Resource, Default)]
pub struct RenderPickerState {
    /// Pending readback shared with main world
    pub pending_readback: Option<Arc<Mutex<PendingReadbackData>>>,
    /// Whether we're waiting for a readback
    pub waiting_for_readback: bool,
}

/// Selection compute parameters (GPU uniform)
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
#[repr(C)]
pub struct SelectionParams {
    /// Number of splats to process
    pub num_splats: u32,
    /// Selection mode (0=mask, 1=rect, 2=sphere, 3=box)
    pub mode: u32,
    /// Use rings mode (0=centers/points, 1=rings/ellipses)
    pub use_rings: u32,
    /// Padding
    pub _padding: u32,
    /// View-projection matrix
    pub view_projection: Mat4,
    /// Model matrix
    pub model_matrix: Mat4,
    /// Rectangle parameters (x1, y1, x2, y2) - NDC space [-1, 1]
    pub rect_params: Vec4,
    /// Sphere parameters (x, y, z, radius) - world space
    pub sphere_params: Vec4,
    /// Box center (x, y, z, 0) - world space
    pub box_center: Vec4,
    /// Box half extents (x, y, z, 0) - world space
    pub box_half_extents: Vec4,
}

/// GPU resources for selection compute (render world resource)
#[derive(Resource)]
pub struct SelectionComputeResources {
    /// Uniform buffer for selection parameters
    pub params_buffer: Buffer,
    /// GPU result buffer (u32 per splat for compute shader output)
    pub result_buffer: Buffer,
    /// Staging buffer for CPU readback
    pub staging_buffer: Buffer,
    /// Bind group for selection compute
    pub bind_group: Option<BindGroup>,
    /// Number of splats this resource can handle
    pub capacity: u32,
}

/// Selection compute pipeline
#[derive(Resource)]
pub struct SelectionComputePipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shader: Handle<Shader>,
}

impl FromWorld for SelectionComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            Some("selection_compute_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // @binding(0): Selection parameters uniform
                    uniform_buffer::<SelectionParams>(false),
                    // @binding(1): Position buffer (read-only)
                    storage_buffer_read_only_sized(false, None),
                    // @binding(2): Result buffer (read-write)
                    bevy::render::render_resource::binding_types::storage_buffer_sized(false, None),
                    // @binding(3): Scale buffer (read-only)
                    storage_buffer_read_only_sized(false, None),
                    // @binding(4): Rotation buffer (read-only)
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );

        // Load embedded shader
        let shader = load_embedded_asset!(asset_server, "../assets/shaders/selection_compute.wgsl");

        Self {
            bind_group_layout,
            shader,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct SelectionComputePipelineKey;

impl SpecializedComputePipeline for SelectionComputePipeline {
    type Key = SelectionComputePipelineKey;

    fn specialize(&self, _key: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("selection_compute_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: self.shader.clone(),
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: true,
        }
    }
}

#[derive(Resource)]
pub struct SelectionPipelineId(pub CachedComputePipelineId);

/// Extract picker request from main world to render world
fn extract_picker_request(
    mut extracted: ResMut<ExtractedPickerRequest>,
    mut render_state: ResMut<RenderPickerState>,
    request: Extract<Res<PickerRequest>>,
    main_pending: Extract<Res<PickerPendingReadback>>,
) {
    if request.active && !render_state.waiting_for_readback {
        extracted.active = true;
        extracted.target_entity = request.target_entity;
        extracted.use_rings = request.use_rings;
        extracted.op = request.op;
        extracted.mode = request.mode;
        extracted.rect = request.rect;
        extracted.sphere = request.sphere;
        extracted.box_params = request.box_params;
        extracted.view_projection = request.view_projection;
        extracted.model_matrix = request.model_matrix;
        extracted.num_splats = request.num_splats;

        // Use shared pending state from main world if available, otherwise create new
        if let Some(ref pending_arc) = main_pending.pending {
            render_state.pending_readback = Some(pending_arc.clone());
        } else {
            // Create new shared pending readback state
            let pending_data = Arc::new(Mutex::new(PendingReadbackData {
                op: request.op,
                target_entity: request.target_entity,
                num_splats: request.num_splats,
                ready: false,
                data: Vec::new(),
            }));
            render_state.pending_readback = Some(pending_data);
        }
    } else {
        extracted.active = false;
    }
}

/// Prepare selection compute pipeline
fn prepare_selection_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedComputePipelines<SelectionComputePipeline>>,
    pipeline: Res<SelectionComputePipeline>,
    request: Res<ExtractedPickerRequest>,
) {
    if !request.active {
        return;
    }

    let pipeline_id =
        pipelines.specialize(&pipeline_cache, &pipeline, SelectionComputePipelineKey);

    commands.insert_resource(SelectionPipelineId(pipeline_id));
}

/// Prepare selection compute resources
fn prepare_selection_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    request: Res<ExtractedPickerRequest>,
    existing_resources: Option<Res<SelectionComputeResources>>,
) {
    if !request.active {
        return;
    }

    let num_splats = request.num_splats;
    let buffer_size = (num_splats as u64) * 4; // u32 per splat

    // Check if we need to (re)create resources
    let need_recreate = existing_resources
        .as_ref()
        .map_or(true, |r| r.capacity < num_splats);

    if need_recreate {
        // Create uniform buffer with default params
        let params = SelectionParams {
            num_splats,
            mode: 1, // Rect
            use_rings: 0, // Centers mode by default
            _padding: 0,
            view_projection: Mat4::IDENTITY,
            model_matrix: Mat4::IDENTITY,
            rect_params: Vec4::ZERO,
            sphere_params: Vec4::ZERO,
            box_center: Vec4::ZERO,
            box_half_extents: Vec4::ZERO,
        };

        let params_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("selection_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let result_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("selection_result_buffer"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("selection_staging_buffer"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        commands.insert_resource(SelectionComputeResources {
            params_buffer,
            result_buffer,
            staging_buffer,
            bind_group: None,
            capacity: num_splats,
        });

        info!(
            "Created selection compute resources for {} splats",
            num_splats
        );
    }
}

/// Prepare selection bind groups
fn prepare_selection_bind_groups(
    render_device: Res<RenderDevice>,
    pipeline: Res<SelectionComputePipeline>,
    request: Res<ExtractedPickerRequest>,
    mut resources: Option<ResMut<SelectionComputeResources>>,
    gpu_buffers_query: Query<&GaussianSplatGpuBuffers>,
) {
    if !request.active {
        return;
    }

    let Some(ref mut resources) = resources else {
        return;
    };

    // Find GPU buffers for target entity
    let gpu_buffers = if let Some(target_entity) = request.target_entity {
        gpu_buffers_query.get(target_entity).ok()
    } else {
        // Use first available entity's buffers
        gpu_buffers_query.iter().next()
    };

    let Some(gpu_buffers) = gpu_buffers else {
        warn!("No GPU buffers found for selection compute");
        return;
    };

    if resources.bind_group.is_none() {
        let bind_group = render_device.create_bind_group(
            Some("selection_compute_bind_group"),
            &pipeline.bind_group_layout,
            &BindGroupEntries::sequential((
                resources.params_buffer.as_entire_binding(),
                gpu_buffers.position_buffer.as_binding(),  // Use as_binding() for offset support
                resources.result_buffer.as_entire_binding(),
                gpu_buffers.scale_buffer.as_binding(),     // Use as_binding() for offset support
                gpu_buffers.rotation_buffer.as_binding(),  // Use as_binding() for offset support
            )),
        );

        resources.bind_group = Some(bind_group);
        info!("Created selection bind group");
    }
}

/// Execute selection compute shader and initiate GPU readback
fn execute_selection_compute(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline_id: Option<Res<SelectionPipelineId>>,
    request: Res<ExtractedPickerRequest>,
    resources: Option<Res<SelectionComputeResources>>,
    mut render_state: ResMut<RenderPickerState>,
    mut extracted_request: ResMut<ExtractedPickerRequest>,
) {
    if !request.active {
        return;
    }

    let Some(pipeline_id) = pipeline_id else {
        return;
    };

    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id.0) else {
        return;
    };

    let Some(resources) = resources else {
        return;
    };

    let Some(bind_group) = &resources.bind_group else {
        return;
    };

    // Update params buffer
    let mode = match request.mode {
        SelectionMode::Mask => 0u32,
        SelectionMode::Rect => 1u32,
        SelectionMode::Sphere => 2u32,
        SelectionMode::Box => 3u32,
    };

    let rect_ndc = request
        .rect
        .map(|r| {
            let (x1, y1, x2, y2) = r.to_ndc();
            Vec4::new(x1, y1, x2, y2)
        })
        .unwrap_or(Vec4::ZERO);

    let params = SelectionParams {
        num_splats: request.num_splats,
        mode,
        use_rings: if request.use_rings { 1 } else { 0 },
        _padding: 0,
        view_projection: request.view_projection,
        model_matrix: request.model_matrix,
        rect_params: rect_ndc,
        sphere_params: request
            .sphere
            .map(|s| Vec4::new(s.center.x, s.center.y, s.center.z, s.radius))
            .unwrap_or(Vec4::ZERO),
        box_center: request
            .box_params
            .map(|b| Vec4::new(b.center.x, b.center.y, b.center.z, 0.0))
            .unwrap_or(Vec4::ZERO),
        box_half_extents: request
            .box_params
            .map(|b| Vec4::new(b.half_extents.x, b.half_extents.y, b.half_extents.z, 0.0))
            .unwrap_or(Vec4::ZERO),
    };

    render_queue.write_buffer(&resources.params_buffer, 0, bytemuck::bytes_of(&params));

    // Create command encoder
    let mut encoder =
        render_device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
            label: Some("selection_compute_encoder"),
        });

    // Execute compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("selection_compute_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);

        // 256 threads per workgroup
        let workgroup_count = (request.num_splats + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Copy results to staging buffer
    let buffer_size = (request.num_splats as u64) * 4;
    encoder.copy_buffer_to_buffer(
        &resources.result_buffer,
        0,
        &resources.staging_buffer,
        0,
        buffer_size,
    );

    // Submit command buffer
    render_queue.submit(Some(encoder.finish()));

    // Synchronous buffer mapping using wgpu_device poll
    let staging_buffer = resources.staging_buffer.clone();
    let num_splats = request.num_splats;
    let pending_data = render_state.pending_readback.clone();

    if let Some(pending) = pending_data {
        let buffer_slice = staging_buffer.slice(..);

        // Use atomic flag for synchronization
        let mapping_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mapping_done_clone = mapping_done.clone();

        // Map the buffer asynchronously
        buffer_slice.map_async(MapMode::Read, move |result| {
            if result.is_ok() {
                mapping_done_clone.store(true, std::sync::atomic::Ordering::Release);
            } else {
                warn!("Failed to map selection staging buffer");
            }
        });

        // Poll device until mapping completes (synchronous wait)
        let wgpu_device = render_device.wgpu_device();
        let timeout = std::time::Duration::from_secs(5);
        let start = std::time::Instant::now();
        let mut poll_count = 0;

        loop {
            let _ = wgpu_device.poll(wgpu::PollType::Wait);
            poll_count += 1;

            if mapping_done.load(std::sync::atomic::Ordering::Acquire) {
                // Read the mapped data
                let data = buffer_slice.get_mapped_range();
                let gpu_results: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                staging_buffer.unmap();

                // Convert u32 results to u8 (0 or 1)
                let results: Vec<u8> = gpu_results.iter().map(|&v| if v != 0 { 1 } else { 0 }).collect();

                // Store results in pending data for main world to read
                if let Ok(mut pending_lock) = pending.lock() {
                    pending_lock.num_splats = num_splats;
                    pending_lock.data = results;
                    pending_lock.ready = true;
                }

                info!(
                    "Selection compute complete: {} splats, {} selected (poll count: {})",
                    num_splats,
                    gpu_results.iter().filter(|&&v| v != 0).count(),
                    poll_count
                );
                break;
            }

            if start.elapsed() > timeout {
                warn!("Timeout waiting for selection buffer mapping!");
                break;
            }

            if poll_count > 10000 {
                warn!("Too many poll attempts for selection buffer!");
                break;
            }

            // Avoid busy waiting
            if poll_count % 100 == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        render_state.waiting_for_readback = false;
    }

    info!(
        "Executed selection compute for {} splats (mode: {:?})",
        request.num_splats, request.mode
    );

    // Mark request as processed
    extracted_request.active = false;
}

/// Setup pending readback when a picker request is activated (main world system)
fn setup_picker_pending(
    mut pending: ResMut<PickerPendingReadback>,
    request: Res<PickerRequest>,
) {
    // Create pending readback state when request becomes active
    if request.active && pending.pending.is_none() {
        let pending_data = Arc::new(Mutex::new(PendingReadbackData {
            op: request.op,
            target_entity: request.target_entity,
            num_splats: request.num_splats,
            ready: false,
            data: Vec::new(),
        }));
        pending.pending = Some(pending_data);
        info!("Created pending readback for {} splats", request.num_splats);
    }
}

/// Poll for picker readback completion (main world system)
fn poll_picker_readback(
    mut pending: ResMut<PickerPendingReadback>,
    mut result: ResMut<PickerResult>,
) {
    // Check if we have pending readback
    let Some(ref pending_arc) = pending.pending else {
        return;
    };

    // Clone the Arc to avoid borrow issues - must be done before any mutable access to pending
    let pending_clone = pending_arc.clone();
    let _ = pending_arc; // Release the borrow
    
    // Try to get the data and extract what we need
    let ready_data = match pending_clone.try_lock() {
        Ok(data) if data.ready && !data.data.is_empty() => {
            Some((data.op, data.target_entity, data.data.clone()))
        }
        _ => None,
    };

    // Process the data outside of the lock
    if let Some((op, target_entity, data)) = ready_data {
        // Transfer results to PickerResult
        result.ready = true;
        result.applied = false;
        result.op = op;
        result.target_entity = target_entity;
        result.results = data;

        info!(
            "Picker readback complete: {} results",
            result.results.len()
        );

        // Clear pending
        pending.pending = None;
    }
}

/// System to apply picker results to splat state (main world)
fn apply_picker_results(
    mut picker_result: ResMut<PickerResult>,
    mut picker_request: ResMut<PickerRequest>,
    mut splat_query: Query<&mut SplatSelectionState>,
) {
    if !picker_result.ready || picker_result.applied {
        return;
    }

    let Some(_target_entity) = picker_result.target_entity else {
        picker_result.applied = true;
        return;
    };

    // Find the splat state for the target entity
    // Note: In render world, entities have different IDs, so this needs proper mapping
    // For now, we apply to all splat states (simple case with single splat entity)
    for mut splat_state in splat_query.iter_mut() {
        if splat_state.states.len() != picker_result.results.len() {
            continue;
        }

        // Apply selection based on operation
        match picker_result.op {
            SelectionOp::Set => {
                // Replace selection
                for (i, &result) in picker_result.results.iter().enumerate() {
                    if let Some(state) = splat_state.states.get_mut(i) {
                        let is_locked =
                            (*state & crate::gaussian_splats::splat_state::LOCKED) != 0;
                        let is_deleted =
                            (*state & crate::gaussian_splats::splat_state::DELETED) != 0;

                        if !is_locked && !is_deleted {
                            if result != 0 {
                                *state |= crate::gaussian_splats::splat_state::SELECTED;
                            } else {
                                *state &= !crate::gaussian_splats::splat_state::SELECTED;
                            }
                        }
                    }
                }
            }
            SelectionOp::Add => {
                // Add to selection
                for (i, &result) in picker_result.results.iter().enumerate() {
                    if result != 0 {
                        if let Some(state) = splat_state.states.get_mut(i) {
                            let is_locked =
                                (*state & crate::gaussian_splats::splat_state::LOCKED) != 0;
                            let is_deleted =
                                (*state & crate::gaussian_splats::splat_state::DELETED) != 0;

                            if !is_locked && !is_deleted {
                                *state |= crate::gaussian_splats::splat_state::SELECTED;
                            }
                        }
                    }
                }
            }
            SelectionOp::Remove => {
                // Remove from selection (skip locked and deleted splats)
                for (i, &result) in picker_result.results.iter().enumerate() {
                    if result != 0 {
                        if let Some(state) = splat_state.states.get_mut(i) {
                            let is_locked =
                                (*state & crate::gaussian_splats::splat_state::LOCKED) != 0;
                            let is_deleted =
                                (*state & crate::gaussian_splats::splat_state::DELETED) != 0;

                            if !is_locked && !is_deleted {
                                *state &= !crate::gaussian_splats::splat_state::SELECTED;
                            }
                        }
                    }
                }
            }
        }

        splat_state.recount();
        splat_state.dirty = true;

        info!(
            "Applied selection: {} selected, {} locked, {} deleted",
            splat_state.num_selected, splat_state.num_locked, splat_state.num_deleted
        );
    }

    picker_result.applied = true;
    picker_request.clear();
}
