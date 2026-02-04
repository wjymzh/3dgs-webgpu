// GPU Radix Sort - 3-Pass Architecture (translated from vrdx)
// Pass 1: Upsweep - local histogram + accumulate to global
// Pass 2: Spine - prefix sum on partition and global histograms  
// Pass 3: Downsweep - scatter using offsets

// Constants (matching vrdx but with smaller sizes for WGSL)
const WG: u32 = 256u;
const RADIX_BITS: u32 = 8u;
const RADIX_SIZE: u32 = 256u;
const RADIX_MASK: u32 = 255u;
const ELEMENTS_PER_THREAD: u32 = 4u;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;  // 1024

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

struct SortParams {
    max_element_count: u32,
    bit_shift: u32,
    pass_index: u32,
    _padding: u32,
}

// ============================================================================
// Pass 1: Upsweep - Count local histogram and accumulate to global
// ============================================================================

@group(0) @binding(0) var<uniform> upsweep_params: SortParams;
// indirect_buffer layout: [vertex_count, instance_count, first_vertex, first_instance]
// We read instance_count (index 1) as the dynamic element count from cull shader
@group(0) @binding(1) var<storage, read> indirect_buffer_upsweep: array<u32>;
@group(0) @binding(2) var<storage, read> keys_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> global_histogram: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> partition_histogram: array<u32>;

var<workgroup> local_histogram: array<atomic<u32>, RADIX_SIZE>;

@compute
@workgroup_size(256, 1, 1)
fn upsweep(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Read dynamic visible count from indirect_buffer[1] (instance_count)
    let num_keys = indirect_buffer_upsweep[1];
    let num_partitions = div_ceil(num_keys, BLOCK_SIZE);
    let partition_id = workgroup_id.x;
    
    if partition_id >= num_partitions {
        return;
    }
    
    let tid = local_id.x;
    let partition_start = partition_id * BLOCK_SIZE;
    let shift = upsweep_params.bit_shift;
    let pass_idx = upsweep_params.pass_index;
    
    // Initialize local histogram
    if tid < RADIX_SIZE {
        atomicStore(&local_histogram[tid], 0u);
    }
    workgroupBarrier();
    
    // Build local histogram
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let key_idx = partition_start + tid * ELEMENTS_PER_THREAD + i;
        if key_idx < num_keys {
            let key = keys_in[key_idx];
            let bin = (key >> shift) & RADIX_MASK;
            atomicAdd(&local_histogram[bin], 1u);
        }
    }
    
    workgroupBarrier();
    
    // Write to partition histogram and accumulate to global histogram
    if tid < RADIX_SIZE {
        let count = atomicLoad(&local_histogram[tid]);
        partition_histogram[RADIX_SIZE * partition_id + tid] = count;
        atomicAdd(&global_histogram[RADIX_SIZE * pass_idx + tid], count);
    }
}

// ============================================================================
// Pass 2: Spine - Prefix sum on partition and global histograms
// ============================================================================

// indirect_buffer layout: [vertex_count, instance_count, first_vertex, first_instance]
@group(0) @binding(0) var<storage, read> indirect_buffer_spine: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_histogram_spine: array<u32>;
@group(0) @binding(2) var<storage, read_write> partition_histogram_spine: array<u32>;
@group(0) @binding(3) var<uniform> spine_params: SortParams;

// Double buffer for data-race-free Hillis-Steele scan
var<workgroup> scan_a: array<u32, 256>;
var<workgroup> scan_b: array<u32, 256>;
var<workgroup> reduction_shared: u32;

@compute
@workgroup_size(256, 1, 1)
fn spine(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Read dynamic visible count from indirect_buffer[1] (instance_count)
    let num_keys = indirect_buffer_spine[1];
    let num_partitions = div_ceil(num_keys, BLOCK_SIZE);
    let bin = workgroup_id.x;
    let tid = local_id.x;
    
    if bin >= RADIX_SIZE {
        return;
    }
    
    // Scan partition histogram for this bin across all partitions
    // Initialize shared reduction
    if tid == 0u {
        reduction_shared = 0u;
    }
    workgroupBarrier();
    
    // Process all partitions for this bin in batches
    let MAX_BATCH_SIZE = 256u;  // Process up to 256 partitions at once
    for (var batch_start = 0u; batch_start < num_partitions; batch_start += MAX_BATCH_SIZE) {
        let partition_idx = batch_start + tid;
        let batch_size = min(MAX_BATCH_SIZE, num_partitions - batch_start);
        
        // Load values for this batch
        if tid < batch_size && partition_idx < num_partitions {
            scan_a[tid] = partition_histogram_spine[RADIX_SIZE * partition_idx + bin];
        } else {
            scan_a[tid] = 0u;
        }
        workgroupBarrier();
        
        // Hillis-Steele inclusive prefix sum with DOUBLE BUFFERING to avoid data race
        var use_a = true;
        var offset = 1u;
        for (var d = 0u; d < 8u; d++) {
            if use_a {
                if tid >= offset {
                    scan_b[tid] = scan_a[tid] + scan_a[tid - offset];
                } else {
                    scan_b[tid] = scan_a[tid];
                }
            } else {
                if tid >= offset {
                    scan_a[tid] = scan_b[tid] + scan_b[tid - offset];
                } else {
                    scan_a[tid] = scan_b[tid];
                }
            }
            workgroupBarrier();
            use_a = !use_a;
            offset <<= 1u;
        }
        
        // After 8 iterations: d=0→B, d=1→A, d=2→B, d=3→A, d=4→B, d=5→A, d=6→B, d=7→A
        // Result is in scan_a!
        
        // Write back as exclusive prefix sum with reduction
        if tid < batch_size && partition_idx < num_partitions {
            var exclusive = reduction_shared;
            if tid > 0u {
                exclusive += scan_a[tid - 1u];  // Fixed: read from scan_a
            }
            partition_histogram_spine[RADIX_SIZE * partition_idx + bin] = exclusive;
        }
        
        // Update reduction for next batch
        workgroupBarrier();
        if tid == 0u && batch_size > 0u {
            reduction_shared += scan_a[batch_size - 1u];  // Fixed: read from scan_a
        }
        workgroupBarrier();
    }
    
    // Bin 0 workgroup also does global histogram prefix sum
    if bin == 0u {
        let pass_idx = spine_params.pass_index;
        scan_a[tid] = global_histogram_spine[RADIX_SIZE * pass_idx + tid];
        workgroupBarrier();
        
        // Hillis-Steele inclusive scan with double buffering
        var use_a = true;
        var offset = 1u;
        for (var d = 0u; d < 8u; d++) {
            if use_a {
                if tid >= offset {
                    scan_b[tid] = scan_a[tid] + scan_a[tid - offset];
                } else {
                    scan_b[tid] = scan_a[tid];
                }
            } else {
                if tid >= offset {
                    scan_a[tid] = scan_b[tid] + scan_b[tid - offset];
                } else {
                    scan_a[tid] = scan_b[tid];
                }
            }
            workgroupBarrier();
            use_a = !use_a;
            offset <<= 1u;
        }
        
        // Convert to exclusive (result in scan_a after 8 iterations)
        var exclusive = 0u;
        if tid > 0u {
            exclusive = scan_a[tid - 1u];  // Fixed: read from scan_a
        }
        global_histogram_spine[RADIX_SIZE * pass_idx + tid] = exclusive;
    }
}

// ============================================================================
// Pass 3: Downsweep - Scatter elements using offsets
// ============================================================================

@group(0) @binding(0) var<uniform> downsweep_params: SortParams;
// indirect_buffer layout: [vertex_count, instance_count, first_vertex, first_instance]
@group(0) @binding(1) var<storage, read> indirect_buffer_downsweep: array<u32>;
@group(0) @binding(2) var<storage, read> global_histogram_downsweep: array<u32>;
@group(0) @binding(3) var<storage, read> partition_histogram_downsweep: array<u32>;
@group(0) @binding(4) var<storage, read> downsweep_keys_in: array<u32>;
@group(0) @binding(5) var<storage, read> downsweep_values_in: array<u32>;
@group(0) @binding(6) var<storage, read_write> downsweep_keys_out: array<u32>;
@group(0) @binding(7) var<storage, read_write> downsweep_values_out: array<u32>;

var<workgroup> local_keys: array<u32, BLOCK_SIZE>;
var<workgroup> local_values: array<u32, BLOCK_SIZE>;
var<workgroup> local_bins: array<u32, BLOCK_SIZE>;

@compute
@workgroup_size(256, 1, 1)
fn downsweep(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Read dynamic visible count from indirect_buffer[1] (instance_count)
    let num_keys = indirect_buffer_downsweep[1];
    let num_partitions = div_ceil(num_keys, BLOCK_SIZE);
    let partition_id = workgroup_id.x;
    
    if partition_id >= num_partitions {
        return;
    }
    
    let tid = local_id.x;
    let partition_start = partition_id * BLOCK_SIZE;
    let shift = downsweep_params.bit_shift;
    
    // Load elements into shared memory
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let key_idx = partition_start + tid * ELEMENTS_PER_THREAD + i;
        let local_idx = tid * ELEMENTS_PER_THREAD + i;
        
        if key_idx < num_keys {
            let key = downsweep_keys_in[key_idx];
            local_keys[local_idx] = key;
            local_values[local_idx] = downsweep_values_in[key_idx];
            local_bins[local_idx] = (key >> shift) & RADIX_MASK;
        } else {
            local_bins[local_idx] = 0xFFFFFFFFu;
        }
    }
    
    workgroupBarrier();
    
    // Thread 0 does sequential scatter to maintain stability
    if tid == 0u {
        var bin_write_pos: array<u32, RADIX_SIZE>;
        
        let pass_idx = downsweep_params.pass_index;
        
        // Initialize write positions from global + partition offsets
        for (var bin = 0u; bin < RADIX_SIZE; bin++) {
            bin_write_pos[bin] = global_histogram_downsweep[RADIX_SIZE * pass_idx + bin] + 
                                 partition_histogram_downsweep[RADIX_SIZE * partition_id + bin];
        }
        
        // Sequential write in input order (stable)
        let partition_end = min(partition_start + BLOCK_SIZE, num_keys);
        for (var i = 0u; i < BLOCK_SIZE; i++) {
            let key_idx = partition_start + i;
            if key_idx < partition_end {
                let bin = local_bins[i];
                if bin != 0xFFFFFFFFu {
                    let write_pos = bin_write_pos[bin];
                    if write_pos < num_keys {
                        downsweep_keys_out[write_pos] = local_keys[i];
                        downsweep_values_out[write_pos] = local_values[i];
                        bin_write_pos[bin]++;
                    }
                }
            }
        }
    }
}
