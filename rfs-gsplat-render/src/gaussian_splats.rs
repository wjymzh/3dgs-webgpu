// Gaussian Splats data structure and management

use bevy::prelude::*;
// Bytemuck traits will be used when needed for GPU upload
use glam::{Vec3, Vec4};

/// Point cloud data structure (CPU side)
#[derive(Component, Clone)]
pub struct GaussianSplatPointCloud {
    pub positions: Vec<Vec4>,  // Use Vec4 instead of Vec3 to match GPU's 16-byte alignment
    pub colors: Vec<Vec4>,
    pub point_size: f32,
}

impl GaussianSplatPointCloud {
    /// Create a new point cloud, automatically validating data length consistency
    pub fn new(positions: Vec<Vec4>, colors: Vec<Vec4>, point_size: f32) -> Self {
        assert_eq!(
            positions.len(),
            colors.len(),
            "positions and colors must have the same length"
        );
        Self {
            positions,
            colors,
            point_size,
        }
    }

    /// Get the number of points
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}


/// A Gaussian Splat representation
/// This is a Bevy component that can be attached to entities
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct GaussianSplats {
    /// Positions of splats (N x 3)
    pub means: Vec<Vec3>,
    /// Rotation quaternions (N x 4)
    pub rotations: Vec<Vec4>,
    /// Log-space scales (N x 3)
    pub log_scales: Vec<Vec3>,
    /// Spherical harmonic coefficients (N x C x 3), C = number of SH coefficients
    pub sh_coeffs: Vec<Vec<Vec3>>,
    /// Raw opacity values (before sigmoid) (N)
    pub raw_opacities: Vec<f32>,
    /// Whether antialiasing was enabled during training
    /// This should be used to set RenderingConfig.antialias when spawning entities
    pub antialiased: bool,
    /// Stored capacity for training (Vec::clone doesn't preserve capacity)
    pub stored_capacity: usize,
}

impl Default for GaussianSplats {
    fn default() -> Self {
        Self {
            means: Vec::new(),
            rotations: Vec::new(),
            log_scales: Vec::new(),
            sh_coeffs: Vec::new(),
            raw_opacities: Vec::new(),
            antialiased: false,
            stored_capacity: 0,
        }
    }
}

impl GaussianSplats {
    /// Create new Gaussian Splats from raw data
    pub fn new(
        means: Vec<Vec3>,
        rotations: Vec<Vec4>,
        log_scales: Vec<Vec3>,
        sh_coeffs: Vec<Vec<Vec3>>,
        raw_opacities: Vec<f32>,
    ) -> Self {
        assert_eq!(means.len(), rotations.len(), "Means and rotations must have same length");
        assert_eq!(means.len(), log_scales.len(), "Means and log_scales must have same length");
        assert_eq!(means.len(), sh_coeffs.len(), "Means and sh_coeffs must have same length");
        assert_eq!(means.len(), raw_opacities.len(), "Means and raw_opacities must have same length");
        
        let len = means.len();
        Self {
            means,
            rotations,
            log_scales,
            sh_coeffs,
            raw_opacities,
            antialiased: false,
            stored_capacity: len,
        }
    }
    
    /// Set antialiased flag
    pub fn with_antialiased(mut self, antialiased: bool) -> Self {
        self.antialiased = antialiased;
        self
    }
    
    /// Number of splats
    pub fn len(&self) -> usize {
        self.means.len()
    }
    
    /// Number of splats (alias for len())
    pub fn num_splats(&self) -> usize {
        self.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.means.is_empty()
    }
    
    /// Calculate the local space bounding box from all splat positions
    /// Returns (min, max) tuple of the axis-aligned bounding box
    pub fn calculate_local_aabb(&self) -> Option<(Vec3, Vec3)> {
        if self.means.is_empty() {
            return None;
        }
        
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        
        for &pos in &self.means {
            min = min.min(pos);
            max = max.max(pos);
        }
        
        Some((min, max))
    }
    
    /// Get SH degree from coefficients count
    pub fn sh_degree(&self) -> u32 {
        if self.sh_coeffs.is_empty() {
            return 0;
        }
        let num_coeffs = self.sh_coeffs[0].len() as u32;
        sh_degree_from_coeffs(num_coeffs)
    }
    
    /// Compute axis-aligned bounding box of the scene
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        if self.means.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        
        let mut min = self.means[0];
        let mut max = self.means[0];
        
        for &pos in &self.means {
            min = min.min(pos);
            max = max.max(pos);
        }
        
        (min, max)
    }
    
    /// Get center of the scene
    pub fn center(&self) -> Vec3 {
        let (min, max) = self.bounding_box();
        (min + max) * 0.5
    }
    
    /// Get size (extent) of the scene
    pub fn size(&self) -> Vec3 {
        let (min, max) = self.bounding_box();
        max - min
    }
    
    /// Get suggested camera distance based on scene size
    pub fn suggested_camera_distance(&self) -> f32 {
        let size = self.size();
        let max_extent = size.x.max(size.y).max(size.z);
        // Camera distance = 2.5x the max extent to comfortably view the whole scene
        // Increased from 1.5x to give more breathing room
        max_extent * 2.5
    }
    
    /// Merge another GaussianSplats into this one
    /// This appends all splats from `other` to `self`
    /// The antialiased flag is preserved if either self or other has it enabled
    pub fn merge(&mut self, other: &GaussianSplats) {
        self.means.extend_from_slice(&other.means);
        self.rotations.extend_from_slice(&other.rotations);
        self.log_scales.extend_from_slice(&other.log_scales);
        self.sh_coeffs.extend_from_slice(&other.sh_coeffs);
        self.raw_opacities.extend_from_slice(&other.raw_opacities);
        // If either dataset was trained with antialiasing, preserve that flag
        // This ensures exported models retain the correct rendering mode
        self.antialiased = self.antialiased || other.antialiased;
    }
    
    /// Create empty GaussianSplats with pre-allocated capacity
    /// Useful for training where splats will be added incrementally
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            means: Vec::with_capacity(capacity),
            rotations: Vec::with_capacity(capacity),
            log_scales: Vec::with_capacity(capacity),
            sh_coeffs: Vec::with_capacity(capacity),
            raw_opacities: Vec::with_capacity(capacity),
            antialiased: false,
            stored_capacity: capacity,  // Store the capacity explicitly
        }
    }
    
    /// Clear all splat data while preserving capacity
    /// Useful for resetting training without reallocating memory
    pub fn clear(&mut self) {
        self.means.clear();
        self.rotations.clear();
        self.log_scales.clear();
        self.sh_coeffs.clear();
        self.raw_opacities.clear();
        // Preserve antialiased flag and stored_capacity for potential restart
    }
    
    /// Update the data in-place without changing capacity
    /// This is optimized for training scenarios where we want to update
    /// existing GPU buffers without recreating them
    /// Returns true if the data was updated successfully
    pub fn update_data(&mut self, other: &GaussianSplats) -> bool {
        // Direct clone_from - reuses capacity if possible
        self.means.clone_from(&other.means);
        self.rotations.clone_from(&other.rotations);
        self.log_scales.clone_from(&other.log_scales);
        self.sh_coeffs.clone_from(&other.sh_coeffs);
        self.raw_opacities.clone_from(&other.raw_opacities);
        self.antialiased = other.antialiased;
        
        true
    }
    
    /// Get the current capacity (pre-allocated size)
    /// Uses stored_capacity because Vec::clone() doesn't preserve capacity
    pub fn capacity(&self) -> usize {
        self.stored_capacity.max(self.means.len())
    }
    
    /// Update the data from tinygsplat_io::GaussianSplatsData (training data)
    /// This is used to update from training data without needing to convert to GaussianSplats first.
    /// Returns true if the update was successful
    pub fn update_from_data(&mut self, data: &tinygsplat_io::GaussianSplatsData) -> bool {
        // Direct assignment - more efficient than clear + extend
        self.means.clone_from(&data.means);
        self.rotations.clone_from(&data.rotations);
        self.log_scales.clone_from(&data.log_scales);
        self.sh_coeffs.clone_from(&data.sh_coeffs);
        self.raw_opacities.clone_from(&data.raw_opacities);
        self.antialiased = data.antialiased;
        
        true
    }
    
    /// Extract a subset of splats based on indices
    /// Returns a new GaussianSplats containing only the splats at the given indices
    pub fn extract_subset(&self, indices: &[usize]) -> Self {
        let mut means = Vec::with_capacity(indices.len());
        let mut rotations = Vec::with_capacity(indices.len());
        let mut log_scales = Vec::with_capacity(indices.len());
        let mut sh_coeffs = Vec::with_capacity(indices.len());
        let mut raw_opacities = Vec::with_capacity(indices.len());
        
        for &idx in indices {
            if idx < self.means.len() {
                means.push(self.means[idx]);
                rotations.push(self.rotations[idx]);
                log_scales.push(self.log_scales[idx]);
                sh_coeffs.push(self.sh_coeffs[idx].clone());
                raw_opacities.push(self.raw_opacities[idx]);
            }
        }
        
        let len = means.len();
        Self {
            means,
            rotations,
            log_scales,
            sh_coeffs,
            raw_opacities,
            antialiased: self.antialiased,
            stored_capacity: len,
        }
    }
    
    /// Duplicate selected splats in place (append copies to the end)
    /// Returns the starting index of the duplicated splats
    pub fn duplicate_selected(&mut self, selection_state: &SplatSelectionState, offset: Option<Vec3>) -> usize {
        let selected_indices: Vec<usize> = selection_state
            .states
            .iter()
            .enumerate()
            .filter_map(|(i, &state)| {
                if state & splat_state::SELECTED != 0 {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        
        let start_index = self.means.len();
        
        for idx in selected_indices {
            if idx < self.means.len() {
                let mut new_mean = self.means[idx];
                if let Some(offset_vec) = offset {
                    new_mean += offset_vec;
                }
                
                self.means.push(new_mean);
                self.rotations.push(self.rotations[idx]);
                self.log_scales.push(self.log_scales[idx]);
                self.sh_coeffs.push(self.sh_coeffs[idx].clone());
                self.raw_opacities.push(self.raw_opacities[idx]);
            }
        }
        
        start_index
    }
    
}

/// Calculate SH degree from number of coefficients
pub fn sh_degree_from_coeffs(num_coeffs: u32) -> u32 {
    // SH degree d has (d+1)^2 coefficients
    // So we need to find d such that (d+1)^2 = num_coeffs
    let d = (num_coeffs as f32).sqrt() as u32;
    if (d + 1) * (d + 1) == num_coeffs {
        d
    } else {
        // If not a perfect square, return the floor
        d.saturating_sub(1)
    }
}

/// Calculate number of SH coefficients for a given degree
pub fn sh_coeffs_for_degree(degree: u32) -> u32 {
    (degree + 1) * (degree + 1)
}

/// Pack mode configuration component
/// When attached to an entity with GaussianSplats, it enables compressed data format
/// **Default: PACK mode is ENABLED** for automatic memory savings
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct PackModeConfig {
    /// Enable pack mode (compress Gaussian data)
    pub enabled: bool,
}

impl Default for PackModeConfig {
    fn default() -> Self {
        Self { enabled: true }  // 🔥 默认启用PACK模式以节省显存
    }
}

impl PackModeConfig {
    pub fn enabled() -> Self {
        Self { enabled: true }
    }
    
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

/// Inverse sigmoid function
pub fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

/// Sigmoid function
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Create example/test Gaussian Splats
pub fn create_test_splats(count: usize) -> GaussianSplats {
    let mut means = Vec::with_capacity(count);
    let mut rotations = Vec::with_capacity(count);
    let mut log_scales = Vec::with_capacity(count);
    let mut sh_coeffs = Vec::with_capacity(count);
    let mut raw_opacities = Vec::with_capacity(count);
    
    // Create a dense 3D grid of splats for better visual coverage
    let grid_size = (count as f32).cbrt() as usize;
    let spacing = 0.1; // Dense spacing
    let offset = -(grid_size as f32 * spacing) / 2.0;
    
    for i in 0..count {
        let x = (i % grid_size) as f32 * spacing + offset;
        let y = ((i / grid_size) % grid_size) as f32 * spacing + offset;
        let z = (i / (grid_size * grid_size)) as f32 * spacing + offset;
        
        means.push(Vec3::new(x, y, z));
        
        // Random-ish rotation based on position
        let angle = (x + y + z) * 0.5;
        let axis = Vec3::new(x, y, z).normalize_or_zero();
        let s = (angle * 0.5).sin();
        let c = (angle * 0.5).cos();
        rotations.push(Vec4::new(c, axis.x * s, axis.y * s, axis.z * s));
        
        // Much smaller scale for dense coverage: e^(-4.0) ≈ 0.018
        log_scales.push(Vec3::new(-4.0, -4.0, -4.0));
        
        // Color gradient based on position
        let color = Vec3::new(
            (x - offset) / (grid_size as f32 * spacing),
            (y - offset) / (grid_size as f32 * spacing),
            (z - offset) / (grid_size as f32 * spacing),
        );
        sh_coeffs.push(vec![color * 0.5]); // Colorful gradient
        
        raw_opacities.push(inverse_sigmoid(0.9)); // Higher opacity for better coverage
    }
    
    GaussianSplats::new(means, rotations, log_scales, sh_coeffs, raw_opacities)
}

// ============================================================================
// Splat Selection State
// ============================================================================

/// Splat state bit flags (matches supersplat's splatState texture format)
/// Each splat has a u8 state value where:
/// - Bit 0 (0x01): Selected
/// - Bit 1 (0x02): Locked  
/// - Bit 2 (0x04): Deleted (hidden)
pub mod splat_state {
    pub const NORMAL: u8 = 0;
    pub const SELECTED: u8 = 1;
    pub const LOCKED: u8 = 2;
    pub const DELETED: u8 = 4;
}

/// Component to store per-splat selection state
/// This is attached to GaussianSplats entities and stores the state of each splat
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct SplatSelectionState {
    /// Per-splat state (one u8 per splat)
    /// Bits: 0=selected, 1=locked, 2=deleted
    pub states: Vec<u8>,
    /// Number of selected splats (cached for performance)
    pub num_selected: u32,
    /// Number of locked splats (cached for performance)
    pub num_locked: u32,
    /// Number of deleted splats (cached for performance)
    pub num_deleted: u32,
    /// Flag indicating state has changed and needs GPU upload
    pub dirty: bool,
}

impl Default for SplatSelectionState {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            num_selected: 0,
            num_locked: 0,
            num_deleted: 0,
            dirty: false,
        }
    }
}

impl SplatSelectionState {
    /// Create a new selection state for N splats (all normal/unselected)
    pub fn new(num_splats: usize) -> Self {
        Self {
            states: vec![splat_state::NORMAL; num_splats],
            num_selected: 0,
            num_locked: 0,
            num_deleted: 0,
            dirty: true,
        }
    }
    
    /// Select splats by indices
    pub fn select(&mut self, indices: &[u32]) {
        for &idx in indices {
            if let Some(state) = self.states.get_mut(idx as usize) {
                if *state & splat_state::SELECTED == 0 {
                    *state |= splat_state::SELECTED;
                    self.num_selected += 1;
                }
            }
        }
        self.dirty = true;
    }
    
    /// Deselect splats by indices
    pub fn deselect(&mut self, indices: &[u32]) {
        for &idx in indices {
            if let Some(state) = self.states.get_mut(idx as usize) {
                if *state & splat_state::SELECTED != 0 {
                    *state &= !splat_state::SELECTED;
                    self.num_selected -= 1;
                }
            }
        }
        self.dirty = true;
    }
    
    /// Set selection (replace current selection with new indices)
    pub fn set_selection(&mut self, indices: &[u32]) {
        // Clear all selected bits
        for state in &mut self.states {
            *state &= !splat_state::SELECTED;
        }
        self.num_selected = 0;
        
        // Select new indices
        self.select(indices);
    }
    
    /// Select all splats (excluding locked and deleted)
    pub fn select_all(&mut self) {
        self.num_selected = 0;
        for state in &mut self.states {
            if *state & (splat_state::LOCKED | splat_state::DELETED) == 0 {
                *state |= splat_state::SELECTED;
                self.num_selected += 1;
            }
        }
        self.dirty = true;
    }
    
    /// Deselect all splats
    pub fn deselect_all(&mut self) {
        for state in &mut self.states {
            *state &= !splat_state::SELECTED;
        }
        self.num_selected = 0;
        self.dirty = true;
    }
    
    /// Invert selection (toggle selected state for all non-locked, non-deleted splats)
    pub fn invert_selection(&mut self) {
        self.num_selected = 0;
        for state in &mut self.states {
            if *state & (splat_state::LOCKED | splat_state::DELETED) == 0 {
                *state ^= splat_state::SELECTED;
                if *state & splat_state::SELECTED != 0 {
                    self.num_selected += 1;
                }
            }
        }
        self.dirty = true;
    }
    
    /// Delete selected splats (mark as deleted)
    pub fn delete_selected(&mut self) {
        for state in &mut self.states {
            if *state & splat_state::SELECTED != 0 {
                *state |= splat_state::DELETED;
                *state &= !splat_state::SELECTED;
                self.num_deleted += 1;
            }
        }
        self.num_selected = 0;
        self.dirty = true;
    }
    
    /// Undelete all splats (remove deleted flag)
    pub fn undelete_all(&mut self) {
        for state in &mut self.states {
            *state &= !splat_state::DELETED;
        }
        self.num_deleted = 0;
        self.dirty = true;
    }
    
    /// Lock selected splats
    pub fn lock_selected(&mut self) {
        for state in &mut self.states {
            if *state & splat_state::SELECTED != 0 {
                *state |= splat_state::LOCKED;
                *state &= !splat_state::SELECTED;
                self.num_locked += 1;
            }
        }
        self.num_selected = 0;
        self.dirty = true;
    }
    
    /// Unlock all splats
    pub fn unlock_all(&mut self) {
        for state in &mut self.states {
            *state &= !splat_state::LOCKED;
        }
        self.num_locked = 0;
        self.dirty = true;
    }
    
    /// Get indices of selected splats
    pub fn get_selected_indices(&self) -> Vec<u32> {
        self.states.iter()
            .enumerate()
            .filter(|(_, &s)| s & splat_state::SELECTED != 0)
            .map(|(i, _)| i as u32)
            .collect()
    }
    
    /// Check if a splat is selected
    pub fn is_selected(&self, index: usize) -> bool {
        self.states.get(index).map(|s| s & splat_state::SELECTED != 0).unwrap_or(false)
    }
    
    /// Check if a splat is locked
    pub fn is_locked(&self, index: usize) -> bool {
        self.states.get(index).map(|s| s & splat_state::LOCKED != 0).unwrap_or(false)
    }
    
    /// Check if a splat is deleted
    pub fn is_deleted(&self, index: usize) -> bool {
        self.states.get(index).map(|s| s & splat_state::DELETED != 0).unwrap_or(false)
    }
    
    /// Update counts from state array
    pub fn recount(&mut self) {
        self.num_selected = 0;
        self.num_locked = 0;
        self.num_deleted = 0;
        for &state in &self.states {
            if state & splat_state::SELECTED != 0 { self.num_selected += 1; }
            if state & splat_state::LOCKED != 0 { self.num_locked += 1; }
            if state & splat_state::DELETED != 0 { self.num_deleted += 1; }
        }
    }
}

/// Convert from GaussianSplatsData (tinygsplat_io) to GaussianSplats
impl From<tinygsplat_io::GaussianSplatsData> for GaussianSplats {
    fn from(data: tinygsplat_io::GaussianSplatsData) -> Self {
        let len = data.means.len();
        Self {
            means: data.means,
            rotations: data.rotations,
            log_scales: data.log_scales,
            sh_coeffs: data.sh_coeffs,
            raw_opacities: data.raw_opacities,
            antialiased: data.antialiased,
            stored_capacity: len,
        }
    }
}

