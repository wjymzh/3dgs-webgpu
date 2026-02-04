// splat_state.rs - Selection operation types and utilities
// Similar to supersplat's splat-state.ts
// 
// Note: SplatSelectionState component is defined in gaussian_splats.rs
// This module provides selection operation types used by gpu_picker.rs

use bevy::prelude::*;

/// Selection operation mode (matches supersplat's op parameter)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionOp {
    /// Replace current selection with new selection
    #[default]
    Set,
    /// Add to current selection
    Add,
    /// Remove from current selection
    Remove,
}

/// Selection mode for GPU intersection tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionMode {
    /// Select using a 2D mask texture (for lasso/brush)
    Mask,
    /// Select using a screen-space rectangle
    #[default]
    Rect,
    /// Select using a world-space sphere
    Sphere,
    /// Select using a world-space box
    Box,
}

/// Rectangle selection parameters (screen-space, normalized 0-1)
#[derive(Debug, Clone, Copy, Default)]
pub struct RectParams {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl RectParams {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }
    
    /// Convert from pixel coordinates to normalized coordinates
    pub fn from_pixels(x1: f32, y1: f32, x2: f32, y2: f32, width: f32, height: f32) -> Self {
        Self {
            x1: x1.min(x2) / width,
            y1: y1.min(y2) / height,
            x2: x1.max(x2) / width,
            y2: y1.max(y2) / height,
        }
    }
    
    /// Convert to NDC coordinates (normalized device coordinates, -1 to 1)
    pub fn to_ndc(&self) -> (f32, f32, f32, f32) {
        (
            self.x1 * 2.0 - 1.0,
            self.y1 * 2.0 - 1.0,
            self.x2 * 2.0 - 1.0,
            self.y2 * 2.0 - 1.0,
        )
    }
}

/// Sphere selection parameters (world-space)
#[derive(Debug, Clone, Copy, Default)]
pub struct SphereParams {
    pub center: Vec3,
    pub radius: f32,
}

/// Box selection parameters (world-space AABB)
#[derive(Debug, Clone, Copy, Default)]
pub struct BoxParams {
    pub center: Vec3,
    pub half_extents: Vec3,
}

/// Splat state bit constants for GPU shader compatibility
/// These match the definitions in gaussian_splats.rs splat_state module
pub mod state_bits {
    /// Splat is selected (bit 0)
    pub const SELECTED: u32 = 1;
    /// Splat is locked (bit 1)
    pub const LOCKED: u32 = 2;
    /// Splat is deleted (bit 2)
    pub const DELETED: u32 = 4;
}
