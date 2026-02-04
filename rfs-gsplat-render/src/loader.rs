// File loader for Gaussian Splats from various formats
// This module wraps tinygsplat_io functions to convert between GaussianSplatsData and GaussianSplats
//
// Note: This module is only available with the "native" feature (not on wasm)

#![cfg(feature = "native")]

use std::path::Path;
use crate::gaussian_splats::GaussianSplats;

/// Load Gaussian Splats from a PLY file
pub fn load_ply_file(path: impl AsRef<Path>) -> Result<GaussianSplats, String> {
    let data = tinygsplat_io::load_ply_file(path)?;
    Ok(convert_from_data(data))
}

/// Load Gaussian Splats from a .splat file
pub fn load_splat_file(path: impl AsRef<Path>) -> Result<GaussianSplats, String> {
    let data = tinygsplat_io::load_splat_file(path)?;
    Ok(convert_from_data(data))
}

/// Load Gaussian Splats from a compressed PLY file
pub fn load_compress_ply_file(path: impl AsRef<Path>) -> Result<GaussianSplats, String> {
    let data = tinygsplat_io::load_compress_ply_file(path)?;
    Ok(convert_from_data(data))
}

/// Detect file format and load accordingly
pub fn load_gaussian_file(path: impl AsRef<Path>) -> Result<GaussianSplats, String> {
    let data = tinygsplat_io::load_gaussian_file(path)?;
    Ok(convert_from_data(data))
}

/// Load Gaussian Splats from a SPZ file (Niantic's compressed format)
pub fn load_spz_file(path: impl AsRef<Path>) -> Result<GaussianSplats, String> {
    let data = tinygsplat_io::load_spz_file(path)?;
    Ok(convert_from_data(data))
}

/// Save Gaussian Splats to a SPZ file (Niantic's compressed format)
pub fn save_spz_file(path: impl AsRef<Path>, splats: &GaussianSplats) -> Result<(), String> {
    let data = convert_to_data(splats);
    tinygsplat_io::save_spz_file(path, &data)
}

/// Save Gaussian Splats to a .splat file
pub fn save_splat_file(path: impl AsRef<Path>, splats: &GaussianSplats) -> Result<(), String> {
    let data = convert_to_data(splats);
    tinygsplat_io::save_splat_file(path, &data)
}

/// Save Gaussian Splats to a regular PLY file
pub fn save_ply_file(path: impl AsRef<Path>, splats: &GaussianSplats) -> Result<(), String> {
    let data = convert_to_data(splats);
    tinygsplat_io::save_ply_file(path, &data)
}

/// Save Gaussian Splats to a compressed PLY file
pub fn save_compress_ply_file(path: impl AsRef<Path>, splats: &GaussianSplats) -> Result<(), String> {
    let data = convert_to_data(splats);
    tinygsplat_io::save_compress_ply_file(path, &data)
}

/// Load Gaussian Splats from a SOG file (compressed format with k-means quantization)
pub fn load_sog_file(path: impl AsRef<Path>) -> Result<GaussianSplats, String> {
    let data = tinygsplat_io::load_sog_file(path)?;
    Ok(convert_from_data(data))
}

/// Save Gaussian Splats to a SOG file (compressed format with k-means quantization)
pub fn save_sog_file(path: impl AsRef<Path>, splats: &GaussianSplats) -> Result<(), String> {
    let data = convert_to_data(splats);
    tinygsplat_io::save_sog_file(path, &data)
}

/// Save Gaussian Splats to SOG format in memory (returns a ZIP archive as bytes)
pub fn save_sog_to_memory(splats: &GaussianSplats) -> Result<Vec<u8>, String> {
    let data = convert_to_data(splats);
    tinygsplat_io::save_sog_to_memory(&data)
}

/// Convert from tinygsplat_io::GaussianSplatsData to rfs-gsplat-render::GaussianSplats
fn convert_from_data(data: tinygsplat_io::GaussianSplatsData) -> GaussianSplats {
    let antialiased = data.antialiased;
    GaussianSplats::new(
        data.means,
        data.rotations,
        data.log_scales,
        data.sh_coeffs,
        data.raw_opacities,
    ).with_antialiased(antialiased)
}

/// Convert from rfs-gsplat-render::GaussianSplats to tinygsplat_io::GaussianSplatsData
fn convert_to_data(splats: &GaussianSplats) -> tinygsplat_io::GaussianSplatsData {
    tinygsplat_io::GaussianSplatsData::new(
        splats.means.clone(),
        splats.rotations.clone(),
        splats.log_scales.clone(),
        splats.sh_coeffs.clone(),
        splats.raw_opacities.clone(),
    ).with_antialiased(splats.antialiased)
}
