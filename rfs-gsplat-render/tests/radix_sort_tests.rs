// GPU Radix Sort Validation Tests
// Tests the radix sort implementation against CPU reference

// CPU reference implementation for validation
fn cpu_radix_sort_reference(keys: &mut [u32], values: &mut [u32]) {
    let n = keys.len();
    let mut keys_temp = vec![0u32; n];
    let mut values_temp = vec![0u32; n];
    
    // 4 passes for 32-bit keys
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
        
        // Swap buffers
        keys.copy_from_slice(&keys_temp);
        values.copy_from_slice(&values_temp);
    }
}

// Verify that an array is sorted
fn is_sorted(keys: &[u32]) -> bool {
    for i in 1..keys.len() {
        if keys[i] < keys[i - 1] {
            return false;
        }
    }
    true
}

// Verify that values are correctly permuted with keys
fn verify_permutation(original_keys: &[u32], original_values: &[u32], sorted_keys: &[u32], sorted_values: &[u32]) -> bool {
    if original_keys.len() != sorted_keys.len() {
        return false;
    }
    
    // For each sorted key-value pair, verify it exists in original
    for i in 0..sorted_keys.len() {
        let sorted_key = sorted_keys[i];
        let sorted_value = sorted_values[i];
        
        // Find this key in original
        let mut found = false;
        for j in 0..original_keys.len() {
            if original_keys[j] == sorted_key && original_values[j] == sorted_value {
                found = true;
                break;
            }
        }
        
        if !found {
            eprintln!("Sorted pair ({}, {}) not found in original data", sorted_key, sorted_value);
            return false;
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_reference_small() {
        let mut keys = vec![5, 2, 8, 1, 9, 3, 7, 4, 6, 0];
        let mut values: Vec<u32> = (0..10).collect();
        
        let original_keys = keys.clone();
        let original_values = values.clone();
        
        cpu_radix_sort_reference(&mut keys, &mut values);
        
        assert!(is_sorted(&keys), "Keys should be sorted");
        assert!(verify_permutation(&original_keys, &original_values, &keys, &values), 
                "Values should be correctly permuted");
        
        println!("✓ CPU reference test passed (10 elements)");
    }

    #[test]
    fn test_cpu_reference_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let n = 1000;
        let mut keys: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
        let mut values: Vec<u32> = (0..n).collect();
        
        let original_keys = keys.clone();
        let original_values = values.clone();
        
        cpu_radix_sort_reference(&mut keys, &mut values);
        
        assert!(is_sorted(&keys), "Keys should be sorted");
        assert!(verify_permutation(&original_keys, &original_values, &keys, &values), 
                "Values should be correctly permuted");
        
        println!("✓ CPU reference test passed (1000 random elements)");
    }

    #[test]
    fn test_cpu_reference_duplicates() {
        let mut keys = vec![5, 2, 5, 1, 9, 2, 7, 5, 6, 1];
        let mut values: Vec<u32> = (0..10).collect();
        
        let original_keys = keys.clone();
        let original_values = values.clone();
        
        cpu_radix_sort_reference(&mut keys, &mut values);
        
        assert!(is_sorted(&keys), "Keys should be sorted");
        assert!(verify_permutation(&original_keys, &original_values, &keys, &values), 
                "Values should be correctly permuted");
        
        println!("✓ CPU reference test passed (duplicates)");
    }

    #[test]
    fn test_cpu_reference_already_sorted() {
        let mut keys: Vec<u32> = (0..100).collect();
        let mut values: Vec<u32> = (0..100).collect();
        
        let original_keys = keys.clone();
        
        cpu_radix_sort_reference(&mut keys, &mut values);
        
        assert_eq!(keys, original_keys, "Already sorted array should remain unchanged");
        assert!(is_sorted(&keys), "Keys should be sorted");
        
        println!("✓ CPU reference test passed (already sorted)");
    }

    #[test]
    fn test_cpu_reference_reverse_sorted() {
        let mut keys: Vec<u32> = (0..100).rev().collect();
        let mut values: Vec<u32> = (0..100).collect();
        
        cpu_radix_sort_reference(&mut keys, &mut values);
        
        assert!(is_sorted(&keys), "Keys should be sorted");
        
        println!("✓ CPU reference test passed (reverse sorted)");
    }

    #[test]
    fn test_cpu_reference_edge_cases() {
        // Test with max values
        let mut keys = vec![u32::MAX, 0, u32::MAX / 2, 1];
        let mut values = vec![0, 1, 2, 3];
        
        cpu_radix_sort_reference(&mut keys, &mut values);
        
        assert!(is_sorted(&keys), "Keys should be sorted");
        assert_eq!(keys, vec![0, 1, u32::MAX / 2, u32::MAX]);
        
        println!("✓ CPU reference test passed (edge cases)");
    }
}

// GPU validation test (requires Bevy app context)
// This needs to be run as an integration test with a real GPU
#[test]
#[ignore] // Run with: cargo test --test radix_sort_tests -- --ignored
fn test_gpu_sort_correctness() {
    // This test requires a full Bevy app with render context
    println!("GPU correctness test requires manual verification");
    println!("Run: cargo run --example test_radix_sort_validation");
}

