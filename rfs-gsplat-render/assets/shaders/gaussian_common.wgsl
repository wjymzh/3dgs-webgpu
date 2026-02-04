// Constants
const SQRT_8: f32 = 2.8284271247461903;  // sqrt(8) standard deviations
const SH_C0: f32 = 0.28209479177387814;  // sqrt(1/(4*pi))


// Inverse sigmoid function
fn inverse_sigmoid(x: f32) -> f32 {
    return log(x / (1.0 - x));
}

// Sigmoid function
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

