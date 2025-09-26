/// Utility functions for the xLSTM library.

/// Sigmoid activation function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Hyperbolic tangent activation function  
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// SiLU (Swish) activation function: x * sigmoid(x)
pub fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// GELU activation function (approximation)
pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Stabilized exponential function to prevent overflow
/// Uses the identity: exp(x) = exp(x - c) * exp(c) where c = max(x)
pub fn exp_stabilized(x: f64, stabilizer: f64) -> f64 {
    (x - stabilizer).exp()
}

/// Apply stabilized exponential to prevent numerical overflow
/// For matrices, finds the max value and uses it as stabilizer
pub fn stabilize_and_exp(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    values.iter().map(|&x| (x - max_val).exp()).collect()
}

/// Safe maximum operation for numerical stability
pub fn safe_max(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.max(b)
    }
}

/// Safe division with epsilon to prevent division by zero
pub fn safe_div(numerator: f64, denominator: f64, epsilon: f64) -> f64 {
    numerator / (denominator + epsilon)
}

/// Find the largest divisor of `size` that is <= `max_blocks`
/// This ensures we can create BlockDiagonal layers with valid block counts
pub fn find_suitable_num_blocks(size: usize, desired_blocks: usize) -> usize {
    if desired_blocks == 0 {
        return 1;
    }

    // Start from the desired number and work down to find a divisor
    for num_blocks in (1..=desired_blocks).rev() {
        if size % num_blocks == 0 {
            return num_blocks;
        }
    }

    // Fallback to 1 if no suitable divisor found
    1
}

/// Find divisors of a number, useful for determining valid block counts
pub fn find_divisors(n: usize) -> Vec<usize> {
    let mut divisors = Vec::new();
    for i in 1..=n {
        if n % i == 0 {
            divisors.push(i);
        }
    }
    divisors
}

/// Xavier uniform initialization for weight matrices
pub fn xavier_uniform(rows: usize, cols: usize) -> ndarray::Array2<f64> {
    use ndarray::Array2;
    use rand::Rng;
    use rand::rng;
    use rand_distr::Uniform;

    let fan_in = cols as f64;
    let fan_out = rows as f64;
    let limit = (6.0 / (fan_in + fan_out)).sqrt();

    let mut rng = rng();
    let dist = Uniform::new(-limit, limit).unwrap();

    Array2::from_shape_fn((rows, cols), |_| rng.sample(dist))
}

/// Initialize bias vectors to zero
pub fn zeros_bias(size: usize) -> ndarray::Array2<f64> {
    ndarray::Array2::zeros((size, 1))
}

#[cfg(test)]
mod tests {
    use rand_distr::num_traits::Float;

    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(1000.0) > 0.99);
        assert!(sigmoid(-1000.0) < 0.01);
    }

    #[test]
    fn test_silu() {
        let x = 1.0;
        let expected = x * sigmoid(x);
        assert!((silu(x) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gelu() {
        // GELU should be approximately 0 at x=0, and close to x for large positive x
        assert!(gelu(0.0).abs() < 0.1);
        assert!((gelu(5.0) - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_exp_stabilized() {
        let x = 100.0;
        let stabilizer = 99.0;
        let result = exp_stabilized(x, stabilizer);
        let expected = (x - stabilizer).exp();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_stabilize_and_exp() {
        let values = vec![1.0, 2.0, 3.0];
        let result = stabilize_and_exp(&values);

        // Check that the relative proportions are maintained
        let sum: f64 = result.iter().sum();
        assert!(sum > 0.0);

        // Largest value should correspond to largest result
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_find_suitable_num_blocks() {
        assert_eq!(find_suitable_num_blocks(8, 4), 4); // 8 is divisible by 4
        assert_eq!(find_suitable_num_blocks(6, 4), 3); // 6 is not divisible by 4, but is by 3
        assert_eq!(find_suitable_num_blocks(5, 4), 1); // 5 is prime, only divisible by 1 and 5
        assert_eq!(find_suitable_num_blocks(12, 5), 4); // 12 is not divisible by 5, but is by 4
        assert_eq!(find_suitable_num_blocks(1, 4), 1); // 1 is only divisible by 1
    }

    #[test]
    fn test_find_divisors() {
        assert_eq!(find_divisors(1), vec![1]);
        assert_eq!(find_divisors(6), vec![1, 2, 3, 6]);
        assert_eq!(find_divisors(8), vec![1, 2, 4, 8]);
        assert_eq!(find_divisors(12), vec![1, 2, 3, 4, 6, 12]);
    }

    #[test]
    fn test_xavier_uniform() {
        let weights = xavier_uniform(10, 5);
        assert_eq!(weights.shape(), &[10, 5]);

        // Check that values are within expected range
        let limit = (6.0 / 15.0).sqrt();
        for &val in weights.iter() {
            assert!(val >= -limit && val <= limit);
        }
    }
}
