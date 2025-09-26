use ndarray::Array2;

/// Layer Normalization implementation.
///
/// Normalizes inputs across the feature dimension for each sample independently.
/// This helps stabilize training and improve convergence in deep networks.
///
/// # Mathematical Operation
///
/// For input x ∈ R^(d, n):
/// ```
/// μ = mean(x, dim=0)  // Mean across features for each sample
/// σ² = var(x, dim=0)   // Variance across features for each sample  
/// y = γ ⊙ (x - μ) / √(σ² + ε) + β
/// ```
///
/// where γ and β are learnable parameters, ⊙ is element-wise multiplication.
///
/// # Example
///
/// ```rust
/// use xlstm_rust::layers::LayerNorm;
/// use ndarray::arr2;
///
/// let mut ln = LayerNorm::new(4);
/// let input = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
/// let output = ln.forward(&input);
/// assert_eq!(output.shape(), &[4, 1]);
/// ```
#[derive(Clone)]
pub struct LayerNorm {
    /// Learnable scale parameter γ: (normalized_shape, 1)
    pub weight: Array2<f64>,
    /// Learnable shift parameter β: (normalized_shape, 1)  
    pub bias: Array2<f64>,
    /// Small constant for numerical stability
    pub epsilon: f64,
    /// Size of the normalized dimension
    pub normalized_shape: usize,
}

impl LayerNorm {
    /// Create a new LayerNorm layer
    ///
    /// # Arguments
    /// * `normalized_shape` - Size of the feature dimension to normalize
    pub fn new(normalized_shape: usize) -> Self {
        // Initialize weight (γ) to 1 and bias (β) to 0
        let weight = Array2::ones((normalized_shape, 1));
        let bias = Array2::zeros((normalized_shape, 1));

        LayerNorm {
            weight,
            bias,
            epsilon: 1e-5,
            normalized_shape,
        }
    }

    /// Create LayerNorm with custom epsilon
    pub fn with_epsilon(normalized_shape: usize, epsilon: f64) -> Self {
        let mut ln = Self::new(normalized_shape);
        ln.epsilon = epsilon;
        ln
    }

    /// Forward pass through layer normalization
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (normalized_shape, batch_size)
    ///
    /// # Returns
    /// * Normalized output tensor of same shape as input
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            input.nrows(),
            self.normalized_shape,
            "Input feature size {} doesn't match expected {}",
            input.nrows(),
            self.normalized_shape
        );

        let batch_size = input.ncols();
        let mut output = Array2::zeros(input.raw_dim());

        // Normalize each sample independently
        for sample_idx in 0..batch_size {
            let sample = input.column(sample_idx);

            // Compute mean and variance across features
            let mean = sample.mean().unwrap_or(0.0);
            let variance = sample.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / self.normalized_shape as f64;

            let std_dev = (variance + self.epsilon).sqrt();

            // Normalize: (x - μ) / σ
            let normalized_sample = sample.mapv(|x| (x - mean) / std_dev);

            // Scale and shift: γ * normalized + β
            let weight_col = self.weight.column(0);
            let bias_col = self.bias.column(0);

            let final_sample = &normalized_sample * &weight_col + &bias_col;

            // Store result
            output.column_mut(sample_idx).assign(&final_sample);
        }

        output
    }

    /// Get the total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weight.len() + self.bias.len()
    }

    /// Get reference to weight (γ) parameter
    pub fn get_weight(&self) -> &Array2<f64> {
        &self.weight
    }

    /// Get reference to bias (β) parameter
    pub fn get_bias(&self) -> &Array2<f64> {
        &self.bias
    }

    /// Set the weight parameter (useful for initialization)
    pub fn set_weight(&mut self, weight: Array2<f64>) {
        assert_eq!(weight.shape(), self.weight.shape(), "Weight shape mismatch");
        self.weight = weight;
    }

    /// Set the bias parameter (useful for initialization)
    pub fn set_bias(&mut self, bias: Array2<f64>) {
        assert_eq!(bias.shape(), self.bias.shape(), "Bias shape mismatch");
        self.bias = bias;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_layer_norm_creation() {
        let ln = LayerNorm::new(4);
        assert_eq!(ln.normalized_shape, 4);
        assert_eq!(ln.epsilon, 1e-5);
        assert_eq!(ln.weight.shape(), &[4, 1]);
        assert_eq!(ln.bias.shape(), &[4, 1]);

        // Check default initialization
        for &w in ln.weight.iter() {
            assert!((w - 1.0).abs() < 1e-10);
        }
        for &b in ln.bias.iter() {
            assert!(b.abs() < 1e-10);
        }
    }

    #[test]
    fn test_layer_norm_forward() {
        let ln = LayerNorm::new(3);

        // Input with mean=2, std≈0.816
        let input = arr2(&[[1.0], [2.0], [3.0]]);
        let output = ln.forward(&input);

        // After normalization, should have mean≈0, std≈1
        let output_col = output.column(0);
        let mean = output_col.mean().unwrap();
        let variance = output_col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 3.0;

        assert!(
            mean.abs() < 1e-10,
            "Mean should be close to 0, got {}",
            mean
        );
        assert!(
            (variance.sqrt() - 1.0).abs() < 1e-6,
            "Std should be close to 1, got {}",
            variance.sqrt()
        );
    }

    #[test]
    fn test_layer_norm_batch() {
        let ln = LayerNorm::new(2);

        // Batch of 3 samples
        let input = arr2(&[
            [1.0, 4.0, 7.0], // Sample 1: mean=2.5
            [3.0, 6.0, 9.0], // Sample 2: mean=7.5
        ]);

        let output = ln.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);

        // Each sample should be independently normalized
        for sample_idx in 0..3 {
            let sample = output.column(sample_idx);
            let mean = sample.mean().unwrap();
            assert!(
                mean.abs() < 1e-10,
                "Sample {} mean should be ~0, got {}",
                sample_idx,
                mean
            );
        }
    }

    #[test]
    fn test_layer_norm_with_scale_and_shift() {
        let mut ln = LayerNorm::new(2);

        // Set custom weight and bias
        ln.weight = arr2(&[[2.0], [3.0]]); // Scale by 2 and 3
        ln.bias = arr2(&[[1.0], [-1.0]]); // Shift by 1 and -1

        let input = arr2(&[[0.0], [2.0]]); // Mean=1, will normalize to [-1, 1]
        let output = ln.forward(&input);

        // After normalization: [0-1, 2-1] / 1 = [-1, 1]
        // After scale and shift: [-1*2+1, 1*3-1] = [-1, 2]
        let expected = arr2(&[[-1.0], [2.0]]);

        for i in 0..2 {
            assert!(
                (output[[i, 0]] - expected[[i, 0]]).abs() < 1e-6,
                "Output[{}] should be {}, got {}",
                i,
                expected[[i, 0]],
                output[[i, 0]]
            );
        }
    }

    #[test]
    fn test_layer_norm_epsilon() {
        let ln = LayerNorm::with_epsilon(1, 1.0); // Large epsilon

        // Input with zero variance (constant)
        let input = arr2(&[[5.0]]);
        let output = ln.forward(&input);

        // Should not crash and should return reasonable values
        assert!(output[[0, 0]].is_finite());
    }

    #[test]
    fn test_num_parameters() {
        let ln = LayerNorm::new(10);
        assert_eq!(ln.num_parameters(), 20); // 10 weights + 10 biases
    }

    #[test]
    #[should_panic(expected = "Input feature size 3 doesn't match expected 4")]
    fn test_wrong_input_size() {
        let ln = LayerNorm::new(4);
        let input = arr2(&[[1.0], [2.0], [3.0]]); // Wrong size: 3 instead of 4
        ln.forward(&input);
    }
}
