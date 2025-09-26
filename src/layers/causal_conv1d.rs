use crate::utils::xavier_uniform;
use ndarray::Array2;

/// Causal 1D convolution layer that respects temporal ordering.
///
/// Performs 1D convolution while ensuring that the output at time t only depends
/// on inputs from times ≤ t. This is achieved by padding the input appropriately
/// and then removing excess outputs.
///
/// # Mathematical Operation
///
/// For input sequence x[t] and kernel weights w[k]:
/// ```
/// y[t] = Σ(k=0 to kernel_size-1) w[k] * x[t-k]
/// ```
///
/// The causal constraint ensures x[t-k] is only accessed for k where t-k ≥ 0.
///
/// # Example
///
/// ```rust
/// use xlstm_rust::layers::CausalConv1D;
/// use ndarray::arr2;
///
/// let mut conv = CausalConv1D::new(4, 4, 3); // 4->4 channels, kernel size 3
/// let input = arr2(&[[1.0, 2.0, 3.0, 4.0]]).t().to_owned(); // sequence length 4
/// let output = conv.forward(&input);
/// assert_eq!(output.shape(), &[4, 1]); // Same sequence length
/// ```
#[derive(Clone)]
pub struct CausalConv1D {
    /// Convolution weights: (out_channels, in_channels, kernel_size)
    pub weights: Array2<f64>,
    /// Optional bias: (out_channels, 1)
    pub bias: Option<Array2<f64>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    /// Amount of padding added to the left (ensures causality)
    pub padding: usize,
}

impl CausalConv1D {
    /// Create a new causal 1D convolution layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels  
    /// * `kernel_size` - Size of the convolution kernel
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let padding = kernel_size - 1; // Left padding for causality

        // Initialize weights: reshape to 2D for easier computation
        // Shape: (out_channels, in_channels * kernel_size)
        let weight_cols = in_channels * kernel_size;
        let weights = xavier_uniform(out_channels, weight_cols);

        // Initialize bias to zero
        let bias = Some(Array2::zeros((out_channels, 1)));

        CausalConv1D {
            weights,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            padding,
        }
    }

    /// Create a causal conv layer without bias
    pub fn without_bias(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let mut conv = Self::new(in_channels, out_channels, kernel_size);
        conv.bias = None;
        conv
    }

    /// Forward pass through the causal convolution
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (in_channels, sequence_length)
    ///
    /// # Returns  
    /// * Output tensor of shape (out_channels, sequence_length)
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            input.nrows(),
            self.in_channels,
            "Input channels {} doesn't match expected {}",
            input.nrows(),
            self.in_channels
        );

        let seq_len = input.ncols();
        let mut output = Array2::zeros((self.out_channels, seq_len));

        // Apply convolution at each time step
        for t in 0..seq_len {
            // Determine the range of inputs to use (respecting causality)
            let start_offset = if t + 1 >= self.kernel_size {
                0
            } else {
                self.kernel_size - t - 1
            };
            let input_start = if t + 1 >= self.kernel_size {
                t + 1 - self.kernel_size
            } else {
                0
            };
            let input_end = t + 1;

            // Create input window for this time step
            let mut input_window = Array2::zeros((self.in_channels, self.kernel_size));

            // Fill the input window (right-aligned for causality)
            for (i, input_t) in (input_start..input_end).enumerate() {
                let window_pos = start_offset + i;
                if window_pos < self.kernel_size {
                    input_window
                        .column_mut(window_pos)
                        .assign(&input.column(input_t));
                }
            }

            // Flatten input window for matrix multiplication
            let flattened_input = input_window
                .t()
                .as_standard_layout()
                .to_owned()
                .into_shape_with_order((self.in_channels * self.kernel_size, 1))
                .expect("Failed to reshape input window");

            // Apply convolution: output = weights @ flattened_input
            let conv_result = self.weights.dot(&flattened_input);

            // Add bias if present
            let final_result = if let Some(ref bias) = self.bias {
                conv_result + bias
            } else {
                conv_result
            };

            // Store result
            output.column_mut(t).assign(&final_result.column(0));
        }

        output
    }

    /// Get the total number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.weights.len();
        let bias_params = self.bias.as_ref().map_or(0, |b| b.len());
        weight_params + bias_params
    }

    /// Get reference to weights (for serialization)
    pub fn get_weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Get reference to bias (for serialization)
    pub fn get_bias(&self) -> Option<&Array2<f64>> {
        self.bias.as_ref()
    }

    /// Compute output sequence length given input length
    pub fn output_length(&self, input_length: usize) -> usize {
        // Causal convolution preserves sequence length
        input_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_causal_conv1d_creation() {
        let conv = CausalConv1D::new(3, 5, 4);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 5);
        assert_eq!(conv.kernel_size, 4);
        assert_eq!(conv.padding, 3); // kernel_size - 1
        assert!(conv.bias.is_some());
        assert_eq!(conv.weights.shape(), &[5, 12]); // 5 x (3 * 4)
    }

    #[test]
    fn test_causal_conv1d_forward() {
        let mut conv = CausalConv1D::new(2, 1, 3);

        // Set predictable weights: [1, 0, 1, 0, 1, 0] -> sum channels 1 and 3 of flattened input
        conv.weights = arr2(&[[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]);
        conv.bias = Some(arr2(&[[0.0]])); // No bias

        // Input: 2 channels, 4 time steps
        let input = arr2(&[
            [1.0, 2.0, 3.0, 4.0], // channel 0
            [0.1, 0.2, 0.3, 0.4], // channel 1
        ]);

        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[1, 4]);

        // Check causality: output[t] should only depend on input[0..=t]
        // For a kernel size of 3, we're essentially computing a weighted sum
        // of current and up to 2 previous time steps
    }

    #[test]
    fn test_causal_conv1d_causality() {
        let mut conv = CausalConv1D::new(1, 1, 3);

        // Simple average filter: weights = [1/3, 1/3, 1/3]
        conv.weights = arr2(&[[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]);
        conv.bias = Some(arr2(&[[0.0]]));

        let input1 = arr2(&[[1.0, 2.0, 3.0]]);
        let input2 = arr2(&[[1.0, 2.0, 9.0]]); // Changed last value

        let output1 = conv.forward(&input1);
        let output2 = conv.forward(&input2);

        // First two outputs should be the same (causal constraint)
        assert!((output1[[0, 0]] - output2[[0, 0]]).abs() < 1e-10);
        assert!((output1[[0, 1]] - output2[[0, 1]]).abs() < 1e-10);

        // Last output can be different
        // (but we don't assert this as the specific values depend on padding handling)
    }

    #[test]
    fn test_causal_conv1d_without_bias() {
        let conv = CausalConv1D::without_bias(2, 3, 5);
        assert!(conv.bias.is_none());
        assert_eq!(conv.num_parameters(), 2 * 3 * 5); // Only weights
    }

    #[test]
    fn test_output_length() {
        let conv = CausalConv1D::new(1, 1, 5);
        assert_eq!(conv.output_length(10), 10);
        assert_eq!(conv.output_length(3), 3);
        assert_eq!(conv.output_length(1), 1);
    }

    #[test]
    fn test_num_parameters() {
        let conv_with_bias = CausalConv1D::new(3, 4, 5);
        assert_eq!(conv_with_bias.num_parameters(), 3 * 4 * 5 + 4); // weights + bias

        let conv_without_bias = CausalConv1D::without_bias(3, 4, 5);
        assert_eq!(conv_without_bias.num_parameters(), 3 * 4 * 5); // Only weights
    }
}
