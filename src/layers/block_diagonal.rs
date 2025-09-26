use crate::utils::xavier_uniform;
use ndarray::Array2;

/// Block diagonal linear layer that applies multiple smaller linear transformations in parallel.
///
/// This is equivalent to having `num_blocks` separate linear layers and concatenating their outputs.
/// It's more efficient than separate layers while providing the same functionality.
///
/// # Mathematical Operation
///
/// For input x ∈ R^(input_size, batch_size), applies:
/// ```
/// y = [W₁x; W₂x; ...; Wₙx] + [b₁; b₂; ...; bₙ]
/// ```
/// where Wᵢ ∈ R^(block_out_size, input_size) and bᵢ ∈ R^(block_out_size, 1)
///
/// # Example
///
/// ```rust
/// use xlstm_rust::layers::BlockDiagonal;
/// use ndarray::arr2;
///
/// let mut layer = BlockDiagonal::new(4, 12, 3, true); // 3 blocks of 4->4
/// let input = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
/// let output = layer.forward(&input);
/// assert_eq!(output.shape(), &[12, 1]); // 3 * 4 = 12
/// ```
#[derive(Clone)]
pub struct BlockDiagonal {
    /// Weight matrices for each block: Vec of (block_out_size, input_size)
    pub weights: Vec<Array2<f64>>,
    /// Bias vectors for each block: Vec of (block_out_size, 1)
    pub biases: Option<Vec<Array2<f64>>>,
    pub input_size: usize,
    pub output_size: usize,
    pub num_blocks: usize,
    pub block_out_size: usize,
}

impl BlockDiagonal {
    /// Create a new BlockDiagonal layer
    ///
    /// # Arguments
    /// * `input_size` - Size of input features
    /// * `output_size` - Total output size (must be divisible by num_blocks)
    /// * `num_blocks` - Number of parallel blocks
    /// * `bias` - Whether to include bias terms
    pub fn new(input_size: usize, output_size: usize, num_blocks: usize, bias: bool) -> Self {
        assert!(
            output_size % num_blocks == 0,
            "Output size {} must be divisible by number of blocks {}",
            output_size,
            num_blocks
        );

        let block_out_size = output_size / num_blocks;

        // Initialize weights using Xavier uniform
        let mut weights = Vec::new();
        for _ in 0..num_blocks {
            weights.push(xavier_uniform(block_out_size, input_size));
        }

        // Initialize biases if requested
        let biases = if bias {
            Some(
                (0..num_blocks)
                    .map(|_| Array2::zeros((block_out_size, 1)))
                    .collect(),
            )
        } else {
            None
        };

        BlockDiagonal {
            weights,
            biases,
            input_size,
            output_size,
            num_blocks,
            block_out_size,
        }
    }

    /// Forward pass through the block diagonal layer
    ///
    /// # Arguments  
    /// * `input` - Input tensor of shape (input_size, batch_size)
    ///
    /// # Returns
    /// * Output tensor of shape (output_size, batch_size)
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            input.nrows(),
            self.input_size,
            "Input size {} doesn't match expected {}",
            input.nrows(),
            self.input_size
        );

        let batch_size = input.ncols();
        let mut output = Array2::zeros((self.output_size, batch_size));

        // Apply each block and stack results
        for (block_idx, weight) in self.weights.iter().enumerate() {
            let start_idx = block_idx * self.block_out_size;
            let end_idx = start_idx + self.block_out_size;

            // Matrix multiplication for this block
            let block_output = weight.dot(input);

            // Add bias if present
            let block_output = if let Some(ref biases) = self.biases {
                let bias = &biases[block_idx];
                // Broadcast bias across batch dimension
                let broadcasted_bias = bias.broadcast((self.block_out_size, batch_size)).unwrap();
                block_output + broadcasted_bias
            } else {
                block_output
            };

            // Place in output tensor
            output
                .slice_mut(ndarray::s![start_idx..end_idx, ..])
                .assign(&block_output);
        }

        output
    }

    /// Get the total number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.weights.iter().map(|w| w.len()).sum::<usize>();
        let bias_params = if let Some(ref biases) = self.biases {
            biases.iter().map(|b| b.len()).sum::<usize>()
        } else {
            0
        };
        weight_params + bias_params
    }

    /// Get reference to weight matrices (for serialization or inspection)
    pub fn get_weights(&self) -> &[Array2<f64>] {
        &self.weights
    }

    /// Get reference to bias vectors (for serialization or inspection)
    pub fn get_biases(&self) -> Option<&[Array2<f64>]> {
        self.biases.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_block_diagonal_creation() {
        let layer = BlockDiagonal::new(4, 12, 3, true);
        assert_eq!(layer.input_size, 4);
        assert_eq!(layer.output_size, 12);
        assert_eq!(layer.num_blocks, 3);
        assert_eq!(layer.block_out_size, 4);
        assert_eq!(layer.weights.len(), 3);
        assert!(layer.biases.is_some());
        assert_eq!(layer.biases.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_block_diagonal_forward() {
        let mut layer = BlockDiagonal::new(2, 6, 3, false);

        // Set predictable weights for testing
        for i in 0..3 {
            layer.weights[i] = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // Identity for each block
        }

        let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let output = layer.forward(&input);

        assert_eq!(output.shape(), &[6, 2]);

        // Each block should output the input (identity transformation)
        // So output should be [input; input; input]
        let expected = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [1.0, 2.0],
            [3.0, 4.0],
        ]);

        for i in 0..6 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_block_diagonal_with_bias() {
        let mut layer = BlockDiagonal::new(2, 4, 2, true);

        // Set weights to identity
        layer.weights[0] = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        layer.weights[1] = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

        // Set biases
        layer.biases.as_mut().unwrap()[0] = arr2(&[[1.0], [2.0]]);
        layer.biases.as_mut().unwrap()[1] = arr2(&[[3.0], [4.0]]);

        let input = arr2(&[[1.0], [1.0]]);
        let output = layer.forward(&input);

        let expected = arr2(&[[2.0], [3.0], [4.0], [5.0]]); // input + bias for each block

        for i in 0..4 {
            assert!((output[[i, 0]] - expected[[i, 0]]).abs() < 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "Output size 7 must be divisible by number of blocks 3")]
    fn test_invalid_block_configuration() {
        BlockDiagonal::new(4, 7, 3, true); // 7 is not divisible by 3
    }

    #[test]
    fn test_num_parameters() {
        let layer_no_bias = BlockDiagonal::new(3, 6, 2, false);
        assert_eq!(layer_no_bias.num_parameters(), 2 * 3 * 3); // 2 blocks * 3*3 weights each

        let layer_with_bias = BlockDiagonal::new(3, 6, 2, true);
        assert_eq!(layer_with_bias.num_parameters(), 2 * 3 * 3 + 2 * 3); // weights + biases
    }
}
