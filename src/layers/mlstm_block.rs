use crate::layers::{BlockDiagonal, CausalConv1D, LayerNorm};
use crate::utils::{find_suitable_num_blocks, safe_div, sigmoid, silu};
use ndarray::Array2;

/// Cache for mLSTM block forward pass (used in training)
#[derive(Clone)]
pub struct MLSTMBlockCache {
    pub input: Array2<f64>,
    pub left_branch: Array2<f64>,
    pub right_branch: Array2<f64>,
    pub conv_output: Array2<f64>,
    pub skip_connection: Array2<f64>,
    pub q: Array2<f64>,
    pub k: Array2<f64>,
    pub v: Array2<f64>,
    pub i_gate: Array2<f64>,
    pub f_gate: Array2<f64>,
    pub o_gate: Array2<f64>,
    pub ct: Array2<f64>,
    pub nt: Array2<f64>,
    pub ht: Array2<f64>,
}

/// Matrix LSTM (mLSTM) block implementation.
///
/// The mLSTM block features matrix-based memory with exponential gating
/// and attention-like mechanisms. Key components:
///
/// 1. **Dual Branch Processing**: Left branch (gated) and right branch (activation)
/// 2. **Causal Convolution**: Temporal processing with causality constraints  
/// 3. **Query-Key-Value Mechanism**: Attention-like memory interaction
/// 4. **Matrix Memory States**: ct (cell) and nt (normalizer) tensors
/// 5. **Exponential Gating**: For improved gradient flow
///
/// # Architecture
///
/// ```text
/// Input -> LayerNorm -> [Left Branch] -> Conv1D -> Q,K,V projections
///                   \-> [Right Branch] -> SiLU activation
///                                    \-> Element-wise multiply -> Output
/// ```
///
/// # Example
///
/// ```rust
/// use xlstm_rust::layers::MLSTMBlock;
/// use ndarray::arr2;
///
/// let mut block = MLSTMBlock::new(8, 2.0, 4); // input_size=8, factor=2.0, depth=4  
/// let input = arr2(&[[1.0; 8]]).t().to_owned();
/// let output = block.forward(&input);
/// assert_eq!(output.shape(), &[8, 1]); // Same size as input
/// ```
#[derive(Clone)]
pub struct MLSTMBlock {
    pub input_size: usize,
    pub hidden_size: usize,

    // Normalization layers
    pub ln_input: LayerNorm,

    // Branch projections
    pub left_proj: BlockDiagonal,
    pub right_proj: BlockDiagonal,

    // Convolution and skip connection
    pub conv: CausalConv1D,
    pub skip_proj: BlockDiagonal,

    // Query, Key, Value projections
    pub q_proj: BlockDiagonal,
    pub k_proj: BlockDiagonal,
    pub v_proj: BlockDiagonal,

    // Gates
    pub i_gate: BlockDiagonal,
    pub f_gate: BlockDiagonal,
    pub o_gate: BlockDiagonal,

    // Gate normalizations
    pub ln_i: LayerNorm,
    pub ln_f: LayerNorm,
    pub ln_o: LayerNorm,

    // Memory state normalizations
    pub ln_c: LayerNorm,
    pub ln_n: LayerNorm,

    // Output processing
    pub group_norm: LayerNorm,
    pub ln_output: LayerNorm,
    pub output_proj: BlockDiagonal,
    pub ln_proj: LayerNorm,

    // Memory states (persistent across time steps)
    pub ct_prev: Array2<f64>, // Cell state
    pub nt_prev: Array2<f64>, // Normalizer state
}

impl MLSTMBlock {
    /// Create a new mLSTM block
    ///
    /// # Arguments
    /// * `input_size` - Size of input features
    /// * `factor` - Hidden size multiplier (hidden_size = input_size * factor)
    /// * `depth` - Desired depth parameter for BlockDiagonal projections (will be adjusted to fit)
    pub fn new(input_size: usize, factor: f64, depth: usize) -> Self {
        let hidden_size = (input_size as f64 * factor) as usize;
        let conv_kernel_size = std::cmp::max(input_size / 10, 1);

        // Find suitable block counts that divide the sizes evenly
        let qkv_blocks = find_suitable_num_blocks(hidden_size, depth);
        let output_blocks = find_suitable_num_blocks(input_size, 1); // For output projection

        MLSTMBlock {
            input_size,
            hidden_size,

            // Input normalization
            ln_input: LayerNorm::new(input_size),

            // Branch projections (single block for simplicity)
            left_proj: BlockDiagonal::new(input_size, hidden_size, 1, true),
            right_proj: BlockDiagonal::new(input_size, hidden_size, 1, true),

            // Convolution and skip
            conv: CausalConv1D::new(hidden_size, hidden_size, conv_kernel_size),
            skip_proj: BlockDiagonal::new(hidden_size, hidden_size, 1, true),

            // Q, K, V projections with suitable block counts
            q_proj: BlockDiagonal::new(hidden_size, hidden_size, qkv_blocks, true),
            k_proj: BlockDiagonal::new(hidden_size, hidden_size, qkv_blocks, true),
            v_proj: BlockDiagonal::new(hidden_size, hidden_size, qkv_blocks, true),

            // Gates (single block for simplicity)
            i_gate: BlockDiagonal::new(hidden_size, hidden_size, 1, true),
            f_gate: BlockDiagonal::new(hidden_size, hidden_size, 1, true),
            o_gate: BlockDiagonal::new(hidden_size, hidden_size, 1, true),

            // Gate normalizations
            ln_i: LayerNorm::new(hidden_size),
            ln_f: LayerNorm::new(hidden_size),
            ln_o: LayerNorm::new(hidden_size),

            // Memory normalizations
            ln_c: LayerNorm::new(hidden_size),
            ln_n: LayerNorm::new(hidden_size),

            // Output processing
            group_norm: LayerNorm::new(hidden_size),
            ln_output: LayerNorm::new(hidden_size),
            output_proj: BlockDiagonal::new(hidden_size, input_size, output_blocks, true),
            ln_proj: LayerNorm::new(input_size),

            // Initialize memory states
            ct_prev: Array2::zeros((hidden_size, 1)),
            nt_prev: Array2::zeros((hidden_size, 1)),
        }
    }

    /// Forward pass through the mLSTM block
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (input_size, batch_size)
    ///
    /// # Returns
    /// * Output tensor of shape (input_size, batch_size)
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let (output, _) = self.forward_with_cache(input);
        output
    }

    /// Forward pass with caching for training
    pub fn forward_with_cache(&mut self, input: &Array2<f64>) -> (Array2<f64>, MLSTMBlockCache) {
        assert_eq!(input.nrows(), self.input_size);

        // Input normalization
        let normed_input = self.ln_input.forward(input);

        // Branch processing
        let left_branch = self.left_proj.forward(&normed_input);
        let right_branch = self.right_proj.forward(&normed_input);
        let right_activated = right_branch.map(|x| silu(*x));

        // Convolution on left branch (NO transpose needed - conv expects (channels, sequence_len))
        let conv_output = self.conv.forward(&left_branch);
        let conv_activated = conv_output.map(|x| silu(*x));

        // Skip connection
        let skip_connection = self.skip_proj.forward(&conv_activated);

        // Q, K, V projections
        let q = self.q_proj.forward(&conv_activated);
        let k = self.k_proj.forward(&conv_activated);
        let v = self.v_proj.forward(&left_branch);

        // Gates with exponential activation
        let i_raw = self.i_gate.forward(&conv_activated);
        let f_raw = self.f_gate.forward(&conv_activated);
        let o_raw = self.o_gate.forward(&conv_activated);

        let i_gate = self.ln_i.forward(&i_raw).map(|x| x.exp());
        let f_gate = self.ln_f.forward(&f_raw).map(|x| x.exp());
        let o_gate = self.ln_o.forward(&o_raw).map(|x| sigmoid(*x));

        // Update memory states
        let ct = &f_gate * &self.ct_prev + &i_gate * &v * &k;
        let ct_normed = self.ln_c.forward(&ct);
        // Take mean across batch for persistent state
        let ct_mean = ct_normed
            .mean_axis(ndarray::Axis(1))
            .unwrap()
            .insert_axis(ndarray::Axis(1));
        self.ct_prev = ct_mean;

        let nt = &f_gate * &self.nt_prev + &i_gate * &k;
        let nt_normed = self.ln_n.forward(&nt);
        let nt_mean = nt_normed
            .mean_axis(ndarray::Axis(1))
            .unwrap()
            .insert_axis(ndarray::Axis(1));
        self.nt_prev = nt_mean;

        // Compute output with safe division
        let batch_size = input.ncols();
        let mut ht = Array2::zeros((self.hidden_size, batch_size));

        for col in 0..batch_size {
            for row in 0..self.hidden_size {
                let ct_q = ct_normed[[row, col]] * q[[row, col]];
                let nt_q = nt_normed[[row, col]] * q[[row, col]];
                let ratio = safe_div(ct_q, nt_q.max(1e-6), 1e-6);
                ht[[row, col]] = o_gate[[row, col]] * ratio;
            }
        }

        // Final processing
        let residual_ht = &ht + &skip_connection;
        let grouped = self.group_norm.forward(&residual_ht);
        let final_left = &grouped * &right_activated;
        let output_normed = self.ln_output.forward(&final_left);
        let projected = self.output_proj.forward(&output_normed);
        let final_output = self.ln_proj.forward(&projected);

        // Create cache
        let cache = MLSTMBlockCache {
            input: input.clone(),
            left_branch,
            right_branch: right_activated,
            conv_output: conv_activated,
            skip_connection,
            q,
            k,
            v,
            i_gate,
            f_gate,
            o_gate,
            ct: ct_normed,
            nt: nt_normed,
            ht,
        };

        (final_output, cache)
    }

    /// Reset the internal memory states
    pub fn reset_state(&mut self) {
        self.ct_prev.fill(0.0);
        self.nt_prev.fill(0.0);
    }

    /// Get the number of parameters in this block
    pub fn num_parameters(&self) -> usize {
        self.ln_input.num_parameters()
            + self.left_proj.num_parameters()
            + self.right_proj.num_parameters()
            + self.conv.num_parameters()
            + self.skip_proj.num_parameters()
            + self.q_proj.num_parameters()
            + self.k_proj.num_parameters()
            + self.v_proj.num_parameters()
            + self.i_gate.num_parameters()
            + self.f_gate.num_parameters()
            + self.o_gate.num_parameters()
            + self.ln_i.num_parameters()
            + self.ln_f.num_parameters()
            + self.ln_o.num_parameters()
            + self.ln_c.num_parameters()
            + self.ln_n.num_parameters()
            + self.group_norm.num_parameters()
            + self.ln_output.num_parameters()
            + self.output_proj.num_parameters()
            + self.ln_proj.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mlstm_block_creation() {
        let block = MLSTMBlock::new(8, 2.0, 4);
        assert_eq!(block.input_size, 8);
        assert_eq!(block.hidden_size, 16); // 8 * 2.0
        assert_eq!(block.ct_prev.shape(), &[16, 1]);
        assert_eq!(block.nt_prev.shape(), &[16, 1]);
    }

    #[test]
    fn test_mlstm_block_forward() {
        let mut block = MLSTMBlock::new(4, 2.0, 2);
        let input = arr2(&[[1.0], [0.5], [-0.3], [0.8]]);

        let output = block.forward(&input);
        assert_eq!(output.shape(), &[4, 1]);

        // Output should be different from input (non-identity transformation)
        let mut different = false;
        for i in 0..4 {
            if (output[[i, 0]] - input[[i, 0]]).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "Output should be different from input");
    }

    #[test]
    fn test_mlstm_block_state_persistence() {
        let mut block = MLSTMBlock::new(2, 1.0, 1);
        let input1 = arr2(&[[1.0], [0.0]]);
        let input2 = arr2(&[[0.0], [1.0]]);

        // First forward pass
        let _output1 = block.forward(&input1);
        let state1 = (block.ct_prev.clone(), block.nt_prev.clone());

        // Second forward pass
        let _output2 = block.forward(&input2);
        let state2 = (block.ct_prev.clone(), block.nt_prev.clone());

        // States should have changed
        let mut state_changed = false;
        for i in 0..block.hidden_size {
            if (state1.0[[i, 0]] - state2.0[[i, 0]]).abs() > 1e-10
                || (state1.1[[i, 0]] - state2.1[[i, 0]]).abs() > 1e-10
            {
                state_changed = true;
                break;
            }
        }
        assert!(
            state_changed,
            "Internal states should change between forward passes"
        );
    }

    #[test]
    fn test_mlstm_block_reset_state() {
        let mut block = MLSTMBlock::new(2, 1.0, 1);
        let input = arr2(&[[1.0], [0.0]]);

        // Forward pass to change state
        let _output = block.forward(&input);

        // Reset state
        block.reset_state();

        // Check that states are zero
        for i in 0..block.hidden_size {
            assert_eq!(block.ct_prev[[i, 0]], 0.0);
            assert_eq!(block.nt_prev[[i, 0]], 0.0);
        }
    }

    #[test]
    fn test_mlstm_block_batch_processing() {
        let mut block = MLSTMBlock::new(3, 1.5, 2);
        let batch_input = arr2(&[[1.0, 2.0], [0.5, -0.5], [-0.3, 0.8]]);

        let output = block.forward(&batch_input);
        assert_eq!(output.shape(), &[3, 2]);

        // Each sample should produce valid output
        for col in 0..2 {
            for row in 0..3 {
                assert!(output[[row, col]].is_finite());
            }
        }
    }

    #[test]
    fn test_mlstm_block_small_sizes() {
        // Test with very small input sizes that would cause division issues
        let mut block = MLSTMBlock::new(1, 2.0, 4); // input=1, depth=4 (not divisible)
        let input = arr2(&[[1.0]]);

        let output = block.forward(&input);
        assert_eq!(output.shape(), &[1, 1]);
        assert!(output[[0, 0]].is_finite());
    }
}
