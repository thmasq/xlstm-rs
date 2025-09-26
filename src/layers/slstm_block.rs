use crate::layers::{BlockDiagonal, CausalConv1D, LayerNorm};
use crate::utils::{gelu, safe_max, sigmoid, silu};
use ndarray::Array2;

/// Cache for sLSTM block forward pass (used in training)
#[derive(Clone)]
pub struct SLSTMBlockCache {
    pub input: Array2<f64>,
    pub conv_output: Array2<f64>,
    pub i_gate: Array2<f64>,
    pub f_gate: Array2<f64>,
    pub o_gate: Array2<f64>,
    pub z_gate: Array2<f64>,
    pub ct: Array2<f64>,
    pub nt: Array2<f64>,
    pub ht: Array2<f64>,
    pub mt: Array2<f64>,
    pub left_branch: Array2<f64>,
    pub right_branch: Array2<f64>,
}

/// Scalar LSTM (sLSTM) block implementation.
///
/// The sLSTM block features scalar-based memory with exponential gating
/// and memory stabilization mechanisms. Key components:
///
/// 1. **Causal Convolution**: Temporal processing with SiLU activation
/// 2. **Four Gates**: Input (i), Forget (f), Output (o), and New (z) gates
/// 3. **Recurrent Connections**: Full recurrent processing with previous hidden states
/// 4. **Memory Stabilization**: Uses mt (max tracker) for numerical stability
/// 5. **Feed-Forward Network**: Final processing with left/right branch multiplication
///
/// # Architecture
///
/// ```text
/// Input -> LayerNorm -> Conv1D -> Gates -> Memory Update -> FFN -> Output  
/// ```
///
/// # Example
///
/// ```rust
/// use xlstm_rust::layers::SLSTMBlock;
/// use ndarray::arr2;
///
/// let mut block = SLSTMBlock::new(6, 4); // input_size=6, depth=4
/// let input = arr2(&[[1.0; 6]]).t().to_owned();
/// let output = block.forward(&input);
/// assert_eq!(output.shape(), &[6, 1]); // Same size as input
/// ```
#[derive(Clone)]
pub struct SLSTMBlock {
    pub input_size: usize,
    pub conv_channels: usize,

    // Input processing
    pub ln_input: LayerNorm,
    pub conv: CausalConv1D,

    // Main gates (input-to-hidden)
    pub i_gate: BlockDiagonal,
    pub f_gate: BlockDiagonal,
    pub o_gate: BlockDiagonal,
    pub z_gate: BlockDiagonal,

    // Recurrent gates (hidden-to-hidden)
    pub ri_gate: BlockDiagonal,
    pub rf_gate: BlockDiagonal,
    pub ro_gate: BlockDiagonal,
    pub rz_gate: BlockDiagonal,

    // Gate normalizations
    pub ln_i: LayerNorm,
    pub ln_f: LayerNorm,
    pub ln_o: LayerNorm,
    pub ln_z: LayerNorm,

    // Memory normalizations
    pub group_norm: LayerNorm,
    pub ln_c: LayerNorm,
    pub ln_n: LayerNorm,
    pub ln_h: LayerNorm,

    // Feed-forward network
    pub left_linear: BlockDiagonal,
    pub right_linear: BlockDiagonal,
    pub ln_output: LayerNorm,
    pub output_proj: BlockDiagonal,

    // Memory states (persistent across time steps)
    pub nt_prev: Array2<f64>, // Normalizer state
    pub ct_prev: Array2<f64>, // Cell state
    pub ht_prev: Array2<f64>, // Hidden state
    pub mt_prev: Array2<f64>, // Max tracker state
}

impl SLSTMBlock {
    /// Create a new sLSTM block
    ///
    /// # Arguments
    /// * `input_size` - Size of input features
    /// * `depth` - Depth parameter for BlockDiagonal projections
    pub fn new(input_size: usize, depth: usize) -> Self {
        let conv_kernel_size = std::cmp::max(input_size / 8, 1);
        let ff_size = (input_size as f64 * 4.0 / 3.0) as usize; // 4/3 expansion factor

        SLSTMBlock {
            input_size,
            conv_channels: input_size,

            // Input processing
            ln_input: LayerNorm::new(input_size),
            conv: CausalConv1D::new(input_size, input_size, conv_kernel_size),

            // Main gates
            i_gate: BlockDiagonal::new(input_size, input_size, depth, true),
            f_gate: BlockDiagonal::new(input_size, input_size, depth, true),
            o_gate: BlockDiagonal::new(input_size, input_size, depth, true),
            z_gate: BlockDiagonal::new(input_size, input_size, depth, true),

            // Recurrent gates (no bias as per original implementation)
            ri_gate: BlockDiagonal::new(input_size, input_size, depth, false),
            rf_gate: BlockDiagonal::new(input_size, input_size, depth, false),
            ro_gate: BlockDiagonal::new(input_size, input_size, depth, false),
            rz_gate: BlockDiagonal::new(input_size, input_size, depth, false),

            // Gate normalizations
            ln_i: LayerNorm::new(input_size),
            ln_f: LayerNorm::new(input_size),
            ln_o: LayerNorm::new(input_size),
            ln_z: LayerNorm::new(input_size),

            // Memory normalizations
            group_norm: LayerNorm::new(input_size),
            ln_c: LayerNorm::new(input_size),
            ln_n: LayerNorm::new(input_size),
            ln_h: LayerNorm::new(input_size),

            // Feed-forward network
            left_linear: BlockDiagonal::new(input_size, ff_size, 1, true),
            right_linear: BlockDiagonal::new(input_size, ff_size, 1, true),
            ln_output: LayerNorm::new(ff_size),
            output_proj: BlockDiagonal::new(ff_size, input_size, 1, true),

            // Initialize memory states
            nt_prev: Array2::zeros((input_size, 1)),
            ct_prev: Array2::zeros((input_size, 1)),
            ht_prev: Array2::zeros((input_size, 1)),
            mt_prev: Array2::zeros((input_size, 1)),
        }
    }

    /// Forward pass through the sLSTM block
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
    pub fn forward_with_cache(&mut self, input: &Array2<f64>) -> (Array2<f64>, SLSTMBlockCache) {
        assert_eq!(input.nrows(), self.input_size);

        // Input normalization
        let normed_input = self.ln_input.forward(input);

        // Convolution (transpose for conv, then back)
        let input_transposed = normed_input.t().to_owned();
        let conv_output = self.conv.forward(&input_transposed);
        let conv_activated = conv_output.t().map(|x| silu(*x));

        // Gate computations with recurrent connections
        let i_raw = &self.i_gate.forward(&conv_activated) + &self.ri_gate.forward(&self.ht_prev);
        let f_raw = &self.f_gate.forward(&conv_activated) + &self.rf_gate.forward(&self.ht_prev);

        let i_norm = self.ln_i.forward(&i_raw);
        let f_norm = self.ln_f.forward(&f_raw);

        // Memory stabilization using mt (max tracker)
        let batch_size = input.ncols();
        let mut m_new = Array2::zeros((self.input_size, batch_size));
        let mut i_gate = Array2::zeros((self.input_size, batch_size));
        let mut f_gate = Array2::zeros((self.input_size, batch_size));

        for col in 0..batch_size {
            for row in 0..self.input_size {
                let log_i = i_norm[[row, col]];
                let log_f = f_norm[[row, col]] + self.mt_prev[[row, 0]];

                // Compute new max tracker
                let m_t = safe_max(log_f, log_i);
                m_new[[row, col]] = m_t;

                // Stabilized gates
                i_gate[[row, col]] = (log_i - m_t).exp();
                f_gate[[row, col]] = (log_f - m_t).exp();
            }
        }

        // Update mt with mean across batch
        let mt_mean = m_new
            .mean_axis(ndarray::Axis(1))
            .unwrap()
            .insert_axis(ndarray::Axis(1));
        self.mt_prev = mt_mean;

        // Output and Z gates
        let o_raw = &self.o_gate.forward(&normed_input) + &self.ro_gate.forward(&self.ht_prev);
        let z_raw = &self.z_gate.forward(&normed_input) + &self.rz_gate.forward(&self.ht_prev);

        let o_gate = self.ln_o.forward(&o_raw).map(|x| sigmoid(*x));
        let z_gate = self.ln_z.forward(&z_raw).map(|x| x.tanh());

        // Memory state updates
        let ct = &f_gate * &self.ct_prev + &i_gate * &z_gate;
        let ct_normed = self.ln_c.forward(&ct);
        let ct_mean = ct_normed
            .mean_axis(ndarray::Axis(1))
            .unwrap()
            .insert_axis(ndarray::Axis(1));
        self.ct_prev = ct_mean;

        let nt = &f_gate * &self.nt_prev + &i_gate;
        let nt_normed = self.ln_n.forward(&nt);
        let nt_mean = nt_normed
            .mean_axis(ndarray::Axis(1))
            .unwrap()
            .insert_axis(ndarray::Axis(1));
        self.nt_prev = nt_mean;

        // Hidden state computation with safe division
        let mut ht = Array2::zeros((self.input_size, batch_size));
        for col in 0..batch_size {
            for row in 0..self.input_size {
                let ct_val = ct_normed[[row, col]];
                let nt_val = nt_normed[[row, col]].max(1e-8);
                let o_val = o_gate[[row, col]];
                ht[[row, col]] = o_val * (ct_val / nt_val);
            }
        }

        let ht_normed = self.ln_h.forward(&ht);
        let ht_mean = ht_normed
            .mean_axis(ndarray::Axis(1))
            .unwrap()
            .insert_axis(ndarray::Axis(1));
        self.ht_prev = ht_mean;

        // Feed-forward network
        let grouped_output = self.group_norm.forward(&ht_normed);

        let left_branch = self.left_linear.forward(&grouped_output);
        let right_branch = self.right_linear.forward(&grouped_output);
        let right_activated = right_branch.map(|x| gelu(*x));

        let combined = &left_branch * &right_activated;
        let output_normed = self.ln_output.forward(&combined);
        let final_output = self.output_proj.forward(&output_normed);

        // Create cache
        let cache = SLSTMBlockCache {
            input: input.clone(),
            conv_output: conv_activated,
            i_gate,
            f_gate,
            o_gate,
            z_gate,
            ct: ct_normed,
            nt: nt_normed,
            ht: ht_normed,
            mt: m_new,
            left_branch,
            right_branch: right_activated,
        };

        (final_output, cache)
    }

    /// Reset the internal memory states  
    pub fn reset_state(&mut self) {
        self.nt_prev.fill(0.0);
        self.ct_prev.fill(0.0);
        self.ht_prev.fill(0.0);
        self.mt_prev.fill(0.0);
    }

    /// Get the number of parameters in this block
    pub fn num_parameters(&self) -> usize {
        self.ln_input.num_parameters()
            + self.conv.num_parameters()
            + self.i_gate.num_parameters()
            + self.f_gate.num_parameters()
            + self.o_gate.num_parameters()
            + self.z_gate.num_parameters()
            + self.ri_gate.num_parameters()
            + self.rf_gate.num_parameters()
            + self.ro_gate.num_parameters()
            + self.rz_gate.num_parameters()
            + self.ln_i.num_parameters()
            + self.ln_f.num_parameters()
            + self.ln_o.num_parameters()
            + self.ln_z.num_parameters()
            + self.group_norm.num_parameters()
            + self.ln_c.num_parameters()
            + self.ln_n.num_parameters()
            + self.ln_h.num_parameters()
            + self.left_linear.num_parameters()
            + self.right_linear.num_parameters()
            + self.ln_output.num_parameters()
            + self.output_proj.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_slstm_block_creation() {
        let block = SLSTMBlock::new(6, 3);
        assert_eq!(block.input_size, 6);
        assert_eq!(block.conv_channels, 6);
        assert_eq!(block.nt_prev.shape(), &[6, 1]);
        assert_eq!(block.ct_prev.shape(), &[6, 1]);
        assert_eq!(block.ht_prev.shape(), &[6, 1]);
        assert_eq!(block.mt_prev.shape(), &[6, 1]);
    }

    #[test]
    fn test_slstm_block_forward() {
        let mut block = SLSTMBlock::new(4, 2);
        let input = arr2(&[[1.0], [0.5], [-0.3], [0.8]]);

        let output = block.forward(&input);
        assert_eq!(output.shape(), &[4, 1]);

        // Output should be finite
        for &val in output.iter() {
            assert!(val.is_finite(), "Output should be finite, got {}", val);
        }
    }

    #[test]
    fn test_slstm_block_state_persistence() {
        let mut block = SLSTMBlock::new(3, 2);
        let input1 = arr2(&[[1.0], [0.0], [0.5]]);
        let input2 = arr2(&[[0.0], [1.0], [-0.5]]);

        // First forward pass
        let _output1 = block.forward(&input1);
        let state1 = (
            block.nt_prev.clone(),
            block.ct_prev.clone(),
            block.ht_prev.clone(),
            block.mt_prev.clone(),
        );

        // Second forward pass
        let _output2 = block.forward(&input2);
        let state2 = (
            block.nt_prev.clone(),
            block.ct_prev.clone(),
            block.ht_prev.clone(),
            block.mt_prev.clone(),
        );

        // At least one state should have changed
        let mut state_changed = false;
        for i in 0..block.input_size {
            if (state1.0[[i, 0]] - state2.0[[i, 0]]).abs() > 1e-10
                || (state1.1[[i, 0]] - state2.1[[i, 0]]).abs() > 1e-10
                || (state1.2[[i, 0]] - state2.2[[i, 0]]).abs() > 1e-10
                || (state1.3[[i, 0]] - state2.3[[i, 0]]).abs() > 1e-10
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
    fn test_slstm_block_reset_state() {
        let mut block = SLSTMBlock::new(3, 2);
        let input = arr2(&[[1.0], [0.0], [0.5]]);

        // Forward pass to change state
        let _output = block.forward(&input);

        // Reset state
        block.reset_state();

        // Check that all states are zero
        for i in 0..block.input_size {
            assert_eq!(block.nt_prev[[i, 0]], 0.0);
            assert_eq!(block.ct_prev[[i, 0]], 0.0);
            assert_eq!(block.ht_prev[[i, 0]], 0.0);
            assert_eq!(block.mt_prev[[i, 0]], 0.0);
        }
    }

    #[test]
    fn test_slstm_block_batch_processing() {
        let mut block = SLSTMBlock::new(2, 1);
        let batch_input = arr2(&[[1.0, -1.0, 0.5], [0.0, 1.0, -0.5]]);

        let output = block.forward(&batch_input);
        assert_eq!(output.shape(), &[2, 3]);

        // Each sample should produce valid finite output
        for col in 0..3 {
            for row in 0..2 {
                assert!(
                    output[[row, col]].is_finite(),
                    "Output[{}, {}] should be finite, got {}",
                    row,
                    col,
                    output[[row, col]]
                );
            }
        }
    }

    #[test]
    fn test_slstm_memory_stabilization() {
        let mut block = SLSTMBlock::new(2, 1);
        let input = arr2(&[[100.0], [100.0]]); // Large input values

        let output = block.forward(&input);

        // Should not overflow/underflow despite large inputs
        for &val in output.iter() {
            assert!(
                val.is_finite(),
                "Output should remain finite with large inputs"
            );
        }

        // States should also remain finite
        for &val in block.mt_prev.iter() {
            assert!(val.is_finite(), "Max tracker should remain finite");
        }
    }
}
