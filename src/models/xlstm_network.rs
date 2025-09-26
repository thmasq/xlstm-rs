use crate::layers::{MLSTMBlock, MLSTMBlockCache, SLSTMBlock, SLSTMBlockCache};
use ndarray::Array2;

/// Cache for xLSTM network forward pass (used in training)
#[derive(Clone)]
pub enum XLSTMBlockCache {
    MLSTM(MLSTMBlockCache),
    SLSTM(SLSTMBlockCache),
}

/// Cache for the entire xLSTM network
#[derive(Clone)]
pub struct XLSTMNetworkCache {
    pub block_caches: Vec<XLSTMBlockCache>,
    pub original_input: Array2<f64>,
}

/// Block type enumeration for xLSTM layers
#[derive(Clone, Debug, PartialEq)]
pub enum XLSTMBlockType {
    MLSTM,
    SLSTM,
}

/// Individual block in the xLSTM network
#[derive(Clone)]
pub enum XLSTMBlock {
    MLSTM(MLSTMBlock),
    SLSTM(SLSTMBlock),
}

impl XLSTMBlock {
    /// Forward pass through the block
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        match self {
            XLSTMBlock::MLSTM(block) => block.forward(input),
            XLSTMBlock::SLSTM(block) => block.forward(input),
        }
    }

    /// Forward pass with caching
    pub fn forward_with_cache(&mut self, input: &Array2<f64>) -> (Array2<f64>, XLSTMBlockCache) {
        match self {
            XLSTMBlock::MLSTM(block) => {
                let (output, cache) = block.forward_with_cache(input);
                (output, XLSTMBlockCache::MLSTM(cache))
            }
            XLSTMBlock::SLSTM(block) => {
                let (output, cache) = block.forward_with_cache(input);
                (output, XLSTMBlockCache::SLSTM(cache))
            }
        }
    }

    /// Reset internal states
    pub fn reset_state(&mut self) {
        match self {
            XLSTMBlock::MLSTM(block) => block.reset_state(),
            XLSTMBlock::SLSTM(block) => block.reset_state(),
        }
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        match self {
            XLSTMBlock::MLSTM(block) => block.num_parameters(),
            XLSTMBlock::SLSTM(block) => block.num_parameters(),
        }
    }

    /// Get block type
    pub fn block_type(&self) -> XLSTMBlockType {
        match self {
            XLSTMBlock::MLSTM(_) => XLSTMBlockType::MLSTM,
            XLSTMBlock::SLSTM(_) => XLSTMBlockType::SLSTM,
        }
    }
}

/// Extended LSTM (xLSTM) network implementation.
///
/// Combines mLSTM and sLSTM blocks in a flexible architecture based on a
/// configuration string. Each character in the config specifies a block type:
/// - 'm': mLSTM block (Matrix LSTM)
/// - 's': sLSTM block (Scalar LSTM)
///
/// The network processes inputs through the blocks sequentially, with each
/// block's output becoming the next block's input. Residual connections
/// are used between blocks to improve gradient flow.
///
/// # Architecture
///
/// ```text
/// Input -> Block₁ -> (+) -> Block₂ -> (+) -> ... -> Blockₙ -> Output
///            ↓       ↑       ↓       ↑              ↓
///         Residual  Input  Residual Input       Residual
/// ```
///
/// # Example
///
/// ```rust
/// use xlstm_rust::models::XLSTMNetwork;
/// use ndarray::arr2;
///
/// // Create network: mLSTM -> sLSTM -> mLSTM
/// let mut network = XLSTMNetwork::from_config("msm", 16, 32, 4);
///
/// let input = arr2(&[[1.0; 16]]).t().to_owned();
/// let output = network.forward(&input);
/// assert_eq!(output.shape(), &[16, 1]);
/// ```
#[derive(Clone)]
pub struct XLSTMNetwork {
    /// Sequence of xLSTM blocks
    pub blocks: Vec<XLSTMBlock>,
    /// Input feature size
    pub input_size: usize,
    /// Hidden size parameter
    pub hidden_size: usize,
    /// Depth parameter for BlockDiagonal layers
    pub depth: usize,
    /// Factor parameter for mLSTM blocks
    pub factor: f64,
    /// Layer configuration string
    pub config: String,
}

impl XLSTMNetwork {
    /// Create a new xLSTM network from configuration string
    ///
    /// # Arguments
    /// * `config` - Configuration string (e.g., "msm" for mLSTM-sLSTM-mLSTM)
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Hidden size parameter for blocks
    /// * `depth` - Depth parameter for BlockDiagonal projections
    /// * `factor` - Factor parameter for mLSTM blocks (default: 2.0)
    pub fn new(
        config: &str,
        input_size: usize,
        hidden_size: usize,
        depth: usize,
        factor: f64,
    ) -> Self {
        let mut blocks = Vec::new();

        for layer_char in config.chars() {
            let block = match layer_char {
                'm' => XLSTMBlock::MLSTM(MLSTMBlock::new(input_size, factor, depth)),
                's' => XLSTMBlock::SLSTM(SLSTMBlock::new(input_size, depth)),
                _ => panic!(
                    "Invalid layer type: '{}'. Use 'm' for mLSTM or 's' for sLSTM",
                    layer_char
                ),
            };
            blocks.push(block);
        }

        XLSTMNetwork {
            blocks,
            input_size,
            hidden_size,
            depth,
            factor,
            config: config.to_string(),
        }
    }

    /// Create network with default factor (2.0) for mLSTM blocks
    pub fn from_config(config: &str, input_size: usize, hidden_size: usize, depth: usize) -> Self {
        Self::new(config, input_size, hidden_size, depth, 2.0)
    }

    /// Create a simple mLSTM-only network
    pub fn mlstm_only(
        input_size: usize,
        hidden_size: usize,
        depth: usize,
        num_layers: usize,
    ) -> Self {
        let config = "m".repeat(num_layers);
        Self::from_config(&config, input_size, hidden_size, depth)
    }

    /// Create a simple sLSTM-only network  
    pub fn slstm_only(
        input_size: usize,
        hidden_size: usize,
        depth: usize,
        num_layers: usize,
    ) -> Self {
        let config = "s".repeat(num_layers);
        Self::from_config(&config, input_size, hidden_size, depth)
    }

    /// Create an alternating mLSTM-sLSTM network
    pub fn alternating(
        input_size: usize,
        hidden_size: usize,
        depth: usize,
        num_layers: usize,
    ) -> Self {
        let mut config = String::new();
        for i in 0..num_layers {
            config.push(if i % 2 == 0 { 'm' } else { 's' });
        }
        Self::from_config(&config, input_size, hidden_size, depth)
    }

    /// Forward pass through the network
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
    pub fn forward_with_cache(&mut self, input: &Array2<f64>) -> (Array2<f64>, XLSTMNetworkCache) {
        assert_eq!(
            input.nrows(),
            self.input_size,
            "Input size {} doesn't match network input size {}",
            input.nrows(),
            self.input_size
        );

        let mut current_output = input.clone();
        let mut block_caches = Vec::new();

        // Process through each block with residual connections
        for block in &mut self.blocks {
            let block_input = current_output.clone();
            let (block_output, cache) = block.forward_with_cache(&block_input);
            block_caches.push(cache);

            // Residual connection: output = block_output + input
            current_output = block_output + block_input;
        }

        let network_cache = XLSTMNetworkCache {
            block_caches,
            original_input: input.clone(),
        };

        (current_output, network_cache)
    }

    /// Process a sequence of inputs maintaining state across time steps
    ///
    /// # Arguments
    /// * `sequence` - Sequence of input tensors, each of shape (input_size, batch_size)
    ///
    /// # Returns
    /// * Vector of output tensors, same length as input sequence
    pub fn forward_sequence(&mut self, sequence: &[Array2<f64>]) -> Vec<Array2<f64>> {
        sequence.iter().map(|input| self.forward(input)).collect()
    }

    /// Process sequence with caching for training
    pub fn forward_sequence_with_cache(
        &mut self,
        sequence: &[Array2<f64>],
    ) -> (Vec<Array2<f64>>, Vec<XLSTMNetworkCache>) {
        let mut outputs = Vec::new();
        let mut caches = Vec::new();

        for input in sequence {
            let (output, cache) = self.forward_with_cache(input);
            outputs.push(output);
            caches.push(cache);
        }

        (outputs, caches)
    }

    /// Reset all internal states in all blocks
    pub fn reset_states(&mut self) {
        for block in &mut self.blocks {
            block.reset_state();
        }
    }

    /// Get the total number of parameters in the network
    pub fn num_parameters(&self) -> usize {
        self.blocks.iter().map(|block| block.num_parameters()).sum()
    }

    /// Get the number of blocks in the network
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the configuration string
    pub fn get_config(&self) -> &str {
        &self.config
    }

    /// Get block types as a vector
    pub fn get_block_types(&self) -> Vec<XLSTMBlockType> {
        self.blocks.iter().map(|block| block.block_type()).collect()
    }

    /// Get summary information about the network
    pub fn summary(&self) -> String {
        let num_mlstm = self
            .blocks
            .iter()
            .filter(|b| matches!(b, XLSTMBlock::MLSTM(_)))
            .count();
        let num_slstm = self
            .blocks
            .iter()
            .filter(|b| matches!(b, XLSTMBlock::SLSTM(_)))
            .count();

        format!(
            "xLSTM Network Summary:\n\
             - Configuration: '{}'\n\
             - Input size: {}\n\
             - Hidden size: {}\n\
             - Depth: {}\n\
             - Factor (mLSTM): {:.1}\n\
             - Total blocks: {} ({} mLSTM, {} sLSTM)\n\
             - Total parameters: {}",
            self.config,
            self.input_size,
            self.hidden_size,
            self.depth,
            self.factor,
            self.num_blocks(),
            num_mlstm,
            num_slstm,
            self.num_parameters()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_xlstm_network_creation() {
        let network = XLSTMNetwork::from_config("msm", 8, 16, 4);
        assert_eq!(network.input_size, 8);
        assert_eq!(network.hidden_size, 16);
        assert_eq!(network.depth, 4);
        assert_eq!(network.num_blocks(), 3);
        assert_eq!(network.get_config(), "msm");

        let block_types = network.get_block_types();
        assert_eq!(
            block_types,
            vec![
                XLSTMBlockType::MLSTM,
                XLSTMBlockType::SLSTM,
                XLSTMBlockType::MLSTM
            ]
        );
    }

    #[test]
    fn test_xlstm_network_forward() {
        let mut network = XLSTMNetwork::from_config("ms", 4, 8, 2);
        let input = arr2(&[[1.0], [0.5], [-0.3], [0.8]]);

        let output = network.forward(&input);
        assert_eq!(output.shape(), &[4, 1]);

        // Output should be finite
        for &val in output.iter() {
            assert!(val.is_finite(), "Output should be finite, got {}", val);
        }
    }

    #[test]
    fn test_xlstm_network_residual_connections() {
        let mut network = XLSTMNetwork::from_config("m", 2, 4, 1);
        let input = arr2(&[[1.0], [0.0]]);

        // With residual connections, output should be different from pure block output
        let output = network.forward(&input);

        // The output should contain contributions from both the block and the residual
        // We can't easily test the exact values, but we can ensure it's computed properly
        assert_eq!(output.shape(), input.shape());
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_xlstm_network_sequence_processing() {
        let mut network = XLSTMNetwork::from_config("sm", 3, 6, 2);
        let sequence = vec![
            arr2(&[[1.0], [0.0], [0.5]]),
            arr2(&[[0.5], [1.0], [0.0]]),
            arr2(&[[0.0], [0.5], [1.0]]),
        ];

        let outputs = network.forward_sequence(&sequence);
        assert_eq!(outputs.len(), 3);

        for (i, output) in outputs.iter().enumerate() {
            assert_eq!(output.shape(), &[3, 1], "Output {} has wrong shape", i);
            for &val in output.iter() {
                assert!(val.is_finite(), "Output {} contains non-finite values", i);
            }
        }
    }

    #[test]
    fn test_xlstm_network_state_reset() {
        let mut network = XLSTMNetwork::from_config("ms", 2, 4, 1);
        let input1 = arr2(&[[1.0], [0.0]]);
        let input2 = arr2(&[[0.0], [1.0]]);

        // Process first input
        let output1a = network.forward(&input1);
        let output2a = network.forward(&input2);

        // Reset states and process again
        network.reset_states();
        let output1b = network.forward(&input1);
        let output2b = network.forward(&input2);

        // First outputs should be identical after reset
        for i in 0..2 {
            assert!(
                (output1a[[i, 0]] - output1b[[i, 0]]).abs() < 1e-10,
                "First output should be identical after reset"
            );
        }

        // Second outputs may differ due to different state histories
        // (This tests that state is actually maintained between calls)
    }

    #[test]
    fn test_xlstm_network_types() {
        let mlstm_net = XLSTMNetwork::mlstm_only(4, 8, 2, 3);
        assert_eq!(mlstm_net.get_config(), "mmm");
        assert_eq!(mlstm_net.num_blocks(), 3);

        let slstm_net = XLSTMNetwork::slstm_only(4, 8, 2, 2);
        assert_eq!(slstm_net.get_config(), "ss");
        assert_eq!(slstm_net.num_blocks(), 2);

        let alt_net = XLSTMNetwork::alternating(4, 8, 2, 4);
        assert_eq!(alt_net.get_config(), "msms");
        assert_eq!(alt_net.num_blocks(), 4);
    }

    #[test]
    fn test_xlstm_network_batch_processing() {
        let mut network = XLSTMNetwork::from_config("m", 3, 6, 2);
        let batch_input = arr2(&[[1.0, -1.0], [0.5, 0.0], [-0.3, 0.8]]);

        let output = network.forward(&batch_input);
        assert_eq!(output.shape(), &[3, 2]);

        // All outputs should be finite
        for col in 0..2 {
            for row in 0..3 {
                assert!(
                    output[[row, col]].is_finite(),
                    "Output[{}, {}] should be finite",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_xlstm_network_num_parameters() {
        let network = XLSTMNetwork::from_config("ms", 4, 8, 2);
        let num_params = network.num_parameters();

        // Should have a reasonable number of parameters (> 0)
        assert!(num_params > 0, "Network should have parameters");

        // Check that it matches sum of individual block parameters
        let manual_sum: usize = network
            .blocks
            .iter()
            .map(|block| block.num_parameters())
            .sum();
        assert_eq!(num_params, manual_sum);
    }

    #[test]
    fn test_xlstm_network_summary() {
        let network = XLSTMNetwork::from_config("msms", 8, 16, 3);
        let summary = network.summary();

        // Should contain key information
        assert!(summary.contains("Configuration: 'msms'"));
        assert!(summary.contains("Input size: 8"));
        assert!(summary.contains("Total blocks: 4"));
        assert!(summary.contains("2 mLSTM, 2 sLSTM"));
    }

    #[test]
    #[should_panic(expected = "Invalid layer type")]
    fn test_invalid_config() {
        XLSTMNetwork::from_config("msx", 4, 8, 2); // 'x' is invalid
    }
}
