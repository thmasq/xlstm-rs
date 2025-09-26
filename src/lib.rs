//! # xLSTM Rust Library
//!
//! A complete Rust implementation of xLSTM (Extended Long Short-Term Memory) networks
//! based on the paper "xLSTM: Extended Long Short-Term Memory" by Beck et al.
//!
//! ## Core Components
//!
//! - **mLSTM Blocks**: Matrix-based LSTM with exponential gating and memory mixing
//! - **sLSTM Blocks**: Scalar LSTM with memory stabilization and exponential gating  
//! - **Block Diagonal Layers**: Efficient parallel linear transformations
//! - **Causal Convolutions**: Temporal convolutions that respect causal ordering
//! - **xLSTM Networks**: Flexible stacking of mLSTM and sLSTM blocks
//!
//! ## Quick Start
//!
//! ```rust
//! use xlstm_rust::models::XLSTMNetwork;
//! use ndarray::arr2;
//!
//! // Create a mixed xLSTM with mLSTM and sLSTM blocks
//! let mut network = XLSTMNetwork::from_config("ms", 32, 64, 4);
//!
//! // Process a sequence
//! let input = arr2(&[[1.0; 32]]).t().to_owned();
//! let output = network.forward(&input);
//! ```

pub mod layers;
pub mod models;
pub mod utils;

// Re-export commonly used items
pub use layers::{BlockDiagonal, CausalConv1D, LayerNorm, MLSTMBlock, SLSTMBlock};
pub use models::XLSTMNetwork;
pub use utils::{exp_stabilized, gelu, sigmoid, silu};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_library_integration() {
        let mut network = models::XLSTMNetwork::from_config("m", 4, 8, 2);
        let input = arr2(&[[1.0, 0.5, -0.3, 0.8]]).t().to_owned();

        let output = network.forward(&input);
        assert_eq!(output.shape(), &[4, 1]);
    }
}
