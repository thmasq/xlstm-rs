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

// Core modules
pub mod layers;
pub mod models;
pub mod utils;

// Training system modules
pub mod loss;
pub mod optimizers;
pub mod schedulers;
pub mod training;

// Re-export commonly used core items
pub use layers::{BlockDiagonal, CausalConv1D, LayerNorm, MLSTMBlock, SLSTMBlock};
pub use models::XLSTMNetwork;
pub use utils::{
    exp_stabilized, find_suitable_num_blocks, gelu, safe_div, safe_max, sigmoid, silu,
};

// Re-export training system components
pub use loss::{BCELoss, CrossEntropyLoss, HuberLoss, LossFunction, MAELoss, MSELoss};
pub use optimizers::{Adagrad, Adam, AdamW, Optimizer, RMSprop, SGD};
pub use schedulers::{
    AnnealStrategy, ConstantLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicalLR,
    CyclicalMode, ExponentialLR, LearningRateScheduler, LinearLR, MultiStepLR, OneCycleLR,
    PlateauMode, PolynomialLR, ReduceLROnPlateau, ScaleMode, StepLR, WarmupScheduler,
};
pub use training::{
    EarlyStopper, TrainingConfig, TrainingMetrics, XLSTMTrainer, create_adam_exponential_trainer,
    create_adam_step_trainer, create_adam_trainer, create_basic_trainer,
    create_rmsprop_cosine_trainer,
};

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

    #[test]
    fn test_training_integration() {
        // Test that training components work together
        let model = models::XLSTMNetwork::from_config("s", 2, 4, 1);

        // Create trainer
        let trainer = create_adam_trainer(model, 0.01);

        // Check components are properly initialized
        assert_eq!(trainer.optimizer.get_learning_rate(), 0.01);
        assert_eq!(trainer.config.epochs, 100); // Default

        // Test that we can create different configurations
        let _sgd_trainer =
            create_basic_trainer(models::XLSTMNetwork::from_config("m", 2, 4, 1), 0.1);

        let _step_trainer = create_adam_step_trainer(
            models::XLSTMNetwork::from_config("ms", 2, 4, 1),
            0.001,
            10,
            0.5,
        );
    }

    #[test]
    fn test_loss_functions_integration() {
        let predictions = arr2(&[[0.5, 0.8], [0.2, 0.9]]);
        let targets = arr2(&[[0.6, 0.7], [0.1, 1.0]]);

        // Test different loss functions
        let mse = MSELoss;
        assert!(mse.compute_loss(&predictions, &targets) > 0.0);

        let mae = MAELoss;
        assert!(mae.compute_loss(&predictions, &targets) > 0.0);

        let huber = HuberLoss::new(1.0);
        assert!(huber.compute_loss(&predictions, &targets) > 0.0);
    }

    #[test]
    fn test_optimizers_integration() {
        let mut param = arr2(&[[1.0, 2.0]]);
        let gradient = arr2(&[[0.1, 0.2]]);

        // Test different optimizers
        let mut sgd = SGD::new(0.01);
        sgd.update("test", &mut param, &gradient);

        let mut adam = Adam::new(0.001);
        adam.update("test", &mut param, &gradient);

        let mut rmsprop = RMSprop::new(0.001);
        rmsprop.update("test", &mut param, &gradient);
    }

    #[test]
    fn test_schedulers_integration() {
        let base_lr = 0.01;

        let mut step_lr = StepLR::new(10, 0.1);
        assert_eq!(step_lr.get_lr(0, base_lr), base_lr);

        let mut cosine_lr = CosineAnnealingLR::new(100, 0.001);
        assert_eq!(cosine_lr.get_lr(0, base_lr), base_lr);

        let mut one_cycle = OneCycleLR::new(0.1, 100);
        assert!(one_cycle.get_lr(0, base_lr) > 0.0);
    }
}
