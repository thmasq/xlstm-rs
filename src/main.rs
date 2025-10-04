/*!
Financial Time Series Forecasting with xLSTM

This example demonstrates how to use xLSTM for financial forecasting tasks.
Shows basic usage patterns with synthetic data.

Author: Mudit Bhargava (Ported to Rust)
Date: October 2025
*/

use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Distribution, Tensor, backend::Backend},
};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use xlstm::{LstmType, XLstmconfig};

type MyBackend = Autodiff<NdArray<f32>>;

fn main() {
    println!("xLSTM Financial Forecasting Example");
    println!("====================================\n");

    // Hyperparameters
    let input_size = 10; // Number of features (OHLCV + indicators)
    let hidden_size = 128;
    let num_layers = 2;
    let num_blocks = 4;
    let output_size = 1; // Predict next price
    let dropout = 0.2;

    let batch_size = 32;
    let seq_length = 60;
    let num_epochs = 5;
    let learning_rate = 0.001;

    // Device
    let device = Default::default();

    // Create model with alternating blocks
    println!("Creating xLSTM model with alternating sLSTM/mLSTM blocks...");
    let config = XLstmconfig::new(input_size, hidden_size, num_layers, num_blocks, output_size)
        .with_dropout(dropout)
        .with_lstm_type(LstmType::Alternate)
        .with_use_projection(true);

    let mut model = config.init::<MyBackend>(&device);
    model.print_architecture();
    println!();

    // Create optimizer
    let mut optim = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .init();

    println!("Starting training...\n");

    // Training loop
    for epoch in 0..num_epochs {
        // Generate synthetic batch
        let input = Tensor::<MyBackend, 3>::random(
            [batch_size, seq_length, input_size],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // Generate synthetic targets (next timestep values)
        let targets = Tensor::<MyBackend, 2>::random(
            [batch_size, output_size],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // Forward pass
        let (predictions, _) = model.predict_last(input, None);

        // Compute loss (MSE)
        let loss = mse_loss(predictions.clone(), targets.clone());
        let loss_value = loss.clone().into_scalar();

        // Backward pass
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(learning_rate, model, grads);

        println!(
            "Epoch [{}/{}], Loss: {:.6}",
            epoch + 1,
            num_epochs,
            loss_value
        );
    }

    println!("\nTraining completed!");
    println!("\nModel can now be used for inference on real financial data.");
    println!("Features to include:");
    println!("  - OHLCV data (Open, High, Low, Close, Volume)");
    println!("  - Technical indicators (SMA, EMA, RSI, MACD, etc.)");
    println!("  - Returns and volatility measures");
}

/// Simple MSE loss function
fn mse_loss<B: Backend>(predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = predictions - targets;
    let squared = diff.clone() * diff;
    squared.mean()
}
