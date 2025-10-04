#![feature(slice_pattern)]
#![recursion_limit = "256"]

/*!
Financial Time Series Forecasting with xLSTM using Embeddings

This example demonstrates how to use xLSTM for financial forecasting
using pre-computed embeddings from `embeddings_128.csv`.

Author: Mudit Bhargava (Ported to Rust)
Date: October 2025
*/

use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Tensor, backend::Backend},
};
use burn_autodiff::Autodiff;
use burn_wgpu::{Wgpu, WgpuDevice};
use core::slice::SlicePattern;
use csv::ReaderBuilder;
use plotters::prelude::*;
use serde::Deserialize;
use std::error::Error;
use xlstm::{LstmType, XLstmconfig};

type MyBackend = Autodiff<Wgpu>;

type DataLoadResult = Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn Error>>;

type SequenceData<B> = (
    Tensor<B, 3>,    // train_x: [num_samples, seq_length, input_size]
    Tensor<B, 2>,    // train_y: [num_samples, output_size]
    Tensor<B, 3>,    // test_x: [num_samples, seq_length, input_size]
    Tensor<B, 2>,    // test_y: [num_samples, output_size]
    Vec<(f32, f32)>, // test_prices: (current_price, next_price)
);

#[derive(Deserialize)]
#[allow(dead_code)]
struct EmbeddingRecord {
    trading_date: String,
    trading_code: String,
    company_name: String,
    last_price: f32,
    #[serde(flatten)]
    embeddings: std::collections::HashMap<String, f32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("xLSTM Financial Forecasting with Embeddings");
    println!("==========================================\n");

    // Load and prepare data
    println!("Loading embeddings_128.csv...");
    let (embeddings, prices) = load_data("embeddings_128.csv")?;
    println!("Loaded {} records", prices.len());

    // Hyperparameters
    let input_size = 128; // Embedding dimensions
    let hidden_size = 128;
    let num_layers = 2;
    let num_blocks = 4;
    let output_size = 1; // Predict next price
    let dropout = 0.2;

    let seq_length = 20; // Use 20 days to predict next day
    let batch_size = 8;
    let num_epochs = 20;
    let learning_rate = 0.0001;
    let train_split = 0.8;

    // Device
    let device = WgpuDevice::default();

    // Create sequences (now returns stacked tensors)
    println!("Creating sequences (seq_length={seq_length})...");
    let (train_x, train_y, test_x, test_y, test_prices) =
        create_sequences(&embeddings, &prices, seq_length, train_split, &device);

    let num_train = train_x.dims()[0];
    let num_test = test_x.dims()[0];

    println!("Training samples: {}", num_train);
    println!("Testing samples: {}\n", num_test);

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

    // Training loop with vectorized batching
    let num_train_batches = num_train.div_ceil(batch_size);

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0f32;
        let mut num_batches = 0;

        for batch_idx in 0..num_train_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(num_train);

            if start_idx >= end_idx {
                break;
            }

            // Vectorized batch extraction using slicing (no clones!)
            let input_batch =
                train_x
                    .clone()
                    .slice([start_idx..end_idx, 0..seq_length, 0..input_size]);
            let target_batch = train_y.clone().slice([start_idx..end_idx, 0..output_size]);

            // Forward pass
            let (predictions, _) = model.predict_last(input_batch, None);

            // Fused MSE loss computation (single kernel chain)
            let loss = ((predictions - target_batch).powf_scalar(2.0)).mean();

            // Extract loss value before backward
            let loss_value: <MyBackend as Backend>::FloatElem = loss.clone().into_scalar();
            let loss_f32 = num_traits::ToPrimitive::to_f32(&loss_value).unwrap_or(0.0);

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads);

            total_loss += loss_f32;
            num_batches += 1;
        }

        let avg_loss = total_loss / num_batches as f32;

        // Validation
        if epoch % 5 == 0 && num_test > 0 {
            let val_loss = evaluate(
                &model,
                &test_x,
                &test_y,
                seq_length,
                input_size,
                output_size,
            );
            println!(
                "Epoch [{:2}/{}], Train Loss: {:.6}, Val Loss: {:.6}",
                epoch + 1,
                num_epochs,
                avg_loss,
                val_loss
            );
        } else {
            println!(
                "Epoch [{:2}/{}], Train Loss: {:.6}",
                epoch + 1,
                num_epochs,
                avg_loss
            );
        }
    }

    println!("\nTraining completed!");
    println!("\nModel Performance:");
    println!("  - Input: 128-dim embeddings representing market state");
    println!("  - Architecture: {num_blocks} alternating sLSTM/mLSTM blocks");
    println!("  - Sequence length: {seq_length} days");
    println!("  - Output: Next day price prediction");

    // Make predictions on test set
    println!("\n\nGenerating predictions on test set...");
    let (predictions, actuals) =
        make_predictions(&model, &test_x, &test_prices, seq_length, input_size);

    // Calculate metrics
    let mse = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;
    let rmse = mse.sqrt();
    let mae = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f32>()
        / predictions.len() as f32;

    println!("\nPrediction Metrics:");
    println!("  RMSE: {rmse:.4}");
    println!("  MAE:  {mae:.4}");

    // Create visualizations
    println!("\nCreating visualizations...");
    plot_predictions(&predictions, &actuals, "predictions_vs_actual.png")?;
    plot_scatter(&predictions, &actuals, "prediction_scatter.png")?;

    println!("\nVisualizations saved:");
    println!("  - predictions_vs_actual.png: Time series comparison");
    println!("  - prediction_scatter.png: Prediction accuracy scatter plot");

    Ok(())
}

fn load_data(path: &str) -> DataLoadResult {
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;

    let mut embeddings = Vec::new();
    let mut prices = Vec::new();

    for result in reader.deserialize() {
        let record: EmbeddingRecord = result?;

        // Extract embeddings in order (emb_0 to emb_127)
        let mut emb_vec = Vec::with_capacity(128);
        for i in 0..128 {
            let key = format!("emb_{i}");
            if let Some(&value) = record.embeddings.get(&key) {
                emb_vec.push(value);
            }
        }

        if emb_vec.len() == 128 {
            embeddings.push(emb_vec);
            prices.push(record.last_price);
        }
    }

    Ok((embeddings, prices))
}

fn create_sequences<B: Backend>(
    embeddings: &[Vec<f32>],
    prices: &[f32],
    seq_length: usize,
    train_split: f32,
    device: &B::Device,
) -> SequenceData<B> {
    let num_sequences = embeddings.len() - seq_length;
    let input_size = 128;

    // Pre-allocate flat arrays for all sequences
    let mut x_data = Vec::with_capacity(num_sequences * seq_length * input_size);
    let mut y_data = Vec::with_capacity(num_sequences);
    let mut price_pairs = Vec::with_capacity(num_sequences);

    // Create sliding windows
    for i in 0..num_sequences {
        // Input: sequence of embeddings (flattened)
        for j in 0..seq_length {
            x_data.extend_from_slice(&embeddings[i + j]);
        }

        // Target: next price (normalized as relative change)
        let current_price = prices[i + seq_length - 1];
        let next_price = prices[i + seq_length];
        let target = (next_price - current_price) / current_price; // Relative change

        y_data.push(target);
        price_pairs.push((current_price, next_price));
    }

    // Split into train/test
    let split_idx = (num_sequences as f32 * train_split) as usize;

    // Create stacked tensors for train set
    let train_x = Tensor::<B, 1>::from_floats(
        x_data[..(split_idx * seq_length * input_size)].as_slice(),
        device,
    )
    .reshape([split_idx, seq_length, input_size]);
    let train_y =
        Tensor::<B, 1>::from_floats(y_data[..split_idx].as_slice(), device).reshape([split_idx, 1]);

    // Create stacked tensors for test set
    let test_len = num_sequences - split_idx;
    let test_x = Tensor::<B, 1>::from_floats(
        x_data[(split_idx * seq_length * input_size)..].as_slice(),
        device,
    )
    .reshape([test_len, seq_length, input_size]);
    let test_y =
        Tensor::<B, 1>::from_floats(y_data[split_idx..].as_slice(), device).reshape([test_len, 1]);

    let test_prices = price_pairs[split_idx..].to_vec();

    (train_x, train_y, test_x, test_y, test_prices)
}

fn evaluate<B: Backend>(
    model: &xlstm::XLstm<B>,
    test_x: &Tensor<B, 3>,
    test_y: &Tensor<B, 2>,
    seq_length: usize,
    input_size: usize,
    output_size: usize,
) -> f32
where
    <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive,
{
    let num_test = test_x.dims()[0];
    if num_test == 0 {
        return 0.0;
    }

    let batch_size = 32.min(num_test);
    let mut total_loss = 0.0f32;
    let mut num_batches = 0;

    for i in (0..num_test).step_by(batch_size) {
        let end_idx = (i + batch_size).min(num_test);

        // Vectorized batch extraction using slicing
        let input_batch = test_x
            .clone()
            .slice([i..end_idx, 0..seq_length, 0..input_size]);
        let target_batch = test_y.clone().slice([i..end_idx, 0..output_size]);

        let (predictions, _) = model.predict_last(input_batch, None);
        let loss = mse_loss(predictions, target_batch);

        let loss_value: <B as Backend>::FloatElem = loss.into_scalar();
        total_loss += num_traits::ToPrimitive::to_f32(&loss_value).unwrap_or(0.0);
        num_batches += 1;
    }

    total_loss / num_batches as f32
}

fn make_predictions<B: Backend>(
    model: &xlstm::XLstm<B>,
    test_x: &Tensor<B, 3>,
    test_prices: &[(f32, f32)],
    seq_length: usize,
    input_size: usize,
) -> (Vec<f32>, Vec<f32>)
where
    <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive,
{
    let num_test = test_x.dims()[0];
    let mut predictions = Vec::with_capacity(num_test);
    let mut actuals = Vec::with_capacity(num_test);

    // Process in batches for efficiency
    let batch_size = 32;
    for i in (0..num_test).step_by(batch_size) {
        let end_idx = (i + batch_size).min(num_test);

        // Vectorized batch extraction
        let input_batch = test_x
            .clone()
            .slice([i..end_idx, 0..seq_length, 0..input_size]);
        let (pred_batch, _) = model.predict_last(input_batch, None);

        // Extract predictions from batch
        let batch_len = end_idx - i;
        for j in 0..batch_len {
            let pred_value: <B as Backend>::FloatElem =
                pred_batch.clone().slice([j..(j + 1), 0..1]).into_scalar();
            let pred_relative = num_traits::ToPrimitive::to_f32(&pred_value).unwrap_or(0.0);

            // Convert back to actual price
            let (current_price, actual_next_price) = test_prices[i + j];
            let predicted_price = current_price * (1.0 + pred_relative);

            predictions.push(predicted_price);
            actuals.push(actual_next_price);
        }
    }

    (predictions, actuals)
}

fn plot_predictions(
    predictions: &[f32],
    actuals: &[f32],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(filename, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_val = predictions
        .iter()
        .chain(actuals.iter())
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = predictions
        .iter()
        .chain(actuals.iter())
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let margin = (max_val - min_val) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("xLSTM Price Predictions vs Actual", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..predictions.len(), (min_val - margin)..(max_val + margin))?;

    chart
        .configure_mesh()
        .x_desc("Time Step")
        .y_desc("Price")
        .draw()?;

    // Plot actual prices
    chart
        .draw_series(LineSeries::new(
            actuals.iter().enumerate().map(|(i, &v)| (i, v)),
            &BLUE,
        ))?
        .label("Actual")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // Plot predictions
    chart
        .draw_series(LineSeries::new(
            predictions.iter().enumerate().map(|(i, &v)| (i, v)),
            &RED,
        ))?
        .label("Predicted")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("  Saved: {filename}");

    Ok(())
}

fn plot_scatter(
    predictions: &[f32],
    actuals: &[f32],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(filename, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_val = predictions
        .iter()
        .chain(actuals.iter())
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = predictions
        .iter()
        .chain(actuals.iter())
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let margin = (max_val - min_val) * 0.1;
    let range = (min_val - margin)..(max_val + margin);

    let mut chart = ChartBuilder::on(&root)
        .caption("Prediction Accuracy Scatter Plot", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(range.clone(), range)?;

    chart
        .configure_mesh()
        .x_desc("Actual Price")
        .y_desc("Predicted Price")
        .draw()?;

    // Plot ideal line (y=x)
    chart
        .draw_series(LineSeries::new(
            vec![
                (min_val - margin, min_val - margin),
                (max_val + margin, max_val + margin),
            ],
            &BLACK.mix(0.3),
        ))?
        .label("Perfect Prediction")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    // Plot scatter points
    chart
        .draw_series(
            actuals
                .iter()
                .zip(predictions.iter())
                .map(|(&a, &p)| Circle::new((a, p), 3, RED.filled())),
        )?
        .label("Predictions")
        .legend(|(x, y)| Circle::new((x + 10, y), 3, RED.filled()));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("  Saved: {filename}");

    Ok(())
}

/// Simple MSE loss function
fn mse_loss<B: Backend>(predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = predictions - targets;
    let squared = diff.clone() * diff;
    squared.mean()
}
