use ndarray::{Array2, arr2};
use xlstm_rs::{
    loss::MSELoss, models::XLSTMNetwork, optimizers::Adam, schedulers::CosineAnnealingLR,
    training::*,
};

/// Simple time series prediction example using xLSTM
fn main() {
    println!("xLSTM Time Series Prediction Example");
    println!("========================================\n");

    // Generate sample time series data (stock price-like)
    let (train_data, test_data) = generate_time_series_data();

    println!("1. Generated time series data:");
    println!("   Training sequences: {}", train_data.0.len());
    println!("   Test sequence length: {}", test_data.len());

    // Create xLSTM model
    let input_dim = 1; // Single feature (price)
    let hidden_dim = 32;
    let depth = 4;

    println!("\n2. Creating xLSTM model:");
    println!("   Architecture: Mixed mLSTM-sLSTM (msms)");
    println!("   Input dimension: {}", input_dim);
    println!("   Hidden dimension: {}", hidden_dim);
    println!("   Depth: {}", depth);

    let model = XLSTMNetwork::from_config("msms", input_dim, hidden_dim, depth);
    println!("   Total parameters: {}", model.num_parameters());

    // Set up training
    let loss_fn = MSELoss;
    let optimizer = Adam::new(0.001);
    let scheduler = CosineAnnealingLR::new(100, 1e-6);

    let mut trainer = XLSTMTrainer::new(model, loss_fn, optimizer).with_scheduler(scheduler);

    let config = TrainingConfig {
        epochs: 100,
        print_every: 20,
        validation_split: 0.2,
        early_stopping_patience: Some(20),
        gradient_clip_norm: Some(1.0),
        ..Default::default()
    };

    trainer = trainer.with_config(config);

    println!("\n3. Training the model...");
    match trainer.train_on_sequences(&train_data.0, &train_data.1) {
        Ok(metrics) => {
            println!("   Training completed successfully!");

            if let Some(final_metrics) = metrics.last() {
                println!("   Final training loss: {:.6}", final_metrics.train_loss);
                if let Some(val_loss) = final_metrics.val_loss {
                    println!("   Final validation loss: {:.6}", val_loss);
                }
            }

            // Test prediction
            println!("\n4. Testing predictions...");
            test_predictions(&mut trainer, &test_data);

            // Demonstrate different prediction modes
            println!("\n5. Advanced prediction examples...");
            demonstrate_prediction_modes(&mut trainer, &test_data);
        }
        Err(e) => {
            println!("   Training failed: {}", e);
        }
    }

    println!("\n Time series prediction example completed!");
}

fn generate_time_series_data() -> (
    (Vec<Vec<Array2<f64>>>, Vec<Vec<Array2<f64>>>),
    Vec<Array2<f64>>,
) {
    let mut sequences = Vec::new();
    let mut targets = Vec::new();

    // Generate 50 training sequences
    for i in 0..50 {
        let sequence_length = 30;
        let mut sequence = Vec::new();
        let mut target_sequence = Vec::new();

        // Generate a trend with some volatility
        let trend = 0.001 * i as f64; // Small upward trend
        let volatility = 0.1;
        let mut price = 100.0; // Starting price

        for _ in 0..sequence_length {
            // Random walk with trend
            let change = trend + volatility * random_normal();
            price += change;

            let input = arr2(&[[price / 100.0]]); // Normalize
            sequence.push(input);

            // Target is next price change (predict direction)
            let next_change = trend + volatility * random_normal();
            let target = arr2(&[[next_change]]);
            target_sequence.push(target);
        }

        sequences.push(sequence);
        targets.push(target_sequence);
    }

    // Generate test sequence
    let mut test_sequence = Vec::new();
    let mut price = 105.0;

    for _ in 0..50 {
        let change = 0.001 + 0.1 * random_normal();
        price += change;
        let input = arr2(&[[price / 100.0]]);
        test_sequence.push(input);
    }

    ((sequences, targets), test_sequence)
}

fn random_normal() -> f64 {
    use rand::Rng;
    let mut rng = rand::rng();

    // Box-Muller transform for normal distribution
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();

    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn test_predictions(
    trainer: &mut XLSTMTrainer<MSELoss, Adam, CosineAnnealingLR>,
    test_data: &[Array2<f64>],
) {
    // Use first 30 points to predict next 10
    let context_length = 30;
    let prediction_length = 10;

    let context = &test_data[..context_length];

    // Method 1: One-step predictions
    let one_step_predictions = trainer.predict(context);
    println!(
        "   One-step predictions generated: {}",
        one_step_predictions.len()
    );

    // Method 2: Multi-step ahead predictions
    let multi_step_predictions = trainer.predict_next(context, prediction_length);
    println!(
        "   Multi-step predictions generated: {}",
        multi_step_predictions.len()
    );

    // Compare predictions with actual values if available
    if test_data.len() > context_length + prediction_length {
        let actual = &test_data[context_length..context_length + prediction_length];

        println!("\n   Prediction comparison (first 5 steps):");
        println!("   Step | Multi-step | Actual | Error");
        println!("   -----|------------|--------|-------");

        for (i, (pred, actual_val)) in multi_step_predictions
            .iter()
            .zip(actual.iter())
            .take(5)
            .enumerate()
        {
            let error = (pred[[0, 0]] - actual_val[[0, 0]]).abs();
            println!(
                "   {:4} | {:10.4} | {:6.4} | {:.4}",
                i + 1,
                pred[[0, 0]],
                actual_val[[0, 0]],
                error
            );
        }
    }
}

fn demonstrate_prediction_modes(
    trainer: &mut XLSTMTrainer<MSELoss, Adam, CosineAnnealingLR>,
    test_data: &[Array2<f64>],
) {
    println!("   Demonstrating different prediction strategies:");

    // Strategy 1: Rolling window prediction
    println!("\n   Rolling window prediction:");
    let window_size = 10;
    let mut rolling_predictions = Vec::new();

    for i in 0..(test_data.len() - window_size - 1) {
        let window = &test_data[i..i + window_size];
        let pred = trainer.predict_next(window, 1);
        if let Some(prediction) = pred.first() {
            rolling_predictions.push(prediction.clone());
        }
    }

    println!(
        "     Generated {} rolling predictions",
        rolling_predictions.len()
    );

    // Strategy 2: Expanding window prediction
    println!("\n   Expanding window prediction:");
    let mut expanding_predictions = Vec::new();
    let min_window = 15;

    for end_idx in min_window..std::cmp::min(test_data.len(), min_window + 5) {
        let window = &test_data[..end_idx];
        let pred = trainer.predict_next(window, 1);
        if let Some(prediction) = pred.first() {
            expanding_predictions.push(prediction.clone());
            println!(
                "     Window size {}: prediction = {:.4}",
                end_idx,
                prediction[[0, 0]]
            );
        }
    }

    // Strategy 3: Confidence intervals (simplified)
    println!("\n   Generating multiple predictions for uncertainty estimation:");
    let context = &test_data[..20];

    for _ in 0..3 {
        // Reset model state and predict (would need proper uncertainty in real implementation)
        let pred = trainer.predict_next(context, 1);
        if let Some(prediction) = pred.first() {
            println!("     Sample prediction: {:.4}", prediction[[0, 0]]);
        }
    }

    // Model interpretation
    println!("\n   Model summary:");
    println!("     Architecture: {}", trainer.model.get_config());
    println!("     Block types: {:?}", trainer.model.get_block_types());
    println!("     Parameters per block type:");

    let block_types = trainer.model.get_block_types();
    for (i, block_type) in block_types.iter().enumerate() {
        println!("       Block {}: {:?}", i, block_type);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation() {
        let (train_data, test_data) = generate_time_series_data();

        assert!(!train_data.0.is_empty());
        assert!(!train_data.1.is_empty());
        assert!(!test_data.is_empty());
        assert_eq!(train_data.0.len(), train_data.1.len());
    }

    #[test]
    fn test_model_creation() {
        let model = XLSTMNetwork::from_config("ms", 1, 16, 2);
        assert_eq!(model.input_size, 1);
        assert_eq!(model.hidden_size, 16);
        assert!(model.num_parameters() > 0);
    }
}
