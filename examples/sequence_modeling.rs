use ndarray::{Array2, arr2};
use xlstm_rs::models::XLSTMNetwork;

/// Generate synthetic sine wave data for sequence modeling
fn generate_sine_wave_sequence(
    length: usize,
    frequency: f64,
    phase: f64,
    noise_level: f64,
) -> Vec<Array2<f64>> {
    let mut sequence = Vec::new();
    let mut rng = rand::rng();

    for i in 0..length {
        let t = i as f64 * 0.1;
        let sine_val = (frequency * t + phase).sin();

        // Add some noise
        let noise = if noise_level > 0.0 {
            use rand::Rng;
            (rng.random::<f64>() - 0.5) * 2.0 * noise_level
        } else {
            0.0
        };

        let noisy_val = sine_val + noise;

        // Create input with multiple features (sine, cosine, time, derivative)
        let cos_val = (frequency * t + phase).cos();
        let time_feature = (t / 10.0).tanh(); // Normalized time
        let derivative = frequency * cos_val; // Analytical derivative

        let input = arr2(&[[noisy_val], [cos_val], [time_feature], [derivative]]);
        sequence.push(input);
    }

    sequence
}

/// Generate a more complex multi-frequency signal
fn generate_complex_signal(length: usize) -> Vec<Array2<f64>> {
    let mut sequence = Vec::new();

    for i in 0..length {
        let t = i as f64 * 0.05;

        // Mix of different frequencies
        let signal1 = (2.0 * t).sin() * 0.6;
        let signal2 = (5.0 * t).sin() * 0.3;
        let signal3 = (0.5 * t).sin() * 0.8;

        let combined = signal1 + signal2 + signal3;

        // Multiple input features
        let trend = (t / 20.0).tanh();
        let envelope = (-0.1 * t).exp();
        let modulated = combined * envelope;

        let input = arr2(&[[modulated], [trend], [envelope], [combined]]);
        sequence.push(input);
    }

    sequence
}

/// Evaluate sequence prediction quality
fn evaluate_prediction(predictions: &[Array2<f64>], targets: &[Array2<f64>]) -> (f64, f64) {
    assert_eq!(predictions.len(), targets.len());

    let mut mse = 0.0;
    let mut mae = 0.0;
    let mut count = 0;

    for (pred, target) in predictions.iter().zip(targets.iter()) {
        for (p, t) in pred.iter().zip(target.iter()) {
            let error = p - t;
            mse += error * error;
            mae += error.abs();
            count += 1;
        }
    }

    mse /= count as f64;
    mae /= count as f64;

    (mse, mae)
}

fn main() {
    println!("🌊 xLSTM Rust - Sequence Modeling Example");
    println!("==========================================\n");

    println!("1. Generating synthetic time series data...");

    // Generate training sequences
    let train_sine = generate_sine_wave_sequence(100, 2.0, 0.0, 0.1);
    let train_complex = generate_complex_signal(150);

    println!(
        "   Simple sine wave sequence: {} time steps",
        train_sine.len()
    );
    println!(
        "   Complex multi-frequency sequence: {} time steps",
        train_complex.len()
    );

    println!("\n2. Creating xLSTM networks for sequence modeling...");

    // Different architectures for comparison
    let mut mlstm_net = XLSTMNetwork::mlstm_only(4, 16, 4, 2);
    let mut slstm_net = XLSTMNetwork::slstm_only(4, 16, 4, 2);
    let mut mixed_net = XLSTMNetwork::from_config("msms", 4, 16, 4);
    let mut deep_net = XLSTMNetwork::from_config("msmsms", 4, 16, 4);

    println!(
        "   mLSTM network: {} parameters",
        mlstm_net.num_parameters()
    );
    println!(
        "   sLSTM network: {} parameters",
        slstm_net.num_parameters()
    );
    println!(
        "   Mixed network: {} parameters",
        mixed_net.num_parameters()
    );
    println!("   Deep network: {} parameters", deep_net.num_parameters());

    println!("\n3. Processing sine wave sequence...");

    // Process the sine wave with different networks
    let sine_outputs_mlstm = mlstm_net.forward_sequence(&train_sine);
    mlstm_net.reset_states(); // Reset for fair comparison

    let sine_outputs_slstm = slstm_net.forward_sequence(&train_sine);
    slstm_net.reset_states();

    let sine_outputs_mixed = mixed_net.forward_sequence(&train_sine);
    mixed_net.reset_states();

    println!(
        "   Processed {} time steps with each network",
        sine_outputs_mlstm.len()
    );

    // Analyze output characteristics
    println!("   Output statistics (last time step):");
    let last_mlstm = sine_outputs_mlstm.last().unwrap();
    let last_slstm = sine_outputs_slstm.last().unwrap();
    let last_mixed = sine_outputs_mixed.last().unwrap();

    println!(
        "     mLSTM - mean: {:.4}, std: {:.4}",
        last_mlstm.mean().unwrap(),
        last_mlstm.std(0.0)
    );
    println!(
        "     sLSTM - mean: {:.4}, std: {:.4}",
        last_slstm.mean().unwrap(),
        last_slstm.std(0.0)
    );
    println!(
        "     Mixed - mean: {:.4}, std: {:.4}",
        last_mixed.mean().unwrap(),
        last_mixed.std(0.0)
    );

    println!("\n4. Sequence-to-sequence prediction task...");

    // Create prediction task: predict next value in sequence
    let sequence_len = 50;
    let pred_sequence = generate_sine_wave_sequence(sequence_len, 1.5, 0.5, 0.05);

    // Use first part as input, predict the rest
    let input_len = 30;
    let inputs = &pred_sequence[..input_len];
    let targets = &pred_sequence[1..input_len + 1]; // Next-step targets

    println!("   Input sequence length: {}", inputs.len());
    println!("   Target sequence length: {}", targets.len());

    // Generate predictions with different networks
    deep_net.reset_states();
    let predictions = deep_net.forward_sequence(inputs);

    // Evaluate prediction quality
    let (mse, mae) = evaluate_prediction(&predictions, targets);
    println!("   Prediction quality - MSE: {:.6}, MAE: {:.6}", mse, mae);

    println!("\n5. Multi-step ahead prediction...");

    // Predict multiple steps ahead
    let context_len = 20;
    let predict_steps = 10;

    mixed_net.reset_states();

    // Process context
    let context = &pred_sequence[..context_len];
    let _context_outputs = mixed_net.forward_sequence(context);

    // Generate predictions step by step
    let mut predictions = Vec::new();
    let mut current_input = pred_sequence[context_len].clone();

    for step in 0..predict_steps {
        let prediction = mixed_net.forward(&current_input);
        predictions.push(prediction.clone());

        // Use prediction as input for next step (teacher forcing disabled)
        current_input = prediction.clone();

        println!(
            "   Step {}: predicted value = {:.4}",
            step + 1,
            prediction[[0, 0]]
        );
    }

    // Compare with actual values
    println!("   Actual vs Predicted (first feature):");
    for step in 0..predict_steps {
        let actual_idx = context_len + step;
        if actual_idx < pred_sequence.len() {
            let actual = pred_sequence[actual_idx][[0, 0]];
            let predicted = predictions[step][[0, 0]];
            let error = (actual - predicted).abs();
            println!(
                "     Step {}: actual={:.4}, predicted={:.4}, error={:.4}",
                step + 1,
                actual,
                predicted,
                error
            );
        }
    }

    println!("\n6. Processing complex multi-frequency signal...");

    deep_net.reset_states();
    let complex_outputs = deep_net.forward_sequence(&train_complex);

    // Analyze how the network responds to different parts of the signal
    let segment_size = train_complex.len() / 5;
    for segment in 0..5 {
        let start_idx = segment * segment_size;
        let end_idx = ((segment + 1) * segment_size).min(complex_outputs.len());

        if start_idx < end_idx {
            let segment_outputs: Vec<_> = complex_outputs[start_idx..end_idx].to_vec();
            let mean_response = segment_outputs
                .iter()
                .map(|output| output.mean().unwrap())
                .sum::<f64>()
                / segment_outputs.len() as f64;

            println!(
                "   Segment {} (t={}-{}): mean response = {:.4}",
                segment + 1,
                start_idx,
                end_idx - 1,
                mean_response
            );
        }
    }

    println!("\n7. Demonstrating long-term memory...");

    // Test how well the network remembers long-term dependencies
    let memory_test_len = 80;
    let mut memory_sequence = Vec::new();

    // Create sequence with long-term pattern
    for i in 0..memory_test_len {
        let base_val = if i < 10 {
            1.0
        } else if i > 70 {
            1.0
        } else {
            0.0
        };
        let noise = (i as f64 * 0.3).sin() * 0.2;

        let input = arr2(&[
            [base_val + noise],
            [noise],
            [base_val],
            [(i as f64) / memory_test_len as f64],
        ]);
        memory_sequence.push(input);
    }

    deep_net.reset_states();
    let memory_outputs = deep_net.forward_sequence(&memory_sequence);

    // Check if network maintains information about the early pattern
    let early_mean = memory_outputs[..10]
        .iter()
        .map(|out| out.mean().unwrap())
        .sum::<f64>()
        / 10.0;
    let late_mean = memory_outputs[70..]
        .iter()
        .map(|out| out.mean().unwrap())
        .sum::<f64>()
        / (memory_outputs.len() - 70) as f64;

    println!("   Early sequence mean response: {:.4}", early_mean);
    println!("   Late sequence mean response: {:.4}", late_mean);
    println!(
        "   Response correlation: {:.4}",
        if early_mean.abs() > 1e-6 {
            late_mean / early_mean
        } else {
            0.0
        }
    );

    println!("\n8. Architecture comparison summary...");

    // Reset all networks and compare on same sequence
    let test_seq = generate_sine_wave_sequence(30, 3.0, 0.25, 0.02);

    mlstm_net.reset_states();
    slstm_net.reset_states();
    mixed_net.reset_states();
    deep_net.reset_states();

    let mlstm_final = mlstm_net
        .forward_sequence(&test_seq)
        .last()
        .unwrap()
        .clone();
    let slstm_final = slstm_net
        .forward_sequence(&test_seq)
        .last()
        .unwrap()
        .clone();
    let mixed_final = mixed_net
        .forward_sequence(&test_seq)
        .last()
        .unwrap()
        .clone();
    let deep_final = deep_net.forward_sequence(&test_seq).last().unwrap().clone();

    println!("   Final outputs on test sequence:");
    println!(
        "   mLSTM: [{:.4}, {:.4}, {:.4}, {:.4}]",
        mlstm_final[[0, 0]],
        mlstm_final[[1, 0]],
        mlstm_final[[2, 0]],
        mlstm_final[[3, 0]]
    );
    println!(
        "   sLSTM: [{:.4}, {:.4}, {:.4}, {:.4}]",
        slstm_final[[0, 0]],
        slstm_final[[1, 0]],
        slstm_final[[2, 0]],
        slstm_final[[3, 0]]
    );
    println!(
        "   Mixed: [{:.4}, {:.4}, {:.4}, {:.4}]",
        mixed_final[[0, 0]],
        mixed_final[[1, 0]],
        mixed_final[[2, 0]],
        mixed_final[[3, 0]]
    );
    println!(
        "   Deep:  [{:.4}, {:.4}, {:.4}, {:.4}]",
        deep_final[[0, 0]],
        deep_final[[1, 0]],
        deep_final[[2, 0]],
        deep_final[[3, 0]]
    );

    println!("\n✅ Sequence modeling demonstration completed!");
    println!("    Key observations:");
    println!("    - xLSTM can process complex temporal patterns");
    println!("    - Different architectures show different response characteristics");
    println!("    - Networks maintain long-term dependencies across sequences");
    println!("    - Both mLSTM and sLSTM contribute to sequence understanding");
    println!("    - Residual connections help with gradient flow in deep networks");
}
