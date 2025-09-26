use ndarray::arr2;
use xlstm_rs::models::XLSTMNetwork;

fn main() {
    println!("xLSTM Rust - Basic Usage Example");
    println!("=====================================\n");

    // Create different types of xLSTM networks
    println!("1. Creating different network architectures...");

    // Mixed mLSTM and sLSTM network
    let mut mixed_network = XLSTMNetwork::from_config("msms", 8, 16, 4);
    println!(
        "   Mixed network (msms): {} parameters",
        mixed_network.num_parameters()
    );

    // mLSTM-only network
    let mut mlstm_network = XLSTMNetwork::mlstm_only(8, 16, 4, 3);
    println!(
        "   mLSTM-only network: {} parameters",
        mlstm_network.num_parameters()
    );

    // sLSTM-only network
    let mut slstm_network = XLSTMNetwork::slstm_only(8, 16, 4, 2);
    println!(
        "   sLSTM-only network: {} parameters",
        slstm_network.num_parameters()
    );

    // Alternating network
    let mut alt_network = XLSTMNetwork::alternating(8, 16, 4, 4);
    println!(
        "   Alternating network: {} parameters",
        alt_network.num_parameters()
    );

    println!("\n2. Processing single inputs...");

    // Create sample input (8 features, 1 sample)
    let input = arr2(&[[1.0], [0.5], [-0.3], [0.8], [0.2], [-0.6], [0.9], [-0.1]]);
    println!("   Input shape: {:?}", input.shape());

    // Process through different networks
    let output_mixed = mixed_network.forward(&input);
    let output_mlstm = mlstm_network.forward(&input);
    let output_slstm = slstm_network.forward(&input);
    let output_alt = alt_network.forward(&input);

    println!("   Mixed output shape: {:?}", output_mixed.shape());
    println!("   mLSTM output shape: {:?}", output_mlstm.shape());
    println!("   sLSTM output shape: {:?}", output_slstm.shape());
    println!("   Alternating output shape: {:?}", output_alt.shape());

    println!("\n3. Processing batch inputs...");

    // Create batch input (8 features, 3 samples)
    let batch_input = arr2(&[
        [1.0, -1.0, 0.5],
        [0.5, 0.0, -0.5],
        [-0.3, 0.8, 0.2],
        [0.8, -0.2, 0.9],
        [0.2, 0.6, -0.1],
        [-0.6, 0.4, 0.7],
        [0.9, -0.7, 0.3],
        [-0.1, 0.1, -0.8],
    ]);
    println!("   Batch input shape: {:?}", batch_input.shape());

    let batch_output = mixed_network.forward(&batch_input);
    println!("   Batch output shape: {:?}", batch_output.shape());

    // Print some sample outputs
    println!("   Sample outputs for first input:");
    for (i, &val) in batch_output.column(0).iter().enumerate() {
        println!("     Feature {}: {:.4}", i, val);
    }

    println!("\n4. Processing sequences...");

    // Create a sequence of inputs (time series)
    let sequence = vec![
        arr2(&[[1.0], [0.0], [0.5], [-0.3], [0.8], [0.2], [-0.6], [0.9]]),
        arr2(&[[0.8], [0.2], [0.3], [-0.1], [0.6], [0.4], [-0.4], [0.7]]),
        arr2(&[[0.6], [0.4], [0.1], [0.1], [0.4], [0.6], [-0.2], [0.5]]),
        arr2(&[[0.4], [0.6], [-0.1], [0.3], [0.2], [0.8], [0.0], [0.3]]),
        arr2(&[[0.2], [0.8], [-0.3], [0.5], [0.0], [1.0], [0.2], [0.1]]),
    ];

    println!("   Sequence length: {}", sequence.len());

    // Process sequence (state is maintained across time steps)
    let sequence_outputs = mixed_network.forward_sequence(&sequence);

    println!("   Output sequence length: {}", sequence_outputs.len());
    for (t, output) in sequence_outputs.iter().enumerate() {
        let mean_output = output.mean().unwrap();
        println!("     Time {}: mean output = {:.4}", t, mean_output);
    }

    println!("\n5. Demonstrating state persistence...");

    // Reset network states
    mixed_network.reset_states();

    let input1 = arr2(&[[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]);
    let input2 = arr2(&[[0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]);

    // First run
    let out1a = mixed_network.forward(&input1);
    let out2a = mixed_network.forward(&input2);

    // Reset and run again
    mixed_network.reset_states();
    let out1b = mixed_network.forward(&input1);
    let out2b = mixed_network.forward(&input2);

    // First outputs should be identical (same initial state)
    let diff1 = (&out1a - &out1b).map(|x| x.abs()).sum();
    println!("   Difference in first outputs after reset: {:.8}", diff1);

    // Second outputs may differ (different state history)
    let diff2 = (&out2a - &out2b).map(|x| x.abs()).sum();
    println!("   Difference in second outputs after reset: {:.8}", diff2);

    println!("\n6. Network architecture information...");
    println!("{}", mixed_network.summary());
}
