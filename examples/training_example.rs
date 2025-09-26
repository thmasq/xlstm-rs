use ndarray::{Array2, arr2};
use rand::Rng;
use xlstm_rs::{
    models::XLSTMNetwork,
    training::*,
    loss::*,
    optimizers::*,
    schedulers::*,
};

/// Generate synthetic time series data for training
fn generate_synthetic_data(num_sequences: usize, sequence_length: usize, input_dim: usize) -> (Vec<Vec<Array2<f64>>>, Vec<Vec<Array2<f64>>>) {
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    
    for seq_idx in 0..num_sequences {
        let mut sequence = Vec::new();
        let mut target_sequence = Vec::new();
        
        // Generate a sine wave with some noise
        let frequency = 0.1 + (seq_idx as f64 * 0.05) % 0.3; // Vary frequency per sequence
        let phase = (seq_idx as f64 * 0.1) % (2.0 * std::f64::consts::PI);
        
        for t in 0..sequence_length {
            let time = t as f64 * 0.1;
            
            // Create input with multiple features
            let mut input_data = vec![
                (frequency * time + phase).sin(),
                (frequency * time + phase).cos(),
                (frequency * time + phase + std::f64::consts::PI / 4.0).sin(), // Phase shifted
                time / sequence_length as f64, // Time feature
            ];
            
            // Pad or truncate to desired input dimension
            input_data.resize(input_dim, 0.0);
            
            // Add some noise
            let mut rng = rand::thread_rng();
            for val in input_data.iter_mut() {
                *val += (rng.gen::<f64>() - 0.5) * 0.1;
            }
            
            let input = Array2::from_shape_vec((input_dim, 1), input_data).unwrap();
            
            // Target is the next value in the sine wave (shifted by 1 step)
            let next_time = (t + 1) as f64 * 0.1;
            let target_val = (frequency * next_time + phase).sin();
            let target = arr2(&[[target_val]]);
            
            sequence.push(input);
            target_sequence.push(target);
        }
        
        sequences.push(sequence);
        targets.push(target_sequence);
    }
    
    (sequences, targets)
}

/// Generate test data for evaluation
fn generate_test_sequence(sequence_length: usize, input_dim: usize) -> Vec<Array2<f64>> {
    let mut sequence = Vec::new();
    let frequency = 0.15;
    let phase = 0.5;
    
    for t in 0..sequence_length {
        let time = t as f64 * 0.1;
        
        let mut input_data = vec![
            (frequency * time + phase).sin(),
            (frequency * time + phase).cos(),
            (frequency * time + phase + std::f64::consts::PI / 4.0).sin(),
            time / sequence_length as f64,
        ];
        
        input_data.resize(input_dim, 0.0);
        let input = Array2::from_shape_vec((input_dim, 1), input_data).unwrap();
        sequence.push(input);
    }
    
    sequence
}

fn main() {
    println!("🧠 xLSTM Training Example");
    println!("=========================\n");

    // Configuration
    let input_dim = 4;
    let hidden_dim = 16;
    let depth = 2;
    let num_train_sequences = 100;
    let sequence_length = 50;
    let test_length = 20;

    println!("1. Generating synthetic training data...");
    let (train_sequences, train_targets) = generate_synthetic_data(
        num_train_sequences, 
        sequence_length, 
        input_dim
    );
    
    println!("   Generated {} training sequences of length {}", 
             train_sequences.len(), sequence_length);

    println!("\n2. Creating xLSTM models with different architectures...");

    // Test different xLSTM architectures
    let architectures = vec![
        ("mLSTM only", "mmm"),
        ("sLSTM only", "sss"), 
        ("Mixed (mLSTM-sLSTM)", "msms"),
        ("Deep mixed", "msmsms"),
    ];

    for (name, config) in architectures {
        println!("\n   Testing architecture: {} ({})", name, config);
        
        // Create model
        let model = XLSTMNetwork::from_config(config, input_dim, hidden_dim, depth);
        println!("   Model parameters: {}", model.num_parameters());

        // Test different optimizers and schedulers
        test_optimizer_configurations(&model, &train_sequences, &train_targets, name);
    }

    println!("\n3. Detailed training example with best configuration...");
    detailed_training_example(&train_sequences, &train_targets, input_dim, hidden_dim, depth);

    println!("\n4. Testing prediction capabilities...");
    test_prediction_capabilities(input_dim, hidden_dim, depth, test_length);

    println!("\n✅ Training examples completed successfully!");
}

fn test_optimizer_configurations(
    base_model: &XLSTMNetwork,
    train_sequences: &[Vec<Array2<f64>>], 
    train_targets: &[Vec<Array2<f64>>],
    arch_name: &str
) {
    let configs = vec![
        ("SGD", "sgd"),
        ("Adam", "adam"),
        ("Adam + StepLR", "adam_step"),
        ("RMSprop + Cosine", "rmsprop_cosine"),
    ];

    for (name, config_type) in configs {
        println!("     Testing {}", name);
        
        let mut trainer = match config_type {
            "sgd" => {
                let model = base_model.clone();
                create_basic_trainer(model, 0.01)
            },
            "adam" => {
                let model = base_model.clone();
                create_adam_trainer(model, 0.001)
            },
            "adam_step" => {
                let model = base_model.clone();
                create_adam_step_trainer(model, 0.001, 20, 0.5)
            },
            "rmsprop_cosine" => {
                let model = base_model.clone();
                create_rmsprop_cosine_trainer(model, 0.001, 50, 1e-6)
            },
            _ => panic!("Unknown config type"),
        };

        // Quick training with reduced epochs for comparison
        let config = TrainingConfig {
            epochs: 10,
            print_every: 5,
            validation_split: 0.2,
            early_stopping_patience: None,
            ..Default::default()
        };

        trainer = trainer.with_config(config);
        
        if let Ok(metrics) = trainer.train_on_sequences(train_sequences, train_targets) {
            if let Some(final_metrics) = metrics.last() {
                println!("       Final loss: {:.6}, LR: {:.2e}", 
                         final_metrics.train_loss, final_metrics.learning_rate);
            }
        }
    }
}

fn detailed_training_example(
    train_sequences: &[Vec<Array2<f64>>], 
    train_targets: &[Vec<Array2<f64>>],
    input_dim: usize,
    hidden_dim: usize,
    depth: usize
) {
    println!("   Creating mixed xLSTM network (msms)...");
    let model = XLSTMNetwork::from_config("msms", input_dim, hidden_dim, depth);
    
    println!("   Setting up Adam optimizer with OneCycle scheduler...");
    let loss_fn = MSELoss;
    let optimizer = Adam::new(0.001).with_weight_decay(1e-4);
    let scheduler = OneCycleLR::new(0.01, 100);
    
    let mut trainer = XLSTMTrainer::new(model, loss_fn, optimizer)
        .with_scheduler(scheduler);
    
    let config = TrainingConfig {
        epochs: 50,
        print_every: 10,
        validation_split: 0.2,
        early_stopping_patience: Some(15),
        gradient_clip_norm: Some(1.0),
        save_best_model: true,
        ..Default::default()
    };
    
    trainer = trainer.with_config(config);
    
    println!("   Starting training...");
    match trainer.train_on_sequences(train_sequences, train_targets) {
        Ok(metrics) => {
            println!("   Training completed successfully!");
            
            // Print training summary
            if let (Some(first), Some(last)) = (metrics.first(), metrics.last()) {
                println!("   Initial loss: {:.6} -> Final loss: {:.6}", 
                         first.train_loss, last.train_loss);
                
                if let (Some(first_val), Some(last_val)) = (first.val_loss, last.val_loss) {
                    println!("   Validation: {:.6} -> {:.6}", first_val, last_val);
                }
            }
            
            // Test final model
            let test_sequence = generate_test_sequence(20, input_dim);
            let predictions = trainer.predict(&test_sequence);
            
            println!("   Generated {} predictions", predictions.len());
            
            // Show some prediction samples
            println!("   Sample predictions (first 5 steps):");
            for (i, pred) in predictions.iter().take(5).enumerate() {
                println!("     Step {}: {:.4}", i + 1, pred[[0, 0]]);
            }
        },
        Err(e) => {
            println!("   Training failed: {}", e);
        }
    }
}

fn test_prediction_capabilities(input_dim: usize, hidden_dim: usize, depth: usize, test_length: usize) {
    println!("   Creating a simple model for prediction testing...");
    let model = XLSTMNetwork::from_config("ms", input_dim, hidden_dim, depth);
    let mut trainer = create_adam_trainer(model, 0.001);
    
    // Generate a simple test sequence
    let test_sequence = generate_test_sequence(test_length, input_dim);
    
    println!("   Testing sequence-to-sequence prediction...");
    let predictions = trainer.predict(&test_sequence);
    println!("     Input length: {}, Output length: {}", test_sequence.len(), predictions.len());
    
    println!("   Testing autoregressive prediction...");
    let context_length = 10;
    let pred_length = 5;
    let context = &test_sequence[..context_length];
    let future_predictions = trainer.predict_next(context, pred_length);
    
    println!("     Context length: {}, Predicted {} future steps", 
             context_length, future_predictions.len());
    
    // Compare with actual continuation
    let actual_continuation = &test_sequence[context_length..context_length + pred_length];
    if actual_continuation.len() == future_predictions.len() {
        println!("     Prediction vs Actual (first feature):");
        for (i, (pred, actual)) in future_predictions.iter().zip(actual_continuation.iter()).enumerate() {
            println!("       Step {}: pred={:.4}, actual={:.4}, error={:.4}", 
                     i + 1, pred[[0, 0]], actual[[0, 0]], (pred[[0, 0]] - actual[[0, 0]]).abs());
        }
    }
    
    println!("   Testing different loss functions...");
    test_loss_functions();
}

fn test_loss_functions() {
    let predictions = arr2(&[[0.8, 0.2], [0.6, 0.4]]);
    let targets = arr2(&[[1.0, 0.0], [0.5, 0.5]]);
    
    let loss_functions: Vec<(&str, Box<dyn LossFunction>)> = vec![
        ("MSE", Box::new(MSELoss)),
        ("MAE", Box::new(MAELoss)),
        ("Huber", Box::new(HuberLoss::new(1.0))),
        ("BCE", Box::new(BCELoss)),
    ];
    
    for (name, loss_fn) in loss_functions {
        let loss = loss_fn.compute_loss(&predictions, &targets);
        println!("     {} Loss: {:.6}", name, loss);
    }
}
