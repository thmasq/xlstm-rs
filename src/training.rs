use crate::loss::{LossFunction, MSELoss};
use crate::models::XLSTMNetwork;
use crate::optimizers::{Adam, Optimizer, RMSprop, SGD};
use crate::schedulers::{
    ConstantLR, CosineAnnealingLR, ExponentialLR, LearningRateScheduler, StepLR,
};
use ndarray::Array2;
use std::time::Instant;

/// Training configuration parameters
#[derive(Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub print_every: usize,
    pub validation_split: f64,
    pub early_stopping_patience: Option<usize>,
    pub gradient_clip_norm: Option<f64>,
    pub save_best_model: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            epochs: 100,
            learning_rate: 0.001,
            print_every: 10,
            validation_split: 0.2,
            early_stopping_patience: None,
            gradient_clip_norm: Some(1.0),
            save_best_model: true,
        }
    }
}

/// Training metrics for each epoch
#[derive(Clone, Debug)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub epoch_time: f64,
}

/// Early stopping state tracker
#[derive(Clone)]
pub struct EarlyStopper {
    patience: usize,
    min_delta: f64,
    wait: usize,
    best_loss: f64,
    best_weights: Option<XLSTMNetwork>,
    stopped: bool,
}

impl EarlyStopper {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        EarlyStopper {
            patience,
            min_delta,
            wait: 0,
            best_loss: f64::INFINITY,
            best_weights: None,
            stopped: false,
        }
    }

    pub fn check(&mut self, val_loss: f64, model: &XLSTMNetwork) -> bool {
        if val_loss < self.best_loss - self.min_delta {
            self.best_loss = val_loss;
            self.wait = 0;
            self.best_weights = Some(model.clone());
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped = true;
                return true;
            }
        }
        false
    }

    pub fn restore_best_weights(&self, model: &mut XLSTMNetwork) -> Result<(), String> {
        if let Some(ref best_weights) = self.best_weights {
            *model = best_weights.clone();
            Ok(())
        } else {
            Err("No best weights stored".to_string())
        }
    }

    pub fn get_best_loss(&self) -> f64 {
        self.best_loss
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
}

/// Main trainer struct that combines model, loss, optimizer, and scheduler
pub struct XLSTMTrainer<L, O, S>
where
    L: LossFunction,
    O: Optimizer,
    S: LearningRateScheduler,
{
    pub model: XLSTMNetwork,
    pub loss_fn: L,
    pub optimizer: O,
    pub scheduler: Option<S>,
    pub config: TrainingConfig,
    pub metrics_history: Vec<TrainingMetrics>,
    pub early_stopper: Option<EarlyStopper>,
}

impl<L, O, S> XLSTMTrainer<L, O, S>
where
    L: LossFunction,
    O: Optimizer,
    S: LearningRateScheduler,
{
    pub fn new(model: XLSTMNetwork, loss_fn: L, optimizer: O) -> Self {
        XLSTMTrainer {
            model,
            loss_fn,
            optimizer,
            scheduler: None,
            config: TrainingConfig::default(),
            metrics_history: Vec::new(),
            early_stopper: None,
        }
    }

    pub fn with_scheduler(mut self, scheduler: S) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    pub fn with_config(mut self, config: TrainingConfig) -> Self {
        self.optimizer.set_learning_rate(config.learning_rate);

        // Setup early stopping if configured
        if let Some(patience) = config.early_stopping_patience {
            self.early_stopper = Some(EarlyStopper::new(patience, 1e-6));
        }

        self.config = config;
        self
    }

    /// Train the model on a sequence dataset
    pub fn train_on_sequences(
        &mut self,
        sequences: &[Vec<Array2<f64>>],
        targets: &[Vec<Array2<f64>>],
    ) -> Result<Vec<TrainingMetrics>, String> {
        if sequences.len() != targets.len() {
            return Err("Number of sequences and targets must match".to_string());
        }

        // Split data into train/validation if specified
        let split_idx = if self.config.validation_split > 0.0 {
            ((1.0 - self.config.validation_split) * sequences.len() as f64) as usize
        } else {
            sequences.len()
        };

        let train_sequences = &sequences[..split_idx];
        let train_targets = &targets[..split_idx];

        let (val_sequences, val_targets) = if split_idx < sequences.len() {
            (Some(&sequences[split_idx..]), Some(&targets[split_idx..]))
        } else {
            (None, None)
        };

        println!(
            "Training on {} sequences, validating on {} sequences",
            train_sequences.len(),
            val_sequences.map_or(0, |v| v.len())
        );

        for epoch in 0..self.config.epochs {
            let start_time = Instant::now();

            // Training phase
            let train_loss = self.train_epoch(train_sequences, train_targets)?;

            // Validation phase
            let val_loss = if let (Some(val_seqs), Some(val_targs)) = (val_sequences, val_targets) {
                Some(self.validate_epoch(val_seqs, val_targs)?)
            } else {
                None
            };

            // Update learning rate scheduler
            let current_lr = if let Some(ref mut scheduler) = self.scheduler {
                if let Some(vl) = val_loss {
                    scheduler.step_with_val_loss(vl, epoch);
                } else {
                    scheduler.step(epoch);
                }
                let lr = scheduler.get_lr(epoch, self.config.learning_rate);
                self.optimizer.set_learning_rate(lr);
                lr
            } else {
                self.optimizer.get_learning_rate()
            };

            let epoch_time = start_time.elapsed().as_secs_f64();

            let metrics = TrainingMetrics {
                epoch,
                train_loss,
                val_loss,
                learning_rate: current_lr,
                epoch_time,
            };

            self.metrics_history.push(metrics.clone());

            // Print progress
            if epoch % self.config.print_every == 0 || epoch == self.config.epochs - 1 {
                match val_loss {
                    Some(vl) => println!(
                        "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, lr={:.2e}, time={:.2}s",
                        epoch + 1,
                        self.config.epochs,
                        train_loss,
                        vl,
                        current_lr,
                        epoch_time
                    ),
                    None => println!(
                        "Epoch {}/{}: train_loss={:.6}, lr={:.2e}, time={:.2}s",
                        epoch + 1,
                        self.config.epochs,
                        train_loss,
                        current_lr,
                        epoch_time
                    ),
                }
            }

            // Early stopping check
            if let Some(ref mut early_stopper) = self.early_stopper {
                if let Some(vl) = val_loss {
                    if early_stopper.check(vl, &self.model) {
                        println!(
                            "Early stopping triggered at epoch {} with best validation loss: {:.6}",
                            epoch + 1,
                            early_stopper.get_best_loss()
                        );

                        if self.config.save_best_model {
                            early_stopper.restore_best_weights(&mut self.model)?;
                            println!("Restored best model weights");
                        }
                        break;
                    }
                }
            }
        }

        Ok(self.metrics_history.clone())
    }

    /// Train for one epoch
    fn train_epoch(
        &mut self,
        sequences: &[Vec<Array2<f64>>],
        targets: &[Vec<Array2<f64>>],
    ) -> Result<f64, String> {
        let mut total_loss = 0.0;
        let mut total_samples = 0;

        for (seq_idx, (sequence, target_sequence)) in
            sequences.iter().zip(targets.iter()).enumerate()
        {
            if sequence.len() != target_sequence.len() {
                return Err(format!(
                    "Sequence {} length mismatch: input={}, target={}",
                    seq_idx,
                    sequence.len(),
                    target_sequence.len()
                ));
            }

            let loss = self.train_sequence(sequence, target_sequence)?;
            total_loss += loss;
            total_samples += 1;
        }

        Ok(total_loss / total_samples as f64)
    }

    /// Train on a single sequence
    fn train_sequence(
        &mut self,
        sequence: &[Array2<f64>],
        targets: &[Array2<f64>],
    ) -> Result<f64, String> {
        self.model.reset_states();

        let mut total_loss = 0.0;
        let mut accumulated_gradients = std::collections::HashMap::new();

        // Forward pass through sequence
        for (step, (input, target)) in sequence.iter().zip(targets.iter()).enumerate() {
            let output = self.model.forward(input);

            // Compute loss
            let loss = self.loss_fn.compute_loss(&output, target);
            total_loss += loss;

            // Compute gradients (simplified - would need actual backprop implementation)
            let grad_output = self.loss_fn.compute_gradient(&output, target);

            // In a real implementation, you would backpropagate through the network here
            // For now, we'll simulate gradient computation
            self.accumulate_gradients(input, &grad_output, &mut accumulated_gradients)?;
        }

        // Apply accumulated gradients
        self.apply_gradients(&accumulated_gradients)?;

        Ok(total_loss / sequence.len() as f64)
    }

    /// Validate for one epoch
    fn validate_epoch(
        &mut self,
        sequences: &[Vec<Array2<f64>>],
        targets: &[Vec<Array2<f64>>],
    ) -> Result<f64, String> {
        let mut total_loss = 0.0;
        let mut total_samples = 0;

        for (sequence, target_sequence) in sequences.iter().zip(targets.iter()) {
            let loss = self.validate_sequence(sequence, target_sequence)?;
            total_loss += loss;
            total_samples += 1;
        }

        Ok(total_loss / total_samples as f64)
    }

    /// Validate on a single sequence
    fn validate_sequence(
        &mut self,
        sequence: &[Array2<f64>],
        targets: &[Array2<f64>],
    ) -> Result<f64, String> {
        self.model.reset_states();

        let mut total_loss = 0.0;

        // Forward pass through sequence (no gradient computation)
        for (input, target) in sequence.iter().zip(targets.iter()) {
            let output = self.model.forward(input);
            let loss = self.loss_fn.compute_loss(&output, target);
            total_loss += loss;
        }

        Ok(total_loss / sequence.len() as f64)
    }

    /// Predict on a sequence
    pub fn predict(&mut self, sequence: &[Array2<f64>]) -> Vec<Array2<f64>> {
        self.model.reset_states();

        sequence
            .iter()
            .map(|input| self.model.forward(input))
            .collect()
    }

    /// Predict next values in a sequence (autoregressive)
    pub fn predict_next(&mut self, sequence: &[Array2<f64>], n_steps: usize) -> Vec<Array2<f64>> {
        self.model.reset_states();

        // Process input sequence
        let mut last_output = Array2::zeros((0, 0)); // This would need proper initialization
        for input in sequence {
            last_output = self.model.forward(input);
        }

        // Generate predictions
        let mut predictions = Vec::new();
        let mut current_input = last_output;

        for _ in 0..n_steps {
            let output = self.model.forward(&current_input);
            predictions.push(output.clone());
            current_input = output;
        }

        predictions
    }

    /// Get the latest training metrics
    pub fn get_latest_metrics(&self) -> Option<&TrainingMetrics> {
        self.metrics_history.last()
    }

    /// Get all training metrics
    pub fn get_metrics_history(&self) -> &[TrainingMetrics] {
        &self.metrics_history
    }

    /// Reset training state
    pub fn reset(&mut self) {
        self.model.reset_states();
        self.optimizer.reset();
        if let Some(ref mut scheduler) = self.scheduler {
            scheduler.reset();
        }
        self.metrics_history.clear();
        self.early_stopper = None;
    }

    // Helper methods for gradient computation (simplified)
    fn accumulate_gradients(
        &self,
        _input: &Array2<f64>,
        _grad_output: &Array2<f64>,
        _accumulated_gradients: &mut std::collections::HashMap<String, Array2<f64>>,
    ) -> Result<(), String> {
        // In a real implementation, this would compute gradients through backpropagation
        // For now, we'll return Ok as a placeholder
        Ok(())
    }

    fn apply_gradients(
        &mut self,
        _gradients: &std::collections::HashMap<String, Array2<f64>>,
    ) -> Result<(), String> {
        // In a real implementation, this would apply gradients to model parameters
        // For now, we'll return Ok as a placeholder
        Ok(())
    }

    /// Clip gradients by norm to prevent exploding gradients
    fn _clip_gradients(
        &self,
        gradients: &mut std::collections::HashMap<String, Array2<f64>>,
        max_norm: f64,
    ) {
        if let Some(_clip_norm) = self.config.gradient_clip_norm {
            // Compute total norm
            let mut total_norm = 0.0;
            for grad in gradients.values() {
                total_norm += grad.map(|x| x * x).sum();
            }
            total_norm = total_norm.sqrt();

            // Clip if necessary
            if total_norm > max_norm {
                let clip_coef = max_norm / (total_norm + 1e-6);
                for grad in gradients.values_mut() {
                    *grad = grad.map(|x| x * clip_coef);
                }
            }
        }
    }
}

/// Convenience functions to create common trainer configurations

/// Create a basic trainer with SGD optimizer and MSE loss
pub fn create_basic_trainer(
    model: XLSTMNetwork,
    learning_rate: f64,
) -> XLSTMTrainer<MSELoss, SGD, ConstantLR> {
    let loss_fn = MSELoss;
    let optimizer = SGD::new(learning_rate);

    XLSTMTrainer::new(model, loss_fn, optimizer)
}

/// Create an Adam trainer with MSE loss
pub fn create_adam_trainer(
    model: XLSTMNetwork,
    learning_rate: f64,
) -> XLSTMTrainer<MSELoss, Adam, ConstantLR> {
    let loss_fn = MSELoss;
    let optimizer = Adam::new(learning_rate);

    XLSTMTrainer::new(model, loss_fn, optimizer)
}

/// Create a trainer with Adam optimizer and step learning rate schedule
pub fn create_adam_step_trainer(
    model: XLSTMNetwork,
    learning_rate: f64,
    step_size: usize,
    gamma: f64,
) -> XLSTMTrainer<MSELoss, Adam, StepLR> {
    let loss_fn = MSELoss;
    let optimizer = Adam::new(learning_rate);
    let scheduler = StepLR::new(step_size, gamma);

    XLSTMTrainer::new(model, loss_fn, optimizer).with_scheduler(scheduler)
}

/// Create a trainer with RMSprop and cosine annealing
pub fn create_rmsprop_cosine_trainer(
    model: XLSTMNetwork,
    learning_rate: f64,
    t_max: usize,
    eta_min: f64,
) -> XLSTMTrainer<MSELoss, RMSprop, CosineAnnealingLR> {
    let loss_fn = MSELoss;
    let optimizer = RMSprop::new(learning_rate);
    let scheduler = CosineAnnealingLR::new(t_max, eta_min);

    XLSTMTrainer::new(model, loss_fn, optimizer).with_scheduler(scheduler)
}

/// Create a trainer with Adam and exponential decay
pub fn create_adam_exponential_trainer(
    model: XLSTMNetwork,
    learning_rate: f64,
    gamma: f64,
) -> XLSTMTrainer<MSELoss, Adam, ExponentialLR> {
    let loss_fn = MSELoss;
    let optimizer = Adam::new(learning_rate);
    let scheduler = ExponentialLR::new(gamma);

    XLSTMTrainer::new(model, loss_fn, optimizer).with_scheduler(scheduler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_trainer_creation() {
        let model = XLSTMNetwork::from_config("ms", 4, 8, 2);
        let trainer = create_basic_trainer(model, 0.01);

        assert_eq!(trainer.config.learning_rate, 0.01);
        assert_eq!(trainer.config.epochs, 100);
    }

    #[test]
    fn test_early_stopper() {
        let mut early_stopper = EarlyStopper::new(3, 1e-4);
        let model = XLSTMNetwork::from_config("m", 2, 4, 1);

        // Should not stop initially
        assert!(!early_stopper.check(1.0, &model));

        // Should not stop with improvement
        assert!(!early_stopper.check(0.9, &model));
        assert!(!early_stopper.check(0.8, &model));

        // Should stop after patience epochs without improvement
        assert!(!early_stopper.check(0.85, &model)); // No improvement
        assert!(!early_stopper.check(0.85, &model)); // No improvement
        assert!(early_stopper.check(0.85, &model)); // Should stop now
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig {
            epochs: 50,
            learning_rate: 0.001,
            early_stopping_patience: Some(10),
            ..Default::default()
        };

        let model = XLSTMNetwork::from_config("s", 3, 6, 2);
        let trainer = create_adam_trainer(model, config.learning_rate).with_config(config);

        assert_eq!(trainer.config.epochs, 50);
        assert!(trainer.early_stopper.is_some());
    }

    #[test]
    fn test_prediction() {
        let model = XLSTMNetwork::from_config("m", 2, 4, 1);
        let mut trainer = create_basic_trainer(model, 0.01);

        let sequence = vec![arr2(&[[1.0], [0.0]]), arr2(&[[0.0], [1.0]])];

        let predictions = trainer.predict(&sequence);
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].shape(), &[2, 1]);
    }
}
