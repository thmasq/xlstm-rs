use ndarray::{Array1, Array2};

/// Loss function trait for training neural networks
pub trait LossFunction {
    /// Compute the loss between predictions and targets
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64;

    /// Compute the gradient of the loss with respect to predictions
    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64>;

    /// Compute batch loss for multiple predictions and targets
    /// Default implementation computes average loss across batch
    fn compute_batch_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let batch_size = predictions.ncols();
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let pred_col = predictions
                .column(i)
                .to_owned()
                .insert_axis(ndarray::Axis(1));
            let target_col = targets.column(i).to_owned().insert_axis(ndarray::Axis(1));
            total_loss += self.compute_loss(&pred_col, &target_col);
        }

        total_loss / batch_size as f64
    }

    /// Compute batch gradients for multiple predictions and targets
    /// Default implementation computes gradients for each sample and concatenates
    fn compute_batch_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        let batch_size = predictions.ncols();
        let mut batch_gradients = Array2::zeros(predictions.raw_dim());

        for i in 0..batch_size {
            let pred_col = predictions
                .column(i)
                .to_owned()
                .insert_axis(ndarray::Axis(1));
            let target_col = targets.column(i).to_owned().insert_axis(ndarray::Axis(1));
            let grad = self.compute_gradient(&pred_col, &target_col);
            batch_gradients.column_mut(i).assign(&grad.column(0));
        }

        batch_gradients
    }
}

/// Mean Squared Error loss function
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        squared_diff.sum() / (predictions.len() as f64)
    }

    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let diff = predictions - targets;
        2.0 * diff / (predictions.len() as f64)
    }

    fn compute_batch_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        squared_diff.sum() / (predictions.len() as f64)
    }

    fn compute_batch_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        let diff = predictions - targets;
        2.0 * diff / (predictions.len() as f64)
    }
}

/// Mean Absolute Error loss function
pub struct MAELoss;

impl LossFunction for MAELoss {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        diff.map(|x| x.abs()).sum() / (predictions.len() as f64)
    }

    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let diff = predictions - targets;
        diff.map(|x| {
            if *x > 0.0 {
                1.0
            } else if *x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }) / (predictions.len() as f64)
    }

    fn compute_batch_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        diff.map(|x| x.abs()).sum() / (predictions.len() as f64)
    }

    fn compute_batch_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        let diff = predictions - targets;
        diff.map(|x| {
            if *x > 0.0 {
                1.0
            } else if *x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }) / (predictions.len() as f64)
    }
}

/// Cross-Entropy Loss with softmax
pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let softmax_preds = softmax(predictions);
        let epsilon = 1e-15;
        let log_preds = softmax_preds.map(|x| (x + epsilon).ln());
        -(targets * log_preds).sum() / (predictions.shape()[1] as f64)
    }

    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let softmax_preds = softmax(predictions);
        (softmax_preds - targets) / (predictions.shape()[1] as f64)
    }
}

/// Huber Loss (smooth combination of L1 and L2 loss)
pub struct HuberLoss {
    pub delta: f64,
}

impl HuberLoss {
    pub fn new(delta: f64) -> Self {
        HuberLoss { delta }
    }
}

impl LossFunction for HuberLoss {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        let abs_diff = diff.map(|x| x.abs());

        let loss = abs_diff.map(|&x| {
            if x <= self.delta {
                0.5 * x * x
            } else {
                self.delta * (x - 0.5 * self.delta)
            }
        });

        loss.sum() / (predictions.len() as f64)
    }

    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let diff = predictions - targets;

        let grad = diff.map(|&x| {
            if x.abs() <= self.delta {
                x
            } else {
                self.delta * x.signum()
            }
        });

        grad / (predictions.len() as f64)
    }
}

/// Binary Cross-Entropy Loss
pub struct BCELoss;

impl LossFunction for BCELoss {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let epsilon = 1e-15;
        let clipped_preds = predictions.map(|&x| x.max(epsilon).min(1.0 - epsilon));

        let loss = targets
            .iter()
            .zip(clipped_preds.iter())
            .map(|(&t, &p)| -t * p.ln() - (1.0 - t) * (1.0 - p).ln())
            .sum::<f64>();

        loss / (predictions.len() as f64)
    }

    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let epsilon = 1e-15;
        let clipped_preds = predictions.map(|&x| x.max(epsilon).min(1.0 - epsilon));

        let grad = targets
            .iter()
            .zip(clipped_preds.iter())
            .map(|(&t, &p)| -t / p + (1.0 - t) / (1.0 - p))
            .collect::<Vec<f64>>();

        Array2::from_shape_vec(predictions.raw_dim(), grad).unwrap() / (predictions.len() as f64)
    }
}

/// Numerically stable softmax function
pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(x.raw_dim());

    for (i, col) in x.axis_iter(ndarray::Axis(1)).enumerate() {
        let max_val = col.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let exp_vals: Array1<f64> = col.map(|&val| (val - max_val).exp());
        let sum_exp = exp_vals.sum();

        for (j, &exp_val) in exp_vals.iter().enumerate() {
            result[[j, i]] = exp_val / sum_exp;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss;
        let predictions = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let targets = arr2(&[[1.5, 2.5], [2.5, 3.5]]);

        let loss = loss_fn.compute_loss(&predictions, &targets);
        assert!((loss - 0.25).abs() < 1e-6);

        let gradient = loss_fn.compute_gradient(&predictions, &targets);
        assert_eq!(gradient.shape(), predictions.shape());
    }

    #[test]
    fn test_mae_loss() {
        let loss_fn = MAELoss;
        let predictions = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let targets = arr2(&[[1.5, 2.5], [2.5, 3.5]]);

        let loss = loss_fn.compute_loss(&predictions, &targets);
        assert!((loss - 0.5).abs() < 1e-6);

        let gradient = loss_fn.compute_gradient(&predictions, &targets);
        assert_eq!(gradient.shape(), predictions.shape());
    }

    #[test]
    fn test_huber_loss() {
        let loss_fn = HuberLoss::new(1.0);
        let predictions = arr2(&[[1.0], [3.0]]);
        let targets = arr2(&[[0.0], [0.0]]);

        let loss = loss_fn.compute_loss(&predictions, &targets);
        assert!(loss > 0.0);

        let gradient = loss_fn.compute_gradient(&predictions, &targets);
        assert_eq!(gradient.shape(), predictions.shape());
    }

    #[test]
    fn test_bce_loss() {
        let loss_fn = BCELoss;
        let predictions = arr2(&[[0.8], [0.2]]);
        let targets = arr2(&[[1.0], [0.0]]);

        let loss = loss_fn.compute_loss(&predictions, &targets);
        assert!(loss > 0.0);

        let gradient = loss_fn.compute_gradient(&predictions, &targets);
        assert_eq!(gradient.shape(), predictions.shape());
    }

    #[test]
    fn test_softmax() {
        let input = arr2(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
        let output = softmax(&input);

        // Each column should sum to 1
        for col in output.axis_iter(ndarray::Axis(1)) {
            let sum: f64 = col.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
