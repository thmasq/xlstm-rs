use ndarray::Array1;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;

/// Action distribution trait for different action spaces
pub trait ActionDistribution {
    /// Sample an action from the distribution
    fn sample(&self) -> Array1<f64>;

    /// Compute log probability of an action
    fn log_prob(&self, action: &Array1<f64>) -> f64;

    /// Compute entropy of the distribution
    fn entropy(&self) -> f64;
}

/// Gaussian distribution for continuous action spaces
#[derive(Clone, Debug)]
pub struct GaussianDistribution {
    pub mean: Array1<f64>,
    pub log_std: Array1<f64>,
}

impl GaussianDistribution {
    pub fn new(mean: Array1<f64>, log_std: Array1<f64>) -> Self {
        assert_eq!(
            mean.len(),
            log_std.len(),
            "Mean and log_std must have same length"
        );
        Self { mean, log_std }
    }

    pub fn std(&self) -> Array1<f64> {
        self.log_std.mapv(|x| x.exp())
    }
}

impl ActionDistribution for GaussianDistribution {
    fn sample(&self) -> Array1<f64> {
        let mut rng = rand::rng();
        let std = self.std();

        Array1::from_shape_fn(self.mean.len(), |i| {
            let normal = StandardNormal;
            let noise: f64 = normal.sample(&mut rng);
            self.mean[i] + std[i] * noise
        })
    }

    fn log_prob(&self, action: &Array1<f64>) -> f64 {
        assert_eq!(action.len(), self.mean.len(), "Action dimension mismatch");

        let std = self.std();
        let mut log_prob = 0.0;

        for i in 0..self.mean.len() {
            let diff = action[i] - self.mean[i];
            let var = std[i] * std[i];
            log_prob -= 0.5 * (diff * diff / var + (2.0 * PI * var).ln());
        }

        log_prob
    }

    fn entropy(&self) -> f64 {
        let std = self.std();
        let mut entropy = 0.0;

        for i in 0..std.len() {
            entropy += 0.5 * (2.0 * PI * std[i] * std[i] * std::f64::consts::E).ln();
        }

        entropy
    }
}

/// Categorical distribution for discrete action spaces
#[derive(Clone, Debug)]
pub struct CategoricalDistribution {
    pub logits: Array1<f64>,
}

impl CategoricalDistribution {
    pub fn new(logits: Array1<f64>) -> Self {
        Self { logits }
    }

    pub fn probabilities(&self) -> Array1<f64> {
        let max_logit = self.logits.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits = self.logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        exp_logits / sum_exp
    }
}

impl ActionDistribution for CategoricalDistribution {
    fn sample(&self) -> Array1<f64> {
        let probs = self.probabilities();
        let mut rng = rand::rng();
        let random_val: f64 = rng.random();

        let mut cumulative = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                let mut action = Array1::zeros(1);
                action[0] = i as f64;
                return action;
            }
        }

        // Fallback (should rarely happen)
        let mut action = Array1::zeros(1);
        action[0] = (probs.len() - 1) as f64;
        action
    }

    fn log_prob(&self, action: &Array1<f64>) -> f64 {
        let action_idx = action[0] as usize;
        assert!(action_idx < self.logits.len(), "Invalid action index");

        let probs = self.probabilities();
        probs[action_idx].ln()
    }

    fn entropy(&self) -> f64 {
        let probs = self.probabilities();
        -probs
            .iter()
            .map(|&p| if p > 1e-8 { p * p.ln() } else { 0.0 })
            .sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_gaussian_distribution() {
        let mean = arr1(&[0.0, 1.0]);
        let log_std = arr1(&[-1.0, 0.0]);
        let dist = GaussianDistribution::new(mean, log_std);

        let action = dist.sample();
        assert_eq!(action.len(), 2);

        let log_prob = dist.log_prob(&action);
        assert!(log_prob.is_finite());

        let entropy = dist.entropy();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_categorical_distribution() {
        let logits = arr1(&[1.0, 2.0, 0.5]);
        let dist = CategoricalDistribution::new(logits);

        let probs = dist.probabilities();
        assert!((probs.sum() - 1.0).abs() < 1e-6);

        let action = dist.sample();
        assert_eq!(action.len(), 1);
        assert!(action[0] >= 0.0 && action[0] < 3.0);

        let log_prob = dist.log_prob(&action);
        assert!(log_prob.is_finite());

        let entropy = dist.entropy();
        assert!(entropy > 0.0);
    }
}
