use ndarray::Array2;
use std::collections::HashMap;

/// Optimizer trait for parameter updates during training
pub trait Optimizer {
    /// Update a parameter using its gradient
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>);

    /// Reset optimizer state (clears momentum, running averages, etc.)
    fn reset(&mut self);

    /// Set the learning rate dynamically (for compatibility with schedulers)
    fn set_learning_rate(&mut self, lr: f64);

    /// Get the current learning rate
    fn get_learning_rate(&self) -> f64;

    /// Get optimizer name for logging
    fn name(&self) -> &'static str;
}

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    learning_rate: f64,
    momentum: f64,
    weight_decay: f64,
    dampening: f64,
    nesterov: bool,
    velocity: HashMap<String, Array2<f64>>,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        SGD {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        let mut grad = gradient.clone();

        // Apply weight decay
        if self.weight_decay != 0.0 {
            grad = grad + self.weight_decay * &*param;
        }

        if self.momentum != 0.0 {
            if !self.velocity.contains_key(param_id) {
                self.velocity
                    .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            }

            let velocity = self.velocity.get_mut(param_id).unwrap();
            *velocity = self.momentum * &*velocity + (1.0 - self.dampening) * &grad;

            if self.nesterov {
                grad = &grad + self.momentum * &*velocity;
            } else {
                grad = velocity.clone();
            }
        }

        *param = &*param - self.learning_rate * &grad;
    }

    fn reset(&mut self) {
        self.velocity.clear();
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn name(&self) -> &'static str {
        "SGD"
    }
}

/// Adam optimizer with adaptive learning rates
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    amsgrad: bool,
    t: i32,
    m: HashMap<String, Array2<f64>>,
    v: HashMap<String, Array2<f64>>,
    v_hat_max: HashMap<String, Array2<f64>>, // For AMSGrad
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Adam::with_params(learning_rate, 0.9, 0.999, 1e-8)
    }

    pub fn with_params(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            amsgrad: false,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            v_hat_max: HashMap::new(),
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl Optimizer for Adam {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        self.t += 1;

        let mut grad = gradient.clone();

        // Apply weight decay
        if self.weight_decay != 0.0 {
            grad = grad + self.weight_decay * &*param;
        }

        if !self.m.contains_key(param_id) {
            self.m
                .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            self.v
                .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            if self.amsgrad {
                self.v_hat_max
                    .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            }
        }

        let m_t = self.m.get_mut(param_id).unwrap();
        let v_t = self.v.get_mut(param_id).unwrap();

        // Update biased first moment estimate
        *m_t = self.beta1 * &*m_t + (1.0 - self.beta1) * &grad;

        // Update biased second raw moment estimate
        *v_t = self.beta2 * &*v_t + (1.0 - self.beta2) * (&grad * &grad);

        // Compute bias-corrected first moment estimate
        let m_hat = &*m_t / (1.0 - self.beta1.powi(self.t));

        // Compute bias-corrected second raw moment estimate
        let mut v_hat = &*v_t / (1.0 - self.beta2.powi(self.t));

        // AMSGrad: maintain the maximum of all v_hat until now
        if self.amsgrad {
            let v_hat_max = self.v_hat_max.get_mut(param_id).unwrap();

            // Element-wise maximum
            for (i, j) in ndarray::indices_of(&v_hat) {
                if v_hat[(i, j)] > v_hat_max[(i, j)] {
                    v_hat_max[(i, j)] = v_hat[(i, j)];
                }
            }

            v_hat = v_hat_max.clone();
        }

        let update = self.learning_rate * &m_hat / (&v_hat.map(|x| x.sqrt()) + self.epsilon);
        *param = &*param - &update;
    }

    fn reset(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
        self.v_hat_max.clear();
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn name(&self) -> &'static str {
        "Adam"
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    t: i32,
    m: HashMap<String, Array2<f64>>,
    v: HashMap<String, Array2<f64>>,
}

impl AdamW {
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        AdamW {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn with_params(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> Self {
        AdamW {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        self.t += 1;

        if !self.m.contains_key(param_id) {
            self.m
                .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            self.v
                .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
        }

        let m_t = self.m.get_mut(param_id).unwrap();
        let v_t = self.v.get_mut(param_id).unwrap();

        // Update biased first moment estimate
        *m_t = self.beta1 * &*m_t + (1.0 - self.beta1) * gradient;

        // Update biased second raw moment estimate
        *v_t = self.beta2 * &*v_t + (1.0 - self.beta2) * (gradient * gradient);

        // Compute bias-corrected first moment estimate
        let m_hat = &*m_t / (1.0 - self.beta1.powi(self.t));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &*v_t / (1.0 - self.beta2.powi(self.t));

        // Update parameters with Adam step
        let adam_update = self.learning_rate * &m_hat / (&v_hat.map(|x| x.sqrt()) + self.epsilon);

        // Apply decoupled weight decay
        let weight_decay_update = self.learning_rate * self.weight_decay * &*param;

        *param = &*param - &adam_update - &weight_decay_update;
    }

    fn reset(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn name(&self) -> &'static str {
        "AdamW"
    }
}

/// RMSprop optimizer
pub struct RMSprop {
    learning_rate: f64,
    alpha: f64,
    epsilon: f64,
    weight_decay: f64,
    momentum: f64,
    centered: bool,
    v: HashMap<String, Array2<f64>>,
    momentum_buffer: HashMap<String, Array2<f64>>,
    grad_avg: HashMap<String, Array2<f64>>, // For centered variant
}

impl RMSprop {
    pub fn new(learning_rate: f64) -> Self {
        RMSprop::with_params(learning_rate, 0.99, 1e-8)
    }

    pub fn with_params(learning_rate: f64, alpha: f64, epsilon: f64) -> Self {
        RMSprop {
            learning_rate,
            alpha,
            epsilon,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            v: HashMap::new(),
            momentum_buffer: HashMap::new(),
            grad_avg: HashMap::new(),
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }
}

impl Optimizer for RMSprop {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        let mut grad = gradient.clone();

        // Apply weight decay
        if self.weight_decay != 0.0 {
            grad = grad + self.weight_decay * &*param;
        }

        if !self.v.contains_key(param_id) {
            self.v
                .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            if self.momentum > 0.0 {
                self.momentum_buffer
                    .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            }
            if self.centered {
                self.grad_avg
                    .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            }
        }

        let v_t = self.v.get_mut(param_id).unwrap();

        // Update running average of squared gradients
        *v_t = self.alpha * &*v_t + (1.0 - self.alpha) * (&grad * &grad);

        let mut avg = v_t.clone();

        if self.centered {
            let grad_avg = self.grad_avg.get_mut(param_id).unwrap();
            *grad_avg = self.alpha * &*grad_avg + (1.0 - self.alpha) * &grad;
            avg = &*v_t - (&*grad_avg * &*grad_avg);
        }

        let update = &grad / (&avg.map(|x| x.sqrt()) + self.epsilon);

        if self.momentum > 0.0 {
            let buf = self.momentum_buffer.get_mut(param_id).unwrap();
            *buf = self.momentum * &*buf + &update;
            *param = &*param - self.learning_rate * &*buf;
        } else {
            *param = &*param - self.learning_rate * &update;
        }
    }

    fn reset(&mut self) {
        self.v.clear();
        self.momentum_buffer.clear();
        self.grad_avg.clear();
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn name(&self) -> &'static str {
        "RMSprop"
    }
}

/// Adagrad optimizer with adaptive learning rates
pub struct Adagrad {
    learning_rate: f64,
    epsilon: f64,
    weight_decay: f64,
    initial_accumulator_value: f64,
    sum_squares: HashMap<String, Array2<f64>>,
}

impl Adagrad {
    pub fn new(learning_rate: f64) -> Self {
        Adagrad {
            learning_rate,
            epsilon: 1e-10,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            sum_squares: HashMap::new(),
        }
    }

    pub fn with_params(learning_rate: f64, epsilon: f64, weight_decay: f64) -> Self {
        Adagrad {
            learning_rate,
            epsilon,
            weight_decay,
            initial_accumulator_value: 0.0,
            sum_squares: HashMap::new(),
        }
    }

    pub fn with_initial_accumulator_value(mut self, value: f64) -> Self {
        self.initial_accumulator_value = value;
        self
    }
}

impl Optimizer for Adagrad {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        let mut grad = gradient.clone();

        // Apply weight decay
        if self.weight_decay != 0.0 {
            grad = grad + self.weight_decay * &*param;
        }

        if !self.sum_squares.contains_key(param_id) {
            let mut initial = Array2::zeros(param.raw_dim());
            if self.initial_accumulator_value != 0.0 {
                initial.fill(self.initial_accumulator_value);
            }
            self.sum_squares.insert(param_id.to_string(), initial);
        }

        let sum_sq = self.sum_squares.get_mut(param_id).unwrap();

        // Accumulate squared gradients
        *sum_sq = &*sum_sq + (&grad * &grad);

        // Compute update
        let update = self.learning_rate * &grad / (&sum_sq.map(|x| x.sqrt()) + self.epsilon);
        *param = &*param - &update;
    }

    fn reset(&mut self) {
        self.sum_squares.clear();
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn name(&self) -> &'static str {
        "Adagrad"
    }
}

/// PPO (Proximal Policy Optimization) optimizer for reinforcement learning
///
/// PPO is designed for actor-critic methods where you have:
/// - An actor network (policy) that outputs action probabilities
/// - A critic network (value function) that estimates state values
/// - A clipped objective function to prevent large policy updates
pub struct PPO {
    learning_rate: f64,
    epsilon_clip: f64,
    value_loss_coeff: f64,
    entropy_coeff: f64,
    max_grad_norm: f64,
    gamma: f64,      // Discount factor
    gae_lambda: f64, // GAE lambda for advantage estimation

    // Internal state for Adam-like optimization
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: i32,

    // Per-parameter state
    m: HashMap<String, Array2<f64>>, // First moment estimates
    v: HashMap<String, Array2<f64>>, // Second moment estimates

    // Experience buffer for PPO updates
    pub states: Vec<Array2<f64>>,
    pub actions: Vec<Array2<f64>>,
    pub rewards: Vec<f64>,
    pub values: Vec<f64>,
    pub log_probs: Vec<f64>,
    pub done_flags: Vec<bool>,
}

impl PPO {
    pub fn new(learning_rate: f64) -> Self {
        PPO {
            learning_rate,
            epsilon_clip: 0.2,
            value_loss_coeff: 0.5,
            entropy_coeff: 0.01,
            max_grad_norm: 0.5,
            gamma: 0.99,
            gae_lambda: 0.95,

            // Adam parameters
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,

            m: HashMap::new(),
            v: HashMap::new(),

            // Experience buffer
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            log_probs: Vec::new(),
            done_flags: Vec::new(),
        }
    }

    pub fn with_params(
        learning_rate: f64,
        epsilon_clip: f64,
        value_loss_coeff: f64,
        entropy_coeff: f64,
        max_grad_norm: f64,
        gamma: f64,
        gae_lambda: f64,
    ) -> Self {
        PPO {
            learning_rate,
            epsilon_clip,
            value_loss_coeff,
            entropy_coeff,
            max_grad_norm,
            gamma,
            gae_lambda,

            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,

            m: HashMap::new(),
            v: HashMap::new(),

            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            log_probs: Vec::new(),
            done_flags: Vec::new(),
        }
    }

    /// Store experience for later PPO update
    pub fn store_experience(
        &mut self,
        state: Array2<f64>,
        action: Array2<f64>,
        reward: f64,
        value: f64,
        log_prob: f64,
        done: bool,
    ) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.values.push(value);
        self.log_probs.push(log_prob);
        self.done_flags.push(done);
    }

    /// Compute Generalized Advantage Estimation (GAE)
    pub fn compute_advantages(&self, next_value: f64) -> (Vec<f64>, Vec<f64>) {
        let mut advantages = vec![0.0; self.rewards.len()];
        let mut returns = vec![0.0; self.rewards.len()];

        let mut gae = 0.0;
        let mut next_val = next_value;

        // Compute advantages and returns backwards through the episode
        for i in (0..self.rewards.len()).rev() {
            let mask = if self.done_flags[i] { 0.0 } else { 1.0 };
            let delta = self.rewards[i] + self.gamma * next_val * mask - self.values[i];
            gae = delta + self.gamma * self.gae_lambda * mask * gae;
            advantages[i] = gae;
            returns[i] = gae + self.values[i];
            next_val = self.values[i];
        }

        (advantages, returns)
    }

    /// Compute PPO loss components
    pub fn compute_ppo_loss(
        &self,
        new_log_probs: &[f64],
        old_log_probs: &[f64],
        advantages: &[f64],
        values: &[f64],
        returns: &[f64],
        entropy: f64,
    ) -> (f64, f64, f64) {
        let mut policy_loss = 0.0;
        let mut value_loss = 0.0;

        for i in 0..new_log_probs.len() {
            // PPO clipped objective
            let ratio = (new_log_probs[i] - old_log_probs[i]).exp();
            let surr1 = ratio * advantages[i];
            let surr2 =
                ratio.clamp(1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages[i];
            policy_loss -= surr1.min(surr2);

            // Value function loss (MSE)
            let value_error = returns[i] - values[i];
            value_loss += 0.5 * value_error * value_error;
        }

        let n = new_log_probs.len() as f64;
        policy_loss /= n;
        value_loss /= n;

        // Total loss
        let total_loss =
            policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy;

        (total_loss, policy_loss, value_loss)
    }

    /// Clear the experience buffer
    pub fn clear_buffer(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.values.clear();
        self.log_probs.clear();
        self.done_flags.clear();
    }

    /// Get the current buffer size
    pub fn buffer_size(&self) -> usize {
        self.states.len()
    }

    /// Apply Adam-style parameter update with PPO-specific considerations
    fn apply_adam_update(
        &mut self,
        param_id: &str,
        param: &mut Array2<f64>,
        gradient: &Array2<f64>,
    ) {
        self.t += 1;

        if !self.m.contains_key(param_id) {
            self.m
                .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            self.v
                .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
        }

        let m_t = self.m.get_mut(param_id).unwrap();
        let v_t = self.v.get_mut(param_id).unwrap();

        // Update biased first moment estimate
        *m_t = self.beta1 * &*m_t + (1.0 - self.beta1) * gradient;

        // Update biased second raw moment estimate
        *v_t = self.beta2 * &*v_t + (1.0 - self.beta2) * (gradient * gradient);

        // Compute bias-corrected first moment estimate
        let m_hat = &*m_t / (1.0 - self.beta1.powi(self.t));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &*v_t / (1.0 - self.beta2.powi(self.t));

        // Apply update with gradient clipping
        let update = self.learning_rate * &m_hat / (&v_hat.map(|x| x.sqrt()) + self.epsilon);

        // Gradient clipping
        let update_norm = update.map(|x| x * x).sum().sqrt();
        let clipped_update = if update_norm > self.max_grad_norm {
            update * (self.max_grad_norm / update_norm)
        } else {
            update
        };

        *param = &*param - &clipped_update;
    }
}

impl Optimizer for PPO {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        self.apply_adam_update(param_id, param, gradient);
    }

    fn reset(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
        self.clear_buffer();
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn name(&self) -> &'static str {
        "PPO"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sgd_optimizer() {
        let mut optimizer = SGD::new(0.1);
        let mut param = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let gradient = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        let original_param = param.clone();
        optimizer.update("test_param", &mut param, &gradient);

        let expected = &original_param - 0.1 * &gradient;
        assert!((param - expected).map(|x| x.abs()).sum() < 1e-10);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut optimizer = SGD::new(0.1).with_momentum(0.9);
        let mut param = arr2(&[[1.0], [1.0]]);
        let gradient = arr2(&[[1.0], [1.0]]);

        // First update
        let param_before = param.clone();
        optimizer.update("test", &mut param, &gradient);
        let update1 = &param_before - &param;

        // Second update with same gradient
        let param_before = param.clone();
        optimizer.update("test", &mut param, &gradient);
        let update2 = &param_before - &param;

        // Second update should be larger due to momentum
        assert!(update2.sum() > update1.sum());
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = Adam::new(0.001);
        let mut param = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let gradient = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        let original_param = param.clone();
        optimizer.update("test_param", &mut param, &gradient);

        // Parameter should change
        assert!((param - original_param).map(|x| x.abs()).sum() > 1e-10);

        // Learning rate should be accessible
        assert!((optimizer.get_learning_rate() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_adamw_optimizer() {
        let mut optimizer = AdamW::new(0.001, 0.01);
        let mut param = arr2(&[[1.0], [1.0]]);
        let gradient = arr2(&[[0.1], [0.1]]);

        optimizer.update("test", &mut param, &gradient);

        // Should apply both Adam update and weight decay
        assert!(param.sum() < 2.0); // Original sum was 2.0
        assert_eq!(optimizer.name(), "AdamW");
    }

    #[test]
    fn test_rmsprop_optimizer() {
        let mut optimizer = RMSprop::new(0.01);
        let mut param = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let gradient = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        let original_param = param.clone();
        optimizer.update("test_param", &mut param, &gradient);

        // Parameter should change
        assert!((param - original_param).map(|x| x.abs()).sum() > 1e-10);
    }

    #[test]
    fn test_adagrad_optimizer() {
        let mut optimizer = Adagrad::new(0.1);
        let mut param = arr2(&[[1.0], [1.0]]);
        let gradient = arr2(&[[1.0], [1.0]]);

        // First update
        let param_before = param.clone();
        optimizer.update("test", &mut param, &gradient);
        let update1 = (&param_before - &param).sum();

        // Second update with same gradient
        let param_before = param.clone();
        optimizer.update("test", &mut param, &gradient);
        let update2 = (&param_before - &param).sum();

        // Second update should be smaller due to accumulated gradients
        assert!(update2 < update1);
    }

    #[test]
    fn test_optimizer_reset() {
        let mut optimizer = Adam::new(0.001);
        let mut param = arr2(&[[1.0], [1.0]]);
        let gradient = arr2(&[[0.1], [0.1]]);

        // Make an update to initialize internal state
        optimizer.update("test", &mut param, &gradient);
        assert!(optimizer.t > 0);
        assert!(!optimizer.m.is_empty());

        // Reset should clear state
        optimizer.reset();
        assert_eq!(optimizer.t, 0);
        assert!(optimizer.m.is_empty());
    }
}
