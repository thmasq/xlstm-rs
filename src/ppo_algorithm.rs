use crate::action_distributions::{
    ActionDistribution, CategoricalDistribution, GaussianDistribution,
};
use crate::actor_critic_networks::ActorCriticNetwork;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;

/// PPO hyperparameters
#[derive(Clone, Debug)]
pub struct PPOConfig {
    /// Learning rate for actor
    pub actor_lr: f64,
    /// Learning rate for critic
    pub critic_lr: f64,
    /// PPO clipping parameter
    pub clip_epsilon: f64,
    /// GAE lambda parameter
    pub gae_lambda: f64,
    /// Discount factor gamma
    pub gamma: f64,
    /// Entropy coefficient
    pub entropy_coef: f64,
    /// Value loss coefficient
    pub value_coef: f64,
    /// Number of PPO epochs per update
    pub ppo_epochs: usize,
    /// Mini-batch size
    pub batch_size: usize,
    /// Gradient clipping threshold
    pub max_grad_norm: f64,
    /// Advantage normalization
    pub normalize_advantages: bool,
}

impl Default for PPOConfig {
    fn default() -> Self {
        PPOConfig {
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            clip_epsilon: 0.2,
            gae_lambda: 0.95,
            gamma: 0.99,
            entropy_coef: 0.01,
            value_coef: 1.0,
            ppo_epochs: 4,
            batch_size: 64,
            max_grad_norm: 0.5,
            normalize_advantages: true,
        }
    }
}

/// Trajectory data for training
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub observations: Vec<Array2<f64>>,
    pub actions: Vec<Array1<f64>>,
    pub rewards: Vec<f64>,
    pub values: Vec<f64>,
    pub log_probs: Vec<f64>,
    pub dones: Vec<bool>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            log_probs: Vec::new(),
            dones: Vec::new(),
        }
    }

    pub fn push(
        &mut self,
        obs: Array2<f64>,
        action: Array1<f64>,
        reward: f64,
        value: f64,
        log_prob: f64,
        done: bool,
    ) {
        self.observations.push(obs);
        self.actions.push(action);
        self.rewards.push(reward);
        self.values.push(value);
        self.log_probs.push(log_prob);
        self.dones.push(done);
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn clear(&mut self) {
        self.observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.values.clear();
        self.log_probs.clear();
        self.dones.clear();
    }
}

/// Compute Generalized Advantage Estimation (GAE)
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    gamma: f64,
    gae_lambda: f64,
    next_value: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut returns = vec![0.0; n];

    let mut last_advantage = 0.0;

    // Compute advantages backwards
    for i in (0..n).rev() {
        let next_val = if i == n - 1 {
            if dones[i] { 0.0 } else { next_value }
        } else {
            if dones[i] { 0.0 } else { values[i + 1] }
        };

        let delta = rewards[i] + gamma * next_val - values[i];
        let next_advantage = if i == n - 1 {
            if dones[i] { 0.0 } else { last_advantage }
        } else {
            if dones[i] { 0.0 } else { advantages[i + 1] }
        };

        advantages[i] = delta + gamma * gae_lambda * next_advantage;
        last_advantage = advantages[i];
    }

    // Compute returns
    for i in 0..n {
        returns[i] = advantages[i] + values[i];
    }

    (advantages, returns)
}

/// PPO agent implementation
pub struct PPOAgent {
    pub network: ActorCriticNetwork,
    pub config: PPOConfig,
    pub is_discrete: bool,
    pub trajectory: Trajectory,
}

impl PPOAgent {
    /// Create new PPO agent
    pub fn new(
        xlstm_config: &str,
        input_size: usize,
        hidden_size: usize,
        depth: usize,
        action_dim: usize,
        is_discrete: bool,
        config: PPOConfig,
    ) -> Self {
        let network = ActorCriticNetwork::new(
            xlstm_config,
            input_size,
            hidden_size,
            depth,
            action_dim,
            is_discrete,
        );

        Self {
            network,
            config,
            is_discrete,
            trajectory: Trajectory::new(),
        }
    }

    /// Select action given observation
    pub fn select_action(&mut self, observation: &Array2<f64>) -> (Array1<f64>, f64, f64) {
        let (action_params, log_std, value) = self.network.forward(observation);
        let value_scalar = value[[0, 0]];

        let (action, log_prob) = if self.is_discrete {
            // Discrete action space
            let logits = action_params.column(0).to_owned();
            let dist = CategoricalDistribution::new(logits);
            let action = dist.sample();
            let log_prob = dist.log_prob(&action);
            (action, log_prob)
        } else {
            // Continuous action space
            let mean = action_params.column(0).to_owned();
            let log_std = log_std.expect("Log std should be available for continuous actions");
            let dist = GaussianDistribution::new(mean, log_std);
            let action = dist.sample();
            let log_prob = dist.log_prob(&action);
            (action, log_prob)
        };

        (action, log_prob, value_scalar)
    }

    /// Store trajectory step
    pub fn store_step(
        &mut self,
        obs: Array2<f64>,
        action: Array1<f64>,
        reward: f64,
        value: f64,
        log_prob: f64,
        done: bool,
    ) {
        self.trajectory
            .push(obs, action, reward, value, log_prob, done);
    }

    /// Compute PPO policy loss
    fn compute_policy_loss(
        &self,
        observations: &[Array2<f64>],
        actions: &[Array1<f64>],
        old_log_probs: &[f64],
        advantages: &[f64],
    ) -> f64 {
        let mut total_loss = 0.0;
        let mut total_entropy = 0.0;

        for (i, obs) in observations.iter().enumerate() {
            let (action_params, log_std, _) = {
                let mut net_clone = self.network.clone();
                net_clone.forward(obs)
            };

            let (new_log_prob, entropy) = if self.is_discrete {
                let logits = action_params.column(0).to_owned();
                let dist = CategoricalDistribution::new(logits);
                let log_prob = dist.log_prob(&actions[i]);
                let entropy = dist.entropy();
                (log_prob, entropy)
            } else {
                let mean = action_params.column(0).to_owned();
                let log_std = log_std.expect("Log std required for continuous actions");
                let dist = GaussianDistribution::new(mean, log_std);
                let log_prob = dist.log_prob(&actions[i]);
                let entropy = dist.entropy();
                (log_prob, entropy)
            };

            // Compute ratio and clipped ratio
            let ratio = (new_log_prob - old_log_probs[i]).exp();
            let clipped_ratio = ratio
                .max(1.0 - self.config.clip_epsilon)
                .min(1.0 + self.config.clip_epsilon);

            // PPO clipped objective
            let surr1 = ratio * advantages[i];
            let surr2 = clipped_ratio * advantages[i];
            let policy_loss = -surr1.min(surr2);

            total_loss += policy_loss;
            total_entropy += entropy;
        }

        let n = observations.len() as f64;
        total_loss / n - self.config.entropy_coef * total_entropy / n
    }

    /// Compute value loss
    fn compute_value_loss(&self, observations: &[Array2<f64>], returns: &[f64]) -> f64 {
        let mut total_loss = 0.0;

        for (i, obs) in observations.iter().enumerate() {
            let (_, _, value) = {
                let mut net_clone = self.network.clone();
                net_clone.forward(obs)
            };
            let predicted_value = value[[0, 0]];
            let mse_loss = (predicted_value - returns[i]).powi(2);
            total_loss += mse_loss;
        }

        total_loss / observations.len() as f64
    }

    /// Update networks using PPO
    pub fn update(&mut self, next_value: f64) -> (f64, f64) {
        if self.trajectory.len() == 0 {
            return (0.0, 0.0);
        }

        // Compute advantages and returns using GAE
        let (mut advantages, returns) = compute_gae(
            &self.trajectory.rewards,
            &self.trajectory.values,
            &self.trajectory.dones,
            self.config.gamma,
            self.config.gae_lambda,
            next_value,
        );

        // Normalize advantages if required
        if self.config.normalize_advantages && advantages.len() > 1 {
            let mean = advantages.iter().sum::<f64>() / advantages.len() as f64;
            let var = advantages.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / advantages.len() as f64;
            let std = (var + 1e-8).sqrt();

            for adv in &mut advantages {
                *adv = (*adv - mean) / std;
            }
        }

        let mut avg_policy_loss = 0.0;
        let mut avg_value_loss = 0.0;

        // PPO epochs
        for epoch in 0..self.config.ppo_epochs {
            // Create indices for mini-batching
            let mut indices: Vec<usize> = (0..self.trajectory.len()).collect();
            indices.shuffle(&mut rand::rng());

            // Mini-batch training
            for chunk in indices.chunks(self.config.batch_size) {
                if chunk.len() < self.config.batch_size {
                    continue; // Skip incomplete batches
                }

                // Extract mini-batch data
                let batch_obs: Vec<Array2<f64>> = chunk
                    .iter()
                    .map(|&i| self.trajectory.observations[i].clone())
                    .collect();
                let batch_actions: Vec<Array1<f64>> = chunk
                    .iter()
                    .map(|&i| self.trajectory.actions[i].clone())
                    .collect();
                let batch_old_log_probs: Vec<f64> = chunk
                    .iter()
                    .map(|&i| self.trajectory.log_probs[i])
                    .collect();
                let batch_advantages: Vec<f64> = chunk.iter().map(|&i| advantages[i]).collect();
                let batch_returns: Vec<f64> = chunk.iter().map(|&i| returns[i]).collect();

                // Compute losses
                let policy_loss = self.compute_policy_loss(
                    &batch_obs,
                    &batch_actions,
                    &batch_old_log_probs,
                    &batch_advantages,
                );
                let value_loss = self.compute_value_loss(&batch_obs, &batch_returns);

                avg_policy_loss += policy_loss;
                avg_value_loss += value_loss;

                // Here we would normally compute gradients and update parameters
                // For this implementation, we'll simulate the update
                // In a real implementation, you'd use automatic differentiation
                self.simulate_parameter_update(policy_loss, value_loss);
            }
        }

        let num_updates = self.config.ppo_epochs * (self.trajectory.len() / self.config.batch_size);

        if num_updates > 0 {
            avg_policy_loss /= num_updates as f64;
            avg_value_loss /= num_updates as f64;
        }

        // Clear trajectory for next rollout
        self.trajectory.clear();

        (avg_policy_loss, avg_value_loss)
    }

    /// Simulate parameter update (placeholder for actual gradient-based update)
    fn simulate_parameter_update(&mut self, _policy_loss: f64, _value_loss: f64) {
        // In a real implementation, this would:
        // 1. Compute gradients of the loss w.r.t. network parameters
        // 2. Apply gradient clipping if configured
        // 3. Update parameters using optimizer (Adam, etc.)
        // 4. Update learning rate if scheduled

        // For now, this is a placeholder that would contain the actual
        // gradient computation and parameter update logic

        // Example of what the structure might look like:
        // let gradients = compute_gradients(policy_loss, value_loss, &self.network);
        // let clipped_gradients = clip_gradients(gradients, self.config.max_grad_norm);
        // self.optimizer.step(clipped_gradients);
    }

    /// Reset network states
    pub fn reset_states(&mut self) {
        self.network.reset_states();
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.network.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_ppo_config() {
        let config = PPOConfig::default();
        assert!(config.actor_lr > 0.0);
        assert!(config.clip_epsilon > 0.0);
        assert!(config.gamma > 0.0 && config.gamma <= 1.0);
    }

    #[test]
    fn test_trajectory() {
        let mut traj = Trajectory::new();
        assert_eq!(traj.len(), 0);

        let obs = arr2(&[[1.0], [0.5]]);
        let action = arr1(&[0.0]);
        traj.push(obs, action, 1.0, 0.5, -0.1, false);

        assert_eq!(traj.len(), 1);

        traj.clear();
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_compute_gae() {
        let rewards = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 1.0, 1.5];
        let dones = vec![false, false, true];
        let gamma = 0.99;
        let gae_lambda = 0.95;
        let next_value = 0.0;

        let (advantages, returns) =
            compute_gae(&rewards, &values, &dones, gamma, gae_lambda, next_value);

        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);

        // Returns should be advantages + values
        for i in 0..3 {
            assert!((returns[i] - (advantages[i] + values[i])).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ppo_agent_creation() {
        let config = PPOConfig::default();
        let agent = PPOAgent::new("ms", 4, 8, 2, 2, false, config);

        assert!(!agent.is_discrete);
        assert_eq!(agent.trajectory.len(), 0);
        assert!(agent.num_parameters() > 0);
    }

    #[test]
    fn test_action_selection_continuous() {
        let config = PPOConfig::default();
        let mut agent = PPOAgent::new("s", 3, 6, 1, 2, false, config);

        let obs = arr2(&[[1.0], [0.5], [-0.3]]);
        let (action, log_prob, value) = agent.select_action(&obs);

        assert_eq!(action.len(), 2);
        assert!(log_prob.is_finite());
        assert!(value.is_finite());

        // Actions should be in [-1, 1] due to tanh activation
        for &a in action.iter() {
            assert!(a >= -1.0 && a <= 1.0);
        }
    }

    #[test]
    fn test_action_selection_discrete() {
        let config = PPOConfig::default();
        let mut agent = PPOAgent::new("m", 4, 8, 2, 3, true, config);

        let obs = arr2(&[[1.0], [0.0], [0.5], [-0.2]]);
        let (action, log_prob, value) = agent.select_action(&obs);

        assert_eq!(action.len(), 1);
        assert!(action[0] >= 0.0 && action[0] < 3.0);
        assert!(log_prob.is_finite());
        assert!(value.is_finite());
    }

    #[test]
    fn test_store_and_update() {
        let config = PPOConfig::default();
        let mut agent = PPOAgent::new("s", 2, 4, 1, 1, false, config);

        // Store some trajectory data
        for i in 0..10 {
            let obs = arr2(&[[i as f64], [(i + 1) as f64]]);
            let action = arr1(&[0.5]);
            agent.store_step(obs, action, 1.0, 0.5, -0.1, i == 9);
        }

        assert_eq!(agent.trajectory.len(), 10);

        let (policy_loss, value_loss) = agent.update(0.0);
        assert!(policy_loss.is_finite());
        assert!(value_loss.is_finite());
        assert_eq!(agent.trajectory.len(), 0); // Should be cleared after update
    }
}
