use crate::layers::BlockDiagonal;
use crate::models::XLSTMNetwork;
use ndarray::{Array1, Array2};

/// Actor network using xLSTM for policy learning
#[derive(Clone)]
pub struct ActorNetwork {
    /// xLSTM feature extractor
    pub xlstm: XLSTMNetwork,
    /// Output layer for action means (continuous) or logits (discrete)
    pub action_head: BlockDiagonal,
    /// Log standard deviation for continuous actions (learnable parameter)
    pub log_std: Option<Array1<f64>>,
    /// Action space configuration
    pub action_dim: usize,
    pub is_discrete: bool,
}

impl ActorNetwork {
    /// Create new actor network
    ///
    /// # Arguments
    /// * `xlstm_config` - xLSTM configuration string (e.g., "msm")
    /// * `input_size` - Size of observation features
    /// * `hidden_size` - Hidden size for xLSTM
    /// * `depth` - Depth parameter for xLSTM blocks
    /// * `action_dim` - Dimension of action space
    /// * `is_discrete` - Whether action space is discrete
    pub fn new(
        xlstm_config: &str,
        input_size: usize,
        hidden_size: usize,
        depth: usize,
        action_dim: usize,
        is_discrete: bool,
    ) -> Self {
        let xlstm = XLSTMNetwork::from_config(xlstm_config, input_size, hidden_size, depth);
        let action_head = BlockDiagonal::new(input_size, action_dim, 1, true);

        // For continuous actions, initialize learnable log std
        let log_std = if !is_discrete {
            Some(Array1::zeros(action_dim))
        } else {
            None
        };

        Self {
            xlstm,
            action_head,
            log_std,
            action_dim,
            is_discrete,
        }
    }

    /// Forward pass through actor network
    ///
    /// # Arguments
    /// * `observation` - Observation tensor (input_size, batch_size)
    ///
    /// # Returns
    /// * Action parameters (means for continuous, logits for discrete)
    /// * Log std for continuous actions (None for discrete)
    pub fn forward(&mut self, observation: &Array2<f64>) -> (Array2<f64>, Option<Array1<f64>>) {
        // Extract features using xLSTM
        let features = self.xlstm.forward(observation);

        // Generate action parameters
        let action_params = self.action_head.forward(&features);

        // Apply activation for different action types
        let processed_params = if self.is_discrete {
            action_params // Raw logits for discrete actions
        } else {
            action_params.mapv(|x| x.tanh()) // Tanh for continuous actions
        };

        (processed_params, self.log_std.clone())
    }

    /// Reset internal states
    pub fn reset_states(&mut self) {
        self.xlstm.reset_states();
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let xlstm_params = self.xlstm.num_parameters();
        let head_params = self.action_head.num_parameters();
        let std_params = if let Some(ref log_std) = self.log_std {
            log_std.len()
        } else {
            0
        };
        xlstm_params + head_params + std_params
    }
}

/// Critic network using xLSTM for value estimation
#[derive(Clone)]
pub struct CriticNetwork {
    /// xLSTM feature extractor
    pub xlstm: XLSTMNetwork,
    /// Output layer for value estimation
    pub value_head: BlockDiagonal,
}

impl CriticNetwork {
    /// Create new critic network
    ///
    /// # Arguments
    /// * `xlstm_config` - xLSTM configuration string (e.g., "msm")
    /// * `input_size` - Size of observation features
    /// * `hidden_size` - Hidden size for xLSTM
    /// * `depth` - Depth parameter for xLSTM blocks
    pub fn new(xlstm_config: &str, input_size: usize, hidden_size: usize, depth: usize) -> Self {
        let xlstm = XLSTMNetwork::from_config(xlstm_config, input_size, hidden_size, depth);
        let value_head = BlockDiagonal::new(input_size, 1, 1, true);

        Self { xlstm, value_head }
    }

    /// Forward pass through critic network
    ///
    /// # Arguments
    /// * `observation` - Observation tensor (input_size, batch_size)
    ///
    /// # Returns
    /// * Value estimate (1, batch_size)
    pub fn forward(&mut self, observation: &Array2<f64>) -> Array2<f64> {
        // Extract features using xLSTM
        let features = self.xlstm.forward(observation);

        // Generate value estimate
        self.value_head.forward(&features)
    }

    /// Reset internal states
    pub fn reset_states(&mut self) {
        self.xlstm.reset_states();
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.xlstm.num_parameters() + self.value_head.num_parameters()
    }
}

/// Combined Actor-Critic network for PPO
#[derive(Clone)]
pub struct ActorCriticNetwork {
    pub actor: ActorNetwork,
    pub critic: CriticNetwork,
}

impl ActorCriticNetwork {
    /// Create new actor-critic network
    pub fn new(
        xlstm_config: &str,
        input_size: usize,
        hidden_size: usize,
        depth: usize,
        action_dim: usize,
        is_discrete: bool,
    ) -> Self {
        let actor = ActorNetwork::new(
            xlstm_config,
            input_size,
            hidden_size,
            depth,
            action_dim,
            is_discrete,
        );
        let critic = CriticNetwork::new(xlstm_config, input_size, hidden_size, depth);

        Self { actor, critic }
    }

    /// Forward pass through both networks
    ///
    /// # Returns
    /// * (action_params, log_std, value_estimate)
    pub fn forward(
        &mut self,
        observation: &Array2<f64>,
    ) -> (Array2<f64>, Option<Array1<f64>>, Array2<f64>) {
        let (action_params, log_std) = self.actor.forward(observation);
        let value = self.critic.forward(observation);
        (action_params, log_std, value)
    }

    /// Reset states for both networks
    pub fn reset_states(&mut self) {
        self.actor.reset_states();
        self.critic.reset_states();
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.actor.num_parameters() + self.critic.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_actor_network_continuous() {
        let mut actor = ActorNetwork::new("ms", 8, 16, 2, 3, false);
        let obs = arr2(&[[1.0; 8], [0.5; 8]]).t().to_owned();

        let (action_params, log_std) = actor.forward(&obs);
        assert_eq!(action_params.shape(), &[3, 2]);
        assert!(log_std.is_some());
        assert_eq!(log_std.unwrap().len(), 3);

        // Check tanh activation bounds
        for &val in action_params.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_actor_network_discrete() {
        let mut actor = ActorNetwork::new("s", 6, 12, 3, 4, true);
        let obs = arr2(&[[1.0; 6]]).t().to_owned();

        let (logits, log_std) = actor.forward(&obs);
        assert_eq!(logits.shape(), &[4, 1]);
        assert!(log_std.is_none());
    }

    #[test]
    fn test_critic_network() {
        let mut critic = CriticNetwork::new("m", 5, 10, 2);
        let obs = arr2(&[[0.5; 5], [-0.3; 5], [1.2; 5]]).t().to_owned();

        let values = critic.forward(&obs);
        assert_eq!(values.shape(), &[1, 3]);

        // Values should be finite
        for &val in values.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_actor_critic_network() {
        let mut net = ActorCriticNetwork::new("msm", 4, 8, 2, 2, false);
        let obs = arr2(&[[1.0, -0.5], [0.3, 0.8], [0.0, 1.0], [-0.2, 0.4]]);

        let (action_params, log_std, values) = net.forward(&obs);

        assert_eq!(action_params.shape(), &[2, 2]);
        assert_eq!(values.shape(), &[1, 2]);
        assert!(log_std.is_some());

        let params_count = net.num_parameters();
        assert!(params_count > 0);
    }

    #[test]
    fn test_state_reset() {
        let mut net = ActorCriticNetwork::new("s", 3, 6, 1, 2, true);
        let obs = arr2(&[[1.0], [0.0], [0.5]]);

        // Forward pass to change states
        let _ = net.forward(&obs);

        // Reset should work without errors
        net.reset_states();

        // Should still be able to do forward pass
        let (logits, _, values) = net.forward(&obs);
        assert_eq!(logits.shape(), &[2, 1]);
        assert_eq!(values.shape(), &[1, 1]);
    }
}
