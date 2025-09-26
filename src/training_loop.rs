use crate::ppo_algorithm::{PPOAgent, PPOConfig};
use crate::trading_environment::{
    ActionType, TradingConfig, TradingEnvironment, create_sample_market_data,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Training configuration
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// Number of training episodes
    pub num_episodes: usize,
    /// Maximum steps per episode
    pub max_episode_steps: usize,
    /// Number of steps to collect before PPO update
    pub rollout_steps: usize,
    /// Evaluation frequency (in episodes)
    pub eval_frequency: usize,
    /// Number of evaluation episodes
    pub eval_episodes: usize,
    /// Directory to save models
    pub save_dir: String,
    /// Model save frequency (in episodes)
    pub save_frequency: usize,
    /// Whether to use continuous or discrete actions
    pub use_continuous_actions: bool,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            num_episodes: 1000,
            max_episode_steps: 252, // One trading year
            rollout_steps: 2048,
            eval_frequency: 50,
            eval_episodes: 10,
            save_dir: "models".to_string(),
            save_frequency: 100,
            use_continuous_actions: true,
            seed: 42,
        }
    }
}

/// Training statistics
#[derive(Clone, Debug, Default)]
pub struct TrainingStats {
    pub episode_rewards: Vec<f64>,
    pub episode_lengths: Vec<usize>,
    pub policy_losses: Vec<f64>,
    pub value_losses: Vec<f64>,
    pub portfolio_values: Vec<f64>,
    pub sharpe_ratios: Vec<f64>,
}

impl TrainingStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_episode(
        &mut self,
        reward: f64,
        length: usize,
        portfolio_value: f64,
        sharpe_ratio: f64,
    ) {
        self.episode_rewards.push(reward);
        self.episode_lengths.push(length);
        self.portfolio_values.push(portfolio_value);
        self.sharpe_ratios.push(sharpe_ratio);
    }

    pub fn add_losses(&mut self, policy_loss: f64, value_loss: f64) {
        self.policy_losses.push(policy_loss);
        self.value_losses.push(value_loss);
    }

    pub fn get_recent_stats(&self, n: usize) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if self.episode_rewards.len() >= n {
            let recent_rewards: Vec<f64> =
                self.episode_rewards.iter().rev().take(n).cloned().collect();

            let mean_reward = recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64;
            stats.insert("mean_reward".to_string(), mean_reward);

            let recent_values: Vec<f64> = self
                .portfolio_values
                .iter()
                .rev()
                .take(n)
                .cloned()
                .collect();

            let mean_portfolio_value =
                recent_values.iter().sum::<f64>() / recent_values.len() as f64;
            stats.insert("mean_portfolio_value".to_string(), mean_portfolio_value);

            if !self.sharpe_ratios.is_empty() {
                let recent_sharpe: Vec<f64> =
                    self.sharpe_ratios.iter().rev().take(n).cloned().collect();

                let mean_sharpe = recent_sharpe.iter().sum::<f64>() / recent_sharpe.len() as f64;
                stats.insert("mean_sharpe_ratio".to_string(), mean_sharpe);
            }
        }

        if !self.policy_losses.is_empty() && self.policy_losses.len() >= n {
            let recent_policy_loss =
                self.policy_losses.iter().rev().take(n).sum::<f64>() / n as f64;
            stats.insert("mean_policy_loss".to_string(), recent_policy_loss);
        }

        if !self.value_losses.is_empty() && self.value_losses.len() >= n {
            let recent_value_loss = self.value_losses.iter().rev().take(n).sum::<f64>() / n as f64;
            stats.insert("mean_value_loss".to_string(), recent_value_loss);
        }

        stats
    }
}

/// Main training engine for the stock trading agent
pub struct Trainer {
    pub agent: PPOAgent,
    pub train_env: TradingEnvironment,
    pub eval_env: Option<TradingEnvironment>,
    pub config: TrainingConfig,
    pub stats: TrainingStats,
}

impl Trainer {
    /// Create new trainer
    pub fn new(
        symbols: Vec<String>,
        train_data: HashMap<String, Vec<crate::trading_environment::MarketData>>,
        eval_data: Option<HashMap<String, Vec<crate::trading_environment::MarketData>>>,
        training_config: TrainingConfig,
        ppo_config: PPOConfig,
        trading_config: TradingConfig,
    ) -> Self {
        let action_type = if training_config.use_continuous_actions {
            ActionType::Continuous
        } else {
            ActionType::Discrete
        };

        // Create training environment
        let train_env = TradingEnvironment::new(
            symbols.clone(),
            train_data,
            trading_config.clone(),
            action_type.clone(),
        );

        // Create evaluation environment if data is provided
        let eval_env = eval_data.map(|data| {
            TradingEnvironment::new(
                symbols.clone(),
                data,
                trading_config.clone(),
                action_type.clone(),
            )
        });

        // Create PPO agent
        let xlstm_config = "msm"; // mLSTM-sLSTM-mLSTM configuration as in the paper
        let agent = PPOAgent::new(
            xlstm_config,
            train_env.get_observation_size(),
            128, // Hidden size
            4,   // Depth
            train_env.get_action_size(),
            train_env.is_discrete(),
            ppo_config,
        );

        Self {
            agent,
            train_env,
            eval_env,
            config: training_config,
            stats: TrainingStats::new(),
        }
    }

    /// Run a single episode.
    ///
    /// This is implemented as an associated function that operates on separate mutable references
    /// to the agent and environment to avoid borrowing the entire Trainer (`&mut self`) at once,
    /// which prevents simultaneous mutable borrows of separate fields.
    fn run_episode(
        agent: &mut PPOAgent,
        env: &mut TradingEnvironment,
        config: &TrainingConfig,
        is_training: bool,
    ) -> (f64, HashMap<String, f64>) {
        let mut total_reward = 0.0;
        let mut step_count = 0;

        // Reset environment and agent states
        let mut obs = env.reset();
        agent.reset_states();

        let mut episode_ended = false;

        while !episode_ended && step_count < config.max_episode_steps {
            // Select action
            let (action, log_prob, value) = agent.select_action(&obs);

            // Take step in environment
            let (next_obs, reward, done) = env.step(&action);

            // Store experience if training
            if is_training {
                agent.store_step(obs.clone(), action, reward, value, log_prob, done);
            }

            total_reward += reward;
            obs = next_obs;
            step_count += 1;
            episode_ended = done;
        }

        // Get final episode statistics
        let episode_stats = env.get_episode_stats();

        (total_reward, episode_stats)
    }

    /// Evaluate the agent on the evaluation environment
    fn evaluate(&mut self) -> HashMap<String, f64> {
        if let Some(ref mut eval_env) = self.eval_env {
            let mut eval_stats = Vec::new();

            for _ in 0..self.config.eval_episodes {
                let (_, episode_stats) =
                    Trainer::run_episode(&mut self.agent, eval_env, &self.config, false);
                eval_stats.push(episode_stats);
            }

            // Aggregate evaluation statistics
            let mut aggregated_stats = HashMap::new();

            if !eval_stats.is_empty() {
                for key in eval_stats[0].keys() {
                    let values: Vec<f64> = eval_stats
                        .iter()
                        .filter_map(|stats| stats.get(key))
                        .cloned()
                        .collect();

                    if !values.is_empty() {
                        let mean_value = values.iter().sum::<f64>() / values.len() as f64;
                        aggregated_stats.insert(format!("eval_{}", key), mean_value);
                    }
                }
            }

            aggregated_stats
        } else {
            HashMap::new()
        }
    }

    /// Print training progress
    fn print_progress(
        &self,
        episode: usize,
        recent_stats: &HashMap<String, f64>,
        eval_stats: &HashMap<String, f64>,
        training_time: Duration,
    ) {
        println!("Episode {}/{}", episode, self.config.num_episodes);
        println!("  Training time: {:.2}s", training_time.as_secs_f64());

        if let Some(&mean_reward) = recent_stats.get("mean_reward") {
            println!("  Mean reward (last 10): {:.4}", mean_reward);
        }

        if let Some(&mean_portfolio) = recent_stats.get("mean_portfolio_value") {
            println!("  Mean portfolio value (last 10): ${:.2}", mean_portfolio);
        }

        if let Some(&mean_sharpe) = recent_stats.get("mean_sharpe_ratio") {
            println!("  Mean Sharpe ratio (last 10): {:.4}", mean_sharpe);
        }

        if let Some(&policy_loss) = recent_stats.get("mean_policy_loss") {
            println!("  Mean policy loss: {:.6}", policy_loss);
        }

        if let Some(&value_loss) = recent_stats.get("mean_value_loss") {
            println!("  Mean value loss: {:.6}", value_loss);
        }

        // Print evaluation stats if available
        if !eval_stats.is_empty() {
            println!("  Evaluation results:");
            for (key, value) in eval_stats {
                println!("    {}: {:.4}", key, value);
            }
        }

        println!();
    }

    /// Main training loop
    pub fn train(&mut self) {
        println!(
            "Starting training with {} episodes",
            self.config.num_episodes
        );
        println!("Agent parameters: {}", self.agent.num_parameters());
        println!(
            "Observation space: {}",
            self.train_env.get_observation_size()
        );
        println!("Action space: {}", self.train_env.get_action_size());
        println!("Action type: {:?}", self.train_env.action_type);
        println!();

        let training_start = Instant::now();
        let mut rollout_steps = 0;

        for episode in 1..=self.config.num_episodes {
            let episode_start = Instant::now();

            // Run training episode
            let (episode_reward, episode_stats) =
                Trainer::run_episode(&mut self.agent, &mut self.train_env, &self.config, true);
            rollout_steps += self.agent.trajectory.len();

            // Extract episode statistics
            let portfolio_value = episode_stats
                .get("final_portfolio_value")
                .unwrap_or(&0.0)
                .clone();
            let sharpe_ratio = episode_stats.get("sharpe_ratio").unwrap_or(&0.0).clone();

            // Store episode statistics
            self.stats.add_episode(
                episode_reward,
                self.agent.trajectory.len(),
                portfolio_value,
                sharpe_ratio,
            );

            // PPO update when enough steps collected
            if rollout_steps >= self.config.rollout_steps || episode % 10 == 0 {
                // Get next value for GAE calculation
                let obs = self.train_env.get_observation();
                let (_, _, next_value) = self.agent.select_action(&obs);

                // Update agent
                let (policy_loss, value_loss) = self.agent.update(next_value);
                self.stats.add_losses(policy_loss, value_loss);

                rollout_steps = 0;
            }

            // Evaluation
            let eval_stats = if episode % self.config.eval_frequency == 0 {
                self.evaluate()
            } else {
                HashMap::new()
            };

            // Print progress
            if episode % 10 == 0 {
                let recent_stats = self.stats.get_recent_stats(10);
                let episode_time = episode_start.elapsed();
                self.print_progress(episode, &recent_stats, &eval_stats, episode_time);
            }

            // Save model
            if episode % self.config.save_frequency == 0 {
                self.save_model(episode);
            }
        }

        let total_training_time = training_start.elapsed();
        println!(
            "Training completed in {:.2} minutes",
            total_training_time.as_secs_f64() / 60.0
        );

        // Final evaluation
        if self.eval_env.is_some() {
            println!("Running final evaluation...");
            let final_eval_stats = self.evaluate();
            println!("Final evaluation results:");
            for (key, value) in final_eval_stats {
                println!("  {}: {:.4}", key, value);
            }
        }

        // Save final model
        self.save_model(self.config.num_episodes);
    }

    /// Save model (placeholder implementation)
    fn save_model(&self, episode: usize) {
        // In a real implementation, this would serialize and save the model parameters
        println!("Model saved at episode {} (placeholder)", episode);

        // You would implement actual model serialization here, for example:
        // 1. Serialize network parameters to file
        // 2. Save training statistics
        // 3. Save configuration

        // Example structure:
        // let model_path = format!("{}/model_episode_{}.json", self.config.save_dir, episode);
        // serialize_model(&self.agent.network, &model_path);
    }
}

/// Convenience function to create a complete training setup with sample data
pub fn create_sample_training_setup() -> Trainer {
    // Create sample stock symbols (similar to the paper: NVIDIA, Apple, Microsoft, Google, Amazon)
    let symbols = vec![
        "NVDA".to_string(),
        "AAPL".to_string(),
        "MSFT".to_string(),
        "GOOGL".to_string(),
        "AMZN".to_string(),
    ];

    // Generate sample training data (in practice, you'd load real market data)
    let train_data = create_sample_market_data(&symbols, 1000); // ~4 years of daily data
    let eval_data = Some(create_sample_market_data(&symbols, 252)); // 1 year of evaluation data

    // Configure training
    let training_config = TrainingConfig {
        num_episodes: 500,
        max_episode_steps: 252, // One trading year
        rollout_steps: 1024,
        eval_frequency: 25,
        eval_episodes: 5,
        use_continuous_actions: true,
        ..Default::default()
    };

    // Configure PPO
    let ppo_config = PPOConfig {
        actor_lr: 3e-4,
        critic_lr: 3e-4,
        clip_epsilon: 0.2,
        gae_lambda: 0.95,
        gamma: 0.99,
        entropy_coef: 1e-3, // Lower entropy for trading
        ppo_epochs: 4,
        batch_size: 32,
        ..Default::default()
    };

    // Configure trading environment
    let trading_config = TradingConfig {
        initial_balance: 1_000_000.0,
        transaction_cost: 0.001, // 0.1% transaction cost
        turbulence_threshold: 100.0,
        lookback_window: 30,
        ..Default::default()
    };

    Trainer::new(
        symbols,
        train_data,
        eval_data,
        training_config,
        ppo_config,
        trading_config,
    )
}

/// Main function to run the training
pub fn main() {
    println!("xLSTM-PPO Stock Trading Agent");
    println!("==============================\n");

    // Create training setup
    let mut trainer = create_sample_training_setup();

    // Start training
    trainer.train();

    println!("Training completed successfully!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert!(config.num_episodes > 0);
        assert!(config.rollout_steps > 0);
        assert!(config.eval_frequency > 0);
    }

    #[test]
    fn test_training_stats() {
        let mut stats = TrainingStats::new();

        for i in 1..=20 {
            stats.add_episode(i as f64, 100, 1000000.0 + i as f64 * 1000.0, 0.1);
        }

        let recent_stats = stats.get_recent_stats(10);
        assert!(recent_stats.contains_key("mean_reward"));
        assert!(recent_stats.contains_key("mean_portfolio_value"));
    }

    #[test]
    fn test_trainer_creation() {
        let symbols = vec!["TEST".to_string()];
        let train_data = create_sample_market_data(&symbols, 100);
        let training_config = TrainingConfig::default();
        let ppo_config = PPOConfig::default();
        let trading_config = TradingConfig::default();

        let trainer = Trainer::new(
            symbols,
            train_data,
            None,
            training_config,
            ppo_config,
            trading_config,
        );

        assert!(trainer.agent.num_parameters() > 0);
        assert!(trainer.train_env.get_observation_size() > 0);
    }

    #[test]
    fn test_sample_training_setup() {
        let trainer = create_sample_training_setup();
        assert_eq!(trainer.train_env.symbols.len(), 5);
        assert!(trainer.eval_env.is_some());
        assert_eq!(trainer.config.num_episodes, 500);
    }

    #[test]
    fn test_single_episode() {
        let mut trainer = create_sample_training_setup();
        trainer.config.max_episode_steps = 10; // Short episode for testing

        // Call the associated function with explicit mutable field references
        let (reward, stats) = Trainer::run_episode(
            &mut trainer.agent,
            &mut trainer.train_env,
            &trainer.config,
            true,
        );

        assert!(reward.is_finite());
        assert!(!stats.is_empty());
        assert!(trainer.agent.trajectory.len() > 0);
    }
}
