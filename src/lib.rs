// xLSTM-PPO Stock Trading Library
//
// This library implements a Proximal Policy Optimization (PPO) based stock trading system
// using Extended Long Short-Term Memory (xLSTM) networks as described in:
// "A Deep Reinforcement Learning Approach to Automated Stock Trading, using xLSTM Networks"

pub mod layers;
pub mod models;
pub mod utils;

// PPO-specific modules
pub mod action_distributions;
pub mod actor_critic_networks;
pub mod ppo_algorithm;
pub mod trading_environment;
pub mod training_loop;

// Re-export commonly used types and functions
pub use action_distributions::{ActionDistribution, CategoricalDistribution, GaussianDistribution};
pub use actor_critic_networks::{ActorCriticNetwork, ActorNetwork, CriticNetwork};
pub use ppo_algorithm::{PPOAgent, PPOConfig, Trajectory};
pub use trading_environment::{
    ActionType, MarketData, Portfolio, TradingConfig, TradingEnvironment, create_sample_market_data,
};
pub use training_loop::{Trainer, TrainingConfig, TrainingStats, create_sample_training_setup};

// Re-export core xLSTM components
pub use layers::{BlockDiagonal, CausalConv1D, LayerNorm, MLSTMBlock, SLSTMBlock};
pub use models::XLSTMNetwork;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Quick start function to run a sample training
pub fn run_sample_training() {
    println!("xLSTM-PPO Stock Trading Agent v{}", VERSION);
    println!("=====================================\n");

    // Create and run sample training
    let mut trainer = create_sample_training_setup();
    trainer.train();
}

/// Create a basic trading agent with default configuration
pub fn create_default_agent(
    symbols: Vec<String>,
    observation_size: usize,
    action_size: usize,
    is_discrete: bool,
) -> PPOAgent {
    let xlstm_config = "msm"; // mLSTM-sLSTM-mLSTM as in the paper
    let ppo_config = PPOConfig::default();

    PPOAgent::new(
        xlstm_config,
        observation_size,
        128, // hidden size
        4,   // depth
        action_size,
        is_discrete,
        ppo_config,
    )
}

/// Create a basic trading environment with sample data
pub fn create_default_environment(
    symbols: Vec<String>,
    use_continuous_actions: bool,
) -> TradingEnvironment {
    let data = create_sample_market_data(&symbols, 252); // 1 year of data
    let config = TradingConfig::default();
    let action_type = if use_continuous_actions {
        ActionType::Continuous
    } else {
        ActionType::Discrete
    };

    TradingEnvironment::new(symbols, data, config, action_type)
}

/// Utility function to create xLSTM network for custom applications
pub fn create_xlstm_network(
    config: &str,
    input_size: usize,
    hidden_size: usize,
    depth: usize,
) -> XLSTMNetwork {
    XLSTMNetwork::from_config(config, input_size, hidden_size, depth)
}
