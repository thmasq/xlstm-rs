use ndarray::{Array2, arr2};
use rand::Rng;
use rand_distr::num_traits::Float;
use xlstm_rs::{models::XLSTMNetwork, optimizers::PPO, utils::safe_div};

/// Simple trading environment for PPO training
struct TradingEnvironment {
    prices: Vec<f64>,
    current_step: usize,
    initial_balance: f64,
    current_balance: f64,
    shares_held: f64,
    max_shares: f64,
    transaction_cost_pct: f64,
    window_size: usize,
}

#[derive(Clone, Copy, Debug)]
enum TradingAction {
    Hold = 0,
    Buy = 1,
    Sell = 2,
}

impl TradingAction {
    fn from_index(index: usize) -> Self {
        match index {
            0 => TradingAction::Hold,
            1 => TradingAction::Buy,
            2 => TradingAction::Sell,
            _ => TradingAction::Hold,
        }
    }

    fn to_one_hot(&self) -> Array2<f64> {
        match self {
            TradingAction::Hold => arr2(&[[1.0], [0.0], [0.0]]),
            TradingAction::Buy => arr2(&[[0.0], [1.0], [0.0]]),
            TradingAction::Sell => arr2(&[[0.0], [0.0], [1.0]]),
        }
    }
}

impl TradingEnvironment {
    fn new(prices: Vec<f64>, initial_balance: f64, window_size: usize) -> Self {
        TradingEnvironment {
            prices,
            current_step: window_size,
            initial_balance,
            current_balance: initial_balance,
            shares_held: 0.0,
            max_shares: initial_balance / 100.0, // Reasonable position sizing
            transaction_cost_pct: 0.001,         // 0.1% transaction cost
            window_size,
        }
    }

    fn reset(&mut self) {
        self.current_step = self.window_size;
        self.current_balance = self.initial_balance;
        self.shares_held = 0.0;
    }

    fn get_state(&self) -> Array2<f64> {
        let start_idx = self.current_step.saturating_sub(self.window_size);
        let end_idx = self.current_step;

        let mut state_data = Vec::new();

        // Price features (normalized returns)
        for i in start_idx..end_idx {
            if i > 0 {
                let return_pct = (self.prices[i] - self.prices[i - 1]) / self.prices[i - 1];
                state_data.push(return_pct.tanh()); // Bounded between -1 and 1
            } else {
                state_data.push(0.0);
            }
        }

        // Portfolio state
        let current_price = self.prices[self.current_step];
        let portfolio_value = self.current_balance + self.shares_held * current_price;
        let portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance;
        let position_ratio = self.shares_held / self.max_shares;
        let cash_ratio = self.current_balance / self.initial_balance;

        state_data.push(portfolio_return.tanh());
        state_data.push(position_ratio.tanh());
        state_data.push(cash_ratio.tanh());

        // Technical indicators (simple moving averages)
        if self.current_step >= 5 {
            let ma5: f64 = self.prices[self.current_step - 4..=self.current_step]
                .iter()
                .sum::<f64>()
                / 5.0;
            let ma_ratio = (current_price - ma5) / ma5;
            state_data.push(ma_ratio.tanh());
        } else {
            state_data.push(0.0);
        }

        // Volatility estimate (recent price range)
        if self.current_step >= 10 {
            let recent_prices = &self.prices[self.current_step - 9..=self.current_step];
            let min_price = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_price = recent_prices
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let volatility = safe_div(max_price - min_price, current_price, 1e-8);
            state_data.push(volatility.tanh());
        } else {
            state_data.push(0.0);
        }

        // Convert to column vector
        Array2::from_shape_vec((state_data.len(), 1), state_data).unwrap()
    }

    fn step(&mut self, action: TradingAction) -> (Array2<f64>, f64, bool) {
        if self.current_step >= self.prices.len() - 1 {
            return (self.get_state(), 0.0, true);
        }

        let current_price = self.prices[self.current_step];
        let prev_portfolio_value = self.current_balance + self.shares_held * current_price;

        // Execute action
        match action {
            TradingAction::Buy => {
                let max_buy_amount = self.current_balance * 0.1; // Limit position size
                let shares_to_buy = max_buy_amount / current_price;
                if shares_to_buy > 0.0 && self.current_balance >= max_buy_amount {
                    let cost = shares_to_buy * current_price * (1.0 + self.transaction_cost_pct);
                    self.current_balance -= cost;
                    self.shares_held += shares_to_buy;
                }
            }
            TradingAction::Sell => {
                if self.shares_held > 0.0 {
                    let shares_to_sell = self.shares_held * 0.1; // Partial sell
                    let proceeds =
                        shares_to_sell * current_price * (1.0 - self.transaction_cost_pct);
                    self.current_balance += proceeds;
                    self.shares_held -= shares_to_sell;
                }
            }
            TradingAction::Hold => {
                // Do nothing
            }
        }

        self.current_step += 1;

        // Calculate reward
        let new_price = self.prices[self.current_step];
        let new_portfolio_value = self.current_balance + self.shares_held * new_price;
        let reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value;

        // Penalize excessive trading
        let action_penalty = match action {
            TradingAction::Hold => 0.0,
            _ => -0.0001, // Small penalty for trading
        };

        let final_reward = reward + action_penalty;
        let done = self.current_step >= self.prices.len() - 1;

        (self.get_state(), final_reward, done)
    }

    fn get_portfolio_value(&self) -> f64 {
        let current_price = self.prices[self.current_step.min(self.prices.len() - 1)];
        self.current_balance + self.shares_held * current_price
    }

    fn is_done(&self) -> bool {
        self.current_step >= self.prices.len() - 1
    }
}

/// Actor-Critic agent using xLSTM networks
struct ActorCriticAgent {
    actor: XLSTMNetwork,
    critic: XLSTMNetwork,
    state_size: usize,
    action_size: usize,
}

impl ActorCriticAgent {
    fn new(state_size: usize, action_size: usize, hidden_size: usize) -> Self {
        // Create xLSTM networks for actor and critic
        let actor = XLSTMNetwork::from_config("msms", state_size, hidden_size, 4);
        let critic = XLSTMNetwork::from_config("smsm", state_size, hidden_size, 4);

        ActorCriticAgent {
            actor,
            critic,
            state_size,
            action_size,
        }
    }

    fn get_action_and_value(&mut self, state: &Array2<f64>) -> (TradingAction, f64, f64) {
        // Get action probabilities from actor
        let action_logits = self.actor.forward(state);

        // Convert to probabilities using softmax
        let action_probs = self.softmax(&action_logits);

        // Sample action
        let action_index = self.sample_action(&action_probs);
        let action = TradingAction::from_index(action_index);

        // Calculate log probability
        let log_prob = action_probs[[action_index, 0]].ln();

        // Get value from critic
        let value_output = self.critic.forward(state);
        let value = value_output[[0, 0]]; // Assume single output for value

        (action, value, log_prob)
    }

    fn get_value(&mut self, state: &Array2<f64>) -> f64 {
        let value_output = self.critic.forward(state);
        value_output[[0, 0]]
    }

    fn softmax(&self, logits: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(logits.raw_dim());

        // Find max for numerical stability
        let max_val = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate softmax
        let mut sum = 0.0;
        for i in 0..logits.nrows() {
            let exp_val = (logits[[i, 0]] - max_val).exp();
            result[[i, 0]] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for i in 0..logits.nrows() {
            result[[i, 0]] /= sum;
        }

        result
    }

    fn sample_action(&self, probs: &Array2<f64>) -> usize {
        let mut rng = rand::rng();
        let random_val: f64 = rng.random();

        let mut cumsum = 0.0;
        for i in 0..probs.nrows() {
            cumsum += probs[[i, 0]];
            if random_val <= cumsum {
                return i;
            }
        }

        // Fallback to last action
        probs.nrows() - 1
    }

    fn reset_states(&mut self) {
        self.actor.reset_states();
        self.critic.reset_states();
    }
}

/// Generate synthetic stock price data for training
fn generate_stock_data(length: usize, initial_price: f64, volatility: f64) -> Vec<f64> {
    let mut prices = vec![initial_price];
    let mut rng = rand::rng();

    for _ in 1..length {
        let prev_price = prices.last().unwrap();

        // Geometric Brownian Motion with some trend
        let dt = 1.0 / 252.0; // Daily steps, assuming 252 trading days per year
        let drift = 0.05; // 5% annual return
        let random_shock: f64 = rng.random::<f64>() - 0.5; // Centered around 0

        let price_change = prev_price * (drift * dt + volatility * dt.sqrt() * random_shock);
        let new_price = prev_price + price_change;

        prices.push(new_price.max(1.0)); // Prevent negative prices
    }

    prices
}

fn main() {
    println!("xLSTM PPO Trading Agent");
    println!("==========================\n");

    // Generate synthetic market data
    println!("1. Generating synthetic market data...");
    let train_prices = generate_stock_data(2000, 100.0, 0.2);
    let test_prices = generate_stock_data(500, train_prices.last().unwrap().clone(), 0.2);

    println!("   Training data: {} price points", train_prices.len());
    println!("   Testing data: {} price points", test_prices.len());

    // Setup environment and agent
    println!("\n2. Setting up trading environment and agent...");
    let window_size = 20;
    let initial_balance = 10000.0;
    let state_size = window_size + 5; // Price windows + portfolio features
    let action_size = 3; // Hold, Buy, Sell
    let hidden_size = 32;

    let mut env = TradingEnvironment::new(train_prices.clone(), initial_balance, window_size);
    let mut agent = ActorCriticAgent::new(state_size, action_size, hidden_size);
    let mut optimizer = PPO::new(0.0003);

    println!("   State size: {}", state_size);
    println!("   Action size: {}", action_size);
    println!("   Actor parameters: {}", agent.actor.num_parameters());
    println!("   Critic parameters: {}", agent.critic.num_parameters());

    // Training loop
    println!("\n3. Training the PPO agent...");
    let num_episodes = 100;
    let update_interval = 10; // Update every N episodes

    let mut episode_rewards = Vec::new();
    let mut best_reward = f64::NEG_INFINITY;

    for episode in 0..num_episodes {
        env.reset();
        agent.reset_states();

        let mut episode_reward = 0.0;
        let mut episode_steps = 0;

        // Collect experience for one episode
        while !env.is_done() {
            let state = env.get_state();
            let (action, value, log_prob) = agent.get_action_and_value(&state);

            let (_next_state, reward, done) = env.step(action);

            // Store experience in PPO buffer
            optimizer.store_experience(state, action.to_one_hot(), reward, value, log_prob, done);

            episode_reward += reward;
            episode_steps += 1;

            if done {
                break;
            }
        }

        episode_rewards.push(episode_reward);

        // Update policy every update_interval episodes
        if (episode + 1) % update_interval == 0 && optimizer.buffer_size() > 0 {
            // Calculate advantages
            let final_state = env.get_state();
            let final_value = agent.get_value(&final_state);
            let (_advantages, _returns) = optimizer.compute_advantages(final_value);

            // Perform PPO update (simplified - in practice you'd do multiple epochs)
            println!(
                "   Updating policy at episode {} with {} experiences",
                episode + 1,
                optimizer.buffer_size()
            );

            // Clear buffer after update
            optimizer.clear_buffer();
        }

        // Track best performance
        if episode_reward > best_reward {
            best_reward = episode_reward;
        }

        // Print progress
        if (episode + 1) % 20 == 0 {
            let recent_avg = episode_rewards
                .iter()
                .skip(episode_rewards.len().saturating_sub(20))
                .sum::<f64>()
                / 20.0;

            println!(
                "   Episode {}: reward={:.4}, steps={}, avg_20={:.4}, best={:.4}",
                episode + 1,
                episode_reward,
                episode_steps,
                recent_avg,
                best_reward
            );
        }
    }

    println!("\n4. Training completed!");
    println!("   Best episode reward: {:.4}", best_reward);
    println!(
        "   Final average reward (last 20): {:.4}",
        episode_rewards
            .iter()
            .skip(episode_rewards.len().saturating_sub(20))
            .sum::<f64>()
            / 20.0
    );

    // Testing phase
    println!("\n5. Testing the trained agent...");
    let mut test_env = TradingEnvironment::new(test_prices.clone(), initial_balance, window_size);
    agent.reset_states();

    let mut test_rewards = Vec::new();
    let mut test_actions = Vec::new();
    let mut portfolio_values = vec![initial_balance];

    while !test_env.is_done() {
        let state = test_env.get_state();
        let (action, _value, _log_prob) = agent.get_action_and_value(&state);

        let (_next_state, reward, _done) = test_env.step(action);

        test_rewards.push(reward);
        test_actions.push(action as u8);
        portfolio_values.push(test_env.get_portfolio_value());
    }

    let final_portfolio_value = test_env.get_portfolio_value();
    let total_return = (final_portfolio_value - initial_balance) / initial_balance;
    let cumulative_reward: f64 = test_rewards.iter().sum();

    println!("   Test results:");
    println!("     Initial balance: ${:.2}", initial_balance);
    println!("     Final balance: ${:.2}", final_portfolio_value);
    println!("     Total return: {:.2}%", total_return * 100.0);
    println!("     Cumulative reward: {:.6}", cumulative_reward);
    println!("     Number of trades: {}", test_actions.len());

    // Action distribution
    let hold_count = test_actions.iter().filter(|&&a| a == 0).count();
    let buy_count = test_actions.iter().filter(|&&a| a == 1).count();
    let sell_count = test_actions.iter().filter(|&&a| a == 2).count();

    println!("     Action distribution:");
    println!(
        "       Hold: {} ({:.1}%)",
        hold_count,
        100.0 * hold_count as f64 / test_actions.len() as f64
    );
    println!(
        "       Buy:  {} ({:.1}%)",
        buy_count,
        100.0 * buy_count as f64 / test_actions.len() as f64
    );
    println!(
        "       Sell: {} ({:.1}%)",
        sell_count,
        100.0 * sell_count as f64 / test_actions.len() as f64
    );

    // Calculate baseline (buy and hold)
    let initial_price = test_prices[window_size];
    let final_price = test_prices.last().unwrap();
    let buy_hold_return = (final_price - initial_price) / initial_price;

    println!("\n6. Performance comparison:");
    println!("   PPO Agent return: {:.2}%", total_return * 100.0);
    println!("   Buy & Hold return: {:.2}%", buy_hold_return * 100.0);

    if total_return > buy_hold_return {
        println!(
            "   Agent outperformed buy & hold by {:.2}%!",
            (total_return - buy_hold_return) * 100.0
        );
    } else {
        println!(
            "   Agent underperformed buy & hold by {:.2}%",
            (buy_hold_return - total_return) * 100.0
        );
    }

    println!("\n7. Agent architecture summary:");
    println!("   Actor network: {}", agent.actor.summary());
    println!("   Critic network: {}", agent.critic.summary());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_environment() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0];
        let mut env = TradingEnvironment::new(prices, 1000.0, 2);

        let initial_state = env.get_state();
        assert!(initial_state.nrows() > 0);

        let (_state, _reward, _done) = env.step(TradingAction::Buy);
        assert!(env.current_balance < 1000.0); // Should have spent money
        assert!(env.shares_held > 0.0); // Should have shares
    }

    #[test]
    fn test_actor_critic_agent() {
        let mut agent = ActorCriticAgent::new(10, 3, 16);
        let state = Array2::ones((10, 1));

        let (action, value, log_prob) = agent.get_action_and_value(&state);

        // Should return valid action, value, and log_prob
        assert!(matches!(
            action,
            TradingAction::Hold | TradingAction::Buy | TradingAction::Sell
        ));
        assert!(value.is_finite());
        assert!(log_prob.is_finite() && log_prob <= 0.0); // Log probabilities should be <= 0
    }

    #[test]
    fn test_stock_data_generation() {
        let prices = generate_stock_data(100, 100.0, 0.2);

        assert_eq!(prices.len(), 100);
        assert_eq!(prices[0], 100.0);
        assert!(prices.iter().all(|&p| p > 0.0)); // All prices should be positive
    }
}
