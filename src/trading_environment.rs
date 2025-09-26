use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stock market data point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketData {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub adj_close: f64,
    pub volume: f64,
}

/// Stock trading environment configuration
#[derive(Clone, Debug)]
pub struct TradingConfig {
    /// Initial balance in USD
    pub initial_balance: f64,
    /// Transaction cost rate (e.g., 0.001 for 0.1%)
    pub transaction_cost: f64,
    /// Maximum allowed turbulence index to avoid high-risk trading
    pub turbulence_threshold: f64,
    /// Lookback window for features
    pub lookback_window: usize,
    /// Risk-free rate for Sharpe ratio calculation
    pub risk_free_rate: f64,
    /// Maximum number of shares to hold per stock
    pub max_shares_per_stock: f64,
}

impl Default for TradingConfig {
    fn default() -> Self {
        TradingConfig {
            initial_balance: 1_000_000.0,
            transaction_cost: 0.001,
            turbulence_threshold: 100.0,
            lookback_window: 30,
            risk_free_rate: 0.02,
            max_shares_per_stock: 1000.0,
        }
    }
}

/// Trading action types
#[derive(Clone, Debug)]
pub enum ActionType {
    /// Discrete actions: 0=sell, 1=hold, 2=buy for each stock
    Discrete,
    /// Continuous actions: percentage of portfolio to allocate to each stock
    Continuous,
}

/// Portfolio state
#[derive(Clone, Debug)]
pub struct Portfolio {
    pub cash: f64,
    pub shares: HashMap<String, f64>,
    pub total_value: f64,
    pub prev_total_value: f64,
}

impl Portfolio {
    pub fn new(initial_cash: f64, symbols: &[String]) -> Self {
        let mut shares = HashMap::new();
        for symbol in symbols {
            shares.insert(symbol.clone(), 0.0);
        }

        Self {
            cash: initial_cash,
            shares,
            total_value: initial_cash,
            prev_total_value: initial_cash,
        }
    }

    pub fn update_total_value(&mut self, prices: &HashMap<String, f64>) {
        self.prev_total_value = self.total_value;
        self.total_value = self.cash;

        for (symbol, &shares) in &self.shares {
            if let Some(&price) = prices.get(symbol) {
                self.total_value += shares * price;
            }
        }
    }

    pub fn get_portfolio_weights(&self, prices: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut weights = HashMap::new();

        if self.total_value > 0.0 {
            for (symbol, &shares) in &self.shares {
                if let Some(&price) = prices.get(symbol) {
                    let value = shares * price;
                    weights.insert(symbol.clone(), value / self.total_value);
                }
            }
        }

        weights
    }
}

/// Stock trading environment
pub struct TradingEnvironment {
    pub config: TradingConfig,
    pub symbols: Vec<String>,
    pub data: HashMap<String, Vec<MarketData>>,
    pub portfolio: Portfolio,
    pub current_step: usize,
    pub max_steps: usize,
    pub action_type: ActionType,
    pub episode_returns: Vec<f64>,
}

impl TradingEnvironment {
    /// Create new trading environment
    pub fn new(
        symbols: Vec<String>,
        data: HashMap<String, Vec<MarketData>>,
        config: TradingConfig,
        action_type: ActionType,
    ) -> Self {
        // Ensure all symbols have the same data length
        let min_len = data.values().map(|v| v.len()).min().unwrap_or(0);
        let max_steps = if min_len > config.lookback_window {
            min_len - config.lookback_window
        } else {
            0
        };

        let portfolio = Portfolio::new(config.initial_balance, &symbols);

        Self {
            config,
            symbols,
            data,
            portfolio,
            current_step: 0,
            max_steps,
            action_type,
            episode_returns: Vec::new(),
        }
    }

    /// Reset environment for new episode
    pub fn reset(&mut self) -> Array2<f64> {
        self.current_step = 0;
        self.portfolio = Portfolio::new(self.config.initial_balance, &self.symbols);
        self.episode_returns.clear();
        self.get_observation()
    }

    /// Get current observation (features)
    pub fn get_observation(&self) -> Array2<f64> {
        let feature_size = self.symbols.len() * 6 + 1; // OHLCAV + portfolio weight per stock + turbulence
        let mut features = Array2::zeros((feature_size, 1));

        let current_idx = self.current_step + self.config.lookback_window;
        if current_idx >= self.max_steps + self.config.lookback_window {
            return features; // Return zeros for invalid steps
        }

        // Current prices for portfolio value calculation
        let mut current_prices = HashMap::new();
        for (i, symbol) in self.symbols.iter().enumerate() {
            if let Some(symbol_data) = self.data.get(symbol) {
                if current_idx < symbol_data.len() {
                    let data_point = &symbol_data[current_idx];
                    current_prices.insert(symbol.clone(), data_point.adj_close);

                    // Stock features: OHLCAV normalized
                    let base_idx = i * 6;
                    features[[base_idx, 0]] = data_point.open / data_point.adj_close - 1.0;
                    features[[base_idx + 1, 0]] = data_point.high / data_point.adj_close - 1.0;
                    features[[base_idx + 2, 0]] = data_point.low / data_point.adj_close - 1.0;
                    features[[base_idx + 3, 0]] = data_point.close / data_point.adj_close - 1.0;
                    features[[base_idx + 4, 0]] = data_point.adj_close;
                    features[[base_idx + 5, 0]] = (data_point.volume / 1e6).ln(); // Log normalized volume
                }
            }
        }

        // Update portfolio value with current prices
        let mut portfolio_copy = self.portfolio.clone();
        portfolio_copy.update_total_value(&current_prices);

        // Portfolio weights
        let weights = portfolio_copy.get_portfolio_weights(&current_prices);
        for (i, symbol) in self.symbols.iter().enumerate() {
            let weight_idx = self.symbols.len() * 6 + i;
            if weight_idx < feature_size - 1 {
                features[[weight_idx, 0]] = weights.get(symbol).unwrap_or(&0.0).clone();
            }
        }

        // Turbulence index (simplified calculation)
        let turbulence = self.calculate_turbulence_index(current_idx);
        features[[feature_size - 1, 0]] = turbulence;

        features
    }

    /// Calculate turbulence index for market risk assessment
    fn calculate_turbulence_index(&self, current_idx: usize) -> f64 {
        let lookback = 30.min(current_idx);
        if lookback < 2 {
            return 0.0;
        }

        let mut returns = Vec::new();

        for symbol in &self.symbols {
            if let Some(symbol_data) = self.data.get(symbol) {
                let mut symbol_returns = Vec::new();

                for i in (current_idx - lookback + 1)..=current_idx {
                    if i > 0 && i < symbol_data.len() {
                        let prev_price = symbol_data[i - 1].adj_close;
                        let curr_price = symbol_data[i].adj_close;
                        let ret = (curr_price / prev_price - 1.0).abs();
                        symbol_returns.push(ret);
                    }
                }

                if !symbol_returns.is_empty() {
                    let avg_return =
                        symbol_returns.iter().sum::<f64>() / symbol_returns.len() as f64;
                    returns.push(avg_return);
                }
            }
        }

        if returns.is_empty() {
            return 0.0;
        }

        // Simple turbulence: weighted average of absolute returns
        let turbulence = returns.iter().sum::<f64>() * 100.0;
        turbulence
    }

    /// Execute trading action
    pub fn step(&mut self, action: &Array1<f64>) -> (Array2<f64>, f64, bool) {
        let current_idx = self.current_step + self.config.lookback_window;

        // Get current prices
        let mut current_prices = HashMap::new();
        for symbol in &self.symbols {
            if let Some(symbol_data) = self.data.get(symbol) {
                if current_idx < symbol_data.len() {
                    current_prices.insert(symbol.clone(), symbol_data[current_idx].adj_close);
                }
            }
        }

        // Update portfolio value before action
        self.portfolio.update_total_value(&current_prices);

        // Calculate reward BEFORE executing action
        let reward = self.calculate_reward(&current_prices);

        // Check turbulence condition
        let turbulence = self.calculate_turbulence_index(current_idx);
        let should_trade = turbulence <= self.config.turbulence_threshold;

        if should_trade {
            self.execute_trading_action(action, &current_prices);
        }

        // Update portfolio value after action
        self.portfolio.update_total_value(&current_prices);

        // Move to next step
        self.current_step += 1;

        // Check if episode is done
        let done = self.current_step >= self.max_steps;

        // Get next observation
        let next_obs = if done {
            Array2::zeros((self.get_observation_size(), 1))
        } else {
            self.get_observation()
        };

        self.episode_returns.push(reward);

        (next_obs, reward, done)
    }

    /// Execute the actual trading action
    fn execute_trading_action(&mut self, action: &Array1<f64>, prices: &HashMap<String, f64>) {
        match self.action_type {
            ActionType::Discrete => {
                self.execute_discrete_action(action, prices);
            }
            ActionType::Continuous => {
                self.execute_continuous_action(action, prices);
            }
        }
    }

    /// Execute discrete trading actions (sell/hold/buy)
    fn execute_discrete_action(&mut self, action: &Array1<f64>, prices: &HashMap<String, f64>) {
        for (i, symbol) in self.symbols.iter().enumerate() {
            if i >= action.len() {
                break;
            }

            let action_idx = action[i].round() as i32;
            if let Some(&price) = prices.get(symbol) {
                let current_shares = *self.portfolio.shares.get(symbol).unwrap_or(&0.0);

                match action_idx {
                    0 => {
                        // Sell
                        if current_shares > 0.0 {
                            let sell_amount = current_shares * 0.1; // Sell 10% of holdings
                            let proceeds = sell_amount * price;
                            let transaction_cost = proceeds * self.config.transaction_cost;

                            self.portfolio.cash += proceeds - transaction_cost;
                            self.portfolio
                                .shares
                                .insert(symbol.clone(), current_shares - sell_amount);
                        }
                    }
                    2 => {
                        // Buy
                        let available_cash = self.portfolio.cash * 0.1; // Use 10% of available cash
                        if available_cash > 0.0 {
                            let buy_shares = available_cash / price;
                            let total_cost = buy_shares * price;
                            let transaction_cost = total_cost * self.config.transaction_cost;

                            if self.portfolio.cash >= total_cost + transaction_cost {
                                self.portfolio.cash -= total_cost + transaction_cost;
                                self.portfolio
                                    .shares
                                    .insert(symbol.clone(), current_shares + buy_shares);
                            }
                        }
                    }
                    _ => { // Hold (action_idx == 1 or invalid)
                        // Do nothing
                    }
                }
            }
        }
    }

    /// Execute continuous trading actions (portfolio weights)
    fn execute_continuous_action(&mut self, action: &Array1<f64>, prices: &HashMap<String, f64>) {
        // Normalize actions to sum to 1 (portfolio weights)
        let action_sum: f64 = action.iter().map(|&x| x.abs()).sum();
        if action_sum < 1e-8 {
            return; // No trading if all actions are near zero
        }

        let normalized_weights: Vec<f64> = action
            .iter()
            .map(|&x| (x.abs() / action_sum).max(0.0).min(1.0))
            .collect();

        // Calculate target values for each stock
        for (i, symbol) in self.symbols.iter().enumerate() {
            if i >= normalized_weights.len() {
                break;
            }

            if let Some(&price) = prices.get(symbol) {
                let target_value = self.portfolio.total_value * normalized_weights[i];
                let current_shares = *self.portfolio.shares.get(symbol).unwrap_or(&0.0);
                let current_value = current_shares * price;

                let value_diff = target_value - current_value;

                if value_diff.abs() > self.portfolio.total_value * 0.01 {
                    // Only trade if difference > 1%
                    if value_diff > 0.0 {
                        // Buy more shares
                        let buy_value = value_diff;
                        let transaction_cost = buy_value * self.config.transaction_cost;

                        if self.portfolio.cash >= buy_value + transaction_cost {
                            let buy_shares = buy_value / price;
                            self.portfolio.cash -= buy_value + transaction_cost;
                            self.portfolio
                                .shares
                                .insert(symbol.clone(), current_shares + buy_shares);
                        }
                    } else {
                        // Sell shares
                        let sell_value = -value_diff;
                        let sell_shares = (sell_value / price).min(current_shares);
                        let proceeds = sell_shares * price;
                        let transaction_cost = proceeds * self.config.transaction_cost;

                        self.portfolio.cash += proceeds - transaction_cost;
                        self.portfolio
                            .shares
                            .insert(symbol.clone(), current_shares - sell_shares);
                    }
                }
            }
        }
    }

    /// Calculate reward based on portfolio performance
    fn calculate_reward(&self, _prices: &HashMap<String, f64>) -> f64 {
        let portfolio_return = if self.portfolio.prev_total_value > 0.0 {
            (self.portfolio.total_value - self.portfolio.prev_total_value)
                / self.portfolio.prev_total_value
        } else {
            0.0
        };

        // Simple reward: portfolio return scaled by 100
        portfolio_return * 100.0
    }

    /// Get the size of the observation space
    pub fn get_observation_size(&self) -> usize {
        self.symbols.len() * 6 + 1 // OHLCAV per stock + turbulence
    }

    /// Get the size of the action space
    pub fn get_action_size(&self) -> usize {
        self.symbols.len()
    }

    /// Check if action space is discrete
    pub fn is_discrete(&self) -> bool {
        matches!(self.action_type, ActionType::Discrete)
    }

    /// Get episode statistics
    pub fn get_episode_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if !self.episode_returns.is_empty() {
            let total_return = (self.portfolio.total_value - self.config.initial_balance)
                / self.config.initial_balance;
            let avg_return =
                self.episode_returns.iter().sum::<f64>() / self.episode_returns.len() as f64;

            let returns_std = {
                let mean = avg_return;
                let variance = self
                    .episode_returns
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / self.episode_returns.len() as f64;
                variance.sqrt()
            };

            let sharpe_ratio = if returns_std > 1e-8 {
                (avg_return - self.config.risk_free_rate / 252.0) / returns_std
            } else {
                0.0
            };

            stats.insert("total_return".to_string(), total_return);
            stats.insert("avg_daily_return".to_string(), avg_return);
            stats.insert("volatility".to_string(), returns_std);
            stats.insert("sharpe_ratio".to_string(), sharpe_ratio);
            stats.insert(
                "final_portfolio_value".to_string(),
                self.portfolio.total_value,
            );
        }

        stats
    }
}

/// Helper function to load market data from CSV-like format
pub fn create_sample_market_data(
    symbols: &[String],
    days: usize,
) -> HashMap<String, Vec<MarketData>> {
    let mut data = HashMap::new();

    for symbol in symbols {
        let mut symbol_data = Vec::new();
        let mut base_price = 100.0;

        for day in 0..days {
            // Generate synthetic market data with some randomness
            use rand::Rng;
            let mut rng = rand::rng();

            let daily_return = rng.random_range(-0.05..0.05); // +/- 5% daily moves
            let new_price = base_price * (1.0 + daily_return);

            let high = new_price * (1.0 + rng.random_range(0.0..0.02));
            let low = new_price * (1.0 - rng.random_range(0.0..0.02));
            let volume = rng.random_range(1_000_000.0..10_000_000.0);

            symbol_data.push(MarketData {
                date: format!("2024-{:02}-{:02}", (day / 30) + 1, (day % 30) + 1),
                open: base_price,
                high,
                low,
                close: new_price,
                adj_close: new_price,
                volume,
            });

            base_price = new_price;
        }

        data.insert(symbol.clone(), symbol_data);
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_config() {
        let config = TradingConfig::default();
        assert!(config.initial_balance > 0.0);
        assert!(config.transaction_cost >= 0.0);
        assert!(config.lookback_window > 0);
    }

    #[test]
    fn test_portfolio() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let mut portfolio = Portfolio::new(100000.0, &symbols);

        assert_eq!(portfolio.cash, 100000.0);
        assert_eq!(portfolio.shares.len(), 2);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 150.0);
        prices.insert("GOOGL".to_string(), 2800.0);

        portfolio.shares.insert("AAPL".to_string(), 100.0);
        portfolio.update_total_value(&prices);

        assert_eq!(portfolio.total_value, 100000.0 + 100.0 * 150.0);
    }

    #[test]
    fn test_sample_market_data() {
        let symbols = vec!["TEST".to_string()];
        let data = create_sample_market_data(&symbols, 100);

        assert_eq!(data.len(), 1);
        assert_eq!(data.get("TEST").unwrap().len(), 100);

        let first_day = &data.get("TEST").unwrap()[0];
        assert!(first_day.adj_close > 0.0);
        assert!(first_day.volume > 0.0);
    }

    #[test]
    fn test_trading_environment_creation() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let data = create_sample_market_data(&symbols, 252); // 1 year of data
        let config = TradingConfig::default();

        let env = TradingEnvironment::new(symbols.clone(), data, config, ActionType::Continuous);

        assert_eq!(env.symbols, symbols);
        assert!(env.max_steps > 0);
        assert_eq!(env.get_action_size(), 2);
        assert!(!env.is_discrete());
    }

    #[test]
    fn test_environment_reset_and_step() {
        let symbols = vec!["TEST".to_string()];
        let data = create_sample_market_data(&symbols, 100);
        let config = TradingConfig::default();

        let mut env = TradingEnvironment::new(symbols, data, config, ActionType::Discrete);

        let obs = env.reset();
        assert_eq!(obs.shape()[0], env.get_observation_size());
        assert_eq!(obs.shape()[1], 1);

        let action = Array1::from_vec(vec![1.0]); // Hold action
        let (next_obs, reward, done) = env.step(&action);

        assert_eq!(next_obs.shape()[0], env.get_observation_size());
        assert!(reward.is_finite());
        assert!(!done || env.current_step >= env.max_steps);
    }

    #[test]
    fn test_episode_stats() {
        let symbols = vec!["TEST".to_string()];
        let data = create_sample_market_data(&symbols, 50);
        let config = TradingConfig::default();

        let mut env = TradingEnvironment::new(symbols, data, config, ActionType::Continuous);

        env.reset();

        // Simulate some steps
        for _ in 0..10 {
            let action = Array1::from_vec(vec![0.5]);
            let (_, _, done) = env.step(&action);
            if done {
                break;
            }
        }

        let stats = env.get_episode_stats();
        assert!(stats.contains_key("total_return"));
        assert!(stats.contains_key("final_portfolio_value"));
    }
}
