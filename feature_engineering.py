import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(filepath):
    """Load and prepare the trading data"""
    df = pd.read_csv(filepath)
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = df.sort_values("trading_date").reset_index(drop=True)
    return df


def create_price_features(df):
    """Create price-based features"""
    features = pd.DataFrame(index=df.index)

    # Basic price features
    features["price_range"] = df["max_price"] - df["min_price"]
    features["price_range_pct"] = features["price_range"] / df["opening_price"]
    features["close_open_diff"] = df["last_price"] - df["opening_price"]
    features["close_open_pct"] = features["close_open_diff"] / df["opening_price"]

    # Price position within day's range
    features["close_position"] = (df["last_price"] - df["min_price"]) / (
        df["max_price"] - df["min_price"] + 1e-10
    )

    # Spread features
    features["bid_ask_spread"] = df["best_sell_offer"] - df["best_buy_offer"]
    features["bid_ask_spread_pct"] = features["bid_ask_spread"] / df["last_price"]

    return features


def create_returns_features(df, windows=[1, 2, 3, 5, 10, 20, 30, 60, 90]):
    """Create return-based features"""
    features = pd.DataFrame(index=df.index)

    for window in windows:
        # Simple returns
        features[f"return_{window}d"] = df["last_price"].pct_change(window)

        # Log returns
        features[f"log_return_{window}d"] = np.log(
            df["last_price"] / df["last_price"].shift(window)
        )

        # Volatility (rolling std of returns)
        features[f"volatility_{window}d"] = (
            df["last_price"].pct_change().rolling(window).std()
        )

        # Skewness and Kurtosis
        features[f"skew_{window}d"] = (
            df["last_price"].pct_change().rolling(window).skew()
        )
        features[f"kurt_{window}d"] = (
            df["last_price"].pct_change().rolling(window).kurt()
        )

    return features


def create_moving_average_features(df, windows=[5, 10, 20, 50, 200]):
    """Create moving average features"""
    features = pd.DataFrame(index=df.index)

    for window in windows:
        # Simple Moving Average
        ma = df["last_price"].rolling(window).mean()
        features[f"ma_{window}"] = ma
        features[f"price_to_ma_{window}"] = df["last_price"] / ma - 1

        # Exponential Moving Average
        ema = df["last_price"].ewm(span=window, adjust=False).mean()
        features[f"ema_{window}"] = ema
        features[f"price_to_ema_{window}"] = df["last_price"] / ema - 1

    return features


def create_volume_features(df, windows=[5, 10, 20]):
    """Create volume-based features"""
    features = pd.DataFrame(index=df.index)

    # Basic volume features
    features["volume"] = df["total_volume"]
    features["quantity"] = df["total_quantity"]
    features["avg_trade_size"] = df["total_volume"] / (df["total_trades"] + 1)

    for window in windows:
        # Volume moving averages
        features[f"volume_ma_{window}"] = df["total_volume"].rolling(window).mean()
        features[f"volume_ratio_{window}"] = df["total_volume"] / (
            features[f"volume_ma_{window}"] + 1
        )

        # Volume-weighted price
        features[f"vwap_{window}"] = (df["avg_price"] * df["total_volume"]).rolling(
            window
        ).sum() / df["total_volume"].rolling(window).sum()

    # Price-volume relationship
    features["price_volume_trend"] = df["last_price"].pct_change() * df["total_volume"]

    return features


def create_technical_indicators(df):
    """Create technical indicator features"""
    features = pd.DataFrame(index=df.index)

    # RSI (Relative Strength Index) - multiple periods
    for period in [7, 14, 21]:
        delta = df["last_price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # MACD - multiple configurations
    for fast, slow in [(12, 26), (5, 35)]:
        ema_fast = df["last_price"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["last_price"].ewm(span=slow, adjust=False).mean()
        features[f"macd_{fast}_{slow}"] = ema_fast - ema_slow
        features[f"macd_signal_{fast}_{slow}"] = (
            features[f"macd_{fast}_{slow}"].ewm(span=9, adjust=False).mean()
        )
        features[f"macd_diff_{fast}_{slow}"] = (
            features[f"macd_{fast}_{slow}"] - features[f"macd_signal_{fast}_{slow}"]
        )

    # Bollinger Bands
    ma_20 = df["last_price"].rolling(20).mean()
    std_20 = df["last_price"].rolling(20).std()
    features["bb_upper"] = ma_20 + (std_20 * 2)
    features["bb_lower"] = ma_20 - (std_20 * 2)
    features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / ma_20
    features["bb_position"] = (df["last_price"] - features["bb_lower"]) / (
        features["bb_upper"] - features["bb_lower"] + 1e-10
    )

    # Stochastic Oscillator
    low_14 = df["min_price"].rolling(14).min()
    high_14 = df["max_price"].rolling(14).max()
    features["stoch_k"] = 100 * (df["last_price"] - low_14) / (high_14 - low_14 + 1e-10)
    features["stoch_d"] = features["stoch_k"].rolling(3).mean()

    # Average True Range (ATR)
    high_low = df["max_price"] - df["min_price"]
    high_close = np.abs(df["max_price"] - df["last_price"].shift())
    low_close = np.abs(df["min_price"] - df["last_price"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features["atr_14"] = true_range.rolling(14).mean()
    features["atr_20"] = true_range.rolling(20).mean()

    # Momentum indicators
    for period in [10, 20, 30]:
        features[f"momentum_{period}"] = df["last_price"] - df["last_price"].shift(
            period
        )
        features[f"roc_{period}"] = (
            (df["last_price"] - df["last_price"].shift(period))
            / df["last_price"].shift(period)
            * 100
        )

    # Williams %R
    for period in [14, 20]:
        highest = df["max_price"].rolling(period).max()
        lowest = df["min_price"].rolling(period).min()
        features[f"williams_r_{period}"] = (
            -100 * (highest - df["last_price"]) / (highest - lowest + 1e-10)
        )

    return features


def create_temporal_features(df):
    """Create time-based features"""
    features = pd.DataFrame(index=df.index)

    features["day_of_week"] = df["trading_date"].dt.dayofweek
    features["day_of_month"] = df["trading_date"].dt.day
    features["month"] = df["trading_date"].dt.month
    features["quarter"] = df["trading_date"].dt.quarter
    features["year"] = df["trading_date"].dt.year
    features["week_of_year"] = df["trading_date"].dt.isocalendar().week

    # Cyclical encoding for day of week
    features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

    # Cyclical encoding for month
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

    return features


def create_lag_features(df, lags=[1, 2, 3, 5, 10]):
    """Create lagged price features"""
    features = pd.DataFrame(index=df.index)

    for lag in lags:
        features[f"price_lag_{lag}"] = df["last_price"].shift(lag)
        features[f"volume_lag_{lag}"] = df["total_volume"].shift(lag)
        features[f"trades_lag_{lag}"] = df["total_trades"].shift(lag)

        # High/Low lags
        features[f"high_lag_{lag}"] = df["max_price"].shift(lag)
        features[f"low_lag_{lag}"] = df["min_price"].shift(lag)

    return features


def create_all_features(filepath, output_path="features_engineered.csv"):
    """Main function to create all features"""
    print("Loading data...")
    df = load_and_prepare_data(filepath)

    print("Creating features...")
    feature_sets = []

    # Create different feature sets
    feature_sets.append(create_price_features(df))
    feature_sets.append(create_returns_features(df))
    feature_sets.append(create_moving_average_features(df))
    feature_sets.append(create_volume_features(df))
    feature_sets.append(create_technical_indicators(df))
    feature_sets.append(create_temporal_features(df))
    feature_sets.append(create_lag_features(df))

    # Combine all features
    all_features = pd.concat(feature_sets, axis=1)

    # Add original columns we want to keep
    final_df = pd.concat(
        [
            df[
                [
                    "trading_date",
                    "trading_code",
                    "company_name",
                    "last_price",
                    "total_volume",
                ]
            ],
            all_features,
        ],
        axis=1,
    )

    print(f"Total features created: {len(all_features.columns)}")
    print(f"Feature columns: {list(all_features.columns)}")

    # Handle missing values
    final_df = final_df.fillna(method="bfill").fillna(0)

    # Save to CSV
    final_df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to {output_path}")

    return final_df


def create_embeddings(features_df, embedding_size=128, output_path="embeddings.csv"):
    """Create fixed-size embeddings using PCA"""
    print("\nCreating embeddings...")

    # Select only numeric columns (excluding date and text columns)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove identifiers
    exclude_cols = ["year", "day_of_month", "week_of_year"]
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    X = features_df[numeric_cols].fillna(0)

    print(f"Number of features available: {X.shape[1]}")

    # Check if we have enough features
    if X.shape[1] < embedding_size:
        print(
            f"WARNING: Only {X.shape[1]} features available, but {embedding_size} dimensions requested."
        )
        print("Options:")
        print(f"  1. Using all {X.shape[1]} features (no dimensionality reduction)")
        print(f"  2. Padding with zeros to reach {embedding_size} dimensions")

        # Option 2: Pad with zeros
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Pad to desired size
        padding = np.zeros((X_scaled.shape[0], embedding_size - X.shape[1]))
        embeddings = np.hstack([X_scaled, padding])
        pca = None

        print(f"Using zero-padding approach. Embeddings shape: {embeddings.shape}")
    else:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA to reduce to embedding_size
        pca = PCA(n_components=embedding_size)
        embeddings = pca.fit_transform(X_scaled)

        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"Embeddings shape: {embeddings.shape}")

    # Create embeddings dataframe
    embedding_cols = [f"emb_{i}" for i in range(embedding_size)]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)

    # Add metadata
    embeddings_df = pd.concat(
        [
            features_df[
                ["trading_date", "trading_code", "company_name", "last_price"]
            ].reset_index(drop=True),
            embeddings_df,
        ],
        axis=1,
    )

    embeddings_df.to_csv(output_path, index=False)
    print(f"Embeddings saved to {output_path}")

    return embeddings_df, pca, scaler


if __name__ == "__main__":
    # Example usage
    input_file = "petr4.csv"  # Replace with your file path

    # Create engineered features
    features_df = create_all_features(input_file, "features_engineered.csv")

    # Create embeddings of size 128
    embeddings_df, pca_model, scaler_model = create_embeddings(
        features_df, embedding_size=128, output_path="embeddings_128.csv"
    )

    print("\nDone! You now have:")
    print("1. features_engineered.csv - All engineered features")
    print("2. embeddings_128.csv - 128-dimensional embeddings")
