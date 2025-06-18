"""
Data Preprocessing Module for Stock Price Prediction

This module handles all data preprocessing tasks including:
- Data cleaning and normalization
- Technical indicator calculation
- Feature engineering
- LSTM sequence preparation
- Train/test data splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataPreprocessor:
    """
    Comprehensive data preprocessing class for stock price prediction.
    
    This class handles all preprocessing steps required to prepare stock data
    for LSTM neural network training, including technical indicators, normalization,
    and sequence creation.
    """
    
    def __init__(self, sequence_length=60, test_size=0.2):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            sequence_length (int): Number of days to use for LSTM sequences
            test_size (float): Proportion of data to use for testing
        """
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        
        logger.info(f"Preprocessor initialized: sequence_length={sequence_length}, test_size={test_size}")
    
    def clean_stock_data(self, stock_data):
        """
        Clean and prepare raw stock data.
        
        Args:
            stock_data (pd.DataFrame): Raw stock data from yfinance
            
        Returns:
            pd.DataFrame: Cleaned stock data
        """
        logger.info("Cleaning stock data...")
        
        # Make a copy to avoid modifying original data
        cleaned_data = stock_data.copy()
        
        # Forward fill missing values
        cleaned_data = cleaned_data.fillna(method='ffill')
        
        # Remove any remaining NaN values
        cleaned_data = cleaned_data.dropna()
        
        # Check for required columns (handle both uppercase and lowercase)
        required_columns_lower = ['open', 'high', 'low', 'close', 'volume']
        required_columns_upper = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if we have lowercase columns
        missing_columns_lower = [col for col in required_columns_lower if col not in cleaned_data.columns]
        missing_columns_upper = [col for col in required_columns_upper if col not in cleaned_data.columns]
        
        if missing_columns_lower and missing_columns_upper:
            raise ValueError(f"Missing required columns. Lowercase missing: {missing_columns_lower}, Uppercase missing: {missing_columns_upper}")
        
        # If we have lowercase columns, convert to uppercase for consistency
        if not missing_columns_lower:
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            cleaned_data = cleaned_data.rename(columns=column_mapping)
        
        logger.info(f"Data cleaned: {len(cleaned_data)} rows, {len(cleaned_data.columns)} columns")
        return cleaned_data
    
    def add_technical_indicators(self, data):
        """
        Add comprehensive technical indicators to the dataset.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with technical indicators added
        """
        logger.info("Adding technical indicators...")
        
        df = data.copy()
        
        # Price-based indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Price Rate of Change
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        logger.info(f"Technical indicators added: {len(df.columns)} total columns")
        return df
    
    def add_sentiment_features(self, data, market_sentiment):
        """
        Add sentiment analysis features to the dataset.
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            market_sentiment (float): Overall market sentiment score
            
        Returns:
            pd.DataFrame: Data with sentiment features added
        """
        logger.info("Adding sentiment features...")
        
        df = data.copy()
        
        # Add market sentiment as a feature
        df['Market_Sentiment'] = market_sentiment
        
        # Create sentiment-based features
        df['Sentiment_MA'] = df['Market_Sentiment'].rolling(window=5).mean()
        df['Sentiment_Volatility'] = df['Market_Sentiment'].rolling(window=10).std().fillna(0)
        
        logger.info("Sentiment features added")
        return df
    
    def normalize_features(self, data):
        """
        Normalize features for neural network training.
        
        Args:
            data (pd.DataFrame): Data with all features
            
        Returns:
            tuple: (normalized_data, feature_columns)
        """
        logger.info("Normalizing features...")
        
        # Remove non-numeric columns and handle NaN values
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill')
        
        # Store feature column names
        self.feature_columns = numeric_data.columns.tolist()
        
        # Normalize the data
        normalized_array = self.scaler.fit_transform(numeric_data)
        normalized_data = pd.DataFrame(normalized_array, columns=self.feature_columns, index=numeric_data.index)
        
        logger.info(f"Features normalized: {len(self.feature_columns)} features")
        return normalized_data, self.feature_columns
    
    def create_lstm_sequences(self, data, target_column='Close'):
        """
        Create sequences for LSTM training.
        
        Args:
            data (pd.DataFrame): Normalized data
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        logger.info("Creating LSTM sequences...")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Get target column index
        target_index = data.columns.get_loc(target_column)
        
        X_sequences = []
        y_sequences = []
        
        # Create sequences
        for i in range(self.sequence_length, len(data)):
            # Use all features for the sequence
            X_sequences.append(data.iloc[i-self.sequence_length:i].values)
            # Predict the target column value
            y_sequences.append(data.iloc[i, target_index])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"LSTM sequences created: X={X_sequences.shape}, y={y_sequences.shape}")
        return X_sequences, y_sequences
    
    def split_train_test(self, X_sequences, y_sequences):
        """
        Split sequences into training and testing sets.
        
        Args:
            X_sequences (np.ndarray): Input sequences
            y_sequences (np.ndarray): Target sequences
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train/test sets...")
        
        # Use train_test_split for random splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences, y_sequences, 
            test_size=self.test_size, 
            random_state=42,
            shuffle=False  # Keep time series order
        )
        
        logger.info(f"Data split: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_complete_pipeline(self, stock_data, market_sentiment=0.0):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            stock_data (pd.DataFrame): Raw stock data
            market_sentiment (float): Market sentiment score
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        try:
            # Step 1: Clean data
            cleaned_data = self.clean_stock_data(stock_data)
            
            # Step 2: Add technical indicators
            enhanced_data = self.add_technical_indicators(cleaned_data)
            
            # Step 3: Add sentiment features
            sentiment_data = self.add_sentiment_features(enhanced_data, market_sentiment)
            
            # Step 4: Normalize features
            normalized_data, feature_names = self.normalize_features(sentiment_data)
            
            # Step 5: Create LSTM sequences
            X_sequences, y_sequences = self.create_lstm_sequences(normalized_data)
            
            # Step 6: Split train/test
            X_train, X_test, y_train, y_test = self.split_train_test(X_sequences, y_sequences)
            
            logger.info("Complete preprocessing pipeline finished successfully")
            return X_train, X_test, y_train, y_test, feature_names
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            raise
    
    def inverse_transform_predictions(self, predictions, feature_name='Close'):
        """
        Convert normalized predictions back to original scale.
        
        Args:
            predictions (np.ndarray): Normalized predictions
            feature_name (str): Name of the feature to inverse transform (case-insensitive)
            
        Returns:
            np.ndarray: Predictions in original scale
        """
        # Try to find the feature name in a case-insensitive way
        feature_index = None
        for i, col in enumerate(self.feature_columns):
            if col.lower() == feature_name.lower():
                feature_index = i
                feature_name = col  # Use the actual column name with correct case
                break
                
        if feature_index is None:
            raise ValueError(f"Feature '{feature_name}' not found in preprocessed data. Available features: {self.feature_columns}")
        
        # Create a dummy array with the same shape as the original features
        dummy_array = np.zeros((len(predictions), len(self.feature_columns)))
        dummy_array[:, feature_index] = predictions.flatten()
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy_array)
        
        # Return only the feature of interest
        return inverse_transformed[:, feature_index]


def main():
    """
    Test the preprocessing module with sample data.
    """
    logger.info("Testing Stock Data Preprocessor...")
    
    try:
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(150, 250, len(dates)),
            'Low': np.random.uniform(50, 150, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Add some trend to make it more realistic
        for i in range(1, len(sample_data)):
            sample_data.iloc[i] = sample_data.iloc[i-1] * (1 + np.random.normal(0, 0.02))
        
        # Initialize preprocessor
        preprocessor = StockDataPreprocessor(sequence_length=30, test_size=0.2)
        
        # Test complete pipeline
        X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_complete_pipeline(
            sample_data, market_sentiment=0.1
        )
        
        print(f"✅ Preprocessing test successful!")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Testing data shape: {X_test.shape}")
        print(f"   Number of features: {len(feature_names)}")
        print(f"   Features: {feature_names[:5]}... (showing first 5)")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
