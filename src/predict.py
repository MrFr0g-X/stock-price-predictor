"""
Stock Price Prediction Module
This module handles making predictions using the trained LSTM model,
including real-time predictions and future price forecasting.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from the same package
from src.data_collection import StockDataCollector
from src.preprocessing import StockDataPreprocessor
from src.model import StockPriceLSTM


class StockPricePredictor:
    """
    A comprehensive class for making stock price predictions using the trained LSTM model.
    This class handles both single predictions and batch predictions.
    """
    
    def __init__(self, stock_symbol, model_path="models/lstm_model.h5"):
        """
        Initialize the predictor with a stock symbol and model path.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            model_path (str): Path to the trained LSTM model
        """
        self.stock_symbol = stock_symbol.upper()
        self.model_path = model_path
        self.lstm_model = None
        self.preprocessor = None
        self.sequence_length = 60
        
        logger.info(f"Stock price predictor initialized for {self.stock_symbol}")
    
    def load_trained_model(self):
        """
        Load the pre-trained LSTM model from disk.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                return False
            
            # Initialize LSTM model instance
            self.lstm_model = StockPriceLSTM(sequence_length=self.sequence_length)
            self.lstm_model.model_save_path = self.model_path  # Set the correct path
            
            # Load the trained model (no arguments)
            success = self.lstm_model.load_model()
            
            if success:
                logger.info("LSTM model loaded successfully")
            else:
                logger.error("Failed to load LSTM model")
            
            return success
            
        except Exception as error_message:
            logger.error(f"Error loading model: {error_message}")
            return False
    
    def prepare_prediction_data(self, stock_data, sentiment_data=None):
        """
        Prepare data for prediction using the same preprocessing pipeline.
        
        Args:
            stock_data (pd.DataFrame): Recent stock data for prediction
            sentiment_data (dict): Optional sentiment data
        
        Returns:
            np.array: Preprocessed data ready for prediction
        """
        try:
            logger.info("Preparing data for prediction")
            
            if stock_data is None or len(stock_data) < self.sequence_length:
                logger.error(f"Insufficient data. Need at least {self.sequence_length} data points")
                return None
            
            # Initialize preprocessor
            self.preprocessor = StockDataPreprocessor()
            
            # Clean the data
            cleaned_data = self.preprocessor.clean_stock_data(stock_data)
            if cleaned_data is None:
                return None
            
            # Add technical indicators
            enhanced_data = self.preprocessor.add_technical_indicators(cleaned_data)
            
            # Add sentiment features
            market_sentiment = 0.0
            if sentiment_data and 'sentiment_score' in sentiment_data:
                market_sentiment = sentiment_data.get('sentiment_score', 0.0)
            
            # Use add_sentiment_features instead of combine_with_sentiment
            combined_data = self.preprocessor.add_sentiment_features(enhanced_data, market_sentiment)
            
            # Normalize features
            normalized_data, _ = self.preprocessor.normalize_features(combined_data)
            if normalized_data is None:
                return None
            
            # Create sequences for prediction (only the last sequence)
            if len(normalized_data) >= self.sequence_length:
                # Take the last sequence_length rows for prediction
                last_sequence = normalized_data[-self.sequence_length:].values
                prediction_data = last_sequence.reshape(1, self.sequence_length, -1)
                
                logger.info(f"Prediction data prepared with shape: {prediction_data.shape}")
                return prediction_data
            else:
                logger.error("Not enough data after preprocessing")
                return None
            
        except Exception as error_message:
            logger.error(f"Error preparing prediction data: {error_message}")
            return None
    
    def make_single_prediction(self, stock_data, sentiment_data=None):
        """
        Make a single price prediction for the next day.
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            sentiment_data (dict): Optional sentiment analysis data
        
        Returns:
            dict: Prediction results with confidence metrics
        """
        try:
            logger.info("Making single stock price prediction")
            print("DEBUG: Starting single prediction")
            
            # Load model if not already loaded
            if self.lstm_model is None:
                print("DEBUG: Model not loaded, loading now...")
                if not self.load_trained_model():
                    print("DEBUG: Failed to load model")
                    return None
            
            # Prepare data for prediction
            print("DEBUG: Preparing prediction data...")
            prediction_data = self.prepare_prediction_data(stock_data, sentiment_data)
            if prediction_data is None:
                print("DEBUG: Prediction data preparation failed")
                return None
            
            # Make prediction using the model
            print(f"DEBUG: Making prediction with data shape {prediction_data.shape}")
            normalized_prediction = self.lstm_model.predict(prediction_data)
            
            if normalized_prediction is None:
                print("DEBUG: Model prediction returned None")
                logger.error("Model prediction failed")
                return None
            
            print(f"DEBUG: Got normalized prediction: {normalized_prediction}")
            
            # Transform prediction back to original scale
            print("DEBUG: Inverse transforming prediction...")
            # Use 'close' - the inverse_transform_predictions method is now case-insensitive
            actual_prediction = self.preprocessor.inverse_transform_predictions(normalized_prediction, feature_name='close')
            
            if actual_prediction is None or len(actual_prediction) == 0:
                print("DEBUG: Inverse transform failed")
                logger.error("Failed to transform prediction to original scale")
                return None
            
            print(f"DEBUG: Actual prediction: {actual_prediction}")
            
            # Get the current price for comparison
            current_price = stock_data['close'].iloc[-1]
            predicted_price = actual_prediction[0]
            
            # Calculate price change and percentage change
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Create prediction result
            prediction_result = {
                'stock_symbol': self.stock_symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': price_change,
                'price_change_percent': price_change_percent,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'confidence': self._calculate_confidence(stock_data, predicted_price)
            }
            
            logger.info(f"Prediction completed: ${predicted_price:.2f} ({price_change_percent:+.2f}%)")
            print(f"DEBUG: Prediction completed successfully")
            return prediction_result
            
        except Exception as error_message:
            print(f"DEBUG: Error in make_single_prediction: {error_message}")
            logger.error(f"Error making single prediction: {error_message}")
            import traceback
            traceback.print_exc()
            return None
    
    def make_multiple_predictions(self, stock_data, sentiment_data=None, days_ahead=5):
        """
        Make predictions for multiple days into the future.
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            sentiment_data (dict): Optional sentiment analysis data
            days_ahead (int): Number of days to predict into the future
        
        Returns:
            list: List of prediction results for each day
        """
        try:
            logger.info(f"Making predictions for {days_ahead} days ahead")
            
            # Load model if not already loaded
            if self.lstm_model is None:
                if not self.load_trained_model():
                    return None
            
            predictions_list = []
            current_data = stock_data.copy()
            
            for day in range(days_ahead):
                logger.info(f"Predicting day {day + 1}/{days_ahead}")
                
                # Prepare data for this prediction
                prediction_data = self.prepare_prediction_data(current_data, sentiment_data)
                if prediction_data is None:
                    break
                
                # Make prediction
                normalized_prediction = self.lstm_model.predict(prediction_data)
                if normalized_prediction is None:
                    break
                
                # Transform back to original scale
                actual_prediction = self.preprocessor.inverse_transform_predictions(normalized_prediction)
                if actual_prediction is None or len(actual_prediction) == 0:
                    break
                
                predicted_price = actual_prediction[0]
                current_price = current_data['close'].iloc[-1]
                
                # Calculate changes
                price_change = predicted_price - current_price
                price_change_percent = (price_change / current_price) * 100
                
                # Create prediction result
                target_date = datetime.now() + timedelta(days=day + 1)
                prediction_result = {
                    'day': day + 1,
                    'stock_symbol': self.stock_symbol,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'price_change': price_change,
                    'price_change_percent': price_change_percent,
                    'target_date': target_date.strftime('%Y-%m-%d'),
                    'confidence': self._calculate_confidence(current_data, predicted_price)
                }
                
                predictions_list.append(prediction_result)
                
                # Update data for next prediction (add predicted day as new data point)
                new_row = current_data.iloc[-1].copy()
                new_row['close'] = predicted_price
                new_row['open'] = predicted_price  # Simplified assumption
                new_row['high'] = predicted_price * 1.02  # Assume 2% intraday range
                new_row['low'] = predicted_price * 0.98
                
                # Add new row to current_data
                new_df = pd.DataFrame([new_row])
                current_data = pd.concat([current_data, new_df], ignore_index=True)
            
            logger.info(f"Generated {len(predictions_list)} predictions")
            return predictions_list
            
        except Exception as error_message:
            logger.error(f"Error making multiple predictions: {error_message}")
            return None
    
    def get_real_time_prediction(self, period="3mo"):
        """
        Fetch the latest data and make a real-time prediction.
        
        Args:
            period (str): Period of historical data to fetch
        
        Returns:
            dict: Real-time prediction result
        """
        try:
            logger.info("Generating real-time prediction")
            
            # Collect fresh data
            data_collector = StockDataCollector(self.stock_symbol)
            stock_data, sentiment_data = data_collector.get_combined_data(period=period)
            
            if stock_data is None:
                logger.error("Failed to fetch fresh stock data")
                return None
            
            # Make prediction with fresh data
            prediction_result = self.make_single_prediction(stock_data, sentiment_data)
            
            if prediction_result:
                prediction_result['data_freshness'] = 'real-time'
                prediction_result['data_period'] = period
            
            return prediction_result
            
        except Exception as error_message:
            logger.error(f"Error making real-time prediction: {error_message}")
            return None
    
    def _calculate_confidence(self, stock_data, predicted_price):
        """
        Calculate a confidence score for the prediction based on historical volatility.
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            predicted_price (float): The predicted price
        
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            # Calculate recent volatility
            recent_prices = stock_data['close'].tail(30)  # Last 30 days
            volatility = recent_prices.std() / recent_prices.mean()
            
            # Calculate how close prediction is to recent price trend
            current_price = stock_data['close'].iloc[-1]
            price_change_ratio = abs(predicted_price - current_price) / current_price
            
            # Confidence decreases with volatility and large price changes
            confidence = max(0.1, 1.0 - (volatility * 2) - (price_change_ratio * 0.5))
            return min(1.0, confidence)
            
        except Exception:
            return 0.5  # Default moderate confidence
    
    def save_predictions(self, predictions, filename=None):
        """
        Save predictions to a CSV file.
        
        Args:
            predictions (list or dict): Prediction results
            filename (str): Optional filename for saving
        """
        try:
            if not predictions:
                logger.warning("No predictions to save")
                return
            
            # Convert single prediction to list
            if isinstance(predictions, dict):
                predictions = [predictions]
            
            # Create DataFrame
            predictions_df = pd.DataFrame(predictions)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"predictions_{self.stock_symbol}_{timestamp}.csv"
            
            # Save to CSV
            predictions_df.to_csv(filename, index=False)
            logger.info(f"Predictions saved to {filename}")
            
        except Exception as error_message:
            logger.error(f"Error saving predictions: {error_message}")


def main():
    """
    Main function for testing the prediction module.
    """
    print("Stock Price Prediction Module")
    print("=" * 50)
    
    # Example usage
    stock_symbol = "AAPL"
    predictor = StockPricePredictor(stock_symbol)
    
    print(f"Testing predictions for {stock_symbol}")
    
    # Test real-time prediction (requires trained model)
    try:
        prediction = predictor.get_real_time_prediction()
        if prediction:
            print("\nReal-time Prediction:")
            print(f"Current Price: ${prediction['current_price']:.2f}")
            print(f"Predicted Price: ${prediction['predicted_price']:.2f}")
            print(f"Change: {prediction['price_change_percent']:+.2f}%")
            print(f"Confidence: {prediction['confidence']:.2f}")
        else:
            print("Real-time prediction failed (model may not be trained yet)")
    except Exception as e:
        print(f"Prediction test failed: {e}")
        print("This is expected if the model hasn't been trained yet")


if __name__ == "__main__":
    main()
