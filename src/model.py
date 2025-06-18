"""
LSTM Model Module for Stock Price Prediction

This module defines, trains, and manages the LSTM neural network for stock price prediction.
Features include:
- LSTM architecture design optimized for time series
- Training with validation and early stopping
- Model evaluation and metrics
- Model saving and loading
- Comprehensive error handling
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import logging
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPriceLSTM:
    """
    LSTM Neural Network for Stock Price Prediction.
    
    This class implements a sophisticated LSTM architecture designed specifically
    for stock price prediction with proper regularization and optimization.
    """
    
    def __init__(self, sequence_length=60, num_features=20, model_save_path="models/lstm_model.h5"):
        """
        Initialize the LSTM model.
        
        Args:
            sequence_length (int): Length of input sequences
            num_features (int): Number of input features
            model_save_path (str): Path to save the trained model
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model_save_path = model_save_path
        self.model = None
        self.training_history = None
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        logger.info(f"LSTM model initialized: seq_len={sequence_length}, features={num_features}")
    
    def build_lstm_architecture(self):
        """
        Build the LSTM neural network architecture.
        
        Returns:
            tensorflow.keras.Model: Compiled LSTM model
        """
        logger.info("Building LSTM architecture...")
        
        try:
            # Clear any existing model
            tf.keras.backend.clear_session()
            
            # Input layer
            inputs = keras.Input(shape=(self.sequence_length, self.num_features), name='input_layer')
            
            # First LSTM layer with dropout
            x = layers.LSTM(
                units=128,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_1'
            )(inputs)
            
            # Batch normalization
            x = layers.BatchNormalization(name='batch_norm_1')(x)
            
            # Second LSTM layer
            x = layers.LSTM(
                units=64,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_2'
            )(x)
            
            # Batch normalization
            x = layers.BatchNormalization(name='batch_norm_2')(x)
            
            # Third LSTM layer (no return_sequences as we want final output)
            x = layers.LSTM(
                units=32,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_3'
            )(x)
            
            # Dense layers with regularization
            x = layers.Dense(64, activation='relu', name='dense_1')(x)
            x = layers.Dropout(0.3, name='dropout_1')(x)
            
            x = layers.Dense(32, activation='relu', name='dense_2')(x)
            x = layers.Dropout(0.2, name='dropout_2')(x)
            
            x = layers.Dense(16, activation='relu', name='dense_3')(x)
            
            # Output layer
            outputs = layers.Dense(1, activation='linear', name='output_layer')(x)
            
            # Create model
            self.model = keras.Model(inputs=inputs, outputs=outputs, name='stock_price_lstm')
            
            # Compile model with appropriate optimizer and loss
            optimizer = keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
            
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            # Print model summary
            self.model.summary()
            
            logger.info("LSTM architecture built and compiled successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to build LSTM architecture: {e}")
            raise
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model with validation and early stopping.
        
        Args:
            X_train (np.ndarray): Training input sequences
            y_train (np.ndarray): Training target values
            X_test (np.ndarray): Testing input sequences
            y_test (np.ndarray): Testing target values
            epochs (int): Maximum number of training epochs
            batch_size (int): Training batch size
            validation_split (float): Fraction of training data to use for validation
            
        Returns:
            dict: Training history
        """
        logger.info(f"Starting model training: epochs={epochs}, batch_size={batch_size}")
        
        try:
            if self.model is None:
                raise ValueError("Model not built. Call build_lstm_architecture() first.")
            
            # Callbacks for better training
            callbacks = [
                # Early stopping to prevent overfitting
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                
                # Reduce learning rate when loss plateaus
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                
                # Save best model during training
                keras.callbacks.ModelCheckpoint(
                    filepath=self.model_save_path.replace('.h5', '_best.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train the model
            logger.info("Training started...")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Important for time series data
            )
            
            # Store training history
            self.training_history = history.history
            self.is_trained = True
            
            # Evaluate on test set
            test_loss, test_mae, test_mape = self.model.evaluate(X_test, y_test, verbose=0)
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Final test loss: {test_loss:.6f}")
            logger.info(f"Final test MAE: {test_mae:.6f}")
            logger.info(f"Final test MAPE: {test_mape:.2f}%")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def evaluate_model_performance(self, X_test, y_test):
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            X_test (np.ndarray): Test input sequences
            y_test (np.ndarray): Test target values
            
        Returns:
            dict: Performance metrics
        """
        logger.info("Evaluating model performance...")
        
        try:
            if self.model is None:
                raise ValueError("Model not trained. Train the model first.")
            
            # Make predictions
            y_pred = self.model.predict(X_test, verbose=0)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate directional accuracy
            y_test_direction = np.diff(y_test.flatten())
            y_pred_direction = np.diff(y_pred.flatten())
            directional_accuracy = np.mean(np.sign(y_test_direction) == np.sign(y_pred_direction)) * 100
            
            # Calculate MAPE manually
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'num_predictions': len(y_test)
            }
            
            # Log results
            logger.info("Model Performance Metrics:")
            logger.info(f"  MSE: {mse:.6f}")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  MAE: {mae:.6f}")
            logger.info(f"  R² Score: {r2:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")
            logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def save_model(self):
        """
        Save the trained model to disk.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False
            
            # Save the model
            self.model.save(self.model_save_path)
            
            # Save training history if available
            if self.training_history:
                history_path = self.model_save_path.replace('.h5', '_history.npy')
                np.save(history_path, self.training_history)
            
            logger.info(f"Model saved successfully to {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self):
        """
        Load a previously trained model from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_save_path):
                logger.warning(f"Model file not found: {self.model_save_path}")
                return False
            
            # Load the model
            self.model = keras.models.load_model(self.model_save_path)
            self.is_trained = True
            
            # Try to load training history
            history_path = self.model_save_path.replace('.h5', '_history.npy')
            if os.path.exists(history_path):
                self.training_history = np.load(history_path, allow_pickle=True).item()
            
            logger.info(f"Model loaded successfully from {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, X_input):
        """
        Make predictions using the trained model.
        
        Args:
            X_input (np.ndarray): Input sequences for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Load or train a model first.")
            
            predictions = self.model.predict(X_input, verbose=0)
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history (loss and metrics).
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.training_history is None:
            logger.warning("No training history available")
            return None
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot training and validation loss
            axes[0, 0].plot(self.training_history['loss'], label='Training Loss')
            axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot training and validation MAE
            axes[0, 1].plot(self.training_history['mae'], label='Training MAE')
            axes[0, 1].plot(self.training_history['val_mae'], label='Validation MAE')
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot training and validation MAPE
            axes[1, 0].plot(self.training_history['mape'], label='Training MAPE')
            axes[1, 0].plot(self.training_history['val_mape'], label='Validation MAPE')
            axes[1, 0].set_title('Mean Absolute Percentage Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAPE (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot learning rate (if available)
            if 'lr' in self.training_history:
                axes[1, 1].plot(self.training_history['lr'])
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Learning Rate')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to plot training history: {e}")
            return None


def main():
    """
    Test the LSTM model with sample data.
    """
    logger.info("Testing LSTM Model...")
    
    try:
        # Create sample data for testing
        sequence_length = 30
        num_features = 10
        num_samples = 1000
        
        # Generate synthetic time series data
        X_sample = np.random.randn(num_samples, sequence_length, num_features)
        y_sample = np.random.randn(num_samples, 1)
        
        # Split into train/test
        split_idx = int(0.8 * num_samples)
        X_train = X_sample[:split_idx]
        y_train = y_sample[:split_idx]
        X_test = X_sample[split_idx:]
        y_test = y_sample[split_idx:]
        
        # Initialize and build model
        lstm_model = StockPriceLSTM(
            sequence_length=sequence_length,
            num_features=num_features,
            model_save_path="test_lstm_model.h5"
        )
        
        # Build architecture
        lstm_model.build_lstm_architecture()
        
        # Train model (reduced epochs for testing)
        history = lstm_model.train_model(
            X_train, y_train, X_test, y_test,
            epochs=5, batch_size=32
        )
        
        # Evaluate model
        metrics = lstm_model.evaluate_model_performance(X_test, y_test)
        
        # Test save/load
        lstm_model.save_model()
        
        print(f"✅ LSTM model test successful!")
        print(f"   Model architecture: {lstm_model.model.count_params()} parameters")
        print(f"   Training history: {len(history['loss'])} epochs")
        print(f"   Test R² score: {metrics['r2_score']:.4f}")
        
        # Clean up test file
        if os.path.exists("test_lstm_model.h5"):
            os.remove("test_lstm_model.h5")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM model test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
