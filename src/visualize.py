"""
Stock Price Prediction Visualization Module

This module provides comprehensive visualization capabilities for stock price predictions,
model performance analysis, and backtesting results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
import os
import sys

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure matplotlib and seaborn
plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


class StockVisualizationEngine:
    """
    A comprehensive visualization engine for stock price prediction analysis,
    model performance metrics, and backtesting results.
    """
    
    def __init__(self, plots_dir='plots'):
        """
        Initialize the visualization engine with plotting configurations.
        
        Args:
            plots_dir (str): Directory to save generated plots
        """
        self.plots_dir = plots_dir
        self.ensure_plots_directory()
        
        # Set plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def ensure_plots_directory(self):
        """Create plots directory if it doesn't exist."""
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            print(f"Created plots directory: {self.plots_dir}")
    
    def plot_stock_price_history(self, data, stock_symbol, save_plot=True, show_plot=True):
        """
        Plot historical stock price data with various indicators.
        
        Args:
            data (pd.DataFrame): Stock price data with OHLCV columns
            stock_symbol (str): Stock symbol for title
            save_plot (bool): Whether to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            str: Path to saved plot file or None
        """
        try:
            # Handle both uppercase and lowercase column names
            column_mapping = {
                'close': 'Close', 'open': 'Open', 'high': 'High', 
                'low': 'Low', 'volume': 'Volume'
            }
            
            # Create a copy of the data to avoid modifying the original
            plot_data = data.copy()
            
            # Check if we need to rename columns (if they're lowercase)
            for lowercase, uppercase in column_mapping.items():
                if lowercase in plot_data.columns and uppercase not in plot_data.columns:
                    plot_data[uppercase] = plot_data[lowercase]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{stock_symbol} Stock Price Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Price and Volume
            ax1 = axes[0, 0]
            ax1.plot(plot_data.index, plot_data['Close'], label='Close Price', color='blue', linewidth=2)
            ax1.plot(plot_data.index, plot_data['Open'], label='Open Price', color='green', alpha=0.7)
            ax1.set_title('Stock Price Over Time')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Volume
            ax2 = axes[0, 1]
            ax2.bar(plot_data.index, plot_data['Volume'], alpha=0.7, color='orange')
            ax2.set_title('Trading Volume')
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: High-Low Range
            ax3 = axes[1, 0]
            ax3.fill_between(plot_data.index, plot_data['Low'], plot_data['High'], alpha=0.3, color='purple', label='High-Low Range')
            ax3.plot(plot_data.index, plot_data['Close'], color='red', linewidth=1, label='Close Price')
            ax3.set_title('Price Range Analysis')
            ax3.set_ylabel('Price ($)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Daily Returns
            if 'Close' in plot_data.columns:
                returns = plot_data['Close'].pct_change().dropna()
                ax4 = axes[1, 1]
                ax4.hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
                ax4.set_title('Daily Returns Distribution')
                ax4.set_xlabel('Returns')
                ax4.set_ylabel('Frequency')
                ax4.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                filename = f"{self.plots_dir}/{stock_symbol}_price_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Price history plot saved: {filename}")
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename if save_plot else None
            
        except Exception as e:
            print(f"Error creating price history plot: {e}")
            return None
    
    def plot_predictions_vs_actual(self, actual_prices, predictions, dates=None, 
                                 stock_symbol="Stock", save_plot=True, show_plot=True):
        """
        Plot actual vs predicted stock prices with performance metrics.
        
        Args:
            actual_prices (list/array): Actual stock prices
            predictions (list/array): Predicted stock prices
            dates (list): Corresponding dates (optional)
            stock_symbol (str): Stock symbol for title
            save_plot (bool): Whether to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            str: Path to saved plot file or None
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{stock_symbol} Prediction Analysis', fontsize=16, fontweight='bold')
            
            # Ensure data is in numpy arrays
            actual_prices = np.array(actual_prices)
            predictions = np.array(predictions)
            
            if dates is None:
                dates = pd.date_range(start='2023-01-01', periods=len(actual_prices), freq='D')
            
            # Plot 1: Time Series Comparison
            ax1 = axes[0, 0]
            ax1.plot(dates, actual_prices, label='Actual Prices', color='blue', linewidth=2)
            ax1.plot(dates, predictions, label='Predicted Prices', color='red', linewidth=2, alpha=0.8)
            ax1.set_title('Actual vs Predicted Prices Over Time')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Scatter Plot
            ax2 = axes[0, 1]
            ax2.scatter(actual_prices, predictions, alpha=0.6, color='green')
            
            # Add perfect prediction line
            min_price, max_price = min(actual_prices.min(), predictions.min()), max(actual_prices.max(), predictions.max())
            ax2.plot([min_price, max_price], [min_price, max_price], 'r--', label='Perfect Prediction')
            
            ax2.set_xlabel('Actual Prices ($)')
            ax2.set_ylabel('Predicted Prices ($)')
            ax2.set_title('Prediction Accuracy Scatter Plot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Prediction Errors
            ax3 = axes[1, 0]
            errors = predictions - actual_prices
            ax3.plot(dates, errors, color='red', linewidth=1)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.fill_between(dates, errors, 0, alpha=0.3, color='red' if errors.mean() > 0 else 'blue')
            ax3.set_title('Prediction Errors Over Time')
            ax3.set_ylabel('Error ($)')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Error Distribution
            ax4 = axes[1, 1]
            ax4.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(errors.mean(), color='red', linestyle='--', label=f'Mean Error: ${errors.mean():.2f}')
            ax4.axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero Error')
            ax4.set_xlabel('Prediction Error ($)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Error Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                filename = f"{self.plots_dir}/{stock_symbol}_predictions_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Predictions analysis plot saved: {filename}")
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename if save_plot else None
            
        except Exception as e:
            print(f"Error creating predictions plot: {e}")
            return None
    
    def plot_model_performance_metrics(self, training_history, save_plot=True, show_plot=True):
        """
        Plot training history and model performance metrics.
        
        Args:
            training_history (dict): Dictionary containing training metrics
            save_plot (bool): Whether to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            str: Path to saved plot file or None
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Model Training Performance', fontsize=16, fontweight='bold')
            
            # Plot 1: Training vs Validation Loss
            ax1 = axes[0, 0]
            if 'loss' in training_history and 'val_loss' in training_history:
                epochs = range(1, len(training_history['loss']) + 1)
                ax1.plot(epochs, training_history['loss'], 'b-', label='Training Loss', linewidth=2)
                ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
                ax1.set_title('Model Loss During Training')
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Training history not available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Training Loss (Data Not Available)')
            
            # Plot 2: Learning Rate Schedule (if available)
            ax2 = axes[0, 1]
            if 'lr' in training_history:
                epochs = range(1, len(training_history['lr']) + 1)
                ax2.plot(epochs, training_history['lr'], 'g-', linewidth=2)
                ax2.set_title('Learning Rate Schedule')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Learning Rate')
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')
            else:
                ax2.text(0.5, 0.5, 'Learning rate data not available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Learning Rate (Data Not Available)')
            
            # Plot 3: Training Metrics Summary
            ax3 = axes[1, 0]
            if 'loss' in training_history and 'val_loss' in training_history:
                final_train_loss = training_history['loss'][-1]
                final_val_loss = training_history['val_loss'][-1]
                min_val_loss = min(training_history['val_loss'])
                
                metrics_data = [final_train_loss, final_val_loss, min_val_loss]
                metrics_labels = ['Final Train Loss', 'Final Val Loss', 'Best Val Loss']
                colors = ['blue', 'red', 'green']
                
                bars = ax3.bar(metrics_labels, metrics_data, color=colors, alpha=0.7)
                ax3.set_title('Training Metrics Summary')
                ax3.set_ylabel('Loss Value')
                
                # Add value labels on bars
                for bar, value in zip(bars, metrics_data):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_data)*0.01,
                           f'{value:.4f}', ha='center', va='bottom')
                           
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Training metrics not available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Training Metrics (Data Not Available)')
            
            # Plot 4: Convergence Analysis
            ax4 = axes[1, 1]
            if 'loss' in training_history and 'val_loss' in training_history:
                train_loss = np.array(training_history['loss'])
                val_loss = np.array(training_history['val_loss'])
                
                # Calculate moving averages
                window_size = max(1, len(train_loss) // 10)
                train_ma = pd.Series(train_loss).rolling(window=window_size).mean()
                val_ma = pd.Series(val_loss).rolling(window=window_size).mean()
                
                epochs = range(1, len(train_loss) + 1)
                ax4.plot(epochs, train_ma, 'b-', label=f'Train Loss (MA-{window_size})', linewidth=2)
                ax4.plot(epochs, val_ma, 'r-', label=f'Val Loss (MA-{window_size})', linewidth=2)
                ax4.set_title('Loss Convergence (Moving Average)')
                ax4.set_xlabel('Epochs')
                ax4.set_ylabel('Loss (Moving Average)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Convergence data not available', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Convergence Analysis (Data Not Available)')
            
            plt.tight_layout()
            
            if save_plot:
                filename = f"{self.plots_dir}/model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Model performance plot saved: {filename}")
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename if save_plot else None
            
        except Exception as e:
            print(f"Error creating model performance plot: {e}")
            return None
    
    def plot_backtesting_results(self, backtest_results, stock_symbol="Stock", 
                               save_plot=True, show_plot=True):
        """
        Visualize backtesting results with portfolio performance and metrics.
        
        Args:
            backtest_results (dict): Dictionary containing backtesting results
            stock_symbol (str): Stock symbol for title
            save_plot (bool): Whether to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            str: Path to saved plot file or None
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{stock_symbol} Backtesting Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Portfolio Value Over Time
            ax1 = axes[0, 0]
            if 'portfolio_history' in backtest_results:
                portfolio_values = backtest_results['portfolio_history']
                dates = pd.date_range(start='2023-01-01', periods=len(portfolio_values), freq='D')
                ax1.plot(dates, portfolio_values, label='Portfolio Value', color='blue', linewidth=2)
                
                if 'benchmark_history' in backtest_results:
                    benchmark_values = backtest_results['benchmark_history']
                    ax1.plot(dates, benchmark_values, label='Buy & Hold Benchmark', color='red', linewidth=2, alpha=0.7)
                
                ax1.set_title('Portfolio Performance Over Time')
                ax1.set_ylabel('Portfolio Value ($)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Portfolio history not available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Portfolio Performance (Data Not Available)')
            
            # Plot 2: Performance Metrics Comparison
            ax2 = axes[0, 1]
            if 'strategies' in backtest_results:
                strategies = backtest_results['strategies']
                strategy_names = list(strategies.keys())
                
                if strategy_names:
                    returns = [strategies[name].get('metrics', {}).get('total_return_percent', 0) for name in strategy_names]
                    colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))
                    
                    bars = ax2.bar(strategy_names, returns, color=colors, alpha=0.7)
                    ax2.set_title('Strategy Returns Comparison')
                    ax2.set_ylabel('Total Return (%)')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, returns):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(returns)*0.01,
                               f'{value:.1f}%', ha='center', va='bottom')
                    
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'Strategy data not available', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Strategy Comparison (Data Not Available)')
            else:
                ax2.text(0.5, 0.5, 'Strategy data not available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Strategy Comparison (Data Not Available)')
            
            # Plot 3: Drawdown Analysis
            ax3 = axes[1, 0]
            if 'portfolio_history' in backtest_results:
                portfolio_values = np.array(backtest_results['portfolio_history'])
                
                # Calculate running maximum and drawdown
                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - running_max) / running_max * 100
                
                dates = pd.date_range(start='2023-01-01', periods=len(drawdown), freq='D')
                ax3.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
                ax3.plot(dates, drawdown, color='red', linewidth=1)
                ax3.set_title('Portfolio Drawdown')
                ax3.set_ylabel('Drawdown (%)')
                ax3.grid(True, alpha=0.3)
                
                # Add max drawdown annotation
                max_drawdown_idx = np.argmin(drawdown)
                max_drawdown_value = drawdown[max_drawdown_idx]
                ax3.annotate(f'Max DD: {max_drawdown_value:.2f}%', 
                           xy=(dates[max_drawdown_idx], max_drawdown_value), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            else:
                ax3.text(0.5, 0.5, 'Drawdown data not available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Drawdown Analysis (Data Not Available)')
            
            # Plot 4: Risk-Return Scatter
            ax4 = axes[1, 1]
            if 'strategies' in backtest_results:
                strategies = backtest_results['strategies']
                
                returns_list = []
                volatility_list = []
                strategy_labels = []
                
                for name, data in strategies.items():
                    metrics = data.get('metrics', {})
                    if 'total_return_percent' in metrics and 'volatility' in metrics:
                        returns_list.append(metrics['total_return_percent'])
                        volatility_list.append(metrics['volatility'])
                        strategy_labels.append(name)
                
                if returns_list and volatility_list:
                    scatter = ax4.scatter(volatility_list, returns_list, 
                                        s=100, alpha=0.7, c=range(len(returns_list)), cmap='viridis')
                    
                    # Add labels for each point
                    for i, label in enumerate(strategy_labels):
                        ax4.annotate(label, (volatility_list[i], returns_list[i]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    ax4.set_xlabel('Volatility (%)')
                    ax4.set_ylabel('Returns (%)')
                    ax4.set_title('Risk-Return Analysis')
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'Risk-return data not available', ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Risk-Return Analysis (Data Not Available)')
            else:
                ax4.text(0.5, 0.5, 'Risk-return data not available', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Risk-Return Analysis (Data Not Available)')
            
            plt.tight_layout()
            
            if save_plot:
                filename = f"{self.plots_dir}/{stock_symbol}_backtesting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Backtesting results plot saved: {filename}")
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename if save_plot else None
            
        except Exception as e:
            print(f"Error creating backtesting results plot: {e}")
            return None
    
    def create_interactive_dashboard(self, stock_data, predictions=None, backtest_results=None, 
                                   stock_symbol="Stock", save_html=True):
        """
        Create an interactive dashboard using Plotly for comprehensive analysis.
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            predictions (dict): Prediction results (optional)
            backtest_results (dict): Backtesting results (optional)
            stock_symbol (str): Stock symbol for title
            save_html (bool): Whether to save as HTML file
            
        Returns:
            str: Path to saved HTML file or None
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Stock Price & Volume', 'Predictions vs Actual', 
                              'Portfolio Performance', 'Risk Metrics',
                              'Technical Indicators', 'Sentiment Analysis'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                vertical_spacing=0.08
            )
            
            # Plot 1: Stock Price and Volume
            fig.add_trace(
                go.Candlestick(x=stock_data.index,
                             open=stock_data['Open'],
                             high=stock_data['High'],
                             low=stock_data['Low'],
                             close=stock_data['Close'],
                             name="OHLC"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=stock_data.index, y=stock_data['Volume'], 
                      name="Volume", opacity=0.3),
                row=1, col=1, secondary_y=True
            )
            
            # Plot 2: Predictions (if available)
            if predictions:
                actual = predictions.get('actual', [])
                predicted = predictions.get('predicted', [])
                dates = predictions.get('dates', stock_data.index[-len(actual):] if actual else [])
                
                if actual and predicted:
                    fig.add_trace(
                        go.Scatter(x=dates, y=actual, mode='lines', 
                                 name='Actual', line=dict(color='blue')),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(x=dates, y=predicted, mode='lines', 
                                 name='Predicted', line=dict(color='red')),
                        row=1, col=2
                    )
            
            # Plot 3: Portfolio Performance (if available)
            if backtest_results and 'portfolio_history' in backtest_results:
                portfolio_values = backtest_results['portfolio_history']
                dates = pd.date_range(start='2023-01-01', periods=len(portfolio_values), freq='D')
                
                fig.add_trace(
                    go.Scatter(x=dates, y=portfolio_values, mode='lines', 
                             name='Portfolio Value', line=dict(color='green')),
                    row=2, col=1
                )
                
                if 'benchmark_history' in backtest_results:
                    benchmark_values = backtest_results['benchmark_history']
                    fig.add_trace(
                        go.Scatter(x=dates, y=benchmark_values, mode='lines', 
                                 name='Benchmark', line=dict(color='orange')),
                        row=2, col=1
                    )
            
            # Plot 4: Risk Metrics (if available)
            if backtest_results and 'strategies' in backtest_results:
                strategies = backtest_results['strategies']
                strategy_names = list(strategies.keys())
                returns = [strategies[name].get('metrics', {}).get('total_return_percent', 0) for name in strategy_names]
                
                fig.add_trace(
                    go.Bar(x=strategy_names, y=returns, name='Strategy Returns'),
                    row=2, col=2
                )
            
            # Plot 5: Technical Indicators
            if 'Close' in stock_data.columns:
                # Simple moving average
                sma_20 = stock_data['Close'].rolling(window=20).mean()
                sma_50 = stock_data['Close'].rolling(window=50).mean()
                
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=sma_20, mode='lines', 
                             name='SMA 20', line=dict(color='purple')),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=sma_50, mode='lines', 
                             name='SMA 50', line=dict(color='brown')),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', 
                             name='Close Price', line=dict(color='black')),
                    row=3, col=1
                )
            
            # Plot 6: Sentiment Analysis (placeholder)
            sentiment_dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
            sentiment_scores = np.random.uniform(-1, 1, 30)  # Random sentiment for demo
            
            fig.add_trace(
                go.Scatter(x=sentiment_dates, y=sentiment_scores, mode='lines+markers', 
                         name='Sentiment Score', line=dict(color='red')),
                row=3, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=2)
            
            # Update layout
            fig.update_layout(
                height=1200,
                title_text=f"{stock_symbol} Comprehensive Analysis Dashboard",
                title_font_size=20,
                showlegend=True
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Price ($)", row=1, col=2)
            fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
            fig.update_yaxes(title_text="Returns (%)", row=2, col=2)
            fig.update_yaxes(title_text="Price ($)", row=3, col=1)
            fig.update_yaxes(title_text="Sentiment Score", row=3, col=2)
            
            if save_html:
                filename = f"{self.plots_dir}/{stock_symbol}_interactive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                fig.write_html(filename)
                print(f"Interactive dashboard saved: {filename}")
                return filename
            
            return None
            
        except Exception as e:
            print(f"Error creating interactive dashboard: {e}")
            return None
    
    def plot_feature_importance(self, feature_names, importance_scores, 
                              model_name="Model", save_plot=True, show_plot=True):
        """
        Plot feature importance scores for the prediction model.
        
        Args:
            feature_names (list): List of feature names
            importance_scores (list): Corresponding importance scores
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            str: Path to saved plot file or None
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Sort features by importance
            sorted_idx = np.argsort(importance_scores)[::-1]
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_scores = [importance_scores[i] for i in sorted_idx]
            
            # Create horizontal bar plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
            bars = plt.barh(range(len(sorted_features)), sorted_scores, color=colors)
            
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.xlabel('Importance Score')
            plt.title(f'{model_name} Feature Importance')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
                plt.text(score + max(sorted_scores)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', va='center', ha='left')
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            if save_plot:
                filename = f"{self.plots_dir}/{model_name.lower()}_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Feature importance plot saved: {filename}")
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename if save_plot else None
            
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
            return None
    
    def create_prediction_report_visualization(self, stock_symbol, prediction_data, 
                                             model_metrics, save_plot=True, show_plot=True):
        """
        Create a comprehensive prediction report with multiple visualizations.
        
        Args:
            stock_symbol (str): Stock symbol
            prediction_data (dict): Dictionary containing prediction results
            model_metrics (dict): Model performance metrics
            save_plot (bool): Whether to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            str: Path to saved plot file or None
        """
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Main title
            fig.suptitle(f'{stock_symbol} Comprehensive Prediction Report', fontsize=20, fontweight='bold')
            
            # Plot 1: Price Prediction (spans 2 columns)
            ax1 = fig.add_subplot(gs[0, :2])
            if 'actual' in prediction_data and 'predicted' in prediction_data:
                dates = prediction_data.get('dates', range(len(prediction_data['actual'])))
                ax1.plot(dates, prediction_data['actual'], label='Actual', linewidth=2, color='blue')
                ax1.plot(dates, prediction_data['predicted'], label='Predicted', linewidth=2, color='red', alpha=0.8)
                ax1.set_title('Price Predictions vs Actual')
                ax1.set_ylabel('Price ($)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Model Metrics Summary
            ax2 = fig.add_subplot(gs[0, 2:])
            if model_metrics:
                metrics_names = list(model_metrics.keys())
                metrics_values = list(model_metrics.values())
                
                bars = ax2.bar(metrics_names, metrics_values, color=['green', 'orange', 'purple', 'red'][:len(metrics_names)])
                ax2.set_title('Model Performance Metrics')
                ax2.set_ylabel('Metric Value')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                
                for bar, value in zip(bars, metrics_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_values)*0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Prediction Accuracy Distribution
            ax3 = fig.add_subplot(gs[1, 0])
            if 'actual' in prediction_data and 'predicted' in prediction_data:
                actual = np.array(prediction_data['actual'])
                predicted = np.array(prediction_data['predicted'])
                accuracy = 1 - np.abs((predicted - actual) / actual)
                
                ax3.hist(accuracy, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax3.set_title('Prediction Accuracy Distribution')
                ax3.set_xlabel('Accuracy Ratio')
                ax3.set_ylabel('Frequency')
                ax3.axvline(accuracy.mean(), color='red', linestyle='--', label=f'Mean: {accuracy.mean():.3f}')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Error Analysis
            ax4 = fig.add_subplot(gs[1, 1])
            if 'actual' in prediction_data and 'predicted' in prediction_data:
                errors = np.array(prediction_data['predicted']) - np.array(prediction_data['actual'])
                ax4.boxplot(errors)
                ax4.set_title('Prediction Error Distribution')
                ax4.set_ylabel('Error ($)')
                ax4.grid(True, alpha=0.3)
            
            # Plot 5: Directional Accuracy
            ax5 = fig.add_subplot(gs[1, 2])
            if 'actual' in prediction_data and 'predicted' in prediction_data and len(prediction_data['actual']) > 1:
                actual = np.array(prediction_data['actual'])
                predicted = np.array(prediction_data['predicted'])
                
                actual_direction = np.diff(actual) > 0
                predicted_direction = np.diff(predicted) > 0
                directional_accuracy = np.mean(actual_direction == predicted_direction)
                
                ax5.pie([directional_accuracy, 1-directional_accuracy], 
                       labels=['Correct Direction', 'Wrong Direction'],
                       colors=['green', 'red'], autopct='%1.1f%%')
                ax5.set_title(f'Directional Accuracy: {directional_accuracy:.1%}')
            
            # Plot 6: Volatility Analysis
            ax6 = fig.add_subplot(gs[1, 3])
            if 'actual' in prediction_data:
                actual_prices = np.array(prediction_data['actual'])
                returns = np.diff(actual_prices) / actual_prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
                
                ax6.plot(returns, alpha=0.7, color='blue')
                ax6.set_title(f'Price Volatility (σ={volatility:.1%})')
                ax6.set_ylabel('Daily Returns')
                ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax6.grid(True, alpha=0.3)
            
            # Plot 7: Model Confidence (if available)
            ax7 = fig.add_subplot(gs[2, :2])
            if 'confidence' in prediction_data:
                confidence_scores = prediction_data['confidence']
                dates = prediction_data.get('dates', range(len(confidence_scores)))
                ax7.fill_between(dates, confidence_scores, alpha=0.3, color='purple')
                ax7.plot(dates, confidence_scores, color='purple', linewidth=2)
                ax7.set_title('Model Prediction Confidence')
                ax7.set_ylabel('Confidence Score')
                ax7.grid(True, alpha=0.3)
            else:
                ax7.text(0.5, 0.5, 'Confidence data not available', ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title('Model Confidence (Data Not Available)')
            
            # Plot 8: Summary Statistics
            ax8 = fig.add_subplot(gs[2, 2:])
            if 'actual' in prediction_data and 'predicted' in prediction_data:
                actual = np.array(prediction_data['actual'])
                predicted = np.array(prediction_data['predicted'])
                
                stats_data = {
                    'Mean Actual': np.mean(actual),
                    'Mean Predicted': np.mean(predicted),
                    'Std Actual': np.std(actual),
                    'Std Predicted': np.std(predicted),
                    'Min Actual': np.min(actual),
                    'Max Actual': np.max(actual)
                }
                
                stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
                
                # Create table
                table_data = []
                for _, row in stats_df.iterrows():
                    table_data.append([row['Metric'], f"{row['Value']:.2f}"])
                
                ax8.axis('tight')
                ax8.axis('off')
                table = ax8.table(cellText=table_data, colLabels=['Statistic', 'Value'],
                                cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                ax8.set_title('Summary Statistics')
            
            if save_plot:
                filename = f"{self.plots_dir}/{stock_symbol}_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Comprehensive prediction report saved: {filename}")
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return filename if save_plot else None
            
        except Exception as e:
            print(f"Error creating comprehensive prediction report: {e}")
            return None


def main():
    """
    Demonstration function showing how to use the StockVisualizationEngine class.
    """
    print("StockVisualizationEngine module loaded successfully!")
    print("This module provides comprehensive visualization capabilities for stock price prediction analysis.")
    
    # Example of creating a visualization engine
    viz_engine = StockVisualizationEngine(plots_dir='plots')
    print(f"Example visualization engine created with plots directory: {viz_engine.plots_dir}")
    
    # Example data for demonstration
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(150, 250, len(dates)),
        'Low': np.random.uniform(50, 150, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    print("Sample visualization capabilities:")
    print("- plot_stock_price_history(): Historical price analysis")
    print("- plot_predictions_vs_actual(): Model prediction accuracy")
    print("- plot_model_performance_metrics(): Training performance")
    print("- plot_backtesting_results(): Trading strategy evaluation")
    print("- create_interactive_dashboard(): Interactive Plotly dashboard")
    print("- plot_feature_importance(): Model feature analysis")
    print("- create_prediction_report_visualization(): Comprehensive reports")


if __name__ == "__main__":
    main()
