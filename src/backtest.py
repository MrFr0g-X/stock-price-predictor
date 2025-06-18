"""
Stock Price Prediction Backtesting Module

This module implements comprehensive backtesting strategies for stock price predictions,
including performance metrics calculation and strategy evaluation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingStrategyBacktester:
    """
    A comprehensive backtesting engine for trading strategies based on stock price predictions.
    
    This class implements various trading strategies and calculates performance metrics
    including returns, Sharpe ratio, maximum drawdown, and other risk metrics.
    """
    
    def __init__(self, initial_capital=10000, commission_rate=0.001):
        """
        Initialize the backtester with trading parameters.
        
        Args:
            initial_capital (float): Initial capital for trading
            commission_rate (float): Commission rate per trade (default: 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.trades = []
        self.portfolio_history = []
        
        logger.info(f"Backtester initialized with ${initial_capital:,.2f} initial capital")
    
    def simple_prediction_strategy(self, actual_prices, predicted_prices, dates=None, 
                                 threshold=0.02, max_position=0.8):
        """
        Implement a simple trading strategy based on price predictions.
        
        Args:
            actual_prices (list/array): Actual stock prices
            predicted_prices (list/array): Predicted stock prices
            dates (list): Corresponding dates (optional)
            threshold (float): Minimum price change threshold for trading (default: 2%)
            max_position (float): Maximum portfolio allocation to stock (default: 80%)
        
        Returns:
            dict: Backtesting results with performance metrics
        """
        try:
            logger.info("Running simple prediction-based trading strategy")
            
            # Convert to numpy arrays
            actual_prices = np.array(actual_prices)
            predicted_prices = np.array(predicted_prices)
            
            if dates is None:
                dates = pd.date_range(start='2023-01-01', periods=len(actual_prices), freq='D')
            
            # Initialize tracking variables
            self.portfolio_value = self.initial_capital
            self.cash = self.initial_capital
            self.shares = 0
            self.trades = []
            self.portfolio_history = []
            
            # Track daily portfolio values
            daily_values = []
            daily_returns = []
            
            for i in range(len(actual_prices)):
                current_price = actual_prices[i]
                current_date = dates[i]
                
                # Calculate predicted price change
                if i < len(predicted_prices):
                    predicted_price = predicted_prices[i]
                    price_change_pct = (predicted_price - current_price) / current_price
                else:
                    price_change_pct = 0
                
                # Trading logic
                if i > 0:  # Skip first day (no prediction available)
                    # Buy signal: predicted increase above threshold
                    if price_change_pct > threshold and self.cash > 0:
                        # Calculate maximum shares we can buy
                        max_shares_to_buy = int((self.cash * max_position) / current_price)
                        if max_shares_to_buy > 0:
                            cost = max_shares_to_buy * current_price
                            commission = cost * self.commission_rate
                            total_cost = cost + commission
                            
                            if total_cost <= self.cash:
                                self.shares += max_shares_to_buy
                                self.cash -= total_cost
                                
                                self.trades.append({
                                    'date': current_date,
                                    'action': 'BUY',
                                    'shares': max_shares_to_buy,
                                    'price': current_price,
                                    'commission': commission,
                                    'reason': f'Predicted {price_change_pct:.2%} increase'
                                })
                    
                    # Sell signal: predicted decrease below negative threshold
                    elif price_change_pct < -threshold and self.shares > 0:
                        # Sell all shares
                        revenue = self.shares * current_price
                        commission = revenue * self.commission_rate
                        net_revenue = revenue - commission
                        
                        self.trades.append({
                            'date': current_date,
                            'action': 'SELL',
                            'shares': self.shares,
                            'price': current_price,
                            'commission': commission,
                            'reason': f'Predicted {price_change_pct:.2%} decrease'
                        })
                        
                        self.cash += net_revenue
                        self.shares = 0
                
                # Calculate current portfolio value
                current_portfolio_value = self.cash + (self.shares * current_price)
                self.portfolio_history.append({
                    'date': current_date,
                    'portfolio_value': current_portfolio_value,
                    'cash': self.cash,
                    'shares': self.shares,
                    'stock_price': current_price,
                    'predicted_change': price_change_pct if i < len(predicted_prices) else 0
                })
                
                daily_values.append(current_portfolio_value)
                
                # Calculate daily return
                if i > 0:
                    daily_return = (current_portfolio_value - daily_values[i-1]) / daily_values[i-1]
                    daily_returns.append(daily_return)
                else:
                    daily_returns.append(0)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                daily_values, daily_returns, actual_prices, dates
            )
            
            # Create results dictionary
            results = {
                'strategy_name': 'Simple Prediction Strategy',
                'initial_capital': self.initial_capital,
                'final_portfolio_value': daily_values[-1],
                'total_return': performance_metrics['total_return'],
                'annualized_return': performance_metrics['annualized_return'],
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'max_drawdown': performance_metrics['max_drawdown'],
                'volatility': performance_metrics['volatility'],
                'num_trades': len(self.trades),
                'win_rate': performance_metrics['win_rate'],
                'trades': self.trades,
                'portfolio_history': self.portfolio_history,
                'daily_values': daily_values,
                'daily_returns': daily_returns
            }
            
            logger.info(f"Strategy completed: {performance_metrics['total_return']:.2%} total return")
            return results
            
        except Exception as error_message:
            logger.error(f"Error in simple prediction strategy: {error_message}")
            return None
    
    def buy_and_hold_strategy(self, actual_prices, dates=None):
        """
        Implement a buy-and-hold strategy for comparison.
        
        Args:
            actual_prices (list/array): Actual stock prices
            dates (list): Corresponding dates (optional)
        
        Returns:
            dict: Buy-and-hold strategy results
        """
        try:
            logger.info("Running buy-and-hold strategy for comparison")
            
            actual_prices = np.array(actual_prices)
            
            if dates is None:
                dates = pd.date_range(start='2023-01-01', periods=len(actual_prices), freq='D')
            
            # Buy all shares on first day
            initial_price = actual_prices[0]
            shares_bought = int(self.initial_capital / initial_price)
            cost = shares_bought * initial_price
            commission = cost * self.commission_rate
            total_cost = cost + commission
            
            # Calculate final value
            final_price = actual_prices[-1]
            final_value = shares_bought * final_price
            commission_sell = final_value * self.commission_rate
            net_final_value = final_value - commission_sell
            
            # Calculate daily values
            daily_values = []
            daily_returns = []
            
            for i, price in enumerate(actual_prices):
                current_value = shares_bought * price
                daily_values.append(current_value)
                
                if i > 0:
                    daily_return = (current_value - daily_values[i-1]) / daily_values[i-1]
                    daily_returns.append(daily_return)
                else:
                    daily_returns.append(0)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                daily_values, daily_returns, actual_prices, dates
            )
            
            results = {
                'strategy_name': 'Buy and Hold Strategy',
                'initial_capital': self.initial_capital,
                'final_portfolio_value': net_final_value,
                'total_return': performance_metrics['total_return'],
                'annualized_return': performance_metrics['annualized_return'],
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'max_drawdown': performance_metrics['max_drawdown'],
                'volatility': performance_metrics['volatility'],
                'num_trades': 2,  # Buy and sell
                'win_rate': 1.0 if net_final_value > self.initial_capital else 0.0,
                'trades': [
                    {
                        'date': dates[0],
                        'action': 'BUY',
                        'shares': shares_bought,
                        'price': initial_price,
                        'commission': commission,
                        'reason': 'Initial buy-and-hold purchase'
                    },
                    {
                        'date': dates[-1],
                        'action': 'SELL',
                        'shares': shares_bought,
                        'price': final_price,
                        'commission': commission_sell,
                        'reason': 'Final sell of buy-and-hold strategy'
                    }
                ],
                'daily_values': daily_values,
                'daily_returns': daily_returns
            }
            
            logger.info(f"Buy-and-hold completed: {performance_metrics['total_return']:.2%} total return")
            return results
            
        except Exception as error_message:
            logger.error(f"Error in buy-and-hold strategy: {error_message}")
            return None
    
    def _calculate_performance_metrics(self, daily_values, daily_returns, actual_prices, dates):
        """
        Calculate comprehensive performance metrics for the strategy.
        
        Args:
            daily_values (list): Daily portfolio values
            daily_returns (list): Daily returns
            actual_prices (list): Actual stock prices
            dates (list): Corresponding dates
        
        Returns:
            dict: Performance metrics
        """
        try:
            daily_values = np.array(daily_values)
            daily_returns = np.array(daily_returns)
            
            # Basic metrics
            total_return = (daily_values[-1] - self.initial_capital) / self.initial_capital
            
            # Annualized return
            if len(dates) > 1:
                days = (dates[-1] - dates[0]).days
                years = days / 365.25
                annualized_return = ((daily_values[-1] / self.initial_capital) ** (1/years)) - 1
            else:
                annualized_return = total_return
            
            # Volatility (annualized)
            volatility = np.std(daily_returns) * np.sqrt(252)  # Assuming daily data
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_returns = daily_returns - (risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            
            # Maximum drawdown
            peak = daily_values[0]
            max_drawdown = 0
            for value in daily_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Win rate (simplified - positive vs negative returns)
            positive_days = np.sum(daily_returns > 0)
            win_rate = positive_days / len(daily_returns) if len(daily_returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
        except Exception as error_message:
            logger.error(f"Error calculating performance metrics: {error_message}")
            return {
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
    
    def compare_strategies(self, actual_prices, predicted_prices, dates=None):
        """
        Compare multiple trading strategies and return comprehensive results.
        
        Args:
            actual_prices (list/array): Actual stock prices
            predicted_prices (list/array): Predicted stock prices
            dates (list): Corresponding dates (optional)
        
        Returns:
            dict: Comparison results for all strategies
        """
        try:
            logger.info("Comparing multiple trading strategies")
            
            # Generate dates if not provided
            if dates is None:
                dates = pd.date_range(start='2023-01-01', periods=len(actual_prices), freq='D')
                
            # Run different strategies
            prediction_strategy = self.simple_prediction_strategy(actual_prices, predicted_prices, dates)
            buy_hold_strategy = self.buy_and_hold_strategy(actual_prices, dates)
            
            # Extract portfolio values for visualization
            if prediction_strategy and 'daily_values' in prediction_strategy:
                prediction_portfolio_values = prediction_strategy['daily_values']
            else:
                prediction_portfolio_values = [self.initial_capital] * len(actual_prices)
                
            if buy_hold_strategy and 'daily_values' in buy_hold_strategy:
                buyhold_portfolio_values = buy_hold_strategy['daily_values']
            else:
                buyhold_portfolio_values = [self.initial_capital] * len(actual_prices)
            
            # Calculate strategy metrics for visualization
            strategies = {}
            if prediction_strategy:
                strategies['Prediction Strategy'] = {
                    'metrics': {
                        'total_return_percent': prediction_strategy['total_return'] * 100,
                        'volatility': prediction_strategy['volatility'] * 100,
                        'sharpe_ratio': prediction_strategy['sharpe_ratio'],
                        'max_drawdown': prediction_strategy['max_drawdown'] * 100
                    }
                }
                
            if buy_hold_strategy:
                strategies['Buy & Hold'] = {
                    'metrics': {
                        'total_return_percent': buy_hold_strategy['total_return'] * 100,
                        'volatility': buy_hold_strategy['volatility'] * 100,
                        'sharpe_ratio': buy_hold_strategy['sharpe_ratio'],
                        'max_drawdown': buy_hold_strategy['max_drawdown'] * 100
                    }
                }
            
            # Create comparison summary
            comparison = {
                'prediction_strategy': prediction_strategy,
                'buy_hold_strategy': buy_hold_strategy,
                'portfolio_history': prediction_portfolio_values,
                'benchmark_history': buyhold_portfolio_values,
                'dates': dates,
                'strategies': strategies,
                'summary': {
                    'prediction_vs_buyhold': {
                        'return_difference': prediction_strategy['total_return'] - buy_hold_strategy['total_return'],
                        'sharpe_difference': prediction_strategy['sharpe_ratio'] - buy_hold_strategy['sharpe_ratio'],
                        'max_drawdown_difference': prediction_strategy['max_drawdown'] - buy_hold_strategy['max_drawdown']
                    }
                }
            }
            
            logger.info("Strategy comparison completed")
            return comparison
            
        except Exception as error_message:
            logger.error(f"Error comparing strategies: {error_message}")
            return None


def main():
    """
    Main function to test the backtesting module.
    """
    print("Testing Backtesting Module...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    actual_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    predicted_prices = actual_prices + np.random.randn(100) * 2
    
    # Initialize backtester
    backtester = TradingStrategyBacktester(initial_capital=10000)
    
    # Run strategy comparison
    results = backtester.compare_strategies(actual_prices, predicted_prices, dates)
    
    if results:
        print("\nBacktesting Results:")
        print("="*50)
        
        for strategy_name, strategy_results in results.items():
            if strategy_name != 'summary':
                print(f"\n{strategy_results['strategy_name']}:")
                print(f"  Total Return: {strategy_results['total_return']:.2%}")
                print(f"  Annualized Return: {strategy_results['annualized_return']:.2%}")
                print(f"  Sharpe Ratio: {strategy_results['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {strategy_results['max_drawdown']:.2%}")
                print(f"  Number of Trades: {strategy_results['num_trades']}")
        
        print(f"\nPrediction Strategy vs Buy & Hold:")
        print(f"  Return Difference: {results['summary']['prediction_vs_buyhold']['return_difference']:.2%}")
        print(f"  Sharpe Difference: {results['summary']['prediction_vs_buyhold']['sharpe_difference']:.3f}")
    
    print("\nBacktesting module test completed!")


if __name__ == "__main__":
    main()
