import numpy as np
from config import STRATEGY_SETTINGS

class OptimalStopping:
    """Optimal stopping theory applied to gambling (Secretary Problem)"""

    def __init__(self):
        self.strategies = STRATEGY_SETTINGS['optimal_stopping']['strategies']

    def calculate_stopping_points(self, historical_data, bankroll, strategy='moderate'):
        """Calculate optimal stop-profit and stop-loss points"""
        if len(historical_data) < 100:
            return self._default_stopping_points(bankroll, strategy)

        data = np.array(historical_data)

        # Calculate statistical properties
        mean_return = np.mean(data - 1)  # Subtract 1 for net return
        volatility = np.std(data - 1)
        max_win = np.max(data)
        max_loss = np.min(data)

        # Strategy-specific calculations
        if strategy == 'conservative':
            stop_profit_factor = 1.5  # 50% gain
            stop_loss_factor = 0.9    # 10% loss
            confidence_level = 0.95
        elif strategy == 'moderate':
            stop_profit_factor = 2.0  # 100% gain
            stop_loss_factor = 0.8    # 20% loss
            confidence_level = 0.90
        elif strategy == 'aggressive':
            stop_profit_factor = 3.0  # 200% gain
            stop_loss_factor = 0.7    # 30% loss
            confidence_level = 0.80
        else:
            return self._default_stopping_points(bankroll, strategy)

        # Calculate stop levels
        stop_profit = bankroll * stop_profit_factor
        stop_loss = bankroll * stop_loss_factor

        # Calculate value-at-risk for stop loss
        var_95 = np.percentile(data - 1, 5)  # 5th percentile for 95% VaR
        expected_shortfall = np.mean((data - 1)[(data - 1) <= var_95])

        return {
            'strategy': strategy,
            'stop_profit': round(stop_profit, 2),
            'stop_loss': round(stop_loss, 2),
            'current_bankroll': bankroll,
            'profit_target': round(stop_profit - bankroll, 2),
            'loss_limit': round(bankroll - stop_loss, 2),
            'confidence_level': confidence_level,
            'var_95': round(var_95, 4),
            'expected_shortfall': round(expected_shortfall, 4),
            'recommended_action': self._get_stopping_recommendation(bankroll, stop_profit, stop_loss)
        }

    def _default_stopping_points(self, bankroll, strategy):
        """Default stopping points when insufficient data"""
        defaults = {
            'conservative': {'profit': 1.2, 'loss': 0.95},
            'moderate': {'profit': 1.5, 'loss': 0.9},
            'aggressive': {'profit': 2.0, 'loss': 0.8}
        }

        params = defaults.get(strategy, defaults['moderate'])

        return {
            'strategy': strategy,
            'stop_profit': round(bankroll * params['profit'], 2),
            'stop_loss': round(bankroll * params['loss'], 2),
            'current_bankroll': bankroll,
            'profit_target': round(bankroll * (params['profit'] - 1), 2),
            'loss_limit': round(bankroll * (1 - params['loss']), 2),
            'confidence_level': 0.5,
            'warning': 'Using default values due to insufficient data',
            'recommended_action': 'Monitor closely and adjust based on experience'
        }

    def _get_stopping_recommendation(self, current_bankroll, stop_profit, stop_loss):
        """Get recommendation based on current position"""
        if current_bankroll >= stop_profit:
            return "STOP - Profit target reached"
        elif current_bankroll <= stop_loss:
            return "STOP - Loss limit reached"
        else:
            distance_to_profit = (stop_profit - current_bankroll) / current_bankroll
            distance_to_loss = (current_bankroll - stop_loss) / current_bankroll

            if distance_to_profit < distance_to_loss:
                return f"Continue - Closer to profit target ({distance_to_profit:.1%})"
            else:
                return f"Caution - Closer to loss limit ({distance_to_loss:.1%})"

    def monte_carlo_strategy_comparison(self, starting_bankroll, strategies_data):
        """Compare different stopping strategies using Monte Carlo"""
        results = {}

        for strategy in self.strategies:
            simulations = []

            for _ in range(STRATEGY_SETTINGS['optimal_stopping']['simulations']):
                bankroll = starting_bankroll
                rounds = 0
                max_rounds = 1000

                while rounds < max_rounds and bankroll > 0:
                    # Simulate a round (simplified random walk)
                    return_pct = np.random.normal(0.01, 0.2)  # 1% mean, 20% volatility
                    bankroll *= (1 + return_pct)
                    rounds += 1

                    # Check stopping conditions
                    stop_points = self.calculate_stopping_points([], bankroll, strategy)

                    if bankroll >= stop_points['stop_profit'] or bankroll <= stop_points['stop_loss']:
                        break

                simulations.append({
                    'final_bankroll': bankroll,
                    'rounds': rounds,
                    'bust': bankroll <= 0
                })

            # Analyze results
            final_bankrolls = [s['final_bankroll'] for s in simulations]
            bust_rate = np.mean([s['bust'] for s in simulations])

            results[strategy] = {
                'avg_final_bankroll': round(np.mean(final_bankrolls), 2),
                'median_final_bankroll': round(np.median(final_bankrolls), 2),
                'bust_rate': round(bust_rate, 4),
                'avg_rounds': round(np.mean([s['rounds'] for s in simulations]), 1),
                'win_rate': round(np.mean([s['final_bankroll'] > starting_bankroll for s in simulations]), 4)
            }

        return results

    def find_optimal_stopping_ratio(self, historical_data, bankroll):
        """Find optimal profit/loss ratio using historical data"""
        if len(historical_data) < 50:
            return {'error': 'Insufficient data'}

        data = np.array(historical_data)
        returns = data - 1  # Net returns

        # Test different profit/loss ratios
        profit_ratios = np.arange(1.1, 3.1, 0.1)  # 10% to 200% profit targets
        loss_ratios = np.arange(0.5, 0.95, 0.05)  # 5% to 50% loss limits

        best_ratio = None
        best_performance = -float('inf')

        for profit_ratio in profit_ratios:
            for loss_ratio in loss_ratios:
                # Simulate strategy
                performance = self._simulate_stopping_strategy(returns, profit_ratio, loss_ratio, bankroll)
                if performance > best_performance:
                    best_performance = performance
                    best_ratio = (profit_ratio, loss_ratio)

        if best_ratio:
            profit_ratio, loss_ratio = best_ratio
            return {
                'optimal_profit_ratio': round(profit_ratio, 2),
                'optimal_loss_ratio': round(loss_ratio, 2),
                'profit_target': round(bankroll * profit_ratio, 2),
                'loss_limit': round(bankroll * loss_ratio, 2),
                'expected_performance': round(best_performance, 4)
            }
        else:
            return {'error': 'Could not find optimal ratio'}

    def _simulate_stopping_strategy(self, returns, profit_ratio, loss_ratio, starting_bankroll):
        """Simulate a stopping strategy and return performance metric"""
        bankroll = starting_bankroll
        trades = 0
        wins = 0

        for return_val in returns:
            if bankroll <= 0:
                break

            # Calculate position size (simplified)
            position = min(bankroll * 0.1, bankroll)  # Max 10% of bankroll

            # Apply return
            pnl = position * return_val
            bankroll += pnl
            trades += 1

            if pnl > 0:
                wins += 1

            # Check stopping conditions
            if bankroll >= starting_bankroll * profit_ratio:
                # Hit profit target
                return (bankroll - starting_bankroll) / starting_bankroll  # Return percentage
            elif bankroll <= starting_bankroll * loss_ratio:
                # Hit loss limit
                return (bankroll - starting_bankroll) / starting_bankroll  # Return percentage

        # If we get here, strategy didn't hit stop conditions
        return (bankroll - starting_bankroll) / starting_bankroll