import numpy as np
from config import STRATEGY_SETTINGS
from strategies.kelly import KellyOptimizer
from strategies.optimal_stopping import OptimalStopping
from strategies.session_simulator import SessionSimulator

class StrategyComparator:
    """Compare different gambling strategies head-to-head"""

    def __init__(self):
        self.kelly = KellyOptimizer()
        self.stopping = OptimalStopping()
        self.simulator = SessionSimulator()
        self.strategies = STRATEGY_SETTINGS['comparator']['strategies']

    def compare_strategies(self, game_type, starting_bankroll, historical_data=None):
        """Compare 7+ strategies head-to-head"""
        strategies = self._define_strategies(game_type, starting_bankroll, historical_data)

        results = {}
        simulations = STRATEGY_SETTINGS['comparator']['simulations']

        for strategy_name, strategy_config in strategies.items():
            print(f"Simulating {strategy_name}...")

            if game_type == 'crash':
                sim_result = self.simulator.simulate_crash_sessions(
                    starting_bankroll,
                    strategy_config.get('cashout', 2.0),
                    strategy_config.get('bet_fraction', 0.1)
                )
            elif game_type == 'dice':
                sim_result = self.simulator.simulate_dice_sessions(
                    starting_bankroll,
                    strategy_config.get('target', 50),
                    strategy_config.get('bet_fraction', 0.1)
                )
            elif game_type == 'slots':
                sim_result = self.simulator.simulate_slot_sessions(
                    starting_bankroll,
                    strategy_config.get('bet_size', 1.0),
                    strategy_config.get('rtp', 0.95)
                )
            else:
                continue

            results[strategy_name] = {
                'config': strategy_config,
                'results': sim_result
            }

        # Rank strategies
        rankings = self._rank_strategies(results)

        return {
            'game_type': game_type,
            'starting_bankroll': starting_bankroll,
            'simulations_per_strategy': simulations,
            'strategy_results': results,
            'rankings': rankings,
            'best_strategy': rankings[0] if rankings else None,
            'worst_strategy': rankings[-1] if rankings else None
        }

    def _define_strategies(self, game_type, starting_bankroll, historical_data):
        """Define the strategies to compare"""
        strategies = {}

        if game_type == 'crash':
            # Strategy 1: Conservative Kelly
            kelly_result = self.kelly.optimize_crash_betting(historical_data or [], starting_bankroll)
            strategies['Conservative Kelly'] = {
                'cashout': kelly_result.get('optimal_cashout', 2.0),
                'bet_fraction': min(kelly_result.get('kelly_fraction', 0.05) / 2, 0.05),
                'description': 'Half of optimal Kelly bet size'
            }

            # Strategy 2: Martingale
            strategies['Martingale'] = {
                'cashout': 2.0,
                'bet_fraction': 0.01,  # Start small, double on loss
                'progression': 'martingale',
                'description': 'Double bet after each loss'
            }

            # Strategy 3: Flat Betting
            strategies['Flat Bet'] = {
                'cashout': 2.0,
                'bet_fraction': 0.05,
                'description': 'Fixed 5% of bankroll per bet'
            }

            # Strategy 4: Aggressive
            strategies['Aggressive'] = {
                'cashout': 10.0,
                'bet_fraction': 0.1,
                'description': 'High cashout, high risk'
            }

            # Strategy 5: Conservative
            strategies['Conservative'] = {
                'cashout': 1.5,
                'bet_fraction': 0.02,
                'description': 'Low cashout, low risk'
            }

            # Strategy 6: Fibonacci
            strategies['Fibonacci'] = {
                'cashout': 2.0,
                'bet_fraction': 0.01,
                'progression': 'fibonacci',
                'description': 'Fibonacci progression on losses'
            }

            # Strategy 7: Random
            strategies['Random'] = {
                'cashout': np.random.uniform(1.1, 5.0),
                'bet_fraction': np.random.uniform(0.01, 0.1),
                'description': 'Random parameters'
            }

        elif game_type == 'dice':
            strategies['Over 50'] = {'target': 50, 'bet_fraction': 0.05}
            strategies['Under 25'] = {'target': 25, 'bet_fraction': 0.05}
            strategies['Over 75'] = {'target': 75, 'bet_fraction': 0.05}

        elif game_type == 'slots':
            strategies['Penny Slots'] = {'bet_size': 0.01, 'rtp': 0.94}
            strategies['Quarter Slots'] = {'bet_size': 0.25, 'rtp': 0.95}
            strategies['Dollar Slots'] = {'bet_size': 1.0, 'rtp': 0.96}

        return strategies

    def _rank_strategies(self, results):
        """Rank strategies by performance"""
        strategy_scores = []

        for strategy_name, data in results.items():
            result = data['results']

            # Scoring criteria (higher is better)
            score = 0

            # Primary: Median final bankroll
            median_bankroll = result.get('median_final_bankroll', 0)
            score += median_bankroll * 0.4

            # Secondary: Survival rate (avoiding bust)
            survival_rate = result.get('survival_rate', 0)
            score += survival_rate * 1000  # Weight heavily

            # Tertiary: Profit rate
            profit_rate = result.get('profit_rate', 0)
            score += profit_rate * 500

            # Penalty for high bust rate
            bust_rate = result.get('bust_rate', 0)
            score -= bust_rate * 2000

            strategy_scores.append({
                'strategy': strategy_name,
                'score': score,
                'median_bankroll': median_bankroll,
                'survival_rate': survival_rate,
                'profit_rate': profit_rate,
                'bust_rate': bust_rate
            })

        # Sort by score (descending)
        strategy_scores.sort(key=lambda x: x['score'], reverse=True)

        return strategy_scores

    def get_strategy_details(self, strategy_name, results):
        """Get detailed analysis of a specific strategy"""
        if strategy_name not in results:
            return None

        data = results[strategy_name]
        result = data['results']

        # Calculate additional metrics
        avg_final = result.get('average_final_bankroll', 0)
        median_final = result.get('median_final_bankroll', 0)
        bust_rate = result.get('bust_rate', 0)
        profit_rate = result.get('profit_rate', 0)

        # Risk assessment
        if bust_rate > 0.5:
            risk = 'EXTREME'
        elif bust_rate > 0.3:
            risk = 'HIGH'
        elif bust_rate > 0.1:
            risk = 'MODERATE'
        else:
            risk = 'LOW'

        # Performance assessment
        if median_final > 100:  # Assuming starting bankroll of 100
            performance = 'EXCELLENT'
        elif median_final > 80:
            performance = 'GOOD'
        elif median_final > 50:
            performance = 'FAIR'
        else:
            performance = 'POOR'

        return {
            'strategy': strategy_name,
            'config': data['config'],
            'metrics': {
                'average_final_bankroll': avg_final,
                'median_final_bankroll': median_final,
                'bust_rate': bust_rate,
                'survival_rate': 1 - bust_rate,
                'profit_rate': profit_rate,
                'average_rounds': result.get('average_rounds', 0)
            },
            'risk_assessment': risk,
            'performance_rating': performance,
            'recommendation': self._get_strategy_recommendation(risk, performance)
        }

    def _get_strategy_recommendation(self, risk, performance):
        """Get recommendation for a strategy"""
        if risk == 'EXTREME':
            return "NOT RECOMMENDED - Too risky"
        elif risk == 'HIGH' and performance in ['POOR', 'FAIR']:
            return "USE WITH CAUTION - High risk, low reward"
        elif performance == 'EXCELLENT':
            return "RECOMMENDED - Good performance"
        elif performance == 'GOOD':
            return "CONSIDER - Decent performance"
        else:
            return "AVOID - Poor risk-reward ratio"

    def compare_with_optimal_stopping(self, game_type, starting_bankroll, historical_data):
        """Compare strategies with different stopping rules"""
        base_strategies = self._define_strategies(game_type, starting_bankroll, historical_data)

        stopping_strategies = ['conservative', 'moderate', 'aggressive']
        comparison_results = {}

        for stop_strategy in stopping_strategies:
            # Modify base strategies with stopping rules
            modified_strategies = {}
            for name, config in base_strategies.items():
                modified_config = config.copy()
                # Add stopping parameters
                stop_params = self.stopping.calculate_stopping_points(
                    historical_data or [], starting_bankroll, stop_strategy
                )
                modified_config['stop_profit'] = stop_params.get('stop_profit', starting_bankroll * 2)
                modified_config['stop_loss'] = stop_params.get('stop_loss', starting_bankroll * 0.5)
                modified_strategies[f"{name} + {stop_strategy.title()} Stop"] = modified_config

            # Run comparison
            results = {}
            for strategy_name, strategy_config in modified_strategies.items():
                if game_type == 'crash':
                    sim_result = self.simulator.simulate_crash_sessions(
                        starting_bankroll,
                        strategy_config.get('cashout', 2.0),
                        strategy_config.get('bet_fraction', 0.1)
                    )
                    results[strategy_name] = sim_result

            comparison_results[stop_strategy] = results

        return comparison_results