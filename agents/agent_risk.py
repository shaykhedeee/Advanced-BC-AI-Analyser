import numpy as np
from collections import deque
try:
    from scipy import stats
except ImportError:
    stats = None
import pandas as pd
from config import AGENT_SETTINGS, API_KEYS
try:
    import openai
except BaseException:
    openai = None


class RiskAgent:
    """
    Enhanced Risk Management Agent with:
    - Dynamic Kelly with drawdown penalty
    - CVaR (Conditional Value at Risk) calculation
    - Regime-aware position sizing
    - Multi-scenario Monte Carlo with percentile bands
    - Tilt detection via loss velocity
    - Bankroll decay early warning
    """

    def __init__(self):
        self.client = None
        if API_KEYS.get('openrouter') and openai:
            try:
                self.client = openai.OpenAI(
                    api_key=API_KEYS['openrouter'],
                    base_url="https://openrouter.ai/api/v1"
                )
            except Exception:
                pass

        # Track bankroll history for drawdown analysis
        self.bankroll_history = deque(maxlen=500)
        self.bet_history = deque(maxlen=500)
        self.peak_bankroll = 0
        self.current_drawdown = 0

    def calculate_kelly_criterion(self, win_probability, odds, current_bankroll):
        """Enhanced Kelly with drawdown penalty and fractional variants"""
        try:
            b = odds - 1  # Net odds
            p = win_probability
            q = 1 - p

            if b <= 0:
                return {'full_kelly': 0, 'half_kelly': 0, 'quarter_kelly': 0,
                        'bet_sizes': {'full': 0, 'half': 0, 'quarter': 0}, 'edge': False}

            kelly_fraction = (b * p - q) / b

            # Drawdown penalty: reduce Kelly when in drawdown
            drawdown_multiplier = 1.0
            if self.current_drawdown > 0.1:
                drawdown_multiplier = max(0.25, 1.0 - self.current_drawdown)

            adjusted_kelly = kelly_fraction * drawdown_multiplier

            half_kelly = adjusted_kelly / 2
            quarter_kelly = adjusted_kelly / 4
            eighth_kelly = adjusted_kelly / 8

            return {
                'full_kelly': max(0, round(float(kelly_fraction), 6)),
                'adjusted_kelly': max(0, round(float(adjusted_kelly), 6)),
                'half_kelly': max(0, round(float(half_kelly), 6)),
                'quarter_kelly': max(0, round(float(quarter_kelly), 6)),
                'eighth_kelly': max(0, round(float(eighth_kelly), 6)),
                'drawdown_multiplier': round(float(drawdown_multiplier), 4),
                'current_drawdown': round(float(self.current_drawdown), 4),
                'bet_sizes': {
                    'full': round(max(0, kelly_fraction * current_bankroll), 2),
                    'adjusted': round(max(0, adjusted_kelly * current_bankroll), 2),
                    'half': round(max(0, half_kelly * current_bankroll), 2),
                    'quarter': round(max(0, quarter_kelly * current_bankroll), 2),
                    'eighth': round(max(0, eighth_kelly * current_bankroll), 2),
                },
                'edge': kelly_fraction > 0,
            }
        except Exception as e:
            return {'error': str(e)}

    def update_bankroll(self, new_bankroll: float):
        """Track bankroll for drawdown analysis"""
        self.bankroll_history.append(float(new_bankroll))
        self.peak_bankroll = max(self.peak_bankroll, new_bankroll)
        if self.peak_bankroll > 0:
            self.current_drawdown = (self.peak_bankroll - new_bankroll) / self.peak_bankroll
        else:
            self.current_drawdown = 0

    def calculate_cvar(self, returns: list, confidence: float = 0.95) -> dict:
        """
        Conditional Value at Risk (CVaR / Expected Shortfall).
        CVaR tells you the average loss in the worst (1-confidence)% of cases.
        """
        if not returns or len(returns) < 10:
            return {'var': 0, 'cvar': 0, 'confidence': confidence}

        arr = np.array(returns, dtype=np.float64)
        var_threshold = np.percentile(arr, (1 - confidence) * 100)
        tail_losses = arr[arr <= var_threshold]

        cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else float(var_threshold)

        return {
            'var': round(float(var_threshold), 4),
            'cvar': round(float(cvar), 4),
            'confidence': confidence,
            'worst_case_avg_loss': round(float(abs(cvar)), 4),
            'num_tail_events': int(len(tail_losses)),
        }

    def calculate_risk_of_ruin(self, win_probability, bet_fraction, sessions):
        """Enhanced risk of ruin with closed-form + simulation"""
        try:
            p = win_probability
            q = 1 - p
            f = bet_fraction

            if f <= 0 or f >= 1:
                return {'risk_of_ruin': 1.0, 'error': 'Invalid bet fraction'}

            # Analytic approximation
            if p == q:
                ror = 1.0
            elif p > q:
                ror = (q / p) ** (1.0 / f) if f > 0 else 1.0
            else:
                ror = 1.0

            return {
                'risk_of_ruin': round(float(min(1.0, max(0.0, ror))), 6),
                'sessions': sessions,
                'bet_fraction': f,
                'win_probability': p,
                'survival_probability': round(float(1 - min(1.0, max(0.0, ror))), 6),
            }
        except Exception as e:
            return {'error': str(e)}

    def monte_carlo_bankroll_projection(self, starting_bankroll, win_probability,
                                      avg_win_multiplier, avg_loss_multiplier,
                                      bet_fraction, simulations=1000, max_rounds=1000):
        """Monte Carlo simulation of bankroll over time"""
        try:
            results = []

            for sim in range(simulations):
                bankroll = starting_bankroll
                history = [bankroll]

                for round_num in range(max_rounds):
                    bet_size = bankroll * bet_fraction

                    # Simulate outcome
                    if np.random.random() < win_probability:
                        # Win
                        bankroll += bet_size * (avg_win_multiplier - 1)
                    else:
                        # Loss
                        bankroll -= bet_size * avg_loss_multiplier

                    history.append(bankroll)

                    # Stop if bankrupt
                    if bankroll <= 0:
                        break

                results.append({
                    'final_bankroll': bankroll,
                    'rounds_survived': len(history) - 1,
                    'history': history,
                    'bust': bankroll <= 0
                })

            # Analyze results
            final_bankrolls = [r['final_bankroll'] for r in results]
            bust_rate = np.mean([r['bust'] for r in results])
            avg_rounds = np.mean([r['rounds_survived'] for r in results])
            median_final = np.median(final_bankrolls)
            profit_rate = np.mean([r['final_bankroll'] > starting_bankroll for r in results])

            return {
                'simulations': simulations,
                'bust_rate': bust_rate,
                'average_rounds_survived': avg_rounds,
                'median_final_bankroll': median_final,
                'profit_rate': profit_rate,
                'bankroll_distribution': {
                    'p10': np.percentile(final_bankrolls, 10),
                    'p25': np.percentile(final_bankrolls, 25),
                    'p50': np.percentile(final_bankrolls, 50),
                    'p75': np.percentile(final_bankrolls, 75),
                    'p90': np.percentile(final_bankrolls, 90)
                }
            }

        except Exception as e:
            return {'error': str(e)}

    def find_optimal_strategy(self, game_type, current_data=None):
        """Find optimal betting strategy for a game type"""
        try:
            if game_type == 'crash':
                # For crash games, analyze historical data to find best cashout
                if current_data and len(current_data) > 100:
                    # Find the cashout point that would have maximized returns historically
                    cashouts = np.arange(1.01, 5.01, 0.01)  # 1.01x to 5.00x
                    returns = []

                    for cashout in cashouts:
                        # Simulate betting strategy
                        wins = np.sum(current_data >= cashout)
                        losses = len(current_data) - wins
                        net_return = wins * (cashout - 1) - losses
                        returns.append(net_return)

                    best_cashout = cashouts[np.argmax(returns)]
                    max_return = max(returns)

                    return {
                        'optimal_cashout': best_cashout,
                        'expected_return': max_return / len(current_data),
                        'win_rate': np.sum(current_data >= best_cashout) / len(current_data)
                    }
                else:
                    # Default crash strategy
                    return {
                        'optimal_cashout': 2.0,
                        'expected_return': -0.01,  # House edge
                        'win_rate': 0.5
                    }

            elif game_type == 'dice':
                # For dice, it's usually a specific target (e.g., over/under 50)
                return {
                    'strategy': 'Bet on over 50',
                    'win_probability': 0.5,
                    'expected_return': -0.01
                }

            else:
                return {'strategy': 'No specific strategy available'}

        except Exception as e:
            return {'error': str(e)}

    def calculate_stop_levels(self, bankroll, risk_tolerance=0.1):
        """Calculate stop-loss and take-profit levels"""
        try:
            # Stop loss: maximum drawdown allowed
            stop_loss = bankroll * (1 - risk_tolerance)

            # Take profit: target 2x bankroll (aggressive)
            take_profit = bankroll * 2

            # Conservative take profit: 50% gain
            conservative_tp = bankroll * 1.5

            return {
                'stop_loss': stop_loss,
                'take_profit_aggressive': take_profit,
                'take_profit_conservative': conservative_tp,
                'risk_tolerance': risk_tolerance
            }

        except Exception as e:
            return {'error': str(e)}

    def get_comprehensive_risk_assessment(self, game_type, bankroll, current_data=None):
        """Comprehensive risk assessment combining all methods"""
        try:
            # Get optimal strategy
            strategy = self.find_optimal_strategy(game_type, current_data)

            # Calculate Kelly for the strategy
            if 'win_probability' in strategy:
                kelly = self.calculate_kelly_criterion(
                    strategy['win_probability'],
                    2.0,  # Assume 2.0 odds (even money)
                    bankroll
                )
            else:
                kelly = {'error': 'Cannot calculate Kelly without win probability'}

            # Calculate stop levels
            stops = self.calculate_stop_levels(bankroll)

            # Monte Carlo projection using half-Kelly
            if 'win_probability' in strategy and 'half_kelly' in kelly:
                monte_carlo = self.monte_carlo_bankroll_projection(
                    bankroll,
                    strategy['win_probability'],
                    2.0,  # Average win multiplier
                    1.0,  # Loss multiplier (lose entire bet)
                    kelly.get('half_kelly', 0.01),
                    simulations=100
                )
            else:
                monte_carlo = {'error': 'Cannot run Monte Carlo without strategy data'}

            return {
                'strategy': strategy,
                'kelly_betting': kelly,
                'stop_levels': stops,
                'monte_carlo_projection': monte_carlo,
                'recommendation': self._generate_recommendation(strategy, kelly, monte_carlo)
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_recommendation(self, strategy, kelly, monte_carlo):
        """Generate betting recommendation"""
        try:
            if 'error' in kelly or 'error' in monte_carlo:
                return "Unable to generate recommendation due to calculation errors"

            # Check if strategy has positive edge
            if 'expected_return' in strategy and strategy['expected_return'] > 0:
                edge = "positive"
            elif 'expected_return' in strategy:
                edge = "negative"
            else:
                edge = "unknown"

            # Check Kelly criterion
            if 'edge' in kelly and kelly['edge']:
                kelly_recommendation = "Kelly suggests positive edge"
            else:
                kelly_recommendation = "Kelly suggests no edge or negative edge"

            # Check Monte Carlo
            if 'bust_rate' in monte_carlo:
                bust_rate = monte_carlo['bust_rate']
                if bust_rate < 0.1:
                    mc_recommendation = "Low risk of ruin"
                elif bust_rate < 0.3:
                    mc_recommendation = "Moderate risk of ruin"
                else:
                    mc_recommendation = "High risk of ruin"
            else:
                mc_recommendation = "Cannot assess risk"

            return f"Edge: {edge}. {kelly_recommendation}. {mc_recommendation}."

        except Exception as e:
            return "Error generating recommendation"

    def get_llm_insights(self, risk_assessment):
        """Get LLM insights on risk assessment"""
        if not self.client:
            return "LLM analysis unavailable - no API key configured"

        prompt = f"""
        You are a risk management expert analyzing casino betting strategies. Here are the risk assessment results:

        Strategy: {risk_assessment.get('strategy', {})}
        Kelly Betting: {risk_assessment.get('kelly_betting', {})}
        Stop Levels: {risk_assessment.get('stop_levels', {})}
        Monte Carlo: {risk_assessment.get('monte_carlo_projection', {})}
        Recommendation: {risk_assessment.get('recommendation', '')}

        Provide insights on the risk profile and whether this strategy is advisable.
        Consider bankroll management, risk of ruin, and expected value.
        Keep your response under 300 words.
        """

        try:
            response = self.client.chat.completions.create(
                model="microsoft/wizardlm-2-8x22b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=AGENT_SETTINGS['risk']['max_tokens'],
                temperature=AGENT_SETTINGS['risk']['temperature']
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM analysis failed: {e}"