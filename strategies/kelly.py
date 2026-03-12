import numpy as np
from config import STRATEGY_SETTINGS


class KellyOptimizer:
    """
    Enhanced Kelly Criterion optimizer with:
    - Multi-objective optimization (EV vs risk)
    - Dynamic fractional Kelly based on drawdown
    - Regime-aware bet sizing
    - Monte Carlo with percentile confidence bands
    - Bankroll growth rate optimization (geometric mean)
    """

    def __init__(self):
        self.cashouts = np.arange(
            STRATEGY_SETTINGS['kelly']['cashouts'][0] / 100,
            STRATEGY_SETTINGS['kelly']['cashouts'][1] / 100,
            0.01
        )
        self.peak_bankroll = 0
        self.current_drawdown = 0

    def optimize_crash_betting(self, historical_data, bankroll):
        """Find optimal cashout using multi-objective: max EV, min risk"""
        if len(historical_data) < 50:
            return {
                'optimal_cashout': 2.0,
                'kelly_fraction': 0.0,
                'expected_value': -0.01,
                'warning': 'Insufficient data for optimization'
            }

        data = np.array(historical_data, dtype=np.float64)
        n = len(data)
        sims = STRATEGY_SETTINGS['kelly'].get('simulations', 1000)

        best_score = -float('inf')
        optimal_cashout = 2.0
        optimal_kelly = 0.0
        best_ev = 0
        best_win_prob = 0

        # Multi-objective search
        candidates = []
        for cashout in self.cashouts:
            wins = np.sum(data >= cashout)
            win_prob = wins / n

            if win_prob == 0 or win_prob == 1:
                continue

            b = cashout - 1
            kelly = (b * win_prob - (1 - win_prob)) / b
            ev = win_prob * (cashout - 1) - (1 - win_prob)

            if kelly <= 0:
                continue

            # Geometric growth rate (the REAL optimal criterion)
            growth = win_prob * np.log(1 + kelly * b) + (1 - win_prob) * np.log(1 - kelly)

            # Risk penalty: higher cashout = higher variance
            risk_penalty = (cashout - 1) * 0.005

            # Multi-objective score
            score = growth - risk_penalty

            candidates.append({
                'cashout': cashout,
                'kelly': kelly,
                'ev': ev,
                'growth': growth,
                'score': score,
                'win_prob': win_prob,
            })

            if score > best_score:
                best_score = score
                optimal_cashout = cashout
                optimal_kelly = max(0, kelly)
                best_ev = ev
                best_win_prob = win_prob

        # Apply drawdown adjustment
        drawdown_mult = 1.0
        if self.current_drawdown > 0.1:
            drawdown_mult = max(0.25, 1.0 - self.current_drawdown)

        adjusted_kelly = optimal_kelly * drawdown_mult

        # Monte Carlo with confidence bands
        ror = self._monte_carlo_risk(adjusted_kelly, best_win_prob, optimal_cashout, sims)

        # Top 3 candidates for comparison
        candidates.sort(key=lambda x: x['score'], reverse=True)

        return {
            'optimal_cashout': round(float(optimal_cashout), 2),
            'kelly_fraction': round(float(optimal_kelly), 6),
            'adjusted_kelly': round(float(adjusted_kelly), 6),
            'drawdown_multiplier': round(float(drawdown_mult), 4),
            'expected_value': round(float(best_ev), 6),
            'geometric_growth': round(float(best_score), 8),
            'win_probability': round(float(best_win_prob), 4),
            'risk_of_ruin': ror,
            'recommended_bet': round(float(bankroll * min(adjusted_kelly, 0.10)), 2),
            'top_candidates': [
                {
                    'cashout': round(c['cashout'], 2),
                    'kelly': round(c['kelly'], 4),
                    'ev': round(c['ev'], 4),
                    'growth': round(c['growth'], 6),
                } for c in candidates[:3]
            ] if candidates else [],
        }

    def update_drawdown(self, bankroll: float):
        """Track drawdown for dynamic adjustment"""
        self.peak_bankroll = max(self.peak_bankroll, bankroll)
        if self.peak_bankroll > 0:
            self.current_drawdown = (self.peak_bankroll - bankroll) / self.peak_bankroll
        else:
            self.current_drawdown = 0

    def _monte_carlo_risk(self, kelly_fraction, win_prob, cashout, simulations):
        """Monte Carlo risk of ruin with percentile bands"""
        if kelly_fraction <= 0:
            return {'risk_of_ruin': 1.0}

        bankroll_start = 100
        ruin_count = 0
        final_bankrolls = []
        bet_frac = min(kelly_fraction, 0.10)
        rounds = 1000

        rng = np.random.default_rng(42)

        for _ in range(simulations):
            br = bankroll_start
            for _ in range(rounds):
                bet = br * bet_frac
                if rng.random() < win_prob:
                    br += bet * (cashout - 1)
                else:
                    br -= bet
                if br <= 0:
                    ruin_count += 1
                    break
            final_bankrolls.append(max(0, br))

        fb = np.array(final_bankrolls)

        return {
            'risk_of_ruin': round(float(ruin_count / simulations), 4),
            'median_final': round(float(np.median(fb)), 2),
            'p10': round(float(np.percentile(fb, 10)), 2),
            'p25': round(float(np.percentile(fb, 25)), 2),
            'p75': round(float(np.percentile(fb, 75)), 2),
            'p90': round(float(np.percentile(fb, 90)), 2),
            'profit_rate': round(float(np.mean(fb > bankroll_start)), 4),
        }

    def get_kelly_progression(self, optimal_kelly, bankroll, rounds=10):
        """Calculate Kelly bet sizes for a progression"""
        progression = []
        current_bankroll = bankroll

        for round_num in range(rounds):
            bet_size = current_bankroll * min(optimal_kelly, 0.10)
            progression.append({
                'round': round_num + 1,
                'bankroll': round(current_bankroll, 2),
                'bet_size': round(bet_size, 2),
                'bet_percentage': round(min(optimal_kelly, 0.10) * 100, 2)
            })
            expected_return = optimal_kelly * 0.01
            current_bankroll *= (1 + expected_return)

        return progression

    def compare_kelly_variants(self, optimal_kelly, bankroll, win_prob=0.5):
        """Compare Full vs Half vs Quarter vs Eighth Kelly"""
        variants = {
            'full_kelly': optimal_kelly,
            'half_kelly': optimal_kelly / 2,
            'quarter_kelly': optimal_kelly / 4,
            'eighth_kelly': optimal_kelly / 8,
        }

        comparison = {}
        for name, fraction in variants.items():
            ror = self._monte_carlo_risk(fraction, win_prob, 2.0, 500)
            comparison[name] = {
                'fraction': round(float(fraction), 6),
                'bet_size': round(float(bankroll * fraction), 2),
                'risk_of_ruin': ror.get('risk_of_ruin', 1.0),
                'profit_rate': ror.get('profit_rate', 0),
                'median_outcome': ror.get('median_final', 0),
            }

        return comparison