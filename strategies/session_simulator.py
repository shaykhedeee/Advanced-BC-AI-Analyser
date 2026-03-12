import numpy as np
import pandas as pd
from config import STRATEGY_SETTINGS
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SimulationResult:
    final_bankroll: float
    rounds_played: int
    max_bankroll: float
    min_bankroll: float
    win_rate: float
    profit: float
    bust: bool

class SessionSimulator:
    """Monte Carlo session simulator for gambling strategies"""

    def __init__(self):
        self.sessions = STRATEGY_SETTINGS['session_simulator']['sessions']
        self.max_rounds = STRATEGY_SETTINGS['session_simulator']['max_rounds']

    def simulate_crash_sessions(self, starting_bankroll, cashout_target, bet_fraction):
        """Simulate multiple crash game sessions"""
        results = []

        for session in range(self.sessions):
            result = self._simulate_single_crash_session(
                starting_bankroll, cashout_target, bet_fraction
            )
            results.append(result)

        return self._analyze_results(results, starting_bankroll)

    def simulate_dice_sessions(self, starting_bankroll, target_number, bet_fraction):
        """Simulate multiple dice game sessions"""
        results = []

        for session in range(self.sessions):
            result = self._simulate_single_dice_session(
                starting_bankroll, target_number, bet_fraction
            )
            results.append(result)

        return self._analyze_results(results, starting_bankroll)

    def simulate_slot_sessions(self, starting_bankroll, bet_size, rtp=0.95):
        """Simulate multiple slot machine sessions"""
        results = []

        for session in range(self.sessions):
            result = self._simulate_single_slot_session(
                starting_bankroll, bet_size, rtp
            )
            results.append(result)

        return self._analyze_results(results, starting_bankroll)

    def _simulate_single_crash_session(self, bankroll, cashout_target, bet_fraction):
        """Simulate one complete crash session"""
        initial_bankroll = bankroll
        max_bankroll = bankroll
        min_bankroll = bankroll
        rounds = 0
        wins = 0

        while rounds < self.max_rounds and bankroll > 0:
            bet_size = bankroll * bet_fraction
            bet_size = min(bet_size, bankroll)  # Can't bet more than you have

            # Simulate crash outcome
            rand = np.random.random()
            crash_point = max(1.00, 0.99 / (1 - rand))

            if crash_point >= cashout_target:
                # Win
                payout = bet_size * cashout_target
                bankroll += payout - bet_size  # Net win
                wins += 1
            else:
                # Loss
                bankroll -= bet_size

            rounds += 1
            max_bankroll = max(max_bankroll, bankroll)
            min_bankroll = min(min_bankroll, bankroll)

            # Stop if bankroll gets too low
            if bankroll < initial_bankroll * 0.1:
                break

        win_rate = wins / rounds if rounds > 0 else 0
        profit = bankroll - initial_bankroll
        bust = bankroll <= 0

        return SimulationResult(
            final_bankroll=round(bankroll, 2),
            rounds_played=rounds,
            max_bankroll=round(max_bankroll, 2),
            min_bankroll=round(min_bankroll, 2),
            win_rate=round(win_rate, 4),
            profit=round(profit, 2),
            bust=bust
        )

    def _simulate_single_dice_session(self, bankroll, target_number, bet_fraction):
        """Simulate one complete dice session"""
        initial_bankroll = bankroll
        max_bankroll = bankroll
        min_bankroll = bankroll
        rounds = 0
        wins = 0

        # Assume betting on "over target_number" for simplicity
        win_probability = (100 - target_number) / 100

        while rounds < self.max_rounds and bankroll > 0:
            bet_size = bankroll * bet_fraction
            bet_size = min(bet_size, bankroll)

            # Simulate dice roll
            roll = np.random.randint(1, 101)

            if roll > target_number:
                # Win (1.98x payout for even money bet)
                payout = bet_size * 1.98
                bankroll += payout - bet_size
                wins += 1
            else:
                # Loss
                bankroll -= bet_size

            rounds += 1
            max_bankroll = max(max_bankroll, bankroll)
            min_bankroll = min(min_bankroll, bankroll)

            if bankroll < initial_bankroll * 0.1:
                break

        win_rate = wins / rounds if rounds > 0 else 0
        profit = bankroll - initial_bankroll
        bust = bankroll <= 0

        return SimulationResult(
            final_bankroll=round(bankroll, 2),
            rounds_played=rounds,
            max_bankroll=round(max_bankroll, 2),
            min_bankroll=round(min_bankroll, 2),
            win_rate=round(win_rate, 4),
            profit=round(profit, 2),
            bust=bust
        )

    def _simulate_single_slot_session(self, bankroll, bet_size, rtp):
        """Simulate one complete slot session"""
        initial_bankroll = bankroll
        max_bankroll = bankroll
        min_bankroll = bankroll
        rounds = 0
        total_bet = 0
        total_win = 0

        while rounds < self.max_rounds and bankroll >= bet_size:
            bankroll -= bet_size
            total_bet += bet_size

            # Simulate slot outcome based on RTP
            rand = np.random.random()
            if rand < rtp:
                # Win - assume average payout of 0.9x bet (slightly below RTP for variance)
                payout = bet_size * 0.9
                bankroll += payout
                total_win += payout
            # Else: loss (already subtracted bet)

            rounds += 1
            max_bankroll = max(max_bankroll, bankroll)
            min_bankroll = min(min_bankroll, bankroll)

        win_rate = total_win / total_bet if total_bet > 0 else 0
        profit = bankroll - initial_bankroll
        bust = bankroll <= 0

        return SimulationResult(
            final_bankroll=round(bankroll, 2),
            rounds_played=rounds,
            max_bankroll=round(max_bankroll, 2),
            min_bankroll=round(min_bankroll, 2),
            win_rate=round(win_rate, 4),
            profit=round(profit, 2),
            bust=bust
        )

    def _analyze_results(self, results: List[SimulationResult], starting_bankroll):
        """Analyze simulation results"""
        if not results:
            return {'error': 'No simulation results'}

        final_bankrolls = [r.final_bankroll for r in results]
        profits = [r.profit for r in results]
        rounds = [r.rounds_played for r in results]
        win_rates = [r.win_rate for r in results]
        busts = [r.bust for r in results]

        analysis = {
            'sessions_simulated': len(results),
            'bust_rate': round(np.mean(busts), 4),
            'survival_rate': round(1 - np.mean(busts), 4),
            'average_final_bankroll': round(np.mean(final_bankrolls), 2),
            'median_final_bankroll': round(np.median(final_bankrolls), 2),
            'average_profit': round(np.mean(profits), 2),
            'median_profit': round(np.median(profits), 2),
            'average_rounds': round(np.mean(rounds), 1),
            'average_win_rate': round(np.mean(win_rates), 4),
            'profit_rate': round(np.mean([p > 0 for p in profits]), 4),
            'bankroll_distribution': {
                'p10': round(np.percentile(final_bankrolls, 10), 2),
                'p25': round(np.percentile(final_bankrolls, 25), 2),
                'p50': round(np.percentile(final_bankrolls, 50), 2),
                'p75': round(np.percentile(final_bankrolls, 75), 2),
                'p90': round(np.percentile(final_bankrolls, 90), 2)
            },
            'profit_distribution': {
                'p10': round(np.percentile(profits, 10), 2),
                'p25': round(np.percentile(profits, 25), 2),
                'p50': round(np.percentile(profits, 50), 2),
                'p75': round(np.percentile(profits, 75), 2),
                'p90': round(np.percentile(profits, 90), 2)
            }
        }

        # Risk assessment
        analysis['risk_assessment'] = self._assess_risk(analysis, starting_bankroll)

        return analysis

    def _assess_risk(self, analysis, starting_bankroll):
        """Assess risk level of the strategy"""
        bust_rate = analysis['bust_rate']
        median_final = analysis['median_final_bankroll']

        if bust_rate > 0.5:
            risk_level = 'EXTREME'
            description = 'Very high risk of losing everything'
        elif bust_rate > 0.3:
            risk_level = 'HIGH'
            description = 'Significant risk of bankroll depletion'
        elif bust_rate > 0.1:
            risk_level = 'MODERATE'
            description = 'Moderate risk with potential for recovery'
        elif median_final < starting_bankroll * 0.9:
            risk_level = 'LOW-MODERATE'
            description = 'Low bust risk but expected losses'
        else:
            risk_level = 'LOW'
            description = 'Low risk with positive expected outcome'

        return {
            'risk_level': risk_level,
            'description': description,
            'bust_probability': f"{bust_rate:.1%}",
            'median_outcome': f"{median_final:.2f} ({median_final/starting_bankroll:.1%} of starting bankroll)"
        }

    def generate_histogram_data(self, results: List[SimulationResult], bins=20):
        """Generate histogram data for visualization"""
        final_bankrolls = [r.final_bankroll for r in results]

        hist, bin_edges = np.histogram(final_bankrolls, bins=bins)

        return {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_centers': [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        }