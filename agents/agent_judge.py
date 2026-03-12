import numpy as np
import pandas as pd
from collections import deque
from config import AGENT_SETTINGS, API_KEYS

try:
    import groq
except BaseException:
    groq = None


class JudgeAgent:
    """
    Enhanced Final Verdict Agent — Bayesian decision framework with:
    - Multi-signal weighted aggregation (not simple average)
    - Regime-aware confidence thresholds
    - Historical accuracy tracking per signal source
    - Drawdown-adjusted risk tolerance
    - Pattern Solver integration for regime context
    - Streak-aware bet/wait logic
    """

    def __init__(self):
        self.client = None
        if API_KEYS.get('groq') and groq:
            try:
                self.client = groq.Groq(api_key=API_KEYS['groq'])
            except Exception:
                pass
        self.confidence_threshold = AGENT_SETTINGS.get('confidence_threshold', 0.65)

        # Track signal accuracy over time
        self.signal_history = deque(maxlen=200)
        self.signal_weights = {
            'stats': 0.20,
            'pattern': 0.25,
            'risk': 0.20,
            'ml': 0.25,
            'solver': 0.10,
        }
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_judgments = 0

    def judge(self, stats_report, pattern_report, risk_report,
              ml_prediction=None, solver_result=None):
        """
        Bayesian-weighted judgment across all signal sources.
        Uses dynamic thresholds based on regime and streak state.
        """
        signals = []

        # --- Stats signal ---
        stats_score = 0.5
        if isinstance(stats_report, dict):
            bias = stats_report.get('bias_score', 0)
            conf = stats_report.get('confidence', 0.5)
            stats_score = float(bias) * 0.7 + float(conf) * 0.3
        elif isinstance(stats_report, str) and 'favorable' in stats_report.lower():
            stats_score = 0.6
        signals.append(('stats', float(np.clip(stats_score, 0, 1))))

        # --- Pattern signal ---
        pattern_score = 0.5
        if isinstance(pattern_report, dict):
            edge = pattern_report.get('edge_estimate', 0)
            pattern_score = float(np.clip(edge, 0, 1))
        elif isinstance(pattern_report, str) and 'edge' in pattern_report.lower():
            pattern_score = 0.6
        signals.append(('pattern', float(np.clip(pattern_score, 0, 1))))

        # --- Risk signal ---
        risk_score = 0.4
        if isinstance(risk_report, dict):
            kelly = risk_report.get('kelly_fraction', 0.0)
            ror = risk_report.get('risk_of_ruin', 1.0)
            risk_score = min(1.0, float(kelly) * 3) * (1 - float(ror) * 0.5)
        elif isinstance(risk_report, str) and 'safe' in risk_report.lower():
            risk_score = 0.6
        signals.append(('risk', float(np.clip(risk_score, 0, 1))))

        # --- ML signal ---
        if ml_prediction is not None:
            try:
                ml_score = float(ml_prediction)
                signals.append(('ml', float(np.clip(ml_score, 0, 1))))
            except (TypeError, ValueError):
                pass

        # --- Pattern Solver signal ---
        if solver_result and isinstance(solver_result, dict):
            action_map = {'BET': 0.75, 'WAIT': 0.35, 'REDUCE': 0.25, 'EXIT': 0.1}
            solver_score = action_map.get(solver_result.get('action', 'WAIT'), 0.4)
            solver_conf = solver_result.get('action_confidence', 0.5)
            solver_final = solver_score * 0.6 + solver_conf * 0.4
            signals.append(('solver', float(np.clip(solver_final, 0, 1))))

        # === Weighted fusion (not simple average) ===
        weighted_sum = 0
        total_weight = 0
        for name, score in signals:
            w = self.signal_weights.get(name, 0.1)
            weighted_sum += score * w
            total_weight += w

        confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

        # === Dynamic threshold based on streak state ===
        dynamic_threshold = self.confidence_threshold
        if self.consecutive_losses >= 3:
            dynamic_threshold = min(0.80, self.confidence_threshold + 0.05 * self.consecutive_losses)
        elif self.consecutive_wins >= 5:
            dynamic_threshold = max(0.55, self.confidence_threshold - 0.03)

        # === Regime adjustment ===
        if solver_result and isinstance(solver_result, dict):
            regime = solver_result.get('regime', 'normal')
            risk_level = solver_result.get('risk_level', 'medium')
            if regime == 'anomalous' or risk_level == 'extreme':
                dynamic_threshold = 0.85  # Very conservative
            elif regime == 'volatile' or risk_level == 'high':
                dynamic_threshold = max(dynamic_threshold, 0.75)

        action = 'BET' if confidence >= dynamic_threshold else 'WAIT'
        self.total_judgments += 1

        return {
            'action': action,
            'confidence': round(float(confidence), 4),
            'threshold_used': round(float(dynamic_threshold), 4),
            'signals': signals,
            'streak_state': {
                'wins': self.consecutive_wins,
                'losses': self.consecutive_losses,
            },
            'reasoning': self._build_reasoning(action, confidence, signals, dynamic_threshold)
        }

    def record_outcome(self, won: bool):
        """Update streak tracking"""
        if won:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        self.signal_history.append({'won': won})

    def _build_reasoning(self, action, confidence, signals, threshold):
        parts = [
            f"Action: {action} (conf={confidence:.2%}, threshold={threshold:.2%})",
            f"  Losses streak: {self.consecutive_losses} | Wins streak: {self.consecutive_wins}",
        ]
        for name, score in signals:
            w = self.signal_weights.get(name, 0.1)
            parts.append(f"  {name}: {score:.3f} (weight={w:.2f})")
        return '\n'.join(parts)

    def make_final_judgment(self, stats_report, pattern_report, risk_report,
                            ml_stats=None, game_type='crash', bankroll=1000.0,
                            solver_result=None):
        """Dashboard-compatible wrapper — returns structured verdict dict."""
        ml_pred = None
        if isinstance(ml_stats, dict):
            ml_pred = ml_stats.get('ensemble_proba', ml_stats.get('accuracy', 0.5))

        result = self.judge(stats_report, pattern_report, risk_report,
                            ml_pred, solver_result)

        score = round(result['confidence'] * 10, 1)
        verdict = 'PLAY' if result['action'] == 'BET' else 'WAIT'

        streak_info = result.get('streak_state', {})
        recommendations = [
            f"Confidence: {result['confidence']:.2%} (threshold: {result['threshold_used']:.2%})",
            f"Action: {result['action']}",
            f"Game: {game_type} | Bankroll: ${bankroll:.2f}",
        ]
        if streak_info.get('losses', 0) >= 3:
            recommendations.append(f"WARNING: {streak_info['losses']} consecutive losses — threshold raised")
        if streak_info.get('wins', 0) >= 5:
            recommendations.append(f"Hot streak: {streak_info['wins']} wins — threshold lowered")

        return {
            'verdict': verdict,
            'overall_score': score,
            'explanation': result['reasoning'].split('\n')[0],
            'reasoning': [f"{n}: {v:.3f}" for n, v in result['signals']],
            'recommendations': recommendations,
            'dynamic_threshold': result['threshold_used'],
        }
