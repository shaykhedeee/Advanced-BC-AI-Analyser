import time
import threading
from datetime import datetime, timedelta
from config import STRATEGY_SETTINGS
from collections import deque

class SessionManager:
    """Real-time session tracking and tilt detection"""

    def __init__(self, data_engine):
        self.data_engine = data_engine
        self.current_session = {
            'start_time': None,
            'start_bankroll': 0,
            'current_bankroll': 0,
            'rounds_played': 0,
            'wins': 0,
            'losses': 0,
            'total_bet': 0,
            'total_won': 0,
            'peak_bankroll': 0,
            'lowest_bankroll': float('inf'),
            'win_streak': 0,
            'loss_streak': 0,
            'current_streak': 0,
            'last_bet_time': None
        }
        self.session_history = deque(maxlen=100)
        self.tilt_indicators = {
            'time_playing': 0,
            'rounds_without_break': 0,
            'drawdown_percentage': 0,
            'bet_frequency': 0,
            'streak_length': 0
        }
        self.alerts = []
        self.monitoring = False
        self.monitor_thread = None

    def start_session(self, starting_bankroll):
        """Start a new gambling session"""
        self.current_session = {
            'start_time': datetime.now(),
            'start_bankroll': starting_bankroll,
            'current_bankroll': starting_bankroll,
            'rounds_played': 0,
            'wins': 0,
            'losses': 0,
            'total_bet': 0,
            'total_won': 0,
            'peak_bankroll': starting_bankroll,
            'lowest_bankroll': starting_bankroll,
            'win_streak': 0,
            'loss_streak': 0,
            'current_streak': 0,
            'last_bet_time': datetime.now()
        }
        self.alerts = []
        self.start_monitoring()

    def end_session(self):
        """End the current session"""
        if self.current_session['start_time']:
            session_summary = self.get_session_summary()
            self.session_history.append(session_summary)
            self.current_session = {k: 0 if isinstance(v, (int, float)) else None
                                  for k, v in self.current_session.items()}
        self.stop_monitoring()

    def record_bet(self, bet_amount, outcome_amount, won=True):
        """Record a bet outcome"""
        self.current_session['rounds_played'] += 1
        self.current_session['total_bet'] += bet_amount
        self.current_session['last_bet_time'] = datetime.now()

        if won:
            self.current_session['wins'] += 1
            self.current_session['total_won'] += outcome_amount
            self.current_session['current_bankroll'] += outcome_amount - bet_amount

            # Update streaks
            if self.current_session['current_streak'] >= 0:
                self.current_session['current_streak'] += 1
            else:
                self.current_session['current_streak'] = 1
            self.current_session['win_streak'] = max(self.current_session['win_streak'],
                                                    self.current_session['current_streak'])
        else:
            self.current_session['losses'] += 1
            self.current_session['current_bankroll'] -= bet_amount

            # Update streaks
            if self.current_session['current_streak'] <= 0:
                self.current_session['current_streak'] -= 1
            else:
                self.current_session['current_streak'] = -1
            self.current_session['loss_streak'] = max(self.current_session['loss_streak'],
                                                     abs(self.current_session['current_streak']))

        # Update peaks and valleys
        self.current_session['peak_bankroll'] = max(self.current_session['peak_bankroll'],
                                                   self.current_session['current_bankroll'])
        self.current_session['lowest_bankroll'] = min(self.current_session['lowest_bankroll'],
                                                     self.current_session['current_bankroll'])

        # Update tilt indicators
        self._update_tilt_indicators()

        # Check for alerts
        self._check_alerts()

    def get_session_summary(self):
        """Get summary of current session"""
        if not self.current_session['start_time']:
            return None

        duration = datetime.now() - self.current_session['start_time']
        profit = self.current_session['current_bankroll'] - self.current_session['start_bankroll']
        win_rate = (self.current_session['wins'] / self.current_session['rounds_played']
                   if self.current_session['rounds_played'] > 0 else 0)

        return {
            'duration': duration,
            'starting_bankroll': self.current_session['start_bankroll'],
            'current_bankroll': self.current_session['current_bankroll'],
            'profit': profit,
            'rounds_played': self.current_session['rounds_played'],
            'wins': self.current_session['wins'],
            'losses': self.current_session['losses'],
            'win_rate': win_rate,
            'total_bet': self.current_session['total_bet'],
            'total_won': self.current_session['total_won'],
            'peak_bankroll': self.current_session['peak_bankroll'],
            'lowest_bankroll': self.current_session['lowest_bankroll'],
            'longest_win_streak': self.current_session['win_streak'],
            'longest_loss_streak': self.current_session['loss_streak'],
            'end_time': datetime.now()
        }

    def _update_tilt_indicators(self):
        """Update tilt detection indicators"""
        if not self.current_session['start_time']:
            return

        # Time playing
        self.tilt_indicators['time_playing'] = (
            datetime.now() - self.current_session['start_time']
        ).total_seconds() / 3600  # Hours

        # Rounds without break
        if self.current_session['last_bet_time']:
            time_since_last_bet = (datetime.now() - self.current_session['last_bet_time']).total_seconds()
            self.tilt_indicators['rounds_without_break'] = self.current_session['rounds_played']

        # Drawdown percentage
        peak = self.current_session['peak_bankroll']
        current = self.current_session['current_bankroll']
        self.tilt_indicators['drawdown_percentage'] = (
            (peak - current) / peak if peak > 0 else 0
        )

        # Bet frequency (bets per hour)
        if self.tilt_indicators['time_playing'] > 0:
            self.tilt_indicators['bet_frequency'] = (
                self.current_session['rounds_played'] / self.tilt_indicators['time_playing']
            )

        # Current streak length
        self.tilt_indicators['streak_length'] = abs(self.current_session['current_streak'])

    def _check_alerts(self):
        """Check for tilt and session alerts"""
        alerts = []

        # Tilt detection thresholds
        tilt_thresholds = STRATEGY_SETTINGS['session_manager']['tilt_threshold']
        fatigue_rounds = STRATEGY_SETTINGS['session_manager']['fatigue_rounds']

        # Drawdown alert
        if self.tilt_indicators['drawdown_percentage'] > tilt_thresholds:
            alerts.append({
                'type': 'DRAWDOWN',
                'severity': 'HIGH',
                'message': f'Drawdown exceeds {tilt_thresholds:.1%} - consider stopping',
                'value': self.tilt_indicators['drawdown_percentage']
            })

        # Time playing alert
        if self.tilt_indicators['time_playing'] > 2:  # 2 hours
            alerts.append({
                'type': 'FATIGUE',
                'severity': 'MEDIUM',
                'message': 'Playing for over 2 hours - take a break',
                'value': self.tilt_indicators['time_playing']
            })

        # Rounds without break
        if self.current_session['rounds_played'] > fatigue_rounds:
            alerts.append({
                'type': 'ROUND_COUNT',
                'severity': 'MEDIUM',
                'message': f'Played {self.current_session["rounds_played"]} rounds - consider a break',
                'value': self.current_session['rounds_played']
            })

        # Streak alerts
        if abs(self.current_session['current_streak']) >= 5:
            streak_type = 'WIN' if self.current_session['current_streak'] > 0 else 'LOSS'
            alerts.append({
                'type': f'{streak_type}_STREAK',
                'severity': 'MEDIUM',
                'message': f'{streak_type} streak of {abs(self.current_session["current_streak"])} - be careful',
                'value': abs(self.current_session['current_streak'])
            })

        # Bankroll alerts
        current_bankroll = self.current_session['current_bankroll']
        start_bankroll = self.current_session['start_bankroll']

        if current_bankroll < start_bankroll * 0.5:
            alerts.append({
                'type': 'BANKROLL_LOW',
                'severity': 'HIGH',
                'message': 'Bankroll down 50% or more - stop immediately',
                'value': current_bankroll
            })

        self.alerts.extend(alerts)

    def get_current_status(self):
        """Get current session status"""
        if not self.current_session['start_time']:
            return {'status': 'NO_ACTIVE_SESSION'}

        summary = self.get_session_summary()
        summary.update({
            'tilt_indicators': self.tilt_indicators.copy(),
            'active_alerts': self.alerts.copy(),
            'should_stop': self._should_stop_session()
        })

        return summary

    def _should_stop_session(self):
        """Determine if session should be stopped"""
        reasons = []

        # High drawdown
        if self.tilt_indicators['drawdown_percentage'] > STRATEGY_SETTINGS['session_manager']['tilt_threshold']:
            reasons.append('high_drawdown')

        # Long playing time
        if self.tilt_indicators['time_playing'] > 3:  # 3 hours
            reasons.append('fatigue')

        # Too many rounds
        if self.current_session['rounds_played'] > STRATEGY_SETTINGS['session_manager']['fatigue_rounds'] * 2:
            reasons.append('too_many_rounds')

        # Severe bankroll depletion
        if self.current_session['current_bankroll'] < self.current_session['start_bankroll'] * 0.3:
            reasons.append('bankroll_depleted')

        return reasons

    def get_session_history(self, limit=10):
        """Get recent session history"""
        return list(self.session_history)[-limit:]

    def start_monitoring(self):
        """Start background monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            self._update_tilt_indicators()
            self._check_alerts()
            time.sleep(30)  # Check every 30 seconds

    def get_recommendation(self):
        """Get current recommendation based on session state"""
        if not self.current_session['start_time']:
            return {'recommendation': 'START_SESSION', 'reason': 'No active session'}

        should_stop = self._should_stop_session()

        if should_stop:
            return {
                'recommendation': 'STOP',
                'reason': f'Stop due to: {", ".join(should_stop)}',
                'urgency': 'HIGH' if 'bankroll_depleted' in should_stop else 'MEDIUM'
            }

        # Check if profitable
        profit = self.current_session['current_bankroll'] - self.current_session['start_bankroll']
        if profit > self.current_session['start_bankroll'] * 0.5:  # Up 50%
            return {
                'recommendation': 'STOP',
                'reason': 'Profit target reached',
                'urgency': 'MEDIUM'
            }

        # Continue playing
        return {
            'recommendation': 'CONTINUE',
            'reason': 'Session within normal parameters',
            'next_check_minutes': 15
        }