import random
import time
import threading
import json
import requests
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime
import websocket
from config import GAME_SETTINGS, SCRAPER_SETTINGS

class UniversalDataEngine:
    """Universal data engine for crash, dice, limbo, and slot games"""

    def __init__(self):
        self.data = {
            'crash': deque(maxlen=10000),
            'dice': deque(maxlen=10000),
            'limbo': deque(maxlen=10000),
            'slots': deque(maxlen=10000),
        }
        self.listeners = []
        self.running = False
        self.thread = None

    @property
    def auto_simulating(self) -> bool:
        """True while background simulation is active."""
        return self.running

    def add_listener(self, listener):
        """Add event listener for data updates"""
        self.listeners.append(listener)

    def add_data_point(self, game_type, value, timestamp=None, **extra_fields):
        """Generic insertion method used by live connectors and monitors."""
        timestamp = timestamp or datetime.now()

        if game_type == 'slots':
            symbols = extra_fields.get('symbols', value if isinstance(value, list) else [])
            payout = extra_fields.get('payout', 0)
            self.add_slot_data(symbols, payout)
            return

        data_point = {
            'timestamp': timestamp,
            'value': value,
            'game_type': game_type,
        }
        data_point.update(extra_fields)

        if game_type not in self.data:
            self.data[game_type] = deque(maxlen=10000)

        self.data[game_type].append(data_point)
        self.notify_listeners(game_type, data_point)

    def notify_listeners(self, game_type, data_point):
        """Notify all listeners of new data"""
        for listener in self.listeners:
            listener(game_type, data_point)

    def simulate_crash_round(self):
        """Simulate a single crash game round using provably fair formula"""
        # Provably fair crash formula: crash_point = max(1.00, 0.99 / (1 - random()))
        rand = random.random()
        crash_point = max(1.00, 0.99 / (1 - rand))
        return round(crash_point, 2)

    def simulate_dice_roll(self):
        """Simulate a dice roll (1-100)"""
        return random.randint(1, 100)

    def simulate_limbo_multiplier(self):
        """Simulate limbo multiplier"""
        rand = random.random()
        multiplier = max(1.00, 1.0 / (1 - rand))
        return round(multiplier, 2)

    def simulate_slot_spin(self):
        """Simulate a slot machine spin"""
        symbols = GAME_SETTINGS['slots']['symbols']
        reels = GAME_SETTINGS['slots']['reels']
        spin_result = [random.choice(symbols) for _ in range(reels)]
        return spin_result

    def calculate_slot_payout(self, spin_result):
        """Calculate payout for slot spin (simple 3-of-a-kind)"""
        if len(set(spin_result)) == 1:  # All symbols match
            symbol = spin_result[0]
            payouts = {
                "🍒": 5, "🍋": 10, "🍊": 15, "🍇": 20,
                "🔔": 25, "⭐": 50, "💎": 100
            }
            return payouts.get(symbol, 0)
        return 0

    def add_crash_data(self, crash_point):
        """Add crash data point"""
        timestamp = datetime.now()
        data_point = {
            'timestamp': timestamp,
            'value': crash_point,
            'game_type': 'crash'
        }
        self.data['crash'].append(data_point)
        self.notify_listeners('crash', data_point)

    def add_dice_data(self, roll):
        """Add dice data point"""
        timestamp = datetime.now()
        data_point = {
            'timestamp': timestamp,
            'value': roll,
            'game_type': 'dice'
        }
        self.data['dice'].append(data_point)
        self.notify_listeners('dice', data_point)

    def add_limbo_data(self, multiplier):
        """Add limbo data point"""
        timestamp = datetime.now()
        data_point = {
            'timestamp': timestamp,
            'value': multiplier,
            'game_type': 'limbo'
        }
        self.data['limbo'].append(data_point)
        self.notify_listeners('limbo', data_point)

    def add_slot_data(self, spin_result, payout):
        """Add slot data point"""
        timestamp = datetime.now()
        data_point = {
            'timestamp': timestamp,
            'symbols': spin_result,
            'payout': payout,
            'game_type': 'slots'
        }
        self.data['slots'].append(data_point)
        self.notify_listeners('slots', data_point)

    def get_dataframe(self, game_type, n_points=None):
        """Get pandas DataFrame for game type"""
        if game_type not in self.data:
            return pd.DataFrame()

        data_list = list(self.data[game_type])
        if n_points:
            data_list = data_list[-n_points:]

        df = pd.DataFrame(data_list)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        return df

    def calculate_observed_rtp(self, game_type):
        """Calculate observed RTP for slots"""
        if game_type != 'slots':
            return None

        total_bets = len(self.data['slots'])
        total_payouts = sum(point['payout'] for point in self.data['slots'])

        if total_bets == 0:
            return 0.0

        return total_payouts / total_bets

    def start_auto_simulation(self, interval=2.0):
        """Start automatic simulation in background thread"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._auto_simulate, args=(interval,))
        self.thread.daemon = True
        self.thread.start()

    def stop_auto_simulation(self):
        """Stop automatic simulation"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _auto_simulate(self, interval):
        """Background simulation loop"""
        while self.running:
            # Simulate one round of each game
            crash_point = self.simulate_crash_round()
            self.add_crash_data(crash_point)

            dice_roll = self.simulate_dice_roll()
            self.add_dice_data(dice_roll)

            limbo_mult = self.simulate_limbo_multiplier()
            self.add_limbo_data(limbo_mult)

            slot_spin = self.simulate_slot_spin()
            payout = self.calculate_slot_payout(slot_spin)
            self.add_slot_data(slot_spin, payout)

            time.sleep(interval)

class LiveScanner:
    """Live data scanner for real casino APIs"""

    def __init__(self, data_engine):
        self.data_engine = data_engine
        self.websocket = None
        self.running = False

    def start_scanning(self):
        """Start live scanning"""
        self.running = True
        # For demo purposes, we'll use simulation
        # In real implementation, this would connect to actual APIs
        self.data_engine.start_auto_simulation()

    def stop_scanning(self):
        """Stop live scanning"""
        self.running = False
        self.data_engine.stop_auto_simulation()

    def scrape_bcgame_api(self):
        """Scrape BC.Game API (fallback methods)"""
        base_url = SCRAPER_SETTINGS['bcgame']['base_url']

        # Method 1: REST API v1
        try:
            response = requests.get(f"{base_url}/api/crash/history", timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Process data...
                return data
        except:
            pass

        # Method 2: REST API v2 with pagination
        try:
            response = requests.get(f"{base_url}/api/crash/history?page=1", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data
        except:
            pass

        # Additional fallback methods would be implemented here
        # For now, return simulation data
        return None

    def verify_provably_fair(self, server_seed, client_seed, nonce, outcome):
        """Verify provably fair hash chain"""
        import hmac
        import hashlib

        message = f"{server_seed}-{client_seed}-{nonce}"
        signature = hmac.new(
            server_seed.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        # Convert first 8 chars of hash to float between 0 and 1
        hash_value = int(signature[:8], 16) / 0xFFFFFFFF

        # Verify against outcome
        expected_outcome = max(1.00, 0.99 / (1 - hash_value))
        return abs(expected_outcome - outcome) < 0.01