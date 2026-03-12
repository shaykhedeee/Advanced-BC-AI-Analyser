import requests
from bs4 import BeautifulSoup
import json
import time
import hmac
import hashlib
import websocket
import socketio
import numpy as np
import pandas as pd
from config import SCRAPER_SETTINGS
import threading

class BCGameScraper:
    """BC.Game scraper with 6 fallback methods"""

    def __init__(self):
        self.base_url = SCRAPER_SETTINGS['bcgame']['base_url']
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_with_fallbacks(self, game_type='crash'):
        """Try all 6 fallback methods in order"""
        methods = [
            self._method_rest_api_v1,
            self._method_rest_api_v2,
            self._method_rest_api_v3,
            self._method_graphql,
            self._method_html_scraping,
            self._method_websocket_snapshot
        ]

        for method in methods:
            try:
                result = method(game_type)
                if result and self._validate_data(result):
                    return result
            except Exception as e:
                print(f"Method {method.__name__} failed: {e}")
                continue

        return None

    def _method_rest_api_v1(self, game_type):
        """REST API v1 - Direct history endpoint"""
        endpoint = f"/api/{game_type}/history"
        url = f"{self.base_url}{endpoint}"

        response = self.session.get(url, timeout=SCRAPER_SETTINGS['bcgame']['timeout'])
        response.raise_for_status()

        return response.json()

    def _method_rest_api_v2(self, game_type):
        """REST API v2 - Paginated endpoint"""
        endpoint = f"/api/{game_type}/history?page=1&limit=100"
        url = f"{self.base_url}{endpoint}"

        response = self.session.get(url, timeout=SCRAPER_SETTINGS['bcgame']['timeout'])
        response.raise_for_status()

        return response.json()

    def _method_rest_api_v3(self, game_type):
        """REST API v3 - Currency-specific endpoint"""
        endpoint = f"/api/{game_type}/history?currency=USD"
        url = f"{self.base_url}{endpoint}"

        response = self.session.get(url, timeout=SCRAPER_SETTINGS['bcgame']['timeout'])
        response.raise_for_status()

        return response.json()

    def _method_graphql(self, game_type):
        """GraphQL endpoint"""
        graphql_url = f"{self.base_url}/graphql"

        query = f"""
        {{
            {game_type}History(first: 100) {{
                edges {{
                    node {{
                        id
                        multiplier
                        createdAt
                    }}
                }}
            }}
        }}
        """

        payload = {
            'query': query,
            'variables': {}
        }

        response = self.session.post(
            graphql_url,
            json=payload,
            timeout=SCRAPER_SETTINGS['bcgame']['timeout']
        )
        response.raise_for_status()

        return response.json()

    def _method_html_scraping(self, game_type):
        """HTML scraping with BeautifulSoup"""
        url = f"{self.base_url}/{game_type}"

        response = self.session.get(url, timeout=SCRAPER_SETTINGS['bcgame']['timeout'])
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for JSON data in script tags or data attributes
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'history' in script.string.lower():
                # Try to extract JSON from script
                try:
                    start = script.string.find('{')
                    end = script.string.rfind('}') + 1
                    json_str = script.string[start:end]
                    return json.loads(json_str)
                except:
                    continue

        return None

    def _method_websocket_snapshot(self, game_type):
        """WebSocket snapshot method"""
        # This is a simplified version - real implementation would handle WebSocket properly
        try:
            ws = websocket.create_connection(
                SCRAPER_SETTINGS['bcgame']['websocket_url'],
                timeout=SCRAPER_SETTINGS['bcgame']['timeout']
            )

            # Send subscription message
            subscribe_msg = {
                "type": "subscribe",
                "channel": f"{game_type}_history"
            }
            ws.send(json.dumps(subscribe_msg))

            # Receive initial snapshot
            response = ws.recv()
            ws.close()

            return json.loads(response)

        except Exception as e:
            print(f"WebSocket method failed: {e}")
            return None

    def _validate_data(self, data):
        """Validate scraped data structure"""
        if not isinstance(data, dict):
            return False

        # Check for common data structures
        if 'data' in data and isinstance(data['data'], list):
            return len(data['data']) > 0
        if 'history' in data and isinstance(data['history'], list):
            return len(data['history']) > 0
        if isinstance(data, list):
            return len(data) > 0

        return False

class UniversalJSONExtractor:
    """Extract values from nested JSON structures"""

    @staticmethod
    def extract_values(data, key_patterns=None):
        """Recursively extract values matching patterns"""
        if key_patterns is None:
            key_patterns = ['multiplier', 'crash_point', 'roll', 'result']

        results = []

        def _extract(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if any(pattern in key.lower() for pattern in key_patterns):
                        results.append((new_path, value))
                    _extract(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    _extract(item, new_path)

        _extract(data)
        return results

class ProvablyFairAnalyzer:
    """Analyze provably fair hash chains"""

    def __init__(self):
        self.hash_algorithm = SCRAPER_SETTINGS['provably_fair']['hash_algorithm']

    def verify_crash_outcome(self, server_seed, client_seed, nonce, outcome):
        """Verify a crash game outcome using HMAC-SHA256"""
        message = f"{server_seed}-{client_seed}-{nonce}"

        # Create HMAC signature
        signature = hmac.new(
            server_seed.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Convert first 8 characters of hash to integer
        hash_int = int(signature[:8], 16)

        # Convert to float between 0 and 1
        hash_float = hash_int / 0xFFFFFFFF

        # Calculate expected crash point
        # Formula: max(1.00, 0.99 / (1 - hash_float))
        if hash_float >= 1.0:
            expected_crash = 1.00
        else:
            expected_crash = max(1.00, 0.99 / (1 - hash_float))

        return round(expected_crash, 2), abs(expected_crash - outcome) < 0.01

    def generate_hash_chain(self, server_seed, client_seed, start_nonce=0, count=100):
        """Generate a chain of hash-verified outcomes"""
        results = []

        for nonce in range(start_nonce, start_nonce + count):
            expected_outcome, _ = self.verify_crash_outcome(server_seed, client_seed, nonce, 0)
            results.append({
                'nonce': nonce,
                'outcome': expected_outcome,
                'hash': self._calculate_hash(server_seed, client_seed, nonce)
            })

        return results

    def _calculate_hash(self, server_seed, client_seed, nonce):
        """Calculate HMAC-SHA256 hash"""
        message = f"{server_seed}-{client_seed}-{nonce}"
        signature = hmac.new(
            server_seed.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

class MultiSourceScraper:
    """Scraper that cross-validates data from multiple sources"""

    def __init__(self):
        self.sources = [
            BCGameScraper(),
            # Could add more casino scrapers here
        ]

    def get_consensus_data(self, game_type='crash', min_sources=2):
        """Get data that appears in multiple sources"""
        all_data = []

        for source in self.sources:
            data = source.scrape_with_fallbacks(game_type)
            if data:
                extracted = UniversalJSONExtractor.extract_values(data)
                all_data.append(set(str(val) for _, val in extracted))

        if len(all_data) < min_sources:
            return None

        # Find intersection of all datasets
        consensus = set.intersection(*all_data)
        return list(consensus)

class RealTimeMonitor:
    """Real-time data monitoring with anomaly detection"""

    def __init__(self, data_engine):
        self.data_engine = data_engine
        self.anomalies = []
        self.monitoring = False
        self.thread = None

    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            # Check for anomalies in recent data
            self._check_anomalies()
            time.sleep(5)  # Check every 5 seconds

    def _check_anomalies(self):
        """Check for statistical anomalies"""
        for game_type in ['crash', 'dice', 'limbo']:
            df = self.data_engine.get_dataframe(game_type, n_points=100)
            if len(df) < 50:
                continue

            values = df['value'].values

            # Z-score based anomaly detection
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                continue

            z_scores = (values - mean) / std
            anomalies = np.abs(z_scores) > 3  # 3 sigma rule

            if anomalies.any():
                anomaly_indices = np.where(anomalies)[0]
                for idx in anomaly_indices:
                    if idx not in [a['index'] for a in self.anomalies]:
                        self.anomalies.append({
                            'game_type': game_type,
                            'index': idx,
                            'value': values[idx],
                            'z_score': z_scores[idx],
                            'timestamp': df.index[idx]
                        })

    def get_recent_anomalies(self, hours=1):
        """Get anomalies from the last N hours"""
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        return [a for a in self.anomalies if a['timestamp'] > cutoff]


class LiveConnector:
    """Live WebSocket connector for BC.Game real-time data"""

    def __init__(self, data_engine):
        self.data_engine = data_engine
        self.sio = None
        self.connected = False
        self.current_game = None

    def connect(self):
        """Connect to BC.Game WebSocket"""
        if not SCRAPER_SETTINGS['bcgame'].get('enable_live_connector', False):
            print("Live connector disabled. Set ENABLE_BCGAME_LIVE=true in .env to enable.")
            self.connected = False
            return

        try:
            import socketio

            sio = socketio.Client()
            self.sio = sio

            @sio.event
            def connect():
                print("Connected to BC.Game WebSocket")
                self.connected = True
                # Join crash game room
                sio.emit('join', {'game': 'crash'})

            @sio.event
            def disconnect():
                print("Disconnected from BC.Game WebSocket")
                self.connected = False

            @sio.event
            def game_start(data):
                """Handle crash game start"""
                if isinstance(data, dict) and 'hash' in data:
                    self.current_game = {
                        'hash': data['hash'],
                        'start_time': pd.Timestamp.now(),
                        'multipliers': []
                    }
                    print(f"Crash game started: {data['hash']}")

            @sio.event
            def game_tick(data):
                """Handle crash multiplier updates"""
                if self.current_game and isinstance(data, dict) and 'multiplier' in data:
                    multiplier = data['multiplier']
                    self.current_game['multipliers'].append(multiplier)
                    # Add to data engine
                    self.data_engine.add_data_point('crash', multiplier, pd.Timestamp.now())

            @sio.event
            def game_end(data):
                """Handle crash game end"""
                if self.current_game and isinstance(data, dict):
                    final_multiplier = data.get('multiplier', max(self.current_game['multipliers']) if self.current_game['multipliers'] else 1.0)
                    self.current_game['end_time'] = pd.Timestamp.now()
                    self.current_game['final_multiplier'] = final_multiplier

                    # Add final point
                    self.data_engine.add_data_point('crash', final_multiplier, pd.Timestamp.now())

                    print(f"Crash game ended: {final_multiplier}x")
                    self.current_game = None

            # Connect to BC.Game
            sio.connect('https://bc.game', transports=['websocket'])

        except Exception as e:
            print(f"Failed to connect to live feed: {e}")
            self.connected = False

    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.sio:
            self.sio.disconnect()
            self.connected = False

    def is_connected(self):
        """Check connection status"""
        return self.connected