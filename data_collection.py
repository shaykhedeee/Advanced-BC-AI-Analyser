import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from config import GAME_SETTINGS

class DataCollector:
    """Central data collection and storage system"""

    def __init__(self, db_path="edge_tracker.db"):
        self.db_path = db_path
        self.init_database()
        self.lock = threading.Lock()

    def init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Crash game table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crash_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    crash_point REAL NOT NULL,
                    server_seed TEXT,
                    client_seed TEXT,
                    nonce INTEGER,
                    hash TEXT,
                    game_session TEXT
                )
            ''')

            # Dice game table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dice_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    roll INTEGER NOT NULL,
                    target INTEGER,
                    win BOOLEAN,
                    payout REAL,
                    server_seed TEXT,
                    client_seed TEXT,
                    nonce INTEGER,
                    hash TEXT,
                    game_session TEXT
                )
            ''')

            # Limbo game table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS limbo_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    multiplier REAL NOT NULL,
                    target REAL,
                    win BOOLEAN,
                    payout REAL,
                    server_seed TEXT,
                    client_seed TEXT,
                    nonce INTEGER,
                    hash TEXT,
                    game_session TEXT
                )
            ''')

            # Slots game table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS slots_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbols TEXT NOT NULL,
                    payout REAL,
                    bet_size REAL,
                    rtp REAL,
                    server_seed TEXT,
                    client_seed TEXT,
                    nonce INTEGER,
                    hash TEXT,
                    game_session TEXT
                )
            ''')

            # ML predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    game_type TEXT NOT NULL,
                    actual_value REAL,
                    predicted_value REAL,
                    model_name TEXT,
                    confidence REAL,
                    features_used TEXT,
                    accuracy REAL
                )
            ''')

            # AI predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    game_type TEXT NOT NULL,
                    actual_value REAL,
                    predicted_direction TEXT,
                    confidence REAL,
                    apis_responded INTEGER,
                    consensus_confidence REAL,
                    reasoning TEXT
                )
            ''')

            # Strategy results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    strategy_name TEXT NOT NULL,
                    game_type TEXT,
                    starting_bankroll REAL,
                    final_bankroll REAL,
                    sessions_simulated INTEGER,
                    bust_rate REAL,
                    profit_rate REAL,
                    median_final_bankroll REAL,
                    average_rounds INTEGER
                )
            ''')

            conn.commit()

    def add_crash_data(self, crash_point, server_seed=None, client_seed=None, nonce=None, hash_val=None, session_id=None):
        """Add crash game data point"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO crash_data (crash_point, server_seed, client_seed, nonce, hash, game_session)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (crash_point, server_seed, client_seed, nonce, hash_val, session_id))
                conn.commit()

    def add_dice_data(self, roll, target=None, win=None, payout=None, server_seed=None, client_seed=None, nonce=None, hash_val=None, session_id=None):
        """Add dice game data point"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO dice_data (roll, target, win, payout, server_seed, client_seed, nonce, hash, game_session)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (roll, target, win, payout, server_seed, client_seed, nonce, hash_val, session_id))
                conn.commit()

    def add_limbo_data(self, multiplier, target=None, win=None, payout=None, server_seed=None, client_seed=None, nonce=None, hash_val=None, session_id=None):
        """Add limbo game data point"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO limbo_data (multiplier, target, win, payout, server_seed, client_seed, nonce, hash, game_session)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (multiplier, target, win, payout, server_seed, client_seed, nonce, hash_val, session_id))
                conn.commit()

    def add_slots_data(self, symbols, payout=None, bet_size=None, rtp=None, server_seed=None, client_seed=None, nonce=None, hash_val=None, session_id=None):
        """Add slots game data point"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO slots_data (symbols, payout, bet_size, rtp, server_seed, client_seed, nonce, hash, game_session)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (json.dumps(symbols), payout, bet_size, rtp, server_seed, client_seed, nonce, hash_val, session_id))
                conn.commit()

    def add_ml_prediction(self, game_type, actual_value, predicted_value, model_name, confidence, features_used, accuracy):
        """Add ML prediction result"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ml_predictions (game_type, actual_value, predicted_value, model_name, confidence, features_used, accuracy)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (game_type, actual_value, predicted_value, model_name, confidence, json.dumps(features_used), accuracy))
                conn.commit()

    def add_ai_prediction(self, game_type, actual_value, predicted_direction, confidence, apis_responded, consensus_confidence, reasoning):
        """Add AI prediction result"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ai_predictions (game_type, actual_value, predicted_direction, confidence, apis_responded, consensus_confidence, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (game_type, actual_value, predicted_direction, confidence, apis_responded, consensus_confidence, reasoning))
                conn.commit()

    def add_strategy_result(self, strategy_name, game_type, starting_bankroll, final_bankroll, sessions_simulated, bust_rate, profit_rate, median_final_bankroll, average_rounds):
        """Add strategy simulation result"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO strategy_results (strategy_name, game_type, starting_bankroll, final_bankroll, sessions_simulated, bust_rate, profit_rate, median_final_bankroll, average_rounds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (strategy_name, game_type, starting_bankroll, final_bankroll, sessions_simulated, bust_rate, profit_rate, median_final_bankroll, average_rounds))
                conn.commit()

    def get_data(self, game_type, limit=None, hours=None):
        """Get data for a specific game type"""
        with sqlite3.connect(self.db_path) as conn:
            if game_type == 'crash':
                query = "SELECT crash_point, timestamp FROM crash_data"
            elif game_type == 'dice':
                query = "SELECT roll, timestamp FROM dice_data"
            elif game_type == 'limbo':
                query = "SELECT multiplier, timestamp FROM limbo_data"
            elif game_type == 'slots':
                query = "SELECT symbols, payout, timestamp FROM slots_data"
            else:
                return pd.DataFrame()

            if hours:
                cutoff = datetime.now() - timedelta(hours=hours)
                query += f" WHERE timestamp > '{cutoff}'"
            elif limit:
                query += f" ORDER BY timestamp DESC LIMIT {limit}"

            return pd.read_sql_query(query, conn)

    def get_ml_predictions(self, game_type=None, limit=1000):
        """Get ML prediction history"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM ml_predictions"
            if game_type:
                query += f" WHERE game_type = '{game_type}'"
            query += " ORDER BY timestamp DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_ai_predictions(self, game_type=None, limit=1000):
        """Get AI prediction history"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM ai_predictions"
            if game_type:
                query += f" WHERE game_type = '{game_type}'"
            query += " ORDER BY timestamp DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_strategy_results(self, strategy_name=None, limit=100):
        """Get strategy simulation results"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM strategy_results"
            if strategy_name:
                query += f" WHERE strategy_name = '{strategy_name}'"
            query += " ORDER BY timestamp DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_statistics(self, game_type):
        """Get basic statistics for a game type"""
        df = self.get_data(game_type, limit=10000)
        if df.empty:
            return {}

        if game_type == 'slots':
            # For slots, we have symbols and payout
            stats = {
                'total_spins': len(df),
                'total_payout': df['payout'].sum(),
                'total_bets': len(df),  # Assuming 1 unit bet per spin
                'observed_rtp': df['payout'].mean(),
                'hit_rate': (df['payout'] > 0).mean(),
                'max_payout': df['payout'].max(),
                'min_payout': df['payout'].min()
            }
        else:
            # For numeric games (crash, dice, limbo)
            values = df['crash_point'] if game_type == 'crash' else \
                     df['roll'] if game_type == 'dice' else df['multiplier']

            stats = {
                'count': len(values),
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75)
            }

        return stats

    def export_data(self, game_type, format='csv', filename=None):
        """Export data to file"""
        df = self.get_data(game_type, limit=100000)
        if df.empty:
            return False

        if filename is None:
            filename = f"{game_type}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if format == 'csv':
            df.to_csv(f"{filename}.csv", index=False)
        elif format == 'json':
            df.to_json(f"{filename}.json", orient='records', date_format='iso')
        elif format == 'excel':
            df.to_excel(f"{filename}.xlsx", index=False)

        return True

    def cleanup_old_data(self, days=30):
        """Clean up data older than N days"""
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            tables = ['crash_data', 'dice_data', 'limbo_data', 'slots_data',
                     'ml_predictions', 'ai_predictions', 'strategy_results']

            for table in tables:
                cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))

            conn.commit()
            return cursor.rowcount

    def get_data_summary(self):
        """Get summary of all stored data"""
        with sqlite3.connect(self.db_path) as conn:
            summary = {}

            tables = ['crash_data', 'dice_data', 'limbo_data', 'slots_data',
                     'ml_predictions', 'ai_predictions', 'strategy_results']

            for table in tables:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                summary[table] = count

            return summary

class DataValidator:
    """Validate data quality and integrity"""

    @staticmethod
    def validate_crash_data(values):
        """Validate crash game data"""
        if not values:
            return False, "No data"

        # Check range
        if any(v < 1.00 or v > 1000.0 for v in values):
            return False, "Values outside expected range (1.00 - 1000.0)"

        # Check for obvious patterns (too many identical values)
        unique_values = len(set(values))
        if unique_values < len(values) * 0.1:  # Less than 10% unique
            return False, "Too many identical values - possible data issue"

        return True, "Valid"

    @staticmethod
    def validate_dice_data(values):
        """Validate dice game data"""
        if not values:
            return False, "No data"

        # Check range
        if any(v < 1 or v > 100 for v in values):
            return False, "Values outside expected range (1 - 100)"

        # Check distribution
        counts = {i: values.count(i) for i in range(1, 101)}
        max_count = max(counts.values())
        if max_count > len(values) * 0.15:  # No number should appear more than 15% of the time
            return False, "Distribution appears biased"

        return True, "Valid"

    @staticmethod
    def validate_limbo_data(values):
        """Validate limbo game data"""
        if not values:
            return False, "No data"

        # Check range
        if any(v < 1.00 or v > 1000.0 for v in values):
            return False, "Values outside expected range (1.00 - 1000.0)"

        return True, "Valid"

    @staticmethod
    def validate_slots_data(symbols_list):
        """Validate slots game data"""
        if not symbols_list:
            return False, "No data"

        # Check symbol format
        for symbols in symbols_list:
            if not isinstance(symbols, list) or len(symbols) != 3:
                return False, "Invalid symbol format"

        return True, "Valid"

class DataExporter:
    """Export data in various formats for analysis"""

    @staticmethod
    def export_for_ml(data, filename, format='numpy'):
        """Export data in format suitable for ML training"""
        if format == 'numpy':
            np.save(filename, data)
        elif format == 'csv':
            pd.DataFrame(data).to_csv(filename, index=False)
        elif format == 'json':
            with open(filename, 'w') as f:
                json.dump(data.tolist(), f)

    @staticmethod
    def create_ml_dataset(data_collector, game_type, window_size=50):
        """Create ML training dataset with sliding windows"""
        df = data_collector.get_data(game_type, limit=10000)
        if df.empty:
            return None, None

        if game_type == 'slots':
            values = df['payout'].values
        else:
            values = df['crash_point'] if game_type == 'crash' else \
                     df['roll'] if game_type == 'dice' else df['multiplier'].values

        X, y = [], []
        for i in range(len(values) - window_size):
            X.append(values[i:i + window_size])
            y.append(values[i + window_size])

        return np.array(X), np.array(y)

class DataBackup:
    """Backup and restore data"""

    def __init__(self, data_collector):
        self.data_collector = data_collector

    def backup_to_file(self, backup_file):
        """Backup database to file"""
        import shutil
        shutil.copy2(self.data_collector.db_path, backup_file)

    def restore_from_file(self, backup_file):
        """Restore database from file"""
        import shutil
        shutil.copy2(backup_file, self.data_collector.db_path)

    def export_all_data(self, export_dir):
        """Export all data to directory"""
        export_path = Path(export_dir)
        export_path.mkdir(exist_ok=True)

        # Export each game type
        for game_type in ['crash', 'dice', 'limbo', 'slots']:
            self.data_collector.export_data(game_type, 'csv', export_path / f"{game_type}_data")

        # Export predictions
        ml_preds = self.data_collector.get_ml_predictions()
        ml_preds.to_csv(export_path / "ml_predictions.csv", index=False)

        ai_preds = self.data_collector.get_ai_predictions()
        ai_preds.to_csv(export_path / "ai_predictions.csv", index=False)

        # Export strategy results
        strategy_results = self.data_collector.get_strategy_results()
        strategy_results.to_csv(export_path / "strategy_results.csv", index=False)

        return True