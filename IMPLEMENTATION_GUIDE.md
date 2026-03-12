# Edge Tracker - Quick Implementation Guide

This file contains copy-paste-ready code snippets for immediate deployment.

---

## STEP 1: Add Requirements

Add to `requirements.txt`:

```
pydantic>=2.0
python-json-logger>=2.0
scipy>=1.11
python-dotenv>=1.0
```

---

## STEP 2: Create `validators.py` (NEW FILE)

```python
"""
validators.py - Data validation with Pydantic
Place in: edge_tracker/validators.py
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CrashDataPoint(BaseModel):
    timestamp: datetime = Field(..., description="Game end time")
    crash_point: float = Field(..., ge=1.00, le=1000.0)
    server_seed: Optional[str] = None
    client_seed: Optional[str] = None
    nonce: Optional[int] = None
    hash_value: Optional[str] = None
    game_session: Optional[str] = None
    
    @validator('crash_point')
    def valid_decimal_places(cls, v):
        if v % 0.01 != 0:
            raise ValueError("Max 2 decimal places")
        return round(v, 2)
    
    @validator('timestamp')
    def not_future(cls, v):
        if v > datetime.now():
            raise ValueError("Cannot be future")
        return v

class DiceDataPoint(BaseModel):
    timestamp: datetime
    roll: int = Field(..., ge=1, le=100)
    target: Optional[int] = Field(None, ge=1, le=100)
    win: Optional[bool] = None
    payout: Optional[float] = Field(None, ge=0)

class LimboDataPoint(BaseModel):
    timestamp: datetime
    multiplier: float = Field(..., ge=1.0, le=1000.0)
    target: Optional[float] = Field(None, ge=1.0)
    win: Optional[bool] = None
    payout: Optional[float] = Field(None, ge=0)

class SlotsDataPoint(BaseModel):
    timestamp: datetime
    symbols: List[str] = Field(..., min_items=3, max_items=3)
    payout: Optional[float] = Field(None, ge=0)
    bet_size: Optional[float] = Field(None, gt=0)
    rtp: Optional[float] = Field(None, ge=0, le=1.0)
    
    @validator('symbols')
    def valid_symbols(cls, v):
        valid = {"🍒", "🍋", "🍊", "🍇", "🔔", "⭐", "💎"}
        for sym in v:
            if sym not in valid:
                raise ValueError(f"Invalid symbol: {sym}")
        return v
```

---

## STEP 3: Create `buffers.py` (NEW FILE)

```python
"""
buffers.py - Smart data buffer management
Place in: edge_tracker/buffers.py
"""

from collections import deque
from datetime import datetime
import logging
import json
import sqlite3

logger = logging.getLogger(__name__)

class SmartDataBuffer:
    def __init__(self, max_size=10000, db_path='edge_tracker.db'):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.db_path = db_path
        self._size_threshold = int(max_size * 0.9)
        self.last_overflow_ts = None
    
    def add(self, data_point):
        if len(self.buffer) >= self._size_threshold:
            self._handle_overflow()
        self.buffer.append(data_point)
        return True
    
    def _handle_overflow(self):
        logger.warning(
            f"Buffer at {len(self.buffer)/self.max_size*100:.1f}% capacity - archiving..."
        )
        self._archive_oldest(int(self.max_size * 0.2))
        self.last_overflow_ts = datetime.now()
    
    def _archive_oldest(self, count):
        archived = []
        for _ in range(count):
            if self.buffer:
                archived.append(self.buffer.popleft())
        
        if archived:
            self._save_to_db(archived)
            logger.info(f"Archived {len(archived)} data points")
    
    def _save_to_db(self, items):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for item in items:
                    cursor.execute('''
                        INSERT INTO data_archive (timestamp, game_type, value, metadata)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        item.get('timestamp').isoformat() if hasattr(item.get('timestamp'), 'isoformat') else str(item.get('timestamp')),
                        item.get('game_type'),
                        item.get('value'),
                        json.dumps(item.get('metadata', {}))
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"Archive failed: {e}")
```

---

## STEP 4: Update `scraper.py` - Add Circuit Breaker

Add to top of `scraper.py`:

```python
import time
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60

class CircuitBreaker:
    def __init__(self, name, config=None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"[{self.name}] Circuit breaker HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker OPEN: {self.name}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self):
        if not self.last_failure_time:
            return True
        return (time.time() - self.last_failure_time) >= self.config.timeout_seconds
    
    def _on_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"[{self.name}] Circuit breaker CLOSED")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(f"[{self.name}] Circuit breaker OPEN")
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
```

---

## STEP 5: Update `scraper.py` - Add Resilient Scraping

Replace `scrape_with_fallbacks()` in BCGameScraper:

```python
def scrape_with_fallbacks_resilient(self, game_type='crash', max_retries=3):
    """Scrape with retry and circuit breaker"""
    methods = [
        ('rest_api_v1', self._method_rest_api_v1),
        ('rest_api_v2', self._method_rest_api_v2),
        ('rest_api_v3', self._method_rest_api_v3),
        ('graphql', self._method_graphql),
        ('html_scraping', self._method_html_scraping),
        ('websocket', self._method_websocket_snapshot),
    ]
    
    for method_name, method_func in methods:
        for attempt in range(max_retries):
            try:
                backoff = 2 ** attempt  # 1, 2, 4 seconds
                
                # Call method
                result = method_func(game_type)
                
                if result and self._validate_data(result):
                    print(f"✓ {method_name} success on attempt {attempt+1}")
                    return result
                    
            except Exception as e:
                print(f"✗ {method_name} attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff)
    
    print("⚠ All scraping methods failed")
    return None
```

---

## STEP 6: Create `anomaly_detector.py` (NEW FILE)

```python
"""
anomaly_detector.py - Advanced anomaly detection
Place in: edge_tracker/anomaly_detector.py
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, sensitivity='normal'):
        self.sensitivity = sensitivity
        self.thresholds = {
            'normal': {'z': 3.0, 'iqr': 1.5}
        }
    
    def detect(self, values):
        values = np.asarray(values, dtype=float)
        if len(values) < 5:
            return {'anomalies': [], 'confidence': 0}
        
        results = {
            'zscore': self._zscore_anomalies(values),
            'iqr': self._iqr_anomalies(values),
            'ensemble': None,
        }
        
        # Try isolation forest if available
        try:
            results['isolation'] = self._isolation_anomalies(values)
        except:
            pass
        
        results['ensemble'] = self._ensemble_vote(results)
        return results
    
    def _zscore_anomalies(self, values):
        z = np.abs(stats.zscore(values))
        threshold = self.thresholds[self.sensitivity]['z']
        anomalies = np.where(z > threshold)[0].tolist()
        return {'anomalies': anomalies, 'method': 'zscore'}
    
    def _iqr_anomalies(self, values):
        Q1, Q3 = np.percentile(values, [25, 75])
        IQR = Q3 - Q1
        mult = self.thresholds[self.sensitivity]['iqr']
        lower = Q1 - (mult * IQR)
        upper = Q3 + (mult * IQR)
        anomalies = np.where((values < lower) | (values > upper))[0].tolist()
        return {'anomalies': anomalies, 'bounds': (lower, upper)}
    
    def _isolation_anomalies(self, values):
        X = values.reshape(-1, 1)
        iso = IsolationForest(contamination=0.1, random_state=42)
        preds = iso.fit_predict(X)
        anomalies = np.where(preds == -1)[0].tolist()
        return {'anomalies': anomalies, 'method': 'isolation'}
    
    def _ensemble_vote(self, results):
        votes = {}
        for method, data in results.items():
            if method != 'ensemble':
                for idx in data.get('anomalies', []):
                    votes[idx] = votes.get(idx, 0) + 1
        
        # Flag if 2+ methods agree
        ensemble = [idx for idx, v in votes.items() if v >= 2]
        return {'anomalies': ensemble, 'method': 'ensemble', 'votes': votes}
```

---

## STEP 7: Update `config.py` - Add Logging Settings

Add to end of `config.py`:

```python
# Logging Settings
LOGGING_SETTINGS = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "edge_tracker.log",
            "maxBytes": 10485760,
            "backupCount": 10,
            "formatter": "json",
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": True
        }
    }
}

# Data Quality Settings
DATA_QUALITY = {
    "buffer_size": 10000,
    "buffer_warning_threshold": 0.9,
    "anomaly_sensitivity": "normal",
    "schema_validation": True,
    "duplicate_window_minutes": 5,
}
```

---

## STEP 8: Update `data_engine.py` - Integrate Smart Buffer

Replace UniversalDataEngine initialization:

```python
from buffers import SmartDataBuffer

class UniversalDataEngine:
    def __init__(self):
        # Use smart buffers instead of plain deques
        self.data = {
            'crash': SmartDataBuffer(max_size=10000),
            'dice': SmartDataBuffer(max_size=10000),
            'limbo': SmartDataBuffer(max_size=10000),
            'slots': SmartDataBuffer(max_size=10000),
        }
        self.listeners = []
        self.running = False
        self.thread = None
    
    def add_crash_data(self, crash_point):
        timestamp = datetime.now()
        data_point = {
            'timestamp': timestamp,
            'value': crash_point,
            'game_type': 'crash'
        }
        self.data['crash'].add(data_point)  # Uses smart buffer
        self.notify_listeners('crash', data_point)
```

---

## DEPLOYMENT CHECKLIST

- [ ] Add validators.py
- [ ] Create buffers.py  
- [ ] Create anomaly_detector.py
- [ ] Update scraper.py with CircuitBreaker
- [ ] Update config.py with logging
- [ ] Update data_engine.py to use SmartDataBuffer
- [ ] Install new requirements
- [ ] Test validation with sample data
- [ ] Monitor logs for 24 hours
- [ ] Enable alerts on validation failures

---

## MONITORING SQL

Create this table for archiving:

```sql
CREATE TABLE IF NOT EXISTS data_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME,
    game_type TEXT,
    value REAL,
    metadata TEXT,
    archived_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_archive_game_type ON data_archive(game_type);
CREATE INDEX idx_archive_timestamp ON data_archive(timestamp);
```

---

## QUICK WINS (30 min implementation)

1. Add logging to config.py ✓
2. Create validators.py ✓
3. Add CircuitBreaker to scraper.py ✓
4. Update scrape_with_fallbacks with retry ✓

