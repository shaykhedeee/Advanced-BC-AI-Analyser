# Edge Tracker - Data Pipeline Monitoring Dashboard

Real-time monitoring queries and KPIs for data quality.

---

## KEY PERFORMANCE INDICATORS (KPIs)

### 1. Data Collection Health

```sql
-- Data collection rate (points per minute)
SELECT 
    game_type,
    COUNT(*) as points,
    strftime('%Y-%m-%d %H:%M', timestamp) as minute
FROM crash_data
WHERE timestamp > datetime('now', '-1 hour')
GROUP BY game_type, minute
ORDER BY minute DESC;

-- Expected: 10-20 points/minute per game type

-- Scraper success rate
SELECT 
    game_type,
    METHOD,
    COUNT(*) as attempts,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
    ROUND(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate
FROM scraper_logs
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY game_type, method
ORDER BY success_rate DESC;

-- Expected: >95% per method (after circuit breaker settles)
```

### 2. Data Quality Metrics

```sql
-- Validation pass rate
SELECT 
    game_type,
    COUNT(*) as total_records,
    SUM(CASE WHEN validation_passed = 1 THEN 1 ELSE 0 END) as passed,
    ROUND(SUM(CASE WHEN validation_passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as validation_rate
FROM data_validation_log
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY game_type;

-- Expected: >99% validation pass rate

-- Duplicate detection
SELECT 
    COUNT(*) as total_duplicates,
    COUNT(DISTINCT data_hash) as unique_records
FROM crash_data
WHERE timestamp > datetime('now', '-1 hour');

-- Expected: <1% duplicates

-- Data freshness (age of newest record)
SELECT 
    game_type,
    (strftime('%s', 'now') - strftime('%s', MAX(timestamp))) as age_seconds,
    MAX(timestamp) as newest_record
FROM crash_data
GROUP BY game_type;

-- Expected: <5 seconds for all game types
```

### 3. Anomaly Detection Metrics

```sql
-- Anomalies per game type (last 24h)
SELECT 
    game_type,
    COUNT(*) as total_anomalies,
    COUNT(DISTINCT DATE(timestamp)) as anomaly_days,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM crash_data WHERE timestamp > datetime('now', '-24 hours')), 2) as anomaly_percentage
FROM anomaly_log
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY game_type;

-- Expected: <2% anomaly rate (outliers are normal)

-- Anomaly severity distribution
SELECT 
    severity,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM anomaly_log WHERE timestamp > datetime('now', '-24 hours')), 2) as percentage
FROM anomaly_log
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY severity;

-- Expected: Mostly 'LOW' severity
```

### 4. API Performance Metrics

```sql
-- API response latencies
SELECT 
    api_name,
    COUNT(*) as calls,
    ROUND(AVG(latency_ms), 2) as avg_latency,
    MIN(latency_ms) as min_latency,
    MAX(latency_ms) as max_latency,
    ROUND(SUM(CASE WHEN latency_ms > 5000 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as timeout_rate
FROM api_calls
WHERE timestamp > datetime('now', '-1 hour')
GROUP BY api_name;

-- Expected: avg <2000ms, timeout_rate <1%

-- API error rate
SELECT 
    api_name,
    error_type,
    COUNT(*) as error_count
FROM api_errors
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY api_name, error_type
ORDER BY error_count DESC;

-- Expected: <5% error rate per API
```

### 5. Buffer Health

```sql
-- Buffer utilization (if implemented)
SELECT 
    game_type,
    current_size,
    max_size,
    ROUND(current_size * 100.0 / max_size, 2) as utilization_percent
FROM buffer_metrics
ORDER BY utilization_percent DESC;

-- Expected: <80% utilization

-- Archive operations
SELECT 
    DATE(archived_at) as date,
    COUNT(*) as records_archived,
    SUM(size_bytes) as total_size_mb
FROM data_archive
GROUP BY DATE(archived_at)
ORDER BY date DESC;

-- Expected: Archives happen when buffer reaches 90%
```

---

## ALERT THRESHOLDS

```sql
-- Alert: Low data collection rate
-- Trigger: <5 points in last 5 minutes
SELECT 
    game_type,
    COUNT(*) as recent_points,
    CASE 
        WHEN COUNT(*) < 5 THEN 'ALERT'
        WHEN COUNT(*) < 10 THEN 'WARNING'
        ELSE 'OK'
    END as status
FROM crash_data
WHERE timestamp > datetime('now', '-5 minutes')
GROUP BY game_type
HAVING COUNT(*) < 10;

-- Alert: High validation failure rate
-- Trigger: >1% failures in last hour
SELECT 
    game_type,
    ROUND(SUM(CASE WHEN validation_passed = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as failure_rate,
    CASE 
        WHEN failure_rate > 5 THEN 'CRITICAL'
        WHEN failure_rate > 1 THEN 'ALERT'
        ELSE 'OK'
    END as status
FROM data_validation_log
WHERE timestamp > datetime('now', '-1 hour')
GROUP BY game_type
HAVING failure_rate > 1;

-- Alert: Anomaly spike
-- Trigger: >10% anomalies in 1 hour
SELECT 
    game_type,
    COUNT(*) as anomaly_count,
    CASE 
        WHEN anomaly_count > 10 THEN 'CRITICAL'
        WHEN anomaly_count > 5 THEN 'ALERT'
        ELSE 'OK'
    END as status
FROM anomaly_log
WHERE timestamp > datetime('now', '-1 hour')
GROUP BY game_type;

-- Alert: Circuit breaker open
-- Trigger: Any circuit breaker in OPEN state
SELECT 
    breaker_name,
    state,
    last_failure_time,
    failure_count
FROM circuit_breakers
WHERE state = 'OPEN';

-- All should be CLOSED for production
```

---

## DASHBOARD QUERIES (Real-time)

### System Health Overview

```sql
-- Last 24 hours summary
SELECT 
    'Data Collection' as metric,
    (SELECT COUNT(*) FROM crash_data WHERE timestamp > datetime('now', '-24 hours')) as value
UNION ALL
SELECT 'Validation Pass Rate', 
    ROUND(SUM(CASE WHEN validation_passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
    FROM data_validation_log WHERE timestamp > datetime('now', '-24 hours')
UNION ALL
SELECT 'Anomalies Detected', 
    COUNT(*) FROM anomaly_log WHERE timestamp > datetime('now', '-24 hours')
UNION ALL
SELECT 'APIs Responding', 
    COUNT(DISTINCT api_name) FROM api_calls WHERE timestamp > datetime('now', '-1 hour')
    AND latency_ms < 5000
UNION ALL
SELECT 'Buffer Utilization %', 
    MAX(utilization_percent) FROM buffer_metrics;
```

### Per-Game-Type Health

```sql
-- Breakdown by game type
SELECT 
    'crash' as game_type,
    COUNT(*) as records_24h,
    (SELECT COUNT(*) FROM anomaly_log WHERE game_type = 'crash' AND timestamp > datetime('now', '-24 hours')) as anomalies,
    (SELECT COUNT(*) FROM crash_data WHERE timestamp > datetime('now', '-1 hour')) as records_1h
UNION ALL
SELECT 'dice', COUNT(*), 
    (SELECT COUNT(*) FROM anomaly_log WHERE game_type = 'dice' AND timestamp > datetime('now', '-24 hours')),
    (SELECT COUNT(*) FROM dice_data WHERE timestamp > datetime('now', '-1 hour'))
FROM crash_data WHERE timestamp > datetime('now', '-24 hours')
UNION ALL
SELECT 'limbo', COUNT(*),
    (SELECT COUNT(*) FROM anomaly_log WHERE game_type = 'limbo' AND timestamp > datetime('now', '-24 hours')),
    (SELECT COUNT(*) FROM limbo_data WHERE timestamp > datetime('now', '-1 hour'))
FROM crash_data WHERE timestamp > datetime('now', '-24 hours');
```

---

## PYTHON MONITORING CLASS

```python
# Add to dashboard.py or create monitoring.py

import sqlite3
from datetime import datetime, timedelta
import logging

class DataPipelineMonitor:
    def __init__(self, db_path="edge_tracker.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def get_health_report(self):
        """Get comprehensive health report"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'collection_metrics': self._get_collection_metrics(cursor),
                'validation_metrics': self._get_validation_metrics(cursor),
                'anomaly_metrics': self._get_anomaly_metrics(cursor),
                'api_metrics': self._get_api_metrics(cursor),
                'alerts': self._check_alerts(cursor),
            }
            
            return report
    
    def _get_collection_metrics(self, cursor):
        cursor.execute('''
            SELECT 
                'crash' as game_type,
                COUNT(*) as count_24h,
                (SELECT COUNT(*) FROM crash_data WHERE timestamp > datetime('now', '-1 hour')) as count_1h
            FROM crash_data WHERE timestamp > datetime('now', '-24 hours')
        ''')
        return dict(cursor.fetchone().keys()) if cursor.fetchone() else {}
    
    def _get_validation_metrics(self, cursor):
        cursor.execute('''
            SELECT 
                ROUND(SUM(CASE WHEN validation_passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pass_rate,
                COUNT(*) as total_checks,
                SUM(CASE WHEN validation_passed = 0 THEN 1 ELSE 0 END) as failures
            FROM data_validation_log 
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def _get_anomaly_metrics(self, cursor):
        cursor.execute('''
            SELECT 
                COUNT(*) as count_24h,
                (SELECT COUNT(*) FROM anomaly_log WHERE timestamp > datetime('now', '-1 hour')) as count_1h,
                (SELECT COUNT(*) FROM anomaly_log 
                 WHERE timestamp > datetime('now', '-1 hour') AND severity = 'HIGH') as high_severity_1h
            FROM anomaly_log 
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def _get_api_metrics(self, cursor):
        cursor.execute('''
            SELECT 
                COUNT(*) as total_calls,
                ROUND(AVG(latency_ms), 2) as avg_latency_ms,
                ROUND(SUM(CASE WHEN latency_ms > 5000 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as timeout_rate
            FROM api_calls 
            WHERE timestamp > datetime('now', '-1 hour')
        ''')
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def _check_alerts(self, cursor):
        """Check for active alerts"""
        alerts = []
        
        # Check collection rate
        cursor.execute('''
            SELECT COUNT(*) as point_count FROM crash_data 
            WHERE timestamp > datetime('now', '-5 minutes')
        ''')
        if cursor.fetchone()[0] < 5:
            alerts.append("⚠️  Low data collection rate (<5 points/5min)")
        
        # Check validation rate
        cursor.execute('''
            SELECT 
                ROUND(SUM(CASE WHEN validation_passed = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
            FROM data_validation_log WHERE timestamp > datetime('now', '-1 hour')
        ''')
        rate = cursor.fetchone()[0]
        if rate and rate > 1:
            alerts.append(f"⚠️  High validation failure rate ({rate}%)")
        
        # Check anomalies
        cursor.execute('''
            SELECT COUNT(*) FROM anomaly_log 
            WHERE timestamp > datetime('now', '-1 hour') AND severity = 'HIGH'
        ''')
        if cursor.fetchone()[0] > 5:
            alerts.append("🚨 High severity anomalies detected")
        
        return alerts
    
    def print_report(self):
        """Print formatted health report"""
        report = self.get_health_report()
        
        print("\n" + "="*60)
        print("📊 DATA PIPELINE HEALTH REPORT")
        print("="*60)
        print(f"Generated: {report['timestamp']}\n")
        
        print("📈 COLLECTION METRICS")
        for k, v in report['collection_metrics'].items():
            print(f"  {k}: {v}")
        
        print("\n✅ VALIDATION METRICS")
        for k, v in report['validation_metrics'].items():
            print(f"  {k}: {v}")
        
        print("\n⚠️  ANOMALIES")
        for k, v in report['anomaly_metrics'].items():
            print(f"  {k}: {v}")
        
        print("\n🌐 API PERFORMANCE")
        for k, v in report['api_metrics'].items():
            print(f"  {k}: {v}")
        
        if report['alerts']:
            print("\n🔔 ACTIVE ALERTS")
            for alert in report['alerts']:
                print(f"  {alert}")
        else:
            print("\n✓ No active alerts")
        
        print("="*60 + "\n")

# Usage:
# monitor = DataPipelineMonitor()
# monitor.print_report()
```

---

## GRAFANA DASHBOARD (Dashboard JSON)

For Grafana integration, sample panels:

```json
{
  "panels": [
    {
      "title": "Data Collection Rate",
      "targets": [
        {
          "expr": "SELECT game_type, COUNT(*) FROM crash_data WHERE timestamp > datetime('now', '-1 hour') GROUP BY game_type"
        }
      ]
    },
    {
      "title": "Validation Pass Rate",
      "targets": [
        {
          "expr": "SELECT game_type, ROUND(SUM(CASE WHEN validation_passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) FROM data_validation_log WHERE timestamp > datetime('now', '-24 hours') GROUP BY game_type"
        }
      ]
    },
    {
      "title": "Anomalies (24h)",
      "targets": [
        {
          "expr": "SELECT game_type, COUNT(*) FROM anomaly_log WHERE timestamp > datetime('now', '-24 hours') GROUP BY game_type"
        }
      ]
    },
    {
      "title": "API Latency",
      "targets": [
        {
          "expr": "SELECT api_name, ROUND(AVG(latency_ms), 2) FROM api_calls WHERE timestamp > datetime('now', '-1 hour') GROUP BY api_name"
        }
      ]
    }
  ]
}
```

---

## ALERTING SETUP (Python)

```python
# alerts.py - Send alerts on metric thresholds

import smtplib
from email.mime.text import MIMEText
import logging

class AlertManager:
    def __init__(self, email_config):
        self.config = email_config
        self.logger = logging.getLogger(__name__)
    
    def alert(self, severity, title, message):
        """Send alert"""
        alert_msg = f"[{severity}] {title}\n\n{message}"
        self.logger.warning(alert_msg)
        
        if severity in ['CRITICAL', 'ALERT']:
            self._send_email(title, alert_msg, severity)
    
    def _send_email(self, subject, body, severity):
        try:
            msg = MIMEText(body)
            msg['Subject'] = f"[{severity}] {subject}"
            msg['From'] = self.config['from_addr']
            msg['To'] = self.config['to_addr']
            
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['username'], self.config['password'])
                server.send_message(msg)
            
            self.logger.info(f"Alert email sent: {subject}")
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
```

---

## RECOMMENDED MONITORING SCHEDULE

- **Every 1 minute**: Collection rate, freshness
- **Every 5 minutes**: Validation rate, buffer utilization
- **Every 15 minutes**: Anomaly count, API performance
- **Every hour**: Full health report, trend analysis
- **Every day**: Storage usage, archive operations

