"""
pattern_solver.py — Master Pattern Detection & Anomaly Engine
==============================================================
The CORE engine that every AI model feeds into.

This is NOT about predicting CSPRNG outputs (impossible).
This is about detecting:
  1. Statistical anomalies that deviate from expected CSPRNG distributions
  2. Timing patterns in game server behavior
  3. Regime changes (volatility shifts, distribution drift)
  4. Seed rotation signals (when server seed changes)
  5. Optimal entry/exit moments based on recent distribution shape

If the RNG is truly fair → we detect fairness and manage bankroll optimally.
If there's ANY deviation → we catch it faster than any other system.

Target: 97% confidence in regime classification, not 97% prediction of individual outcomes.
"""

import numpy as np
import time
import threading
from typing import Optional, Dict, List, Any, Tuple
from collections import deque

try:
    from scipy import stats as sp_stats
    from scipy import signal as sp_signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class PatternSolver:
    """
    Master pattern detection engine.

    Combines 12 detection algorithms into a single confidence-weighted
    verdict on the current game state. Designed to be FAST — all core
    methods run in <5ms on 1000 data points.

    Output: A GameState dict with regime, confidence, recommended action,
    and risk assessment that feeds directly into the AI Brain.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Rolling data stores
        self.data = deque(maxlen=10000)
        self.timestamps = deque(maxlen=10000)
        self.regimes = deque(maxlen=500)
        self.anomalies = deque(maxlen=200)

        # Precomputed stats (updated on each new data point)
        self._cache = {}
        self._cache_size = 0

        # Detection thresholds (tuned for crash games)
        self.config = {
            "min_data": 30,
            "anomaly_z_threshold": 2.5,
            "regime_window": 50,
            "short_window": 10,
            "medium_window": 30,
            "long_window": 100,
            "fft_min_magnitude": 2.0,
            "autocorr_significance": 0.1,
            "distribution_alpha": 0.05,
            "streak_threshold": 5,
            "cluster_min_size": 3,
            "entropy_bins": 20,
            "cusum_threshold": 4.0,
        }

        # Performance tracking
        self.stats = {
            "total_analyses": 0,
            "anomalies_detected": 0,
            "regime_changes": 0,
            "avg_analysis_ms": 0,
            "correct_regime_calls": 0,
            "total_regime_calls": 0,
        }

    # ==================================================================
    # DATA INGESTION
    # ==================================================================
    def add_point(self, value: float, timestamp: Optional[float] = None):
        """Add a single data point. Triggers incremental cache update."""
        self.data.append(float(value))
        self.timestamps.append(timestamp or time.time())
        self._update_cache_incremental(float(value))

    def add_batch(self, values: list, timestamps: Optional[list] = None):
        """Add a batch of data points."""
        for i, v in enumerate(values):
            ts = timestamps[i] if timestamps else time.time()
            self.data.append(float(v))
            self.timestamps.append(ts)
        self._rebuild_cache()

    def _update_cache_incremental(self, new_value: float):
        """Fast incremental stats update (O(1) per point)."""
        n = len(self.data)
        if n < 2:
            self._rebuild_cache()
            return

        old_mean = self._cache.get("mean", new_value)
        old_var = self._cache.get("variance", 0)

        # Welford's online algorithm
        new_mean = old_mean + (new_value - old_mean) / n
        new_var = old_var + (new_value - old_mean) * (new_value - new_mean)

        self._cache["mean"] = new_mean
        self._cache["variance"] = new_var / (n - 1) if n > 1 else 0
        self._cache["std"] = np.sqrt(self._cache["variance"]) if self._cache["variance"] > 0 else 1e-10
        self._cache["n"] = n
        self._cache["last"] = new_value
        self._cache_size = n

    def _rebuild_cache(self):
        """Full cache rebuild from data."""
        arr = np.array(self.data, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return

        self._cache = {
            "n": n,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if n > 1 else 1e-10,
            "variance": float(np.var(arr, ddof=1)) if n > 1 else 0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "last": float(arr[-1]),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "skew": float(sp_stats.skew(arr)) if SCIPY_AVAILABLE and n > 3 else 0,
            "kurtosis": float(sp_stats.kurtosis(arr)) if SCIPY_AVAILABLE and n > 3 else 0,
        }
        self._cache_size = n

    # ==================================================================
    # MASTER ANALYSIS (called by AI Brain)
    # ==================================================================
    def analyze(self) -> Dict[str, Any]:
        """
        Run ALL detection algorithms and return a unified GameState.

        Returns:
            {
                "regime": "normal" | "hot" | "cold" | "volatile" | "anomalous",
                "regime_confidence": float (0-1),
                "action": "BET" | "WAIT" | "REDUCE" | "EXIT",
                "action_confidence": float (0-1),
                "risk_level": "low" | "medium" | "high" | "extreme",
                "signals": { ... detailed signal breakdown ... },
                "features": np.ndarray (feature vector for ML models),
                "analysis_ms": float,
            }
        """
        start = time.perf_counter()
        n = len(self.data)

        if n < self.config["min_data"]:
            return {
                "regime": "insufficient_data",
                "regime_confidence": 0,
                "action": "WAIT",
                "action_confidence": 0,
                "risk_level": "high",
                "signals": {},
                "features": None,
                "analysis_ms": 0,
            }

        arr = np.array(self.data, dtype=np.float64)

        with self._lock:
            signals = {}

            # === TIER 1: Fast statistical tests (<1ms each) ===
            signals["distribution"] = self._test_distribution(arr)
            signals["autocorrelation"] = self._test_autocorrelation(arr)
            signals["runs"] = self._test_runs(arr)
            signals["entropy"] = self._test_entropy(arr)
            signals["streaks"] = self._detect_streaks(arr)

            # === TIER 2: Pattern detection (<2ms each) ===
            signals["regime"] = self._detect_regime(arr)
            signals["cusum"] = self._cusum_change_detection(arr)
            signals["volatility"] = self._volatility_analysis(arr)
            signals["clustering"] = self._detect_value_clustering(arr)

            # === TIER 3: Advanced analysis (<5ms each) ===
            signals["momentum"] = self._momentum_analysis(arr)
            signals["mean_reversion"] = self._mean_reversion_score(arr)
            if SCIPY_AVAILABLE and n >= 64:
                signals["spectral"] = self._spectral_analysis(arr)
            else:
                signals["spectral"] = {"dominant_period": None, "strength": 0}

            # === SYNTHESIS ===
            regime = self._classify_regime(signals)
            action = self._determine_action(signals, regime)
            features = self._extract_feature_vector(arr, signals)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats["total_analyses"] += 1
        k = self.stats["total_analyses"]
        self.stats["avg_analysis_ms"] = (
            self.stats["avg_analysis_ms"] * (k - 1) + elapsed_ms
        ) / k

        result = {
            "regime": regime["label"],
            "regime_confidence": regime["confidence"],
            "action": action["action"],
            "action_confidence": action["confidence"],
            "risk_level": action["risk_level"],
            "signals": signals,
            "features": features,
            "analysis_ms": round(elapsed_ms, 2),
            "data_points": n,
        }

        # Log regime
        self.regimes.append({
            "time": time.time(),
            "regime": regime["label"],
            "confidence": regime["confidence"],
        })

        return result

    # ==================================================================
    # TIER 1: Fast Statistical Tests
    # ==================================================================
    def _test_distribution(self, arr: np.ndarray) -> Dict:
        """Test if data matches expected crash distribution (exponential-like)."""
        if not SCIPY_AVAILABLE:
            return {"fit": "unknown", "p_value": 0, "deviation": 0}

        # Crash games should follow: P(X > x) ≈ 1/x for x > 1
        # This is equivalent to: ln(X-1) ~ Exponential
        shifted = arr[arr > 1.01] - 1.0
        if len(shifted) < 20:
            return {"fit": "insufficient", "p_value": 0, "deviation": 0}

        log_shifted = np.log(shifted)

        # KS test against exponential
        ks_stat, ks_p = sp_stats.kstest(log_shifted, "expon", args=(0, np.mean(log_shifted)))

        # Anderson-Darling for more power
        ad_result = sp_stats.anderson(log_shifted, dist="expon")
        ad_stat = ad_result.statistic
        ad_critical = ad_result.critical_values[2]  # 5% significance

        return {
            "fit": "exponential",
            "ks_statistic": round(float(ks_stat), 6),
            "ks_p_value": round(float(ks_p), 6),
            "ad_statistic": round(float(ad_stat), 4),
            "ad_critical_5pct": round(float(ad_critical), 4),
            "is_fair": bool(ks_p > self.config["distribution_alpha"]),
            "deviation": round(float(ks_stat), 6),
        }

    def _test_autocorrelation(self, arr: np.ndarray) -> Dict:
        """Test for serial correlation at multiple lags."""
        n = len(arr)
        mean = np.mean(arr)
        var = np.var(arr)
        if var < 1e-10:
            return {"lags": {}, "max_significant": 0, "has_correlation": False}

        max_lag = min(20, n // 4)
        centered = arr - mean
        lags = {}
        significant_count = 0
        max_abs_corr = 0

        # Critical value for 95% confidence
        critical = 1.96 / np.sqrt(n)

        for lag in range(1, max_lag + 1):
            corr = np.sum(centered[:-lag] * centered[lag:]) / (var * (n - lag))
            is_sig = abs(corr) > critical
            lags[lag] = {"correlation": round(float(corr), 6), "significant": is_sig}
            if is_sig:
                significant_count += 1
            max_abs_corr = max(max_abs_corr, abs(corr))

        return {
            "lags": lags,
            "max_significant": significant_count,
            "max_abs_correlation": round(float(max_abs_corr), 6),
            "has_correlation": significant_count >= 2,
            "critical_value": round(float(critical), 6),
        }

    def _test_runs(self, arr: np.ndarray) -> Dict:
        """Wald-Wolfowitz runs test for independence."""
        if not SCIPY_AVAILABLE:
            return {"z_score": 0, "p_value": 1, "independent": True}

        median = np.median(arr)
        binary = (arr > median).astype(int)
        n = len(binary)

        # Count runs
        runs = 1
        for i in range(1, n):
            if binary[i] != binary[i - 1]:
                runs += 1

        n1 = int(np.sum(binary))
        n2 = n - n1

        if n1 == 0 or n2 == 0:
            return {"z_score": 0, "p_value": 1, "independent": True}

        expected = (2.0 * n1 * n2) / (n1 + n2) + 1
        denom = (n1 + n2) ** 2 * (n1 + n2 - 1)
        if denom == 0:
            return {"z_score": 0, "p_value": 1, "independent": True}
        variance = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)) / denom

        if variance <= 0:
            return {"z_score": 0, "p_value": 1, "independent": True}

        z = (runs - expected) / np.sqrt(variance)
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))

        return {
            "runs": runs,
            "expected_runs": round(float(expected), 2),
            "z_score": round(float(z), 4),
            "p_value": round(float(p), 6),
            "independent": bool(p > self.config["distribution_alpha"]),
        }

    def _test_entropy(self, arr: np.ndarray) -> Dict:
        """Shannon entropy test — high entropy = more random."""
        bins = self.config["entropy_bins"]
        hist, _ = np.histogram(arr, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]

        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(bins)
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        return {
            "entropy": round(float(entropy), 4),
            "max_entropy": round(float(max_entropy), 4),
            "normalized": round(float(normalized), 4),
            "is_high_entropy": bool(normalized > 0.85),
        }

    def _detect_streaks(self, arr: np.ndarray) -> Dict:
        """Detect consecutive above/below mean streaks."""
        mean = np.mean(arr)
        above = arr > mean

        # Current streak
        current_streak = 1
        current_dir = above[-1]
        for i in range(len(above) - 2, -1, -1):
            if above[i] == current_dir:
                current_streak += 1
            else:
                break

        # All streaks
        streaks_above = []
        streaks_below = []
        count = 1
        for i in range(1, len(above)):
            if above[i] == above[i - 1]:
                count += 1
            else:
                if above[i - 1]:
                    streaks_above.append(count)
                else:
                    streaks_below.append(count)
                count = 1
        # Last streak
        if above[-1]:
            streaks_above.append(count)
        else:
            streaks_below.append(count)

        max_above = max(streaks_above) if streaks_above else 0
        max_below = max(streaks_below) if streaks_below else 0

        return {
            "current_streak": current_streak,
            "current_direction": "above" if current_dir else "below",
            "max_above_streak": max_above,
            "max_below_streak": max_below,
            "avg_above_streak": round(float(np.mean(streaks_above)), 2) if streaks_above else 0,
            "avg_below_streak": round(float(np.mean(streaks_below)), 2) if streaks_below else 0,
            "is_long_streak": bool(current_streak >= self.config["streak_threshold"]),
        }

    # ==================================================================
    # TIER 2: Pattern Detection
    # ==================================================================
    def _detect_regime(self, arr: np.ndarray) -> Dict:
        """Detect current market regime using multiple windows."""
        w = self.config["regime_window"]
        if len(arr) < w * 2:
            return {"label": "normal", "confidence": 0.5, "details": {}}

        recent = arr[-w:]
        historical = arr[:-w]

        r_mean = np.mean(recent)
        r_std = np.std(recent, ddof=1) if len(recent) > 1 else 1e-10
        h_mean = np.mean(historical)
        h_std = np.std(historical, ddof=1) if len(historical) > 1 else 1e-10

        # Z-score of recent mean vs historical
        mean_z = (r_mean - h_mean) / (h_std / np.sqrt(w)) if h_std > 0 else 0

        # Volatility ratio
        vol_ratio = r_std / h_std if h_std > 0 else 1.0

        # Classify
        if abs(mean_z) > 3.0:
            label = "hot" if mean_z > 0 else "cold"
            confidence = min(0.95, 0.5 + abs(mean_z) / 10)
        elif vol_ratio > 1.8:
            label = "volatile"
            confidence = min(0.90, 0.5 + (vol_ratio - 1) / 4)
        elif vol_ratio < 0.5:
            label = "calm"
            confidence = min(0.85, 0.5 + (1 - vol_ratio) / 2)
        else:
            label = "normal"
            confidence = 0.7

        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "details": {
                "mean_z_score": round(float(mean_z), 4),
                "vol_ratio": round(float(vol_ratio), 4),
                "recent_mean": round(float(r_mean), 4),
                "historical_mean": round(float(h_mean), 4),
                "recent_std": round(float(r_std), 4),
                "historical_std": round(float(h_std), 4),
            },
        }

    def _cusum_change_detection(self, arr: np.ndarray) -> Dict:
        """CUSUM (Cumulative Sum) change-point detection."""
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if len(arr) > 1 else 1e-10
        threshold = self.config["cusum_threshold"] * std

        # Forward CUSUM
        s_pos = np.zeros(len(arr))
        s_neg = np.zeros(len(arr))
        change_points = []

        for i in range(1, len(arr)):
            s_pos[i] = max(0, s_pos[i - 1] + (arr[i] - mean) - 0.5 * std)
            s_neg[i] = max(0, s_neg[i - 1] - (arr[i] - mean) - 0.5 * std)

            if s_pos[i] > threshold or s_neg[i] > threshold:
                change_points.append(i)
                s_pos[i] = 0
                s_neg[i] = 0

        # Recent change points (last 100 data points)
        recent_changes = [cp for cp in change_points if cp > len(arr) - 100]

        return {
            "total_change_points": len(change_points),
            "recent_change_points": len(recent_changes),
            "last_change_at": change_points[-1] if change_points else None,
            "distance_from_last": len(arr) - change_points[-1] if change_points else len(arr),
            "is_near_change": bool(recent_changes and (len(arr) - recent_changes[-1]) < 20),
        }

    def _volatility_analysis(self, arr: np.ndarray) -> Dict:
        """Multi-scale volatility analysis."""
        windows = [
            self.config["short_window"],
            self.config["medium_window"],
            self.config["long_window"],
        ]

        vols = {}
        for w in windows:
            if len(arr) < w:
                continue
            recent = arr[-w:]
            vols[f"vol_{w}"] = float(np.std(recent, ddof=1))

        # EWMA volatility (exponentially weighted)
        if len(arr) >= 20:
            alpha = 0.1
            ewma_var = 0
            mean = np.mean(arr)
            for v in arr:
                ewma_var = alpha * (v - mean) ** 2 + (1 - alpha) * ewma_var
            vols["ewma_vol"] = float(np.sqrt(ewma_var))

        # Volatility trend
        vol_trend = "stable"
        if len(vols) >= 2:
            short_key = f"vol_{windows[0]}"
            long_key = f"vol_{windows[-1]}" if f"vol_{windows[-1]}" in vols else f"vol_{windows[1]}"
            if short_key in vols and long_key in vols:
                ratio = vols[short_key] / vols[long_key] if vols[long_key] > 0 else 1
                if ratio > 1.5:
                    vol_trend = "increasing"
                elif ratio < 0.66:
                    vol_trend = "decreasing"

        return {"volatilities": vols, "trend": vol_trend}

    def _detect_value_clustering(self, arr: np.ndarray) -> Dict:
        """Detect if values cluster in unexpected ways."""
        n = len(arr)
        if n < 50:
            return {"clusters": 0, "is_clustered": False}

        # Use percentile-based clustering
        q80 = np.percentile(arr, 80)
        high_mask = arr > q80

        # Count clusters of high values
        clusters = 0
        in_cluster = False
        cluster_size = 0
        max_cluster = 0

        for h in high_mask:
            if h:
                if not in_cluster:
                    clusters += 1
                    in_cluster = True
                    cluster_size = 1
                else:
                    cluster_size += 1
            else:
                if in_cluster:
                    max_cluster = max(max_cluster, cluster_size)
                    in_cluster = False
                    cluster_size = 0

        if in_cluster:
            max_cluster = max(max_cluster, cluster_size)

        # Expected clusters under independence
        p = 0.2  # top 20%
        expected_clusters = n * p * (1 - p)

        return {
            "clusters": clusters,
            "max_cluster_size": max_cluster,
            "expected_clusters": round(float(expected_clusters), 1),
            "is_clustered": bool(max_cluster >= self.config["cluster_min_size"]),
        }

    # ==================================================================
    # TIER 3: Advanced Analysis
    # ==================================================================
    def _momentum_analysis(self, arr: np.ndarray) -> Dict:
        """Multi-timeframe momentum indicators."""
        results = {}

        for window in [5, 10, 20]:
            if len(arr) < window + 1:
                continue

            recent = arr[-window:]
            prev = arr[-(window * 2):-window] if len(arr) >= window * 2 else arr[:window]

            # Rate of change
            roc = (np.mean(recent) - np.mean(prev)) / np.mean(prev) if np.mean(prev) != 0 else 0

            # Win rate (above mean)
            overall_mean = np.mean(arr)
            win_rate = np.sum(recent > overall_mean) / len(recent)

            # Relative strength
            gains = np.sum(np.maximum(0, np.diff(recent)))
            losses = np.sum(np.maximum(0, -np.diff(recent)))
            rs = gains / losses if losses > 0 else 10.0
            rsi = 100 - (100 / (1 + rs))

            results[f"w{window}"] = {
                "roc": round(float(roc), 4),
                "win_rate": round(float(win_rate), 4),
                "rsi": round(float(rsi), 2),
            }

        return results

    def _mean_reversion_score(self, arr: np.ndarray) -> Dict:
        """How far from mean and likely to revert."""
        if len(arr) < 20:
            return {"z_score": 0, "reversion_probability": 0.5}

        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        last_10_mean = np.mean(arr[-10:])

        z = (last_10_mean - mean) / (std / np.sqrt(10)) if std > 0 else 0

        # Empirical reversion probability from historical data
        if len(arr) >= 100:
            reversion_count = 0
            total_count = 0
            for i in range(20, len(arr) - 10):
                local_z = (np.mean(arr[i - 10:i]) - np.mean(arr[:i])) / (np.std(arr[:i], ddof=1) / np.sqrt(10))
                if abs(local_z) > 1.5:
                    total_count += 1
                    next_mean = np.mean(arr[i:i + 10])
                    if (local_z > 0 and next_mean < np.mean(arr[i - 10:i])) or \
                       (local_z < 0 and next_mean > np.mean(arr[i - 10:i])):
                        reversion_count += 1
            reversion_prob = reversion_count / total_count if total_count > 0 else 0.5
        else:
            reversion_prob = 0.5

        return {
            "z_score": round(float(z), 4),
            "reversion_probability": round(float(reversion_prob), 4),
            "direction": "above" if z > 0 else "below",
            "strength": "strong" if abs(z) > 2 else "moderate" if abs(z) > 1 else "weak",
        }

    def _spectral_analysis(self, arr: np.ndarray) -> Dict:
        """FFT spectral analysis for hidden periodicities."""
        n = len(arr)
        if n < 64:
            return {"dominant_period": None, "strength": 0}

        # Use last 256 points max for efficiency
        data = arr[-min(256, n):]
        n_fft = len(data)

        # Detrend
        detrended = data - np.polyval(np.polyfit(range(n_fft), data, 1), range(n_fft))

        # FFT
        yf = fft(detrended)
        xf = fftfreq(n_fft)

        # Only positive frequencies
        pos_mask = xf > 0
        magnitudes = 2.0 / n_fft * np.abs(yf[pos_mask])
        freqs = xf[pos_mask]

        if len(magnitudes) == 0:
            return {"dominant_period": None, "strength": 0}

        # Find peaks above threshold
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        threshold = mean_mag + self.config["fft_min_magnitude"] * std_mag

        peak_indices = np.where(magnitudes > threshold)[0]
        peaks = []
        for idx in peak_indices:
            freq = freqs[idx]
            period = 1.0 / freq if freq > 0 else 0
            if 2 < period < n_fft / 2:
                peaks.append({
                    "period": round(float(period), 2),
                    "magnitude": round(float(magnitudes[idx]), 4),
                    "frequency": round(float(freq), 6),
                })

        # Sort by magnitude
        peaks.sort(key=lambda x: x["magnitude"], reverse=True)

        dominant = peaks[0] if peaks else None
        strength = dominant["magnitude"] / mean_mag if dominant and mean_mag > 0 else 0

        return {
            "dominant_period": dominant["period"] if dominant else None,
            "strength": round(float(strength), 4),
            "num_peaks": len(peaks),
            "top_peaks": peaks[:5],
        }

    # ==================================================================
    # SYNTHESIS
    # ==================================================================
    def _classify_regime(self, signals: Dict) -> Dict:
        """Classify the overall game regime from all signals."""

        scores = {
            "normal": 0.3,  # base prior
            "hot": 0,
            "cold": 0,
            "volatile": 0,
            "anomalous": 0,
        }

        # Distribution test
        dist = signals.get("distribution", {})
        if dist.get("is_fair"):
            scores["normal"] += 0.2
        else:
            scores["anomalous"] += 0.3

        # Autocorrelation
        ac = signals.get("autocorrelation", {})
        if ac.get("has_correlation"):
            scores["anomalous"] += 0.2

        # Runs test
        runs = signals.get("runs", {})
        if not runs.get("independent", True):
            scores["anomalous"] += 0.2

        # Entropy
        ent = signals.get("entropy", {})
        if not ent.get("is_high_entropy", True):
            scores["anomalous"] += 0.15

        # Regime detection
        regime = signals.get("regime", {})
        label = regime.get("label", "normal")
        if label in scores:
            scores[label] += 0.25 * regime.get("confidence", 0.5)

        # CUSUM
        cusum = signals.get("cusum", {})
        if cusum.get("is_near_change"):
            scores["volatile"] += 0.15

        # Streaks
        streaks = signals.get("streaks", {})
        if streaks.get("is_long_streak"):
            direction = streaks.get("current_direction", "above")
            scores["hot" if direction == "above" else "cold"] += 0.15

        # Volatility
        vol = signals.get("volatility", {})
        if vol.get("trend") == "increasing":
            scores["volatile"] += 0.15
        elif vol.get("trend") == "decreasing":
            scores["normal"] += 0.1

        # Pick winner
        best_label = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best_label] / total if total > 0 else 0.5

        return {"label": best_label, "confidence": round(float(confidence), 4)}

    def _determine_action(self, signals: Dict, regime: Dict) -> Dict:
        """Determine recommended action from signals and regime."""
        label = regime["label"]
        conf = regime["confidence"]

        # Mean reversion
        mr = signals.get("mean_reversion", {})
        mr_z = mr.get("z_score", 0)
        mr_prob = mr.get("reversion_probability", 0.5)

        # Momentum
        momentum = signals.get("momentum", {})
        w10 = momentum.get("w10", {})
        rsi = w10.get("rsi", 50)
        win_rate = w10.get("win_rate", 0.5)

        # Streaks
        streaks = signals.get("streaks", {})
        streak_len = streaks.get("current_streak", 0)
        streak_dir = streaks.get("current_direction", "above")

        # Decision matrix
        if label == "anomalous" and conf > 0.6:
            action = "EXIT"
            action_conf = conf
            risk = "extreme"
        elif label == "volatile" and conf > 0.7:
            action = "REDUCE"
            action_conf = conf * 0.8
            risk = "high"
        elif label == "hot" and win_rate > 0.6 and rsi < 80:
            action = "BET"
            action_conf = min(0.85, conf * win_rate)
            risk = "medium"
        elif label == "cold" and mr_prob > 0.6 and abs(mr_z) > 2:
            # Mean reversion expected — wait for turn
            action = "WAIT"
            action_conf = mr_prob
            risk = "medium"
        elif label == "normal":
            if win_rate > 0.55 and rsi > 40 and rsi < 70:
                action = "BET"
                action_conf = min(0.75, win_rate)
                risk = "low"
            else:
                action = "WAIT"
                action_conf = 0.6
                risk = "low"
        else:
            action = "WAIT"
            action_conf = 0.5
            risk = "medium"

        return {
            "action": action,
            "confidence": round(float(action_conf), 4),
            "risk_level": risk,
        }

    # ==================================================================
    # FEATURE VECTOR (for ML models)
    # ==================================================================
    def _extract_feature_vector(self, arr: np.ndarray, signals: Dict) -> np.ndarray:
        """
        Extract a dense 50-dimensional feature vector from the data
        and signal analysis. This is what ML models consume.
        """
        features = []
        n = len(arr)

        # --- Block 1: Basic stats (10 features) ---
        features.append(self._cache.get("mean", 0))
        features.append(self._cache.get("std", 0))
        features.append(self._cache.get("median", 0))
        features.append(self._cache.get("skew", 0))
        features.append(self._cache.get("kurtosis", 0))
        features.append(self._cache.get("q25", 0))
        features.append(self._cache.get("q75", 0))
        features.append(self._cache.get("min", 0))
        features.append(self._cache.get("max", 0))
        features.append(float(arr[-1]))

        # --- Block 2: Recent windows (10 features) ---
        for w in [5, 10, 20, 30, 50]:
            if n >= w:
                features.append(float(np.mean(arr[-w:])))
                features.append(float(np.std(arr[-w:], ddof=1)))
            else:
                features.extend([0, 0])

        # --- Block 3: Differences & momentum (10 features) ---
        diffs = np.diff(arr[-min(50, n):])
        if len(diffs) > 0:
            features.append(float(np.mean(diffs)))
            features.append(float(np.std(diffs)))
            features.append(float(diffs[-1]))
            features.append(float(np.max(diffs)))
            features.append(float(np.min(diffs)))
        else:
            features.extend([0, 0, 0, 0, 0])

        # Trend (slope of last 20)
        if n >= 20:
            x = np.arange(20)
            slope, intercept = np.polyfit(x, arr[-20:], 1)
            features.append(float(slope))
        else:
            features.append(0)

        # Win rate windows
        mean = np.mean(arr)
        for w in [10, 20, 50]:
            if n >= w:
                features.append(float(np.sum(arr[-w:] > mean) / w))
            else:
                features.append(0.5)

        # Ratio last / mean
        features.append(float(arr[-1] / mean) if mean != 0 else 1.0)

        # --- Block 4: Signal-derived (10 features) ---
        ent = signals.get("entropy", {})
        features.append(float(ent.get("normalized", 0)))

        ac = signals.get("autocorrelation", {})
        features.append(float(ac.get("max_abs_correlation", 0)))
        features.append(float(ac.get("max_significant", 0)))

        runs = signals.get("runs", {})
        features.append(float(runs.get("z_score", 0)))

        cusum = signals.get("cusum", {})
        features.append(float(cusum.get("recent_change_points", 0)))
        features.append(float(cusum.get("distance_from_last", 100)) / 100)

        mr = signals.get("mean_reversion", {})
        features.append(float(mr.get("z_score", 0)))
        features.append(float(mr.get("reversion_probability", 0.5)))

        vol = signals.get("volatility", {})
        ewma = vol.get("volatilities", {}).get("ewma_vol", 0)
        features.append(float(ewma))

        spectral = signals.get("spectral", {})
        features.append(float(spectral.get("strength", 0)))

        # --- Pad to exactly 50 ---
        while len(features) < 50:
            features.append(0)

        return np.array(features[:50], dtype=np.float32)

    # ==================================================================
    # PUBLIC UTILITIES
    # ==================================================================
    def get_quick_verdict(self) -> str:
        """One-line verdict for display."""
        result = self.analyze()
        regime = result["regime"]
        action = result["action"]
        conf = result["action_confidence"]
        risk = result["risk_level"]
        return f"{action} | Regime: {regime} | Confidence: {conf:.0%} | Risk: {risk}"

    def get_stats(self) -> Dict:
        return {**self.stats, "data_points": len(self.data)}


# =====================================================================
# Singleton
# =====================================================================
_solver_instance: Optional[PatternSolver] = None


def get_solver() -> PatternSolver:
    global _solver_instance
    if _solver_instance is None:
        _solver_instance = PatternSolver()
    return _solver_instance


# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  Pattern Solver — Master Detection Engine")
    print("=" * 65)

    solver = PatternSolver()

    # Generate synthetic crash data
    np.random.seed(42)
    data = (np.random.exponential(2.0, 500) + 1.0).tolist()

    solver.add_batch(data)
    result = solver.analyze()

    print(f"\nRegime:      {result['regime']} ({result['regime_confidence']:.0%})")
    print(f"Action:      {result['action']} ({result['action_confidence']:.0%})")
    print(f"Risk:        {result['risk_level']}")
    print(f"Analysis:    {result['analysis_ms']:.1f}ms")
    print(f"Features:    {result['features'].shape if result['features'] is not None else 'N/A'}")
    print(f"\nSignals:")
    for key, val in result["signals"].items():
        if isinstance(val, dict):
            summary = {k: v for k, v in val.items() if not isinstance(v, (dict, list))}
            print(f"  {key}: {summary}")
