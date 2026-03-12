import numpy as np
try:
    from scipy import signal
    from scipy import stats as sp_stats
except ImportError:
    signal = None
    sp_stats = None
import pandas as pd
from config import AGENT_SETTINGS, API_KEYS
try:
    import google.generativeai as genai
except BaseException:
    genai = None


class PatternAgent:
    """
    Enhanced Pattern Detection Agent with 9 detection methods:
    1. High-crash clustering
    2. Low-streak detection
    3. Volatility regime change (multi-window)
    4. Repeating sequence detection
    5. FFT cyclical patterns
    6. Markov chain transition matrix
    7. CUSUM change-point detection
    8. Hurst exponent (long-range dependence)
    9. Information mutual correlation
    """

    def __init__(self):
        self.client = None
        if API_KEYS.get('google_gemini') and genai:
            try:
                genai.configure(api_key=API_KEYS['google_gemini'])  # type: ignore[attr-defined]
                self.client = genai.GenerativeModel('gemini-pro')
            except Exception:
                pass

    def detect_patterns(self, data, game_type='crash'):
        """Detect all patterns with 9 detection algorithms"""
        if len(data) < 50:
            return "Insufficient data for pattern analysis"

        values = np.array(data, dtype=np.float64)

        analysis = {
            'high_crash_clustering': self._detect_high_crash_clustering(values),
            'low_streak_detection': self._detect_low_streaks(values),
            'volatility_regime': self._detect_volatility_regime_change(values),
            'repeating_sequences': self._detect_repeating_sequences(values),
            'fft_analysis': self._fft_cyclical_patterns(values),
            'markov_chain': self._markov_transition_matrix(values),
            'cusum_detection': self._cusum_change_detection(values),
            'hurst_exponent': self._hurst_exponent(values),
            'mutual_info': self._mutual_information(values),
        }

        honest_assessment = self._generate_honest_assessment(analysis, game_type)
        edge_estimate = self._compute_edge_estimate(analysis)

        return {
            'patterns': analysis,
            'honest_assessment': honest_assessment,
            'reality_check': self._reality_check_message(),
            'edge_estimate': edge_estimate,
            'total_signals': sum(1 for v in analysis.values()
                                 if isinstance(v, dict) and v.get('detected', False)),
        }

    def _detect_high_crash_clustering(self, values, threshold_percentile=80):
        """Detect clustering of high values (potential hot streaks)"""
        try:
            threshold = np.percentile(values, threshold_percentile)
            high_values = values > threshold

            # Find clusters of high values
            clusters = []
            current_cluster = []

            for i, is_high in enumerate(high_values):
                if is_high:
                    current_cluster.append(i)
                else:
                    if len(current_cluster) >= 3:  # Minimum cluster size
                        clusters.append(current_cluster)
                    current_cluster = []

            # Check last cluster
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)

            cluster_stats = {
                'num_clusters': len(clusters),
                'avg_cluster_size': np.mean([len(c) for c in clusters]) if clusters else 0,
                'max_cluster_size': max([len(c) for c in clusters]) if clusters else 0,
                'total_high_values': np.sum(high_values)
            }

            # Statistical significance test
            expected_cluster_rate = len(values) * (1 - threshold_percentile/100)
            observed_clusters = len(clusters)

            return {
                'clusters': clusters,
                'stats': cluster_stats,
                'conclusion': f'Found {len(clusters)} clusters of {len(values) * (1 - threshold_percentile/100):.0f}+ values'
            }

        except Exception as e:
            return {'error': str(e)}

    def _detect_low_streaks(self, values, threshold_percentile=20):
        """Detect unusually long low streaks"""
        try:
            threshold = np.percentile(values, threshold_percentile)
            low_values = values <= threshold

            # Find streaks of low values
            streaks = []
            current_streak = 0

            for is_low in low_values:
                if is_low:
                    current_streak += 1
                else:
                    if current_streak >= 5:  # Minimum streak length
                        streaks.append(current_streak)
                    current_streak = 0

            # Check final streak
            if current_streak >= 5:
                streaks.append(current_streak)

            streak_stats = {
                'num_streaks': len(streaks),
                'avg_streak_length': np.mean(streaks) if streaks else 0,
                'max_streak_length': max(streaks) if streaks else 0,
                'total_low_values': np.sum(low_values)
            }

            return {
                'streaks': streaks,
                'stats': streak_stats,
                'conclusion': f'Found {len(streaks)} low streaks of 5+ consecutive values'
            }

        except Exception as e:
            return {'error': str(e)}

    def _detect_volatility_regime_change(self, values, window=20):
        """Detect changes in volatility regimes"""
        try:
            if len(values) < window * 2:
                return {'conclusion': 'Insufficient data for regime analysis'}

            # Calculate rolling volatility
            rolling_std = pd.Series(values).rolling(window=window).std()

            # Detect significant changes
            std_changes = rolling_std.pct_change(periods=window)

            # Find regime changes (significant volatility shifts)
            threshold = std_changes.std() * 2  # 2 standard deviations
            regime_changes = std_changes.abs() > threshold

            change_points = np.where(regime_changes)[0]

            return {
                'regime_changes': len(change_points),
                'change_points': change_points.tolist(),
                'avg_volatility': np.mean(rolling_std.dropna()),
                'volatility_trend': 'increasing' if rolling_std.iloc[-1] > rolling_std.iloc[0] else 'decreasing',
                'conclusion': f'Detected {len(change_points)} significant volatility regime changes'
            }

        except Exception as e:
            return {'error': str(e)}

    def _detect_repeating_sequences(self, values, min_length=3, max_length=8):
        """Detect repeating number sequences"""
        try:
            sequences = []

            for length in range(min_length, min(max_length + 1, len(values) // 2)):
                for start in range(len(values) - length * 2 + 1):
                    seq1 = values[start:start + length]
                    seq2 = values[start + length:start + length * 2]

                    if np.allclose(seq1, seq2, rtol=0.1):  # Allow small differences
                        sequences.append({
                            'sequence': seq1.tolist(),
                            'start_pos': start,
                            'length': length
                        })

            return {
                'repeating_sequences': sequences,
                'total_sequences': len(sequences),
                'conclusion': f'Found {len(sequences)} repeating sequences of length {min_length}-{max_length}'
            }

        except Exception as e:
            return {'error': str(e)}

    def _fft_cyclical_patterns(self, values):
        """Use FFT to detect cyclical patterns"""
        try:
            if len(values) < 20:
                return {'conclusion': 'Insufficient data for FFT analysis'}

            # Remove linear trend
            detrended = signal.detrend(values)

            # Apply FFT
            fft = np.fft.fft(detrended)
            freqs = np.fft.fftfreq(len(detrended))

            # Find dominant frequencies
            magnitudes = np.abs(fft)
            peak_indices = np.argsort(magnitudes)[-5:][::-1]  # Top 5 peaks

            dominant_freqs = []
            for idx in peak_indices:
                if freqs[idx] > 0:  # Only positive frequencies
                    period = 1 / freqs[idx] if freqs[idx] != 0 else 0
                    if 2 < period < len(values) / 2:  # Reasonable period range
                        dominant_freqs.append({
                            'frequency': freqs[idx],
                            'period': period,
                            'magnitude': magnitudes[idx]
                        })

            return {
                'dominant_frequencies': dominant_freqs,
                'total_peaks': len(dominant_freqs),
                'detected': len(dominant_freqs) >= 2,
                'conclusion': f'Found {len(dominant_freqs)} significant cyclical patterns'
            }

        except Exception as e:
            return {'error': str(e), 'detected': False}

    # ===================================================
    # NEW: Advanced Pattern Detection Methods
    # ===================================================
    def _markov_transition_matrix(self, values):
        """Build Markov chain transition matrix — detect if transitions deviate from independence"""
        try:
            n = len(values)
            if n < 50:
                return {'conclusion': 'Insufficient data', 'detected': False}

            median = np.median(values)
            states = (values > median).astype(int)  # 0 = below, 1 = above

            # 2x2 transition matrix
            trans = np.zeros((2, 2))
            for i in range(n - 1):
                trans[states[i], states[i + 1]] += 1

            # Normalize to probabilities
            row_sums = trans.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            trans_probs = trans / row_sums

            # Expected under independence: P(next=1) = overall rate
            p1 = np.sum(states) / n
            expected = np.array([[1 - p1, p1], [1 - p1, p1]])

            # Chi-squared deviation
            deviation = np.sum((trans_probs - expected) ** 2)
            detected = deviation > 0.02

            return {
                'transition_matrix': trans_probs.tolist(),
                'p_above': round(float(p1), 4),
                'deviation_from_independence': round(float(deviation), 6),
                'detected': bool(detected),
                'conclusion': f'Transition deviation: {deviation:.4f} — {"non-independent" if detected else "independent"}'
            }
        except Exception as e:
            return {'error': str(e), 'detected': False}

    def _cusum_change_detection(self, values):
        """CUSUM change-point detection for structural breaks"""
        try:
            n = len(values)
            if n < 30:
                return {'conclusion': 'Insufficient data', 'detected': False}

            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else 1e-10
            threshold = 4.0 * std

            s_pos = np.zeros(n)
            s_neg = np.zeros(n)
            change_points = []

            for i in range(1, n):
                s_pos[i] = max(0, s_pos[i - 1] + (values[i] - mean) - 0.5 * std)
                s_neg[i] = max(0, s_neg[i - 1] - (values[i] - mean) - 0.5 * std)

                if s_pos[i] > threshold or s_neg[i] > threshold:
                    change_points.append(i)
                    s_pos[i] = 0
                    s_neg[i] = 0

            recent = [cp for cp in change_points if cp > n - 50]

            return {
                'total_change_points': len(change_points),
                'recent_change_points': len(recent),
                'last_change_at': change_points[-1] if change_points else None,
                'detected': bool(len(recent) > 0),
                'conclusion': f'{len(change_points)} change points ({len(recent)} recent)'
            }
        except Exception as e:
            return {'error': str(e), 'detected': False}

    def _hurst_exponent(self, values):
        """
        R/S Hurst exponent:
        H = 0.5: random walk (no memory)
        H > 0.5: trending (persistent)
        H < 0.5: mean-reverting (anti-persistent)
        """
        try:
            n = len(values)
            if n < 100:
                return {'hurst': 0.5, 'interpretation': 'insufficient data', 'detected': False}

            max_k = min(n // 4, 256)
            min_k = 8
            if max_k <= min_k:
                return {'hurst': 0.5, 'interpretation': 'insufficient data', 'detected': False}

            rs_values = []
            ns = []

            for k in range(min_k, max_k + 1, max(1, (max_k - min_k) // 20)):
                sub_n = n // k
                if sub_n < 2:
                    continue

                rs_list = []
                for i in range(k):
                    chunk = values[i * sub_n:(i + 1) * sub_n]
                    mean_c = np.mean(chunk)
                    deviations = np.cumsum(chunk - mean_c)
                    r = np.max(deviations) - np.min(deviations)
                    s = np.std(chunk, ddof=1)
                    if s > 0:
                        rs_list.append(r / s)

                if rs_list:
                    rs_values.append(np.mean(rs_list))
                    ns.append(sub_n)

            if len(rs_values) < 3:
                return {'hurst': 0.5, 'interpretation': 'insufficient segments', 'detected': False}

            log_ns = np.log(ns)
            log_rs = np.log(rs_values)
            hurst = np.polyfit(log_ns, log_rs, 1)[0]
            hurst = float(np.clip(hurst, 0, 1))

            if hurst > 0.6:
                interp = 'trending (persistent)'
            elif hurst < 0.4:
                interp = 'mean-reverting (anti-persistent)'
            else:
                interp = 'random walk'

            return {
                'hurst': round(hurst, 4),
                'interpretation': interp,
                'detected': bool(abs(hurst - 0.5) > 0.1),
                'conclusion': f'H={hurst:.3f} — {interp}'
            }
        except Exception as e:
            return {'error': str(e), 'hurst': 0.5, 'detected': False}

    def _mutual_information(self, values):
        """Mutual information between x(t) and x(t+1) — non-linear correlation"""
        try:
            n = len(values)
            if n < 50:
                return {'mi': 0, 'detected': False, 'conclusion': 'Insufficient data'}

            x = values[:-1]
            y = values[1:]
            bins = min(20, n // 5)

            # Joint histogram
            hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
            pxy = hist_xy / hist_xy.sum()

            # Marginals
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)

            # MI
            mi = 0.0
            for i in range(bins):
                for j in range(bins):
                    if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                        mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))

            # Normalized MI (0-1)
            hx = -np.sum(px[px > 0] * np.log2(px[px > 0]))
            hy = -np.sum(py[py > 0] * np.log2(py[py > 0]))
            nmi = (2 * mi) / (hx + hy) if (hx + hy) > 0 else 0

            detected = nmi > 0.05

            return {
                'mi': round(float(mi), 6),
                'normalized_mi': round(float(nmi), 6),
                'detected': bool(detected),
                'conclusion': f'MI={mi:.4f}, NMI={nmi:.4f} — {"non-linear dependence" if detected else "independent"}'
            }
        except Exception as e:
            return {'error': str(e), 'mi': 0, 'detected': False}

    def _generate_honest_assessment(self, analysis, game_type):
        """Generate honest assessment of pattern findings"""
        total_patterns = 0
        detected_count = 0

        for key, result in analysis.items():
            if isinstance(result, dict):
                if result.get('detected', False):
                    detected_count += 1
                if 'total_sequences' in result:
                    total_patterns += result['total_sequences']
                elif 'num_clusters' in result:
                    total_patterns += result['num_clusters']
                elif 'num_streaks' in result:
                    total_patterns += result['num_streaks']
                elif 'regime_changes' in result:
                    total_patterns += result.get('regime_changes', 0)
                elif 'total_peaks' in result:
                    total_patterns += result['total_peaks']

        if detected_count == 0:
            assessment = "No significant patterns detected. Data appears random."
        elif detected_count < 3:
            assessment = f"Found {detected_count} minor signals — likely statistical noise."
        elif detected_count < 6:
            assessment = f"Found {detected_count} pattern signals — some may indicate regime shifts."
        else:
            assessment = f"Found {detected_count} strong signals — data shows structural anomalies."

        return assessment

    def _compute_edge_estimate(self, analysis):
        """Compute a 0-1 edge estimate from all pattern signals"""
        scores = []

        # Hurst exponent: H > 0.5 = trending, H < 0.5 = mean-reverting
        hurst = analysis.get('hurst_exponent', {})
        if isinstance(hurst, dict):
            h = hurst.get('hurst', 0.5)
            scores.append(abs(h - 0.5) * 2)  # Distance from random walk

        # Markov: deviation from independence
        markov = analysis.get('markov_chain', {})
        if isinstance(markov, dict) and markov.get('detected'):
            scores.append(0.6)
        else:
            scores.append(0.3)

        # CUSUM change points
        cusum = analysis.get('cusum_detection', {})
        if isinstance(cusum, dict) and cusum.get('detected'):
            scores.append(0.6)
        else:
            scores.append(0.3)

        # Mutual information
        mi = analysis.get('mutual_info', {})
        if isinstance(mi, dict) and mi.get('detected'):
            scores.append(0.65)
        else:
            scores.append(0.3)

        return round(float(np.mean(scores)), 4) if scores else 0.3

    def _reality_check_message(self):
        """Return a reality check message"""
        return """
        REALITY CHECK: In truly random data (like cryptographically secure RNG),
        patterns will ALWAYS appear by chance. The gambler's fallacy is the belief
        that past outcomes influence future ones. Each round is mathematically
        independent. No amount of pattern analysis can predict truly random outcomes.
        """

    def get_llm_insights(self, pattern_results):
        """Get LLM insights on pattern analysis"""
        if not self.client:
            return "LLM analysis unavailable - no API key configured"

        prompt = f"""
        You are a pattern recognition expert analyzing casino game data. Here are the pattern detection results:

        High Crash Clustering: {pattern_results['patterns'].get('high_crash_clustering', {})}
        Low Streaks: {pattern_results['patterns'].get('low_streak_detection', {})}
        Volatility Regimes: {pattern_results['patterns'].get('volatility_regime', {})}
        Repeating Sequences: {pattern_results['patterns'].get('repeating_sequences', {})}
        FFT Analysis: {pattern_results['patterns'].get('fft_analysis', {})}

        Honest Assessment: {pattern_results.get('honest_assessment', '')}

        Explain what these patterns (or lack thereof) mean for game fairness and prediction.
        Be honest about the limitations of pattern analysis on random data.
        Keep your response under 250 words.
        """

        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"LLM analysis failed: {e}"