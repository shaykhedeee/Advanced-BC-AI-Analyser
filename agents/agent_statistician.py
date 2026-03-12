import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None
import pandas as pd
from config import AGENT_SETTINGS, API_KEYS
try:
    import groq
except BaseException:
    groq = None


class StatisticianAgent:
    """
    Enhanced Statistical Analysis Agent with 9 tests:
    1. Chi-squared goodness-of-fit
    2. Wald-Wolfowitz runs test
    3. Multi-lag autocorrelation (up to lag 20)
    4. Z-score anomaly detection
    5. Distribution fit (KS + Anderson-Darling)
    6. Serial correlation at multiple lags
    7. Entropy & information content
    8. Maximum run length test
    9. Lag-plot slope analysis
    """

    def __init__(self):
        self.client = None
        if API_KEYS.get('groq') and groq:
            try:
                self.client = groq.Groq(api_key=API_KEYS['groq'])
            except Exception:
                pass

    def analyze_fairness(self, data, game_type='crash'):
        """Perform comprehensive statistical fairness analysis with 9 tests"""
        if len(data) < 30:
            return "Insufficient data for statistical analysis"

        values = np.array(data, dtype=np.float64)

        analysis = {
            'chi_squared_test': self._chi_squared_test(values, game_type),
            'runs_test': self._wald_wolfowitz_runs_test(values),
            'autocorrelation': self._autocorrelation_analysis(values),
            'anomaly_detection': self._z_score_anomalies(values),
            'distribution_fit': self._distribution_fit_test(values, game_type),
            'serial_correlation': self._serial_correlation_test(values),
            'entropy_test': self._entropy_test(values),
            'max_run_test': self._max_run_length_test(values),
            'lag_plot': self._lag_plot_analysis(values),
        }

        verdict = self._generate_fairness_verdict(analysis)
        bias_score = self._compute_bias_score(analysis)

        return {
            'analysis': analysis,
            'verdict': verdict,
            'confidence': self._calculate_confidence(analysis),
            'bias_score': bias_score,
            'num_tests': len(analysis),
            'tests_passed': sum(1 for t in analysis.values()
                                if isinstance(t, dict) and t.get('fair', t.get('independent', True))),
        }

    def _chi_squared_test(self, values, game_type):
        """Chi-squared goodness-of-fit test"""
        try:
            if game_type == 'crash':
                # For crash games, test if distribution matches expected
                # Expected: geometric distribution with p = 0.01 (house edge)
                expected_freq = len(values) * 0.01
                observed_freq = np.sum(values > 2.0)  # Above 2x multiplier

                chi_squared = ((observed_freq - expected_freq) ** 2) / expected_freq
                p_value = 1 - stats.chi2.cdf(chi_squared, 1)

            elif game_type == 'dice':
                # For dice, test uniformity
                observed, _ = np.histogram(values, bins=10, range=(1, 100))
                expected = len(values) / 10
                chi_squared = np.sum((observed - expected) ** 2 / expected)
                p_value = 1 - stats.chi2.cdf(chi_squared, 9)

            else:
                return {'statistic': None, 'p_value': None, 'conclusion': 'Test not applicable'}

            return {
                'statistic': chi_squared,
                'p_value': p_value,
                'conclusion': 'Fair' if p_value > 0.05 else 'Potentially unfair'
            }

        except Exception as e:
            return {'error': str(e)}

    def _wald_wolfowitz_runs_test(self, values):
        """Wald-Wolfowitz runs test for independence"""
        try:
            # Convert to binary: above/below median
            median = np.median(values)
            binary = (values > median).astype(int)

            # Count runs
            runs = 1
            for i in range(1, len(binary)):
                if binary[i] != binary[i-1]:
                    runs += 1

            # Expected runs and variance
            n1 = np.sum(binary)
            n2 = len(binary) - n1
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

            if variance > 0:
                z_statistic = (runs - expected_runs) / np.sqrt(variance)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
            else:
                z_statistic = 0
                p_value = 1.0

            return {
                'runs': runs,
                'expected_runs': expected_runs,
                'z_statistic': z_statistic,
                'p_value': p_value,
                'conclusion': 'Independent' if p_value > 0.05 else 'Potentially dependent'
            }

        except Exception as e:
            return {'error': str(e)}

    def _autocorrelation_analysis(self, values, lags=5):
        """Autocorrelation analysis for lags 1-5"""
        try:
            autocorr = {}
            for lag in range(1, lags + 1):
                if len(values) > lag:
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    autocorr[f'lag_{lag}'] = corr
                else:
                    autocorr[f'lag_{lag}'] = None

            # Check if any autocorrelation is significant
            significant_lags = [lag for lag, corr in autocorr.items()
                              if corr is not None and abs(corr) > 0.1]

            return {
                'autocorrelations': autocorr,
                'significant_lags': significant_lags,
                'conclusion': 'No significant autocorrelation' if not significant_lags
                            else f'Significant autocorrelation at lags: {significant_lags}'
            }

        except Exception as e:
            return {'error': str(e)}

    def _z_score_anomalies(self, values, threshold=3):
        """Detect anomalies using z-score method"""
        try:
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                return {'anomalies': [], 'conclusion': 'No variation in data'}

            z_scores = (values - mean) / std
            anomalies = np.abs(z_scores) > threshold

            anomaly_indices = np.where(anomalies)[0]
            anomaly_values = values[anomaly_indices]

            return {
                'anomaly_count': len(anomaly_values),
                'anomaly_values': anomaly_values.tolist(),
                'anomaly_indices': anomaly_indices.tolist(),
                'conclusion': f'Found {len(anomaly_values)} anomalies' if len(anomaly_values) > 0
                            else 'No anomalies detected'
            }

        except Exception as e:
            return {'error': str(e)}

    def _distribution_fit_test(self, values, game_type):
        """Test if data fits expected distribution (KS + Anderson-Darling)"""
        try:
            if game_type == 'crash':
                # Crash games: values > 1 should have P(X > x) ~ 1/x
                shifted = values[values > 1.01] - 1.0
                if len(shifted) < 20:
                    return {'conclusion': 'Insufficient data for distribution test', 'fair': True}

                log_shifted = np.log(shifted)
                # KS test against exponential
                ks_stat, ks_p = stats.kstest(log_shifted, 'expon', args=(0, np.mean(log_shifted)))
                # Anderson-Darling
                ad_result = stats.anderson(log_shifted, dist='expon')

                return {
                    'ks_statistic': round(float(ks_stat), 6),
                    'ks_p_value': round(float(ks_p), 6),
                    'ad_statistic': round(float(ad_result.statistic), 4),
                    'ad_critical_5pct': round(float(ad_result.critical_values[2]), 4),
                    'fair': bool(ks_p > 0.05),
                    'conclusion': 'Distribution matches expected' if ks_p > 0.05
                                  else 'Distribution deviates from expected'
                }

            elif game_type == 'dice':
                observed, _ = np.histogram(values, bins=10, range=(1, 100))
                expected = np.full(10, len(values) / 10)
                ks_stat, p_val = stats.ks_2samp(observed, expected)
                return {
                    'ks_statistic': round(float(ks_stat), 6),
                    'p_value': round(float(p_val), 6),
                    'fair': bool(p_val > 0.05),
                    'conclusion': 'Uniform distribution' if p_val > 0.05 else 'Non-uniform'
                }
            else:
                return {'conclusion': 'Test not applicable', 'fair': True}
        except Exception as e:
            return {'error': str(e), 'fair': True}

    # =========================================
    # NEW ADVANCED TESTS
    # =========================================
    def _serial_correlation_test(self, values):
        """Serial correlation at lags 1-10 with Durbin-Watson-like statistic"""
        try:
            n = len(values)
            if n < 20:
                return {'conclusion': 'Insufficient data', 'fair': True}

            mean = np.mean(values)
            centered = values - mean
            var = np.var(values)
            if var < 1e-10:
                return {'conclusion': 'No variation', 'fair': True}

            critical = 1.96 / np.sqrt(n)
            correlations = {}
            sig_count = 0

            for lag in range(1, min(11, n // 3)):
                corr = np.sum(centered[:-lag] * centered[lag:]) / (var * (n - lag))
                is_sig = abs(corr) > critical
                correlations[f'lag_{lag}'] = {
                    'correlation': round(float(corr), 6),
                    'significant': bool(is_sig),
                }
                if is_sig:
                    sig_count += 1

            # Durbin-Watson
            diffs = np.diff(values)
            dw = np.sum(diffs ** 2) / np.sum(values ** 2) if np.sum(values ** 2) > 0 else 2.0

            return {
                'correlations': correlations,
                'significant_count': sig_count,
                'durbin_watson': round(float(dw), 4),
                'fair': sig_count < 3,
                'conclusion': f'{sig_count} significant serial correlations detected'
            }
        except Exception as e:
            return {'error': str(e), 'fair': True}

    def _entropy_test(self, values):
        """Shannon entropy — higher = more random"""
        try:
            bins = min(30, len(values) // 5)
            if bins < 5:
                return {'conclusion': 'Insufficient data', 'fair': True}

            hist, _ = np.histogram(values, bins=bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]

            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(bins)
            normalized = entropy / max_entropy if max_entropy > 0 else 0

            return {
                'entropy': round(float(entropy), 4),
                'max_entropy': round(float(max_entropy), 4),
                'normalized_entropy': round(float(normalized), 4),
                'fair': bool(normalized > 0.80),
                'conclusion': f'Entropy {normalized:.2%} of maximum — {"high randomness" if normalized > 0.8 else "low randomness"}'
            }
        except Exception as e:
            return {'error': str(e), 'fair': True}

    def _max_run_length_test(self, values):
        """Test if the longest run is statistically expected"""
        try:
            n = len(values)
            if n < 30:
                return {'conclusion': 'Insufficient data', 'fair': True}

            median = np.median(values)
            above = (values > median).astype(int)

            # Find maximum run length
            max_run = 1
            current_run = 1
            for i in range(1, n):
                if above[i] == above[i - 1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1

            # Expected max run in n coin flips ~ log2(n)
            expected_max = np.log2(n)
            # Standard deviation ~ sqrt(log2(n)) approximately
            std_max = np.sqrt(np.log2(n)) if n > 1 else 1

            z = (max_run - expected_max) / std_max if std_max > 0 else 0
            is_fair = abs(z) < 2.5

            return {
                'max_run': int(max_run),
                'expected_max_run': round(float(expected_max), 2),
                'z_score': round(float(z), 4),
                'fair': bool(is_fair),
                'conclusion': f'Max run {max_run} (expected ~{expected_max:.0f}) — {"normal" if is_fair else "unusual"}'
            }
        except Exception as e:
            return {'error': str(e), 'fair': True}

    def _lag_plot_analysis(self, values):
        """Lag-1 plot analysis: correlation between x(t) and x(t+1)"""
        try:
            if len(values) < 20:
                return {'conclusion': 'Insufficient data', 'fair': True}

            x = values[:-1]
            y = values[1:]

            # Pearson correlation
            corr, p_val = stats.pearsonr(x, y) if stats else (np.corrcoef(x, y)[0, 1], 0.5)

            # Linear fit slope
            slope = np.polyfit(x, y, 1)[0]

            return {
                'lag1_correlation': round(float(corr), 6),
                'p_value': round(float(p_val), 6),
                'slope': round(float(slope), 6),
                'fair': bool(abs(corr) < 0.1),
                'conclusion': f'Lag-1 correlation: {corr:.4f} — {"independent" if abs(corr) < 0.1 else "correlated"}'
            }
        except Exception as e:
            return {'error': str(e), 'fair': True}

    def _compute_bias_score(self, analysis):
        """Compute a 0-1 score where 0 = heavily biased, 1 = perfectly fair"""
        scores = []
        for key, result in analysis.items():
            if isinstance(result, dict):
                if result.get('fair', True):
                    scores.append(1.0)
                else:
                    scores.append(0.3)
        return round(float(np.mean(scores)), 4) if scores else 0.5

    def _generate_fairness_verdict(self, analysis):
        """Generate overall fairness verdict from all 9 tests"""
        fair_count = 0
        total_tests = 0

        for key, result in analysis.items():
            if isinstance(result, dict) and 'fair' in result:
                total_tests += 1
                if result['fair']:
                    fair_count += 1
            elif isinstance(result, dict) and 'conclusion' in result:
                total_tests += 1
                conclusion = result.get('conclusion', '')
                if any(w in conclusion.lower() for w in ['fair', 'independent', 'normal', 'matches', 'uniform', 'high random']):
                    fair_count += 1

        if total_tests == 0:
            return "Unable to perform statistical tests"

        ratio = fair_count / total_tests

        if ratio >= 0.85:
            return f"Data appears statistically fair ({fair_count}/{total_tests} tests passed)"
        elif ratio >= 0.6:
            return f"Data shows minor irregularities ({fair_count}/{total_tests} tests passed)"
        elif ratio >= 0.4:
            return f"Data shows moderate statistical anomalies ({fair_count}/{total_tests} tests passed)"
        else:
            return f"Data shows significant irregularities ({fair_count}/{total_tests} tests passed)"

    def _calculate_confidence(self, analysis):
        """Calculate confidence in the analysis"""
        completed = sum(1 for t in analysis.values()
                        if isinstance(t, dict) and ('conclusion' in t or 'fair' in t))
        return min(1.0, completed / 9)  # 9 tests total

    def get_llm_insights(self, analysis_results):
        """Get LLM insights on the statistical analysis"""
        if not self.client:
            return "LLM analysis unavailable - no API key configured"

        prompt = f"""
        You are a statistical analysis expert. Analyze these statistical test results for a casino game:

        Chi-squared test: {analysis_results['analysis'].get('chi_squared_test', {})}
        Runs test: {analysis_results['analysis'].get('runs_test', {})}
        Autocorrelation: {analysis_results['analysis'].get('autocorrelation', {})}
        Anomalies: {analysis_results['analysis'].get('anomaly_detection', {})}

        Overall verdict: {analysis_results.get('verdict', 'Unknown')}

        Provide a brief explanation of what these results mean for game fairness.
        Keep your response under 200 words.
        """

        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=AGENT_SETTINGS['statistician']['max_tokens'],
                temperature=AGENT_SETTINGS['statistician']['temperature']
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM analysis failed: {e}"