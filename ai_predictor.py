import os
import json
import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from config import API_KEYS, AI_PREDICTION, GAME_SETTINGS, has_real_api_key


class AIPredictor:
    """Multi-API AI prediction system using Groq, Gemini, OpenRouter, and AI/ML API"""

    def __init__(self):
        self.clients = {}
        self.prediction_history = []
        self.api_stats = {}
        self._init_clients()

    @property
    def active_apis(self):
        return self.clients

    def _init_clients(self):
        """Initialize all AI API clients"""
        try:
            if has_real_api_key("groq"):
                from openai import OpenAI
                self.clients["groq"] = OpenAI(
                    api_key=API_KEYS["groq"],
                    base_url=AI_PREDICTION["groq"]["base_url"]
                )
                self.api_stats["groq"] = {"calls": 0, "success": 0, "total_time": 0, "errors": 0}
                print("[AI] Groq client initialized ({})".format(AI_PREDICTION['groq']['model']))
        except Exception as e:
            print("[AI] Groq init failed: {}".format(e))

        try:
            if has_real_api_key("google_gemini"):
                import importlib
                genai = importlib.import_module("google.generativeai")
                genai.configure(api_key=API_KEYS["google_gemini"])
                self.clients["google_gemini"] = genai.GenerativeModel(
                    model_name=AI_PREDICTION["google_gemini"]["model"]
                )
                self.api_stats["google_gemini"] = {"calls": 0, "success": 0, "total_time": 0, "errors": 0}
                print("[AI] Google Gemini initialized ({})".format(AI_PREDICTION['google_gemini']['model']))
        except BaseException as e:
            print("[AI] Gemini init failed: {}".format(e))

        try:
            if has_real_api_key("openrouter"):
                from openai import OpenAI
                self.clients["openrouter"] = OpenAI(
                    api_key=API_KEYS["openrouter"],
                    base_url=AI_PREDICTION["openrouter"]["base_url"]
                )
                self.api_stats["openrouter"] = {"calls": 0, "success": 0, "total_time": 0, "errors": 0}
                print("[AI] OpenRouter initialized ({})".format(AI_PREDICTION['openrouter']['model']))
        except Exception as e:
            print("[AI] OpenRouter init failed: {}".format(e))

        try:
            if has_real_api_key("aiml_api"):
                from openai import OpenAI
                self.clients["aiml_api"] = OpenAI(
                    api_key=API_KEYS["aiml_api"],
                    base_url=AI_PREDICTION["aiml_api"]["base_url"]
                )
                self.api_stats["aiml_api"] = {"calls": 0, "success": 0, "total_time": 0, "errors": 0}
                print("[AI] AI/ML API initialized ({})".format(AI_PREDICTION['aiml_api']['model']))
        except Exception as e:
            print("[AI] AI/ML API init failed: {}".format(e))

        print("[AI] {} AI APIs connected".format(len(self.clients)))

    def _normalize_confidence(self, value):
        """Normalize confidence to 0..1 whether the source returns 0..1 or 0..100."""
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.5
        if conf > 1.0:
            conf /= 100.0
        return max(0.0, min(1.0, conf))

    def _local_statistical_fallback(self, data, game_type='crash'):
        """Offline-safe heuristic prediction when external APIs are unavailable."""
        values = np.array(data[-100:], dtype=float)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values)) if len(values) > 1 else 0.0
        median_val = float(np.median(values))
        last_val = float(values[-1])

        if game_type == 'crash':
            threshold = max(2.0, mean_val)
            direction = 'above' if last_val < threshold else 'below'
            predicted_value = mean_val if last_val >= threshold else max(median_val, 2.0)
            pattern = 'mean_reversion_crash_band'
        else:
            direction = 'above' if last_val < mean_val else 'below'
            predicted_value = mean_val
            pattern = 'mean_reversion'

        risk_level = 'low' if std_val < mean_val * 0.25 else 'medium' if std_val < mean_val * 0.75 else 'high'

        return {
            'source': 'local_stats',
            'direction': direction,
            'predicted_value': round(predicted_value, 4),
            'confidence': 0.58,
            'pattern': pattern,
            'pattern_detected': pattern,
            'reasoning': (
                f'Offline statistical fallback using mean reversion and volatility analysis. '
                f'mean={mean_val:.4f}, median={median_val:.4f}, std={std_val:.4f}, last={last_val:.4f}.'
            ),
            'risk_level': risk_level,
            'latency': 0.0,
        }

    # ── Query Methods ──

    def _query_openai_compatible(self, client, api_name, prompt):
        """Query an OpenAI-compatible API (Groq, OpenRouter, AI/ML)"""
        cfg = AI_PREDICTION[api_name]
        start = time.time()
        self.api_stats[api_name]["calls"] += 1

        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": "You are a mathematical sequence analysis AI. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=cfg.get("temperature", 0.3),
                max_tokens=cfg.get("max_tokens", 500),
            )
            text = response.choices[0].message.content.strip()

            # Clean markdown fences
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            result = json.loads(text)
            latency = time.time() - start
            self.api_stats[api_name]["success"] += 1
            self.api_stats[api_name]["total_time"] += latency
            result["source"] = api_name
            result["latency"] = latency
            result["confidence"] = self._normalize_confidence(result.get("confidence", 0.5))
            result["pattern_detected"] = result.get("pattern_detected", result.get("pattern", "unknown"))
            return result

        except Exception as e:
            self.api_stats[api_name]["errors"] += 1
            print("[AI] {} error: {}".format(api_name, e))
            return None

    def _query_gemini(self, prompt):
        """Query Google Gemini API"""
        if "google_gemini" not in self.clients:
            return None

        start = time.time()
        self.api_stats["google_gemini"]["calls"] += 1

        try:
            response = self.clients["google_gemini"].generate_content(prompt)
            text = response.text.strip()

            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            result = json.loads(text)
            latency = time.time() - start
            self.api_stats["google_gemini"]["success"] += 1
            self.api_stats["google_gemini"]["total_time"] += latency
            result["source"] = "google_gemini"
            result["latency"] = latency
            result["confidence"] = self._normalize_confidence(result.get("confidence", 0.5))
            result["pattern_detected"] = result.get("pattern_detected", result.get("pattern", "unknown"))
            return result

        except Exception as e:
            self.api_stats["google_gemini"]["errors"] += 1
            print("[AI] Gemini error: {}".format(e))
            return None

    # ── Prompt Building ──

    def _build_prediction_prompt(self, data, game_type, context=None):
        """Build prediction prompt with statistical context"""
        values = np.array(data[-100:])
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        median_val = float(np.median(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))

        recent_10 = [round(float(x), 2) for x in values[-10:]]
        recent_30 = [round(float(x), 2) for x in values[-30:]]

        # Trend analysis
        pct_above = float(np.mean(values > mean_val) * 100)
        last_5_above = sum(1 for x in values[-5:] if x > mean_val)

        prompt = (
            "You are a mathematical sequence analysis AI. "
            "Analyze this {} game data and predict whether the NEXT value "
            "will be ABOVE or BELOW the mean ({:.4f}).\n\n"
            "STATISTICS:\n"
            "- Mean: {:.4f}\n"
            "- Std Dev: {:.4f}\n"
            "- Median: {:.4f}\n"
            "- Min: {:.4f}, Max: {:.4f}\n"
            "- Pct Above Mean: {:.1f}%\n"
            "- Last 5 above mean: {}/5\n\n"
            "LAST 10 VALUES: {}\n"
            "LAST 30 VALUES: {}\n\n"
        ).format(
            game_type, mean_val,
            mean_val, std_val, median_val,
            min_val, max_val,
            pct_above, last_5_above,
            recent_10, recent_30
        )

        if context:
            prompt += "ADDITIONAL CONTEXT: {}\n\n".format(context)

        prompt += (
            "Respond ONLY with valid JSON:\n"
            '{\n'
            '  "direction": "above" or "below",\n'
            '  "predicted_value": <number>,\n'
            '  "confidence": <0-100>,\n'
            '  "pattern": "<description>",\n'
            '  "reasoning": "<brief explanation>"\n'
            '}\n'
        )
        return prompt

    # ── Main Prediction ──

    def predict_next(self, data, game_type='crash', context=None):
        """Query ALL AI APIs in parallel and build consensus prediction"""
        if len(data) < 20:
            return {
                'error': 'Need at least 20 data points',
                'predictions': {},
                'consensus': None
            }

        prompt = self._build_prediction_prompt(data, game_type, context)

        if not self.clients:
            fallback = self._local_statistical_fallback(data, game_type)
            consensus = self._build_consensus({'local_stats': fallback}, data)
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'game_type': game_type,
                'predictions': {'local_stats': fallback},
                'consensus': consensus,
            })
            return {
                'predictions': {'local_stats': fallback},
                'consensus': consensus,
                'apis_responded': 1,
                'api_count': 1,
                'timestamp': datetime.now().isoformat(),
                'mode': 'offline_fallback',
            }

        # Query all APIs in parallel
        results = {}
        threads = []

        def query_api(name):
            if name == "google_gemini":
                r = self._query_gemini(prompt)
            else:
                client = self.clients.get(name)
                if client:
                    r = self._query_openai_compatible(client, name, prompt)
                else:
                    r = None
            if r:
                results[name] = r

        for name in self.clients:
            t = threading.Thread(target=query_api, args=(name,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        if not results:
            return {
                'error': 'All API calls failed',
                'predictions': results,
                'consensus': None
            }

        # Build consensus
        consensus = self._build_consensus(results, data)

        # Store prediction
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'game_type': game_type,
            'predictions': results,
            'consensus': consensus,
        })

        return {
            'predictions': results,
            'consensus': consensus,
            'apis_responded': len(results),
            'api_count': len(results),
            'timestamp': datetime.now().isoformat(),
        }

    def _build_consensus(self, results, data):
        """Build consensus prediction from multiple API responses"""
        above_votes = 0
        below_votes = 0
        predicted_values = []
        confidences = []

        for api_name, r in results.items():
            direction = str(r.get("direction", "")).lower()
            if "above" in direction:
                above_votes += 1
            else:
                below_votes += 1

            pv = r.get("predicted_value")
            if pv is not None:
                try:
                    predicted_values.append(float(pv))
                except (ValueError, TypeError):
                    pass

            confidences.append(self._normalize_confidence(r.get("confidence", 0.5)))

        total = above_votes + below_votes
        agreement = max(above_votes, below_votes) / total if total > 0 else 0

        if above_votes > below_votes:
            direction = "above"
        elif below_votes > above_votes:
            direction = "below"
        else:
            direction = "neutral"

        avg_confidence = float(np.mean(confidences)) if confidences else 0.5
        consensus_value = float(np.mean(predicted_values)) if predicted_values else float(np.mean(data[-50:]))

        risk_level = 'low'
        if agreement < 0.51 or avg_confidence < 0.45:
            risk_level = 'high'
        elif agreement < 0.67 or avg_confidence < 0.6:
            risk_level = 'medium'

        return {
            "direction": direction,
            "predicted_value": round(consensus_value, 4),
            "confidence": round(avg_confidence, 1),
            "agreement": round(agreement, 2),
            "votes": {"above": above_votes, "below": below_votes},
            "apis_responded": total,
            "risk_level": risk_level,
        }

    # ── Accuracy Tracking ──

    def update_accuracy(self, actual_value, data_mean):
        """Update accuracy tracking with actual outcome"""
        if not self.prediction_history:
            return

        last = self.prediction_history[-1]
        consensus = last.get("consensus", {})
        predicted_dir = consensus.get("direction", "")

        actual_dir = "above" if actual_value > data_mean else "below"
        correct = (predicted_dir == actual_dir)

        last["actual_value"] = actual_value
        last["actual_direction"] = actual_dir
        last["correct"] = correct

    def get_accuracy_stats(self):
        """Get accuracy statistics"""
        evaluated = [p for p in self.prediction_history if "correct" in p]
        if not evaluated:
            return {
                "total_predictions": len(self.prediction_history),
                "evaluated": 0,
                "accuracy": 0.0,
                "recent_accuracy": 0.0,
                "avg_confidence": 0.0,
            }

        correct = sum(1 for p in evaluated if p["correct"])
        total = len(evaluated)
        recent = evaluated[-20:]
        recent_correct = sum(1 for p in recent if p["correct"])

        confidences = []
        for p in evaluated:
            c = p.get("consensus", {}).get("confidence", 50)
            confidences.append(float(c))

        return {
            "total_predictions": len(self.prediction_history),
            "evaluated": total,
            "accuracy": round(correct / total, 3) if total > 0 else 0.0,
            "recent_accuracy": round(recent_correct / len(recent), 3) if recent else 0.0,
            "recent_accuracy_20": round(recent_correct / len(recent), 3) if recent else 0.0,
            "avg_confidence": round(float(np.mean(confidences)), 1) if confidences else 0.0,
        }

    # ── Health & Analysis ──

    def get_api_health(self):
        """Get health status of all AI APIs"""
        health = {}
        for api_name, stats in self.api_stats.items():
            total = stats["calls"]
            health[api_name] = {
                "connected": api_name in self.clients,
                "total_calls": total,
                "success": stats["success"],
                "errors": stats["errors"],
                "success_rate": round(stats["success"] / total, 2) if total > 0 else 0,
                "avg_latency": round(stats["total_time"] / total, 2) if total > 0 else 0,
            }
        return health

    def get_detailed_analysis(self, data, game_type='crash'):
        """Deep analysis from best available API"""
        if len(data) < 30:
            return "Need at least 30 data points for detailed analysis"

        values = np.array(data[-200:] if len(data) > 200 else data)

        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        median_val = float(np.median(values))
        last_50 = [round(float(x), 2) for x in values[-50:]]

        prompt = (
            "You are an expert data scientist analyzing {} game sequences.\n\n"
            "DATASET ({} values):\n"
            "Mean: {:.4f}, Std: {:.4f}\n"
            "Min: {:.4f}, Max: {:.4f}\n"
            "Median: {:.4f}\n\n"
            "LAST 50 VALUES: {}\n\n"
            "Analyze this data for:\n"
            "1. Hidden patterns or cycles\n"
            "2. Distribution anomalies\n"
            "3. Serial correlation\n"
            "4. Cluster behavior\n"
            "5. Optimal prediction strategy\n"
            "6. Confidence in predictability\n\n"
            "Provide a structured analysis."
        ).format(
            game_type, len(values),
            mean_val, std_val,
            min_val, max_val,
            median_val,
            last_50
        )

        # Query best available API
        for api_name in ['groq', 'openrouter', 'aiml_api']:
            if api_name not in self.clients:
                continue
            try:
                cfg = AI_PREDICTION[api_name]
                response = self.clients[api_name].chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2000,
                )
                return response.choices[0].message.content
            except Exception as e:
                print("[AI] Analysis via {} failed: {}".format(api_name, e))

        # Try Gemini
        if "google_gemini" in self.clients:
            try:
                response = self.clients["google_gemini"].generate_content(prompt)
                return response.text
            except Exception as e:
                print("[AI] Gemini analysis failed: {}".format(e))

        return "All APIs failed for detailed analysis"


class ContinuousPredictor:
    """Background auto-prediction loop"""

    def __init__(self, ai_predictor, data_engine, ml_brain):
        self.ai_predictor = ai_predictor
        self.data_engine = data_engine
        self.ml_brain = ml_brain
        self.listeners = []
        self.running = False
        self.thread = None

    def add_listener(self, callback):
        self.listeners.append(callback)

    def notify(self, prediction):
        for cb in self.listeners:
            try:
                cb(prediction)
            except Exception as e:
                print("[ContinuousPredictor] Listener error: {}".format(e))

    def start(self, game_type='crash', interval=5.0):
        """Start continuous prediction"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._loop, args=(game_type, interval), daemon=True
        )
        self.thread.start()

    def stop(self):
        """Stop continuous prediction"""
        self.running = False

    def _loop(self, game_type, interval):
        """Main prediction loop"""
        while self.running:
            try:
                df = self.data_engine.get_dataframe(game_type, n_points=100)
                if len(df) < 20:
                    time.sleep(interval)
                    continue

                data = df['value'].tolist()
                prediction = self.ai_predictor.predict_next(data, game_type)

                if prediction and prediction.get('consensus'):
                    self.notify(prediction)

            except Exception as e:
                print("[ContinuousPredictor] Loop error: {}".format(e))

            time.sleep(interval)

    def get_prediction_summary(self):
        """Alias for get_summary() — dashboard compatibility."""
        return self.get_summary()

    def get_summary(self):
        """Get prediction summary"""
        stats = self.ai_predictor.get_accuracy_stats()
        health = self.ai_predictor.get_api_health()

        lines = ["=== CONTINUOUS PREDICTION SUMMARY ==="]
        lines.append("Total Predictions: {}".format(stats['total_predictions']))
        lines.append("Evaluated: {}".format(stats['evaluated']))
        lines.append("Accuracy: {:.1%}".format(stats['accuracy']))
        lines.append("Recent Accuracy (20): {:.1%}".format(stats['recent_accuracy']))
        lines.append("Avg Confidence: {:.1f}%".format(stats['avg_confidence']))
        lines.append("")

        for api, h in health.items():
            status = "OK" if h["connected"] else "DISCONNECTED"
            lines.append("  {}: {} | calls={} success_rate={:.0%}".format(api, status, h['total_calls'], h['success_rate']))

        return "\n".join(lines)
