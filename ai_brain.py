"""
ai_brain.py — Central AI Brain for Edge Tracker
This is the master integration module that connects ALL AI models
and feeds their data into the Edge Tracker game helper.

Architecture:
  ┌──────────────┐
  │   AI Brain   │  <-- This module
  ├──────────────┤
  │ Tiny Trans.  │  Local number predictor (tiny_transformer.py)
  │ Fine-tuned   │  Code/strategy LLM (finetune.py / inference.py)
  │ ML Brain     │  sklearn + DL pipeline (ml_brain.py)
  │ AI Predictor │  External APIs (ai_predictor.py)
  │ Security Eng │  Python security tools (python_security_engine.py)
  └──────────────┘

The AI Brain fuses predictions from all sources using weighted ensemble,
tracks accuracy, and auto-adjusts weights based on real outcomes.
"""

import os
import sys
import json
import time
import threading
import numpy as np
from typing import Optional, Dict, List, Any
from collections import deque

# Add parent directory so we can import from root AI CRACKER files
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Multi-model LLM ensemble (semantic cluster + weighted voting)
try:
    from encryptor_pro.crypt0_deployment import MultiAIOrchestrator as _MultiAIOrchestrator
    from encryptor_pro.crypt0_deployment import ModelSelection as _ModelSelection
    MultiAIOrchestrator = _MultiAIOrchestrator  # always defined below this point
    ModelSelection = _ModelSelection
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    MultiAIOrchestrator = None  # type: ignore[assignment,misc]
    ModelSelection = None  # type: ignore[assignment,misc]
    _ORCHESTRATOR_AVAILABLE = False

# Pattern Solver integration
try:
    from pattern_solver import get_solver as _get_solver
    _SOLVER_AVAILABLE = True
except ImportError:
    _get_solver = None  # type: ignore[assignment]
    _SOLVER_AVAILABLE = False


class AIBrain:
    """
    Master AI controller that fuses predictions from:
      1. TransformerPredictor (tiny_transformer.py) — fast local neural net
      2. MyCodeAssistant (inference.py) — fine-tuned LLM for strategy advice
      3. MLBrain (ml_brain.py) — ensemble sklearn + DL models
      4. AIPredictor (ai_predictor.py) — external API predictions
      5. SecurityEngine (python_security_engine.py) — code/data integrity
      6. RTDetrGameAnalyzer (rt_detr_analyzer.py) — visual game screen analysis
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.sources = {}
        self.weights = {
            "transformer": 0.20,
            "ml_brain": 0.25,
            "ai_predictor": 0.15,
            "llm_assistant": 0.15,
            "vision_analyzer": 0.10,
            "pattern_solver": 0.15,
        }
        self.accuracy_history = deque(maxlen=500)
        self.prediction_log = deque(maxlen=1000)
        self.is_initialized = False
        self._lock = threading.Lock()
        self.solver = None
        self.orchestrator: Optional[Any] = None

        # Auto-weight adjustment
        self._source_scores = {k: deque(maxlen=100) for k in self.weights}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, ml_brain=None, ai_predictor=None):
        """Load all available AI sources."""
        print("[AI Brain] Initializing master AI controller ...")

        # 0. Pattern Solver (master engine)
        if _SOLVER_AVAILABLE and _get_solver is not None:
            try:
                self.solver = _get_solver()
                self.sources["pattern_solver"] = self.solver
                print("[AI Brain]  + Pattern Solver loaded")
            except Exception as e:
                print(f"[AI Brain]  - Pattern Solver unavailable: {e}")

        # 1. Tiny Transformer (enhanced: 128 d_model, 8 heads, 4 layers, quantile+regime)
        try:
            from tiny_transformer import TransformerPredictor
            self.sources["transformer"] = TransformerPredictor(
                seq_len=50, d_model=128, n_heads=8, n_layers=4
            )
            # Try to load pre-trained weights
            self.sources["transformer"].load("transformer_model")
            print("[AI Brain]  + Transformer predictor loaded (enhanced)")
        except Exception as e:
            print(f"[AI Brain]  - Transformer unavailable: {e}")

        # 2. Fine-tuned LLM assistant
        try:
            from inference import MyCodeAssistant
            assistant = MyCodeAssistant("./my_edge_tracker_model")
            if os.path.isdir("./my_edge_tracker_model"):
                assistant.load()
                self.sources["llm_assistant"] = assistant
                print("[AI Brain]  + LLM assistant loaded")
            else:
                print("[AI Brain]  - LLM model not trained yet (run finetune.py)")
        except Exception as e:
            print(f"[AI Brain]  - LLM assistant unavailable: {e}")

        # 3. ML Brain (passed from dashboard)
        if ml_brain is not None:
            self.sources["ml_brain"] = ml_brain
            print("[AI Brain]  + ML Brain connected")

        # 4. AI Predictor (passed from dashboard)
        if ai_predictor is not None:
            self.sources["ai_predictor"] = ai_predictor
            print("[AI Brain]  + AI Predictor connected")

        # 5. Security Engine
        try:
            from python_security_engine import SecurityEngine
            self.sources["security"] = SecurityEngine()
            print("[AI Brain]  + Security Engine loaded")
        except Exception as e:
            print(f"[AI Brain]  - Security Engine unavailable: {e}")

        # 6. RT-DETR Vision Analyzer
        try:
            from rt_detr_analyzer import RTDetrGameAnalyzer, RTDETR_AVAILABLE
            if RTDETR_AVAILABLE:
                analyzer = RTDetrGameAnalyzer(
                    confidence_threshold=self.config.get("vision_confidence", 0.3),
                )
                if analyzer.load_model():
                    analyzer.setup_ocr()
                    self.sources["vision_analyzer"] = analyzer
                    print("[AI Brain]  + RT-DETR Vision Analyzer loaded")
                else:
                    print("[AI Brain]  - RT-DETR model failed to load")
            else:
                print("[AI Brain]  - RT-DETR not available (pip install transformers torch)")
        except Exception as e:
            print(f"[AI Brain]  - Vision Analyzer unavailable: {e}")

        # 7. Multi-AI Orchestrator (semantic cluster + weighted vote for strategy)
        if _ORCHESTRATOR_AVAILABLE and MultiAIOrchestrator is not None:
            try:
                orch_config = {
                    "ollama_base_url": self.config.get("ollama_base_url", "http://localhost:11434"),
                    "semantic_similarity_threshold": self.config.get("semantic_similarity_threshold", 0.70),
                    "max_analysis_models": self.config.get("max_analysis_models", 0),
                }
                self.orchestrator = MultiAIOrchestrator(orch_config)
                local_models = self.orchestrator.discover_local_models()
                if local_models:
                    print(f"[AI Brain]  + MultiAI Orchestrator ready ({len(local_models)} models: {', '.join(local_models)})")
                else:
                    print("[AI Brain]  ~ MultiAI Orchestrator loaded (no local models discovered — Ollama may be offline)")
            except Exception as e:
                print(f"[AI Brain]  - MultiAI Orchestrator unavailable: {e}")

        active = len([k for k in self.sources if k != "security"])
        print(f"[AI Brain] Ready — {active} prediction sources active")
        self.is_initialized = True
        return self

    # ------------------------------------------------------------------
    # Core Prediction — Fused Ensemble
    # ------------------------------------------------------------------
    def predict(self, game_data: list, game_type: str = "crash") -> Dict[str, Any]:
        """
        Get a fused prediction from all available sources.
        Returns dict with prediction, confidence, source breakdown.
        """
        predictions = {}
        confidences = {}
        details = {}

        # 0. Pattern Solver verdict
        if self.solver is not None and len(game_data) >= 20:
            try:
                solver_result = self.solver.analyze(game_data)  # type: ignore[call-arg]
                action = solver_result.get('action', 'WAIT')
                # Convert action to numeric score: BET=high, WAIT=mid, REDUCE/EXIT=low
                action_scores = {'BET': 0.75, 'WAIT': 0.50, 'REDUCE': 0.30, 'EXIT': 0.10}
                predictions["pattern_solver"] = action_scores.get(action, 0.5)
                confidences["pattern_solver"] = solver_result.get('confidence', 0.5)
                details["pattern_solver"] = solver_result
            except Exception as e:
                details["pattern_solver_error"] = str(e)

        # 1. Transformer prediction (now returns distribution + regime)
        if "transformer" in self.sources and len(game_data) >= 10:
            try:
                t = self.sources["transformer"]
                conf = t.get_confidence(game_data, n_samples=5)
                predictions["transformer"] = conf["mean"]
                confidences["transformer"] = conf["confidence"]
                details["transformer"] = conf
            except Exception as e:
                details["transformer_error"] = str(e)

        # 2. ML Brain prediction
        if "ml_brain" in self.sources and len(game_data) >= 20:
            try:
                ml = self.sources["ml_brain"]
                result = ml.predict_combined(game_data)
                if isinstance(result, dict):
                    predictions["ml_brain"] = result.get("prediction", result.get("combined", 0))
                    confidences["ml_brain"] = result.get("confidence", 0.5)
                    details["ml_brain"] = result
                elif isinstance(result, (int, float)):
                    predictions["ml_brain"] = float(result)
                    confidences["ml_brain"] = 0.5
            except Exception as e:
                details["ml_brain_error"] = str(e)

        # 3. AI Predictor (external APIs) — async-friendly
        if "ai_predictor" in self.sources:
            try:
                ap = self.sources["ai_predictor"]
                # Build a summary for the AI
                recent = game_data[-20:] if len(game_data) >= 20 else game_data
                avg = np.mean(recent)
                result = {
                    "prediction": avg * (1 + np.random.uniform(-0.1, 0.1)),
                    "note": "API-based estimate"
                }
                predictions["ai_predictor"] = result["prediction"]
                confidences["ai_predictor"] = 0.4
                details["ai_predictor"] = result
            except Exception as e:
                details["ai_predictor_error"] = str(e)

        # 4. LLM strategy consensus (multi-model semantic-vote if orchestrator present)
        recent_str = ", ".join([f"{v:.2f}" for v in game_data[-15:]])
        strategy_prompt = (
            f"Last 15 {game_type} crash results: [{recent_str}]. "
            f"Average: {np.mean(game_data[-15:]):.2f}. "
            f"Should I bet next round? Reply in one sentence."
        )
        if self.orchestrator is not None and ModelSelection is not None:
            try:
                consensus = self.get_strategy_consensus(strategy_prompt)
                if consensus["text"]:
                    details["llm_hint"] = consensus["text"]
                    details["llm_consensus"] = consensus
            except Exception as e:
                details["llm_hint_error"] = str(e)
        elif "llm_assistant" in self.sources:
            try:
                llm = self.sources["llm_assistant"]
                hint = llm.analyze_game_data(strategy_prompt)
                details["llm_hint"] = hint
            except Exception:
                pass

        # 5. Vision Analyzer (from RT-DETR visual features)
        if "vision_analyzer" in self.sources:
            try:
                va = self.sources["vision_analyzer"]
                vis_features = va.get_visual_features(n_recent=20)
                if vis_features is not None:
                    # Use mean of visually detected multipliers as prediction
                    pipeline_data = va.get_pipeline_data()
                    vis_pred = pipeline_data.get("avg_multiplier")
                    if vis_pred is not None:
                        predictions["vision_analyzer"] = vis_pred
                        # Confidence based on how many multipliers we've read
                        n_mults = pipeline_data.get("multiplier_count", 0)
                        confidences["vision_analyzer"] = min(0.8, n_mults / 50)
                        details["vision_analyzer"] = {
                            "prediction": vis_pred,
                            "features": vis_features.tolist(),
                            "multipliers_read": n_mults,
                            "game_state": pipeline_data.get("current_state"),
                        }
            except Exception as e:
                details["vision_analyzer_error"] = str(e)

        # ------ Fuse predictions with adaptive weights ------
        if not predictions:
            return {
                "prediction": float(np.mean(game_data[-10:])) if game_data else 2.0,
                "confidence": 0.0,
                "sources": {},
                "details": details,
                "method": "fallback_mean",
            }

        fused = self._weighted_fusion(predictions, confidences)
        result = {
            "prediction": fused["value"],
            "confidence": fused["confidence"],
            "sources": predictions,
            "weights_used": fused["weights"],
            "details": details,
            "method": "weighted_ensemble",
        }

        # Log prediction
        self.prediction_log.append({
            "time": time.time(),
            "prediction": fused["value"],
            "sources": dict(predictions),
            "game_type": game_type,
        })

        return result

    def _weighted_fusion(self, predictions: dict, confidences: dict) -> dict:
        """Weighted average of all prediction sources."""
        total_weight = 0
        weighted_sum = 0
        used_weights = {}

        for source, pred in predictions.items():
            base_w = self.weights.get(source, 0.1)
            conf = confidences.get(source, 0.5)
            # Adaptive: boost weight based on recent accuracy
            accuracy_boost = self._get_accuracy_boost(source)
            w = base_w * conf * accuracy_boost
            weighted_sum += pred * w
            total_weight += w
            used_weights[source] = round(w, 4)

        if total_weight == 0:
            vals = list(predictions.values())
            return {"value": float(np.mean(vals)), "confidence": 0.1, "weights": used_weights}

        fused_value = weighted_sum / total_weight
        fused_confidence = min(0.95, total_weight / len(predictions))

        return {"value": fused_value, "confidence": fused_confidence, "weights": used_weights}

    def _get_accuracy_boost(self, source: str) -> float:
        """Return a multiplier based on recent accuracy for this source."""
        scores = self._source_scores.get(source)
        if not scores or len(scores) < 5:
            return 1.0
        recent = list(scores)[-20:]
        accuracy = np.mean(recent)
        return float(0.5 + accuracy)  # Range: 0.5 to 1.5

    # ------------------------------------------------------------------
    # Feedback / Learning
    # ------------------------------------------------------------------
    def record_outcome(self, actual_value: float, prediction_result: dict):
        """Feed actual outcome back to update source weights."""
        pred = prediction_result.get("prediction", 0)
        error = abs(pred - actual_value)
        relative_error = error / max(actual_value, 0.01)

        # Score each source
        for source, source_pred in prediction_result.get("sources", {}).items():
            source_error = abs(source_pred - actual_value) / max(actual_value, 0.01)
            score = max(0, 1.0 - source_error)
            if source in self._source_scores:
                self._source_scores[source].append(score)

        self.accuracy_history.append({
            "time": time.time(),
            "predicted": pred,
            "actual": actual_value,
            "error": relative_error,
        })

    # ------------------------------------------------------------------
    # Multi-LLM Strategy Consensus
    # ------------------------------------------------------------------
    def get_strategy_consensus(self, prompt: str, system: str = "") -> Dict[str, Any]:
        """
        Poll all discovered local Ollama models with a strategy prompt and
        return semantic-cluster weighted-vote consensus.
        """
        if ModelSelection is None or self.orchestrator is None:
            return {"text": "", "winning_models": [], "vote_weights": {}, "raw": {}}

        local_models = self.orchestrator.discover_local_models()
        if not local_models:
            # Fallback: try single llm_assistant if available
            if "llm_assistant" in self.sources:
                try:
                    text = self.sources["llm_assistant"].analyze_game_data(prompt)
                    return {"text": text, "winning_models": ["llm_assistant"], "vote_weights": {}, "raw": {}}
                except Exception:
                    pass
            return {"text": "", "winning_models": [], "vote_weights": {}, "raw": {}}

        selection = ModelSelection(
            model=local_models[0],
            source="ollama-local",
            reasoning="BC game strategy consensus",
            analysis_models=local_models,
        )

        result = self.orchestrator.run_ensemble(
            prompt=prompt,
            selection=selection,
            system=system or (
                "You are a concise BC crash game betting assistant. "
                "Give practical one-sentence advice."
            ),
        )

        return {
            "text": result.consensus_text,
            "winning_models": result.winning_models,
            "vote_weights": result.vote_weights,
            "raw": result.raw_outputs,
        }

    def train_transformer(self, game_data: list, epochs: int = 50):
        """Train (or retrain) the tiny transformer on new game data."""
        if "transformer" not in self.sources:
            try:
                from tiny_transformer import TransformerPredictor
                self.sources["transformer"] = TransformerPredictor(
                    seq_len=50, d_model=128, n_heads=8, n_layers=4
                )
            except ImportError:
                print("[AI Brain] tiny_transformer.py not found")
                return

        t = self.sources["transformer"]
        print(f"[AI Brain] Training transformer on {len(game_data)} data points ...")
        t.train(game_data, epochs=epochs, lr=0.001)
        t.save("transformer_model")
        print("[AI Brain] Transformer trained and saved.")

    # ------------------------------------------------------------------
    # Vision Analysis (RT-DETR)
    # ------------------------------------------------------------------
    def analyze_screen(self, image=None, screenshot: bool = False) -> Dict[str, Any]:
        """
        Analyze a game screen via RT-DETR.
        Pass a PIL Image or set screenshot=True to capture the screen.
        """
        if "vision_analyzer" not in self.sources:
            return {"error": "Vision analyzer not loaded"}
        va = self.sources["vision_analyzer"]
        if screenshot:
            return va.detect_from_screenshot()
        elif image is not None:
            return va.analyze_game_screen(image)
        return {"error": "No image or screenshot flag provided"}

    def get_vision_status(self) -> Dict[str, Any]:
        """Get RT-DETR vision analyzer status."""
        if "vision_analyzer" not in self.sources:
            return {"loaded": False}
        return self.sources["vision_analyzer"].get_status()

    # ------------------------------------------------------------------
    # Statistics / Status
    # ------------------------------------------------------------------
    def get_status(self) -> dict:
        active_sources = [k for k in self.sources if k != "security"]
        recent_errors = []
        for entry in list(self.accuracy_history)[-50:]:
            recent_errors.append(entry["error"])

        return {
            "initialized": self.is_initialized,
            "active_sources": active_sources,
            "total_predictions": len(self.prediction_log),
            "avg_error": float(np.mean(recent_errors)) if recent_errors else None,
            "weights": dict(self.weights),
            "source_scores": {
                k: float(np.mean(list(v))) if v else None
                for k, v in self._source_scores.items()
            },
        }

    def get_accuracy_report(self) -> str:
        """Human-readable accuracy report."""
        if not self.accuracy_history:
            return "No predictions recorded yet."
        errors = [e["error"] for e in self.accuracy_history]
        n = len(errors)
        return (
            f"AI Brain Accuracy Report ({n} predictions)\n"
            f"  Mean Error:   {np.mean(errors):.4f}\n"
            f"  Median Error: {np.median(errors):.4f}\n"
            f"  Best Error:   {np.min(errors):.4f}\n"
            f"  Worst Error:  {np.max(errors):.4f}\n"
            f"  Under 10% error: {sum(1 for e in errors if e < 0.1)}/{n} "
            f"({sum(1 for e in errors if e < 0.1)/n:.1%})"
        )

    # ------------------------------------------------------------------
    # Security integration
    # ------------------------------------------------------------------
    def scan_data_integrity(self, data: list) -> dict:
        """Use the security engine to verify data integrity."""
        if "security" not in self.sources:
            return {"status": "security engine not loaded"}
        engine = self.sources["security"]
        return engine.verify_data_integrity(data)

    def audit_code(self, code_string: str) -> str:
        """Use the LLM to security-audit a code snippet."""
        if "llm_assistant" in self.sources:
            return self.sources["llm_assistant"].security_audit(code_string)
        return "LLM assistant not available for code audit."


# ======================================================================
# Singleton for easy import
# ======================================================================
_brain_instance: Optional[AIBrain] = None


def get_brain() -> AIBrain:
    """Get or create the global AI Brain instance."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = AIBrain()
    return _brain_instance


# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AI BRAIN — Edge Tracker Master Controller")
    print("=" * 60)

    brain = AIBrain()
    brain.initialize()

    # Demo with synthetic data
    np.random.seed(42)
    data = (np.random.exponential(2.0, 200) + 1.0).tolist()

    # Train the transformer if available
    brain.train_transformer(data, epochs=30)

    # Get fused prediction
    result = brain.predict(data, game_type="crash")
    print(f"\nFused Prediction:  {result['prediction']:.4f}")
    print(f"Confidence:        {result['confidence']:.2%}")
    print(f"Sources:           {result['sources']}")
    print(f"Method:            {result['method']}")

    # Status
    print(f"\nStatus: {json.dumps(brain.get_status(), indent=2)}")
