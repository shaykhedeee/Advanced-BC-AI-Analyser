"""
ollama_brain.py — Local Ollama LLM Ensemble Brain for Edge Tracker 2026
========================================================================
Discovers ALL locally installed Ollama models and organises them into 
two speed-optimised triplet ensembles for real-time provably-fair game
analysis.

  SPEED TRIPLET  (< 5 GB each — sub-second local inference)
    • dolphin-phi:latest   1.6 GB — ultra-fast reasoning
    • aeline/halo:latest   4.7 GB — fast chat/analysis
    • dolphin3:latest      4.9 GB — fast instruct

  DEEP TRIPLET   (heavy models — background inference)
    • aeline/phil:latest   4.7 GB — deep reasoning
    • gpt-oss:20b         13.8 GB — large GPT-class model
    • qwen3-coder:30b     18.6 GB — best statistical reasoning

  ALL-6 CONSENSUS — aggregates both triplets for a final verdict.

All models are queried in parallel threads with timeout protection.
Results are fused with confidence-weighted majority voting.
"""

import json
import time
import threading
import re
import numpy as np
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

OLLAMA_BASE = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Speed catalogue — updated on every initialize() call via /api/tags
# ---------------------------------------------------------------------------
_SPEED_FAST  = "fast"    # < 6 GB
_SPEED_HEAVY = "heavy"   # >= 6 GB

_SIZE_THRESHOLD_GB = 6.0

# ---------------------------------------------------------------------------
# Hard-coded triplet preferences (names may differ across users – we match by
# substring so  "dolphin-phi"  matches  "dolphin-phi:latest" etc.)
# ---------------------------------------------------------------------------
TRIPLET_SPEED_PREFERENCES = ["dolphin-phi", "halo", "dolphin3"]
TRIPLET_DEEP_PREFERENCES  = ["phil", "gpt-oss", "qwen3-coder"]


# ===========================================================================
class OllamaModel:
    """Thin wrapper around a single Ollama model."""

    def __init__(self, name: str, size_gb: float = 0.0):
        self.name       = name
        self.size_gb    = size_gb
        self.speed_tier = _SPEED_FAST if size_gb < _SIZE_THRESHOLD_GB else _SPEED_HEAVY
        self.call_count = 0
        self.total_ms   = 0
        self.error_count = 0
        self._lock      = threading.Lock()

    # ------------------------------------------------------------------
    def query(self, prompt: str, timeout_s: float = 20.0) -> Tuple[str, float]:
        """
        Send a prompt to this model and return (response_text, elapsed_ms).
        Uses the Ollama /api/generate endpoint (streaming=False).
        Raises RuntimeError on network / API failure.
        """
        payload = json.dumps({
            "model":  self.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":   0.05,  # near-deterministic for analysis
                "num_predict":   256,
                "top_k":         20,
                "top_p":         0.85,
            }
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data    = payload,
            method  = "POST",
            headers = {"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw       = resp.read().decode()
                data      = json.loads(raw)
                text      = data.get("response", "").strip()
                elapsed   = (time.time() - t0) * 1000
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            with self._lock:
                self.error_count += 1
            raise RuntimeError(f"{self.name}: {exc}") from exc

        with self._lock:
            self.call_count += 1
            self.total_ms   += elapsed

        return text, elapsed

    @property
    def avg_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_ms / self.call_count

    def __repr__(self) -> str:
        return (f"OllamaModel({self.name!r}, "
                f"{self.size_gb:.1f}GB, {self.speed_tier})")


# ===========================================================================
class OllamaTriplet:
    """
    Three models queried in parallel; results fused by confidence-weighted
    majority vote.
    """

    def __init__(self, name: str, models: List[OllamaModel]):
        self.name       = name
        self.models     = models         # exactly 3
        self.call_log   = deque(maxlen=200)
        self._executor  = ThreadPoolExecutor(max_workers=3,
                                             thread_name_prefix=f"triplet_{name}")

    # ------------------------------------------------------------------
    def _query_model(self, model: OllamaModel, prompt: str,
                     timeout_s: float) -> Dict[str, Any]:
        try:
            text, ms = model.query(prompt, timeout_s=timeout_s)
            parsed   = _parse_model_output(text)
            return {"model": model.name, "raw": text, "parsed": parsed,
                    "ms": ms, "ok": True}
        except Exception as exc:
            return {"model": model.name, "raw": "", "parsed": None,
                    "ms": 0, "ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    def analyze(self, prompt: str, timeout_s: float = 25.0) -> Dict[str, Any]:
        """
        Query all 3 models in parallel and return fused result.
        """
        t0      = time.time()
        futures = {self._executor.submit(self._query_model, m, prompt, timeout_s): m
                   for m in self.models}

        responses = []
        for future in as_completed(futures, timeout=timeout_s + 3):
            try:
                responses.append(future.result())
            except Exception as exc:
                m = futures[future]
                responses.append({"model": m.name, "raw": "", "parsed": None,
                                   "ms": 0, "ok": False, "error": str(exc)})

        fused   = _fuse_triplet_responses(responses)
        elapsed = (time.time() - t0) * 1000

        record = {"time": time.time(), "triplet": self.name,
                  "fused": fused, "responses": responses, "total_ms": elapsed}
        self.call_log.append(record)

        return {
            "triplet":    self.name,
            "fused":      fused,
            "responses":  responses,
            "total_ms":   elapsed,
            "ok_count":   sum(1 for r in responses if r["ok"]),
        }


# ===========================================================================
class OllamaBrain:
    """
    Master local LLM brain.

    Usage
    -----
        brain = OllamaBrain()
        brain.initialize()
        result = brain.analyze_game_data(data, game_type="crash")
    """

    def __init__(self):
        self.models:       Dict[str, OllamaModel]  = {}
        self.triplet_fast: Optional[OllamaTriplet] = None
        self.triplet_deep: Optional[OllamaTriplet] = None
        self.is_ready      = False
        self.call_history  = deque(maxlen=500)
        self._lock         = threading.Lock()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def initialize(self) -> "OllamaBrain":
        print("[OllamaBrain] Connecting to Ollama at", OLLAMA_BASE)
        available = self._discover_models()
        if not available:
            print("[OllamaBrain] WARNING — no models found or Ollama not running")
            self.is_ready = False
            return self

        print(f"[OllamaBrain] {len(available)} model(s) discovered:")
        for m in available:
            print(f"  • {m.name:<35} {m.size_gb:.1f} GB  [{m.speed_tier}]")

        self._build_triplets(available)
        self.is_ready = True
        print(f"[OllamaBrain] Ready — Speed triplet: "
              f"{[m.name for m in self.triplet_fast.models] if self.triplet_fast else '—'}")
        print(f"[OllamaBrain]         Deep  triplet: "
              f"{[m.name for m in self.triplet_deep.models] if self.triplet_deep else '—'}")
        return self

    # ------------------------------------------------------------------
    def _discover_models(self) -> List[OllamaModel]:
        """Return OllamaModel objects for all locally installed models."""
        try:
            req  = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            print(f"[OllamaBrain] Cannot reach Ollama: {exc}")
            return []

        models = []
        for entry in data.get("models", []):
            name    = entry["name"]
            size_gb = entry.get("size", 0) / 1e9
            m       = OllamaModel(name, size_gb)
            self.models[name] = m
            models.append(m)
        return models

    # ------------------------------------------------------------------
    def _build_triplets(self, models: List[OllamaModel]):
        """
        Form the two triplets:
          SPEED — prefer fast (< 6 GB) models matching TRIPLET_SPEED_PREFERENCES
          DEEP  — prefer heavy models matching TRIPLET_DEEP_PREFERENCES
        Falls back to size-ordered fill if fewer than 3 preferred matched.
        """
        def match_models(prefs: List[str]) -> List[OllamaModel]:
            result = []
            for pref in prefs:
                for m in models:
                    if pref.lower() in m.name.lower() and m not in result:
                        result.append(m)
                        break
            return result

        # Speed triplet
        fast_candidates = match_models(TRIPLET_SPEED_PREFERENCES)
        if len(fast_candidates) < 3:
            # Fill with smallest remaining models
            by_size = sorted([m for m in models if m not in fast_candidates],
                             key=lambda x: x.size_gb)
            for m in by_size:
                if len(fast_candidates) >= 3:
                    break
                fast_candidates.append(m)
        self.triplet_fast = OllamaTriplet("SPEED", fast_candidates[:3])

        # Deep triplet
        deep_candidates = match_models(TRIPLET_DEEP_PREFERENCES)
        if len(deep_candidates) < 3:
            by_size = sorted([m for m in models if m not in deep_candidates],
                             key=lambda x: -x.size_gb)
            for m in by_size:
                if len(deep_candidates) >= 3:
                    break
                deep_candidates.append(m)
        self.triplet_deep = OllamaTriplet("DEEP", deep_candidates[:3])

    # ------------------------------------------------------------------
    # Core Analysis API
    # ------------------------------------------------------------------
    def analyze_game_data(
        self,
        game_data:     List[float],
        game_type:     str  = "crash",
        fast_only:     bool = False,
        timeout_s:     float = 20.0,
    ) -> Dict[str, Any]:
        """
        Analyse recent game_data with the Ollama ensemble.

        Returns a dict with:
          prediction    — numeric estimate of next result
          confidence    — 0.0–1.0
          recommendation — "HIGH" | "MEDIUM" | "LOW" | "SKIP"
          reasoning     — human-readable summary
          triplet_results — raw triplet outputs
        """
        if not self.is_ready:
            return _empty_result("OllamaBrain not initialised")

        if len(game_data) < 5:
            return _empty_result("Insufficient data (need ≥5 points)")

        prompt = _build_analysis_prompt(game_data, game_type)

        results = {}

        # ---- Speed Triplet (always used — real-time) ----
        if self.triplet_fast:
            t0 = time.time()
            results["speed"] = self.triplet_fast.analyze(prompt, timeout_s=timeout_s)
            results["speed"]["wall_s"] = round(time.time() - t0, 3)

        # ---- Deep Triplet (skipped if fast_only) ----
        if self.triplet_deep and not fast_only:
            # Run deep analysis in background with extended timeout
            deep_timeout = min(timeout_s * 3, 90.0)
            t0 = time.time()
            results["deep"] = self.triplet_deep.analyze(prompt, timeout_s=deep_timeout)
            results["deep"]["wall_s"] = round(time.time() - t0, 3)

        # ---- All-6 Consensus ----
        all_fused_values = []
        all_confidences  = []
        all_reasoning    = []

        for triplet_key, triplet_result in results.items():
            f = triplet_result.get("fused", {})
            if f.get("prediction") is not None:
                all_fused_values.append(f["prediction"])
                all_confidences.append(f["confidence"])
            if f.get("reasoning"):
                all_reasoning.append(f"[{triplet_key.upper()}] {f['reasoning']}")

        if all_fused_values:
            weights = [c + 0.01 for c in all_confidences]
            total_w = sum(weights)
            grand_prediction  = sum(v * w for v, w in zip(all_fused_values, weights)) / total_w
            grand_confidence  = np.mean(all_confidences)
            recommendation    = _confidence_to_recommendation(grand_confidence)
        else:
            grand_prediction = float(np.mean(game_data[-10:])) if game_data else 2.0
            grand_confidence = 0.0
            recommendation   = "SKIP"

        final = {
            "prediction":      round(grand_prediction, 4),
            "confidence":      round(grand_confidence, 4),
            "recommendation":  recommendation,
            "reasoning":       " | ".join(all_reasoning) or "No model consensus reached.",
            "triplet_results": results,
            "models_used":     self._active_model_names(),
        }

        self.call_history.append({"time": time.time(), **final})
        return final

    # ------------------------------------------------------------------
    def query_single(self, model_name: str, prompt: str,
                     timeout_s: float = 30.0) -> Dict[str, Any]:
        """Direct query to a specific model by name."""
        if model_name not in self.models:
            return {"ok": False, "error": f"Model {model_name!r} not found"}
        try:
            text, ms = self.models[model_name].query(prompt, timeout_s=timeout_s)
            return {"ok": True, "model": model_name, "response": text, "ms": ms}
        except Exception as exc:
            return {"ok": False, "model": model_name, "error": str(exc)}

    # ------------------------------------------------------------------
    def query_all_parallel(self, prompt: str,
                           timeout_s: float = 35.0) -> List[Dict[str, Any]]:
        """
        Query ALL 6 models in parallel and return all responses.
        Great for one-off deep analysis where latency is not critical.
        """
        results = []
        with ThreadPoolExecutor(max_workers=6,
                                thread_name_prefix="ollama_all") as pool:
            futures = {pool.submit(m.query, prompt, timeout_s): name
                       for name, m in self.models.items()}
            for f in as_completed(futures, timeout=timeout_s + 5):
                name = futures[f]
                try:
                    text, ms = f.result()
                    results.append({"model": name, "response": text,
                                    "ms": ms, "ok": True})
                except Exception as exc:
                    results.append({"model": name, "response": "",
                                    "ms": 0, "ok": False, "error": str(exc)})
        results.sort(key=lambda r: self.models[r["model"]].size_gb)
        return results

    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            "ready":           self.is_ready,
            "model_count":     len(self.models),
            "models":         {n: {"size_gb": m.size_gb, "tier": m.speed_tier,
                                    "calls": m.call_count,
                                    "avg_ms": round(m.avg_ms, 1),
                                    "errors": m.error_count}
                               for n, m in self.models.items()},
            "triplet_fast":    [m.name for m in self.triplet_fast.models]
                               if self.triplet_fast else [],
            "triplet_deep":    [m.name for m in self.triplet_deep.models]
                               if self.triplet_deep else [],
            "total_analyses":  len(self.call_history),
        }

    # ------------------------------------------------------------------
    def _active_model_names(self) -> List[str]:
        names = []
        if self.triplet_fast:
            names.extend(m.name for m in self.triplet_fast.models)
        if self.triplet_deep:
            for m in self.triplet_deep.models:
                if m.name not in names:
                    names.append(m.name)
        return names


# ===========================================================================
# Prompt Builder
# ===========================================================================
def _build_analysis_prompt(game_data: List[float], game_type: str) -> str:
    recent   = game_data[-30:] if len(game_data) >= 30 else game_data
    n        = len(recent)
    mean_val = float(np.mean(recent))
    std_val  = float(np.std(recent))
    min_val  = float(np.min(recent))
    max_val  = float(np.max(recent))
    median   = float(np.median(recent))

    # Streak detection
    last_5       = recent[-5:]
    low_streak   = sum(1 for v in last_5 if v < mean_val)
    high_streak  = sum(1 for v in last_5 if v >= mean_val)

    data_str = ", ".join(f"{v:.2f}" for v in recent[-15:])

    prompt = f"""You are an expert statistical analyst for provably-fair {game_type} games.

RECENT DATA (last {n} rounds — newest last):
  Values: [{data_str}]

STATISTICS:
  Mean:    {mean_val:.4f}
  Median:  {median:.4f}
  Std Dev: {std_val:.4f}
  Min:     {min_val:.4f}
  Max:     {max_val:.4f}
  Low streak  (last 5 below mean): {low_streak}/5
  High streak (last 5 above mean): {high_streak}/5

YOUR TASK — respond in this EXACT JSON format (no extra text):
{{
  "prediction": <float — estimated next {game_type} value>,
  "confidence": <float 0.0–1.0>,
  "reasoning": "<1–2 sentence explanation>",
  "recommendation": "<HIGH|MEDIUM|LOW|SKIP>"
}}

Rules:
• Base prediction on statistical distribution patterns only.
• {game_type.capitalize()} is provably fair — truly random; never claim predictability.
• Be honest about uncertainty. If data is random, say confidence is low.
• JSON only — no markdown, no preamble."""

    return prompt


# ===========================================================================
# Output Parser
# ===========================================================================
def _parse_model_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model output. Handles models that add markdown fences,
    thinking blocks, or extra prose around the JSON.
    """
    if not text:
        return None

    # Strip <think>...</think> blocks (Qwen3 / chain-of-thought models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Find first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        # Try extracting individual fields
        pred  = _extract_float(text, r'"prediction"\s*:\s*([\d.]+)')
        conf  = _extract_float(text, r'"confidence"\s*:\s*([\d.]+)')
        rec   = _extract_str(text,   r'"recommendation"\s*:\s*"([A-Z]+)"')
        reas  = _extract_str(text,   r'"reasoning"\s*:\s*"([^"]+)"')
        if pred is not None:
            return {"prediction": pred, "confidence": conf or 0.3,
                    "recommendation": rec or "SKIP", "reasoning": reas or ""}
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    # Normalise
    try:
        return {
            "prediction":     float(data.get("prediction", 0.0)),
            "confidence":     float(data.get("confidence", 0.3)),
            "recommendation": str(data.get("recommendation", "SKIP")).upper(),
            "reasoning":      str(data.get("reasoning", "")),
        }
    except (ValueError, TypeError):
        return None


def _extract_float(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_str(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(1) if m else None


# ===========================================================================
# Triplet Fusion
# ===========================================================================
def _fuse_triplet_responses(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fuse 3 model outputs into a single result using confidence-weighted average.
    """
    good = [r for r in responses if r["ok"] and r["parsed"] is not None]

    if not good:
        return {"prediction": None, "confidence": 0.0,
                "recommendation": "SKIP", "reasoning": "All models failed."}

    preds   = [r["parsed"]["prediction"]  for r in good]
    confs   = [r["parsed"]["confidence"]  for r in good]
    recs    = [r["parsed"]["recommendation"] for r in good]
    reasons = [r["parsed"]["reasoning"]   for r in good if r["parsed"]["reasoning"]]

    # Speed penalty for slow responses
    speed_weights = []
    for r in good:
        ms = r.get("ms", 5000)
        w  = max(0.1, 1.0 - ms / 60_000)   # mild penalty for very slow models
        speed_weights.append(w)

    combined_weights = [c * sw for c, sw in zip(confs, speed_weights)]
    total_w = sum(combined_weights) or 1.0

    fused_pred = sum(p * w for p, w in zip(preds, combined_weights)) / total_w
    fused_conf = np.mean(confs)

    # Majority vote on recommendation
    rec_counts: Dict[str, float] = {}
    for r, w in zip(recs, combined_weights):
        rec_counts[r] = rec_counts.get(r, 0) + w
    fused_rec = max(rec_counts, key=rec_counts.get)

    best_reason = max(reasons, key=len) if reasons else ""

    return {
        "prediction":     round(fused_pred, 4),
        "confidence":     round(float(fused_conf), 4),
        "recommendation": fused_rec,
        "reasoning":      best_reason,
        "votes":          len(good),
    }


def _confidence_to_recommendation(conf: float) -> str:
    if conf >= 0.70:
        return "HIGH"
    elif conf >= 0.50:
        return "MEDIUM"
    elif conf >= 0.30:
        return "LOW"
    return "SKIP"


def _empty_result(reason: str) -> Dict[str, Any]:
    return {
        "prediction":      None,
        "confidence":      0.0,
        "recommendation":  "SKIP",
        "reasoning":       reason,
        "triplet_results": {},
        "models_used":     [],
    }


# ===========================================================================
# Module-level singleton
# ===========================================================================
_ollama_brain_instance: Optional[OllamaBrain] = None
_init_lock = threading.Lock()


def get_ollama_brain() -> OllamaBrain:
    """Get or lazily initialise the global OllamaBrain singleton."""
    global _ollama_brain_instance
    with _init_lock:
        if _ollama_brain_instance is None:
            _ollama_brain_instance = OllamaBrain()
            _ollama_brain_instance.initialize()
    return _ollama_brain_instance


# ===========================================================================
# CLI smoke-test
# ===========================================================================
if __name__ == "__main__":
    import random

    print("=" * 65)
    print("  OllamaBrain — Edge Tracker 2026  |  Smoke Test")
    print("=" * 65)

    brain = OllamaBrain()
    brain.initialize()

    if not brain.is_ready:
        print("Ollama not available — exiting.")
        raise SystemExit(1)

    print("\n[Status]")
    status = brain.get_status()
    print(f"  Models loaded : {status['model_count']}")
    print(f"  Speed triplet : {status['triplet_fast']}")
    print(f"  Deep  triplet : {status['triplet_deep']}")

    # Synthetic crash data (realistic provably-fair distribution)
    random.seed(42)
    data = [round(max(1.0, random.expovariate(0.5) + 1.0), 2) for _ in range(50)]
    print(f"\n[Test] Analysing {len(data)} crash rounds (fast-only mode)...")

    t0     = time.time()
    result = brain.analyze_game_data(data, game_type="crash", fast_only=True, timeout_s=20)
    took   = time.time() - t0

    print(f"\n  Prediction    : {result['prediction']}")
    print(f"  Confidence    : {result['confidence']:.2%}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Reasoning     : {result['reasoning'][:120]}...")
    print(f"  Models used   : {result['models_used']}")
    print(f"  Wall time     : {took:.2f}s")

    print("\n[Done] OllamaBrain smoke test complete.")
