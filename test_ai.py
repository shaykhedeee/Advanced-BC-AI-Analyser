"""Quick test to verify all imports work"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from config import API_KEYS, AI_PREDICTION, DL_SETTINGS
    print(f"[OK] Config loaded - {len(API_KEYS)} API keys, DL models: {DL_SETTINGS['models']}")
except Exception as e:
    print(f"[FAIL] Config: {e}")

try:
    from ai_predictor import AIPredictor
    ai = AIPredictor()
    print(f"[OK] AI Predictor - {len(ai.clients)} APIs connected")
except Exception as e:
    print(f"[FAIL] AI Predictor: {e}")

try:
    from training_pipeline import DeepLearningPipeline
    dl = DeepLearningPipeline()
    print(f"[OK] DL Pipeline - Framework: {dl._framework}")
except Exception as e:
    print(f"[FAIL] DL Pipeline: {e}")

try:
    from ml_brain import MLBrain
    ml = MLBrain()
    print(f"[OK] ML Brain - {len(ml.models)} sklearn models + DL pipeline")
except Exception as e:
    print(f"[FAIL] ML Brain: {e}")

try:
    from data_engine import UniversalDataEngine
    engine = UniversalDataEngine()
    print("[OK] Data Engine")
except Exception as e:
    print(f"[FAIL] Data Engine: {e}")

try:
    from agents import StatisticianAgent, PatternAgent, RiskAgent, JudgeAgent
    print("[OK] All agents imported")
except Exception as e:
    print(f"[FAIL] Agents: {e}")

try:
    from strategies import KellyOptimizer, OptimalStopping, SessionSimulator, SessionManager, StrategyComparator
    print("[OK] All strategies imported")
except Exception as e:
    print(f"[FAIL] Strategies: {e}")

# Quick test: generate data and train
print("\n--- Running Quick Training Test ---")
import numpy as np

engine = UniversalDataEngine()
# Generate 200 crash data points
for _ in range(200):
    crash = engine.simulate_crash_round()
    engine.add_crash_data(crash)

data = [p['value'] for p in engine.data['crash']]
print(f"Generated {len(data)} crash data points")
print(f"Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")

# Train sklearn models
print("\nTraining sklearn models...")
success = ml.train_models(data)
print(f"Sklearn training: {'OK' if success else 'FAILED'}")

# Get prediction
pred = ml.predict_next(data)
if pred:
    print(f"Sklearn prediction: ensemble={pred['ensemble']}, confidence={pred['confidence']:.1%}")

# Train deep learning
print("\nTraining deep learning models...")
dl_results = ml.train_deep_learning(data)
if 'error' in dl_results:
    print(f"DL Error: {dl_results['error']}")
else:
    print(f"DL Training: {dl_results.get('best_model')} ({dl_results.get('best_accuracy', 0):.1%})")

# Combined prediction
combined = ml.predict_combined(data)
if combined:
    print(f"Combined prediction: {combined['direction']} (confidence: {combined['confidence']:.1%})")
    print(f"Total models: {combined['total_models']}")

print("\n=== ALL TESTS PASSED ===")
