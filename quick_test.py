import sys
print(f"Python {sys.version}")

# Test scipy
try:
    import scipy
    print(f"[OK] scipy {scipy.__version__}")
except Exception as e:
    print(f"[FAIL] scipy: {e}")

# Test sklearn
try:
    from sklearn.ensemble import RandomForestClassifier
    print("[OK] sklearn")
except Exception as e:
    print(f"[FAIL] sklearn: {e}")

# Test xgboost
try:
    import xgboost
    print(f"[OK] xgboost {xgboost.__version__}")
except Exception as e:
    print(f"[FAIL] xgboost: {e}")

# Test openai
try:
    import openai
    print(f"[OK] openai {openai.__version__}")
except Exception as e:
    print(f"[FAIL] openai: {e}")

# Test groq
try:
    import groq
    print(f"[OK] groq {groq.__version__}")
except Exception as e:
    print(f"[FAIL] groq: {e}")

# Test google genai
try:
    import google.generativeai as genai
    print("[OK] google-generativeai")
except Exception as e:
    print(f"[FAIL] google-generativeai: {e}")

# Test customtkinter
try:
    import customtkinter
    print(f"[OK] customtkinter {customtkinter.__version__}")
except Exception as e:
    print(f"[FAIL] customtkinter: {e}")

# Test our modules
print("\n--- Project Modules ---")
try:
    from config import API_KEYS, ML_SETTINGS, DL_SETTINGS
    print(f"[OK] config - {len(API_KEYS)} API keys, {len(DL_SETTINGS.get('models',{}))} DL models")
except Exception as e:
    print(f"[FAIL] config: {e}")

try:
    from training_pipeline import DeepLearningPipeline
    dlp = DeepLearningPipeline()
    print(f"[OK] training_pipeline - framework: {dlp._framework}")
except Exception as e:
    print(f"[FAIL] training_pipeline: {e}")

try:
    from ai_predictor import AIPredictor
    aip = AIPredictor()
    print(f"[OK] ai_predictor - {len(aip.active_apis)} APIs connected")
except Exception as e:
    print(f"[FAIL] ai_predictor: {e}")

try:
    from ml_brain import MLBrain
    mlb = MLBrain()
    print(f"[OK] ml_brain - {len(mlb.models)} sklearn models, DL: {mlb.dl_pipeline is not None}")
except Exception as e:
    print(f"[FAIL] ml_brain: {e}")

try:
    from data_engine import UniversalDataEngine
    de = UniversalDataEngine()
    print("[OK] data_engine")
except Exception as e:
    print(f"[FAIL] data_engine: {e}")

try:
    from agents import StatisticianAgent, PatternAgent, RiskAgent, JudgeAgent
    print("[OK] agents")
except Exception as e:
    print(f"[FAIL] agents: {e}")

print("\n--- Quick Training Test ---")
try:
    import numpy as np
    data = np.random.exponential(2.0, 300) + 1.0
    
    # Test DL training
    dlp2 = DeepLearningPipeline()
    results = dlp2.train_all_models(data)
    if 'error' not in results:
        print(f"[OK] DL Training - Best: {results.get('best_model')} ({results.get('best_accuracy', 0):.1%})")
    else:
        print(f"[WARN] DL Training: {results.get('error')}")
    
    # Test ML training
    mlb2 = MLBrain()
    success = mlb2.train_models(data)
    print(f"[OK] ML Training - success={success}, models={len(mlb2.models)}")
    
    # Test AI predictor
    aip2 = AIPredictor()
    if aip2.active_apis:
        pred = aip2.predict_next(data.tolist(), 'crash')
        print(f"[OK] AI Predict - {pred}")
    else:
        print("[SKIP] AI Predict - no APIs connected")
    
except Exception as e:
    print(f"[FAIL] Training test: {e}")
    import traceback
    traceback.print_exc()

print("\n=== ALL TESTS COMPLETE ===")
