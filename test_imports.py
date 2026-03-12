#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """Test all major imports"""
    print("Testing Edge Tracker 2026 imports...")
    print("=" * 50)
    
    try:
        print("✓ Testing config...")
        from config import API_KEYS, GAME_SETTINGS, ML_SETTINGS
        print("  Config loaded successfully")
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False
    
    try:
        print("✓ Testing data_engine...")
        from data_engine import UniversalDataEngine, LiveScanner
        print("  Data engine loaded successfully")
    except Exception as e:
        print(f"  ✗ Data engine error: {e}")
        return False
    
    try:
        print("✓ Testing ml_brain...")
        from ml_brain import MLBrain
        print("  ML brain loaded successfully")
    except Exception as e:
        print(f"  ✗ ML brain error: {e}")
        return False
    
    try:
        print("✓ Testing ai_predictor...")
        from ai_predictor import AIPredictor, ContinuousPredictor
        print("  AI predictor loaded successfully")
    except Exception as e:
        print(f"  ✗ AI predictor error: {e}")
        return False
    
    try:
        print("✓ Testing scraper...")
        from scraper import BCGameScraper, RealTimeMonitor
        print("  Scraper loaded successfully")
    except Exception as e:
        print(f"  ✗ Scraper error: {e}")
        return False
    
    try:
        print("✓ Testing agents...")
        from agents import StatisticianAgent, PatternAgent, RiskAgent, JudgeAgent
        print("  Agents loaded successfully")
    except Exception as e:
        print(f"  ✗ Agents error: {e}")
        return False
    
    try:
        print("✓ Testing strategies...")
        from strategies import KellyOptimizer, OptimalStopping, SessionSimulator, SessionManager, StrategyComparator
        print("  Strategies loaded successfully")
    except Exception as e:
        print(f"  ✗ Strategies error: {e}")
        return False
    
    try:
        print("✓ Testing dashboard...")
        from dashboard import EdgeTrackerDashboard
        print("  Dashboard loaded successfully")
    except Exception as e:
        print(f"  ✗ Dashboard error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All imports successful! Edge Tracker 2026 is ready to run.")
    print("You can now launch the application with: python run.py")
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        print("\n❌ Some imports failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All systems go! The application should launch successfully.")