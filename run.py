#!/usr/bin/env python3
"""
Edge Tracker 2026 - Entry Point
Real-Time Statistical Analysis Engine for Provably Fair Games
"""

import sys
import os
from pathlib import Path

# Force UTF-8 output so emoji don't crash on Windows CP1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from dashboard import EdgeTrackerDashboard

def main():
    """Main entry point"""
    print("🎯 Edge Tracker 2026 Pro - AI-Powered Game Analysis")
    print("====================================================")
    print()
    print("Core Systems:")
    print("• 🤖 AI Brain — Master controller fusing all prediction sources")
    print("• 🧠 ML Brain — 5 ensemble sklearn models + deep learning")
    print("• 📡 AI APIs — Groq, Gemini, OpenRouter, AI/ML (4 LLMs)")
    print("• 🔮 Transformer — Custom neural network for sequence prediction")
    print("• 🔒 Security Engine — Python security tools & education")
    print("• 📊 Statistical analysis (chi-squared, runs test, autocorrelation)")
    print("• 🎯 Multi-agent AI analysis (4 specialized agents)")
    print("• 💰 7 mathematical betting strategies")
    print("• 🔗 Provably fair hash chain verification")
    print("• 🌐 Web scraping with 6 fallback methods")
    print()

    # Check for AI model files
    parent_dir = current_dir.parent
    ai_files = ["finetune.py", "inference.py", "tiny_transformer.py", "prepare_dataset.py"]
    found = [f for f in ai_files if (parent_dir / f).exists()]
    print(f"AI Model Files: {len(found)}/{len(ai_files)} found")
    if (parent_dir / "my_edge_tracker_model").exists():
        print("🟢 Fine-tuned model found — LLM assistant available")
    else:
        print("🟡 No fine-tuned model yet — run finetune.py to train")
    print()
    print("Starting application...")
    print()

    try:
        app = EdgeTrackerDashboard()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"\nError starting application: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install customtkinter matplotlib numpy pandas scikit-learn xgboost requests beautifulsoup4 websocket-client")
        sys.exit(1)

if __name__ == "__main__":
    main()