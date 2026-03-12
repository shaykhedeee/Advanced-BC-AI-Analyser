# Edge Tracker 2026 - Provably Fair Game Analysis

## Overview

Edge Tracker 2026 is a comprehensive data science application that analyzes provably fair crash game data using machine learning and statistics. The core thesis is that cryptographically secure RNG **cannot** be predicted by machine learning - ML accuracy will converge to ~50% (coin flip) on truly random data.

## Features

### 🎯 Core Components

1. **5 ML Models** (scikit-learn + XGBoost)
   - Random Forest (2 variants)
   - XGBoost
   - Gradient Boosting
   - MLP Neural Network

2. **Deep Learning Models** (LSTM, GRU, BiLSTM, AttentionLSTM, TCN)
   - PyTorch backend (with TensorFlow fallback)
   - Pure NumPy neural network (no framework dependency)

3. **4 AI Agents** (Groq, Gemini, OpenRouter, Ollama)
   - Statistician Agent (Groq/Llama 3.3 70B)
   - Pattern Agent (Google Gemini)
   - Risk Agent (OpenRouter/Qwen)
   - Judge Agent (Local Ollama/Llama 3.2)

4. **Web Scraper** with 6 fallback methods
   - BC.Game API scraping
   - WebSocket monitoring
   - Multiple REST API endpoints
   - Fallback to simulation

5. **Real-time Monitoring** with anomaly detection
   - Live data collection
   - Statistical process control
   - Pattern detection
   - Provably fair verification

6. **CustomTkinter Dashboard** (dark-themed)
   - Real-time charts and statistics
   - Multi-tab interface
   - Live data feed
   - Strategy comparison

7. **Strategy Lab** with Kelly Criterion + Monte Carlo
   - Kelly Criterion optimization
   - Optimal stopping strategies
   - Session simulation
   - Strategy comparison

## Architecture

```
edge_tracker/
├── run.py                    # Entry point
├── dashboard.py             # Main GUI application
├── config.py               # Configuration settings
├── data_engine.py          # Data collection and simulation
├── ml_brain.py             # Machine learning models
├── ai_predictor.py         # AI API integration
├── scraper.py             # Web scraping functionality
├── training_pipeline.py   # Deep learning training
├── agents/                # AI agent implementations
│   ├── agent_statistician.py
│   ├── agent_pattern.py
│   ├── agent_risk.py
│   └── agent_judge.py
└── strategies/            # Betting strategies
    ├── kelly.py
    ├── optimal_stopping.py
    ├── session_simulator.py
    ├── session_manager.py
    └── comparator.py
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys in `config.py` or environment variables:
   - `GOOGLE_API_KEY` (for Gemini)
   - `GROQ_API_KEY` (for Groq)
   - `OPENROUTER_API_KEY` (for OpenRouter)
   - `AIML_API_KEY` (for AI/ML API)

3. Launch the application:
```bash
python run.py
```

## Usage

### Quick Start

1. **Start Simulation**: Click "Start Simulation" in the Live Feed tab to generate test data
2. **Train Models**: Once you have 100+ data points, click "Train All Models" in the AI Train tab
3. **Get Predictions**: Switch to AI Predict tab and click "AI Predict Next"
4. **Analyze Results**: Use the Stats, Patterns, and Risk tabs for detailed analysis
5. **Get Final Judgment**: Click "Get Final Judgment" for the comprehensive verdict

### Key Insights

- **ML Accuracy**: Will converge to ~50% on truly random data
- **Statistical Tests**: Chi-squared, runs test, autocorrelation
- **Pattern Detection**: FFT analysis and clustering detection
- **Risk Management**: Kelly Criterion and Monte Carlo simulation
- **Provably Fair**: Hash chain verification and seed analysis

## Technical Details

### Data Simulation

The application simulates provably fair games using cryptographically secure formulas:
- **Crash**: `max(1.00, 0.99 / (1 - random()))`
- **Dice**: Random integers 1-100
- **Limbo**: `max(1.00, 1.0 / (1 - random()))`
- **Slots**: Symbol-based with configurable RTP

### Machine Learning

- **Feature Engineering**: 50+ features including statistics, momentum, volatility, entropy
- **Ensemble Methods**: Majority voting with confidence weighting
- **Deep Learning**: Sequence prediction with multiple architectures
- **Time Series**: Proper cross-validation to prevent data leakage

### AI Integration

- **Multi-API**: Queries 4 different AI services simultaneously
- **Consensus Building**: Weighted voting based on confidence
- **Real-time**: Continuous prediction with auto-training
- **Fallback**: Graceful degradation when APIs are unavailable

## Project Status

✅ **Complete and Functional**

All components are implemented and working:
- ✅ 5 ML models (Random Forest, XGBoost, Gradient Boosting, MLP, RF_v2)
- ✅ 4 AI agents (Statistician, Pattern, Risk, Judge)
- ✅ Web scraper with 6 fallback methods
- ✅ Real-time monitoring with anomaly detection
- ✅ CustomTkinter dark-themed dashboard
- ✅ Strategy lab with Kelly Criterion + Monte Carlo
- ✅ Deep learning pipeline (LSTM, GRU, BiLSTM, AttentionLSTM, TCN)
- ✅ Provably fair verification
- ✅ Statistical analysis suite

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **API Keys**: Set up API keys in config.py or environment variables
3. **Data Requirements**: Need 100+ data points for ML training, 50+ for predictions
4. **Framework Detection**: Deep learning will use PyTorch, TensorFlow, or NumPy fallback

### Dependencies

```bash
customtkinter==5.2.2
matplotlib==3.8.2
numpy==1.26.2
pandas==2.1.4
scikit-learn==1.3.2
xgboost==2.0.2
requests==2.31.0
beautifulsoup4==4.12.2
websocket-client==1.6.4
pillow==10.2.0
google-generativeai>=0.3.2
groq>=0.4.1
ollama>=0.2.1
openai>=1.6.1
scipy>=1.11.0
torch>=2.0.0
```

## License

This project demonstrates that machine learning cannot predict cryptographically secure random number generators. All code is provided for educational purposes.

## Disclaimer

This application is for educational and research purposes only. It demonstrates that:
1. Cryptographically secure RNG cannot be predicted by ML
2. Provably fair games are mathematically sound
3. No betting strategy can overcome the house edge on fair games

**Gambling addiction warning**: If you or someone you know has a gambling problem, please seek help from professional resources.