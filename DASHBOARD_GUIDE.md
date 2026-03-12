# Edge Tracker 2026 Dashboard - Complete Guide

## 🎯 Overview

The **Edge Tracker 2026 Dashboard** is a comprehensive real-time analysis platform for provably fair game data, combining:
- **5 Classical ML Models** (Random Forest, XGBoost, Gradient Boosting, MLP)
- **5 Deep Learning Models** (LSTM, GRU, BiLSTM, AttentionLSTM, TCN)
- **4 AI API Agents** (Groq, Gemini, OpenRouter, Ollama)
- **Advanced Statistical Analysis**
- **Real-time Monitoring** with anomaly detection
- **Security Auditing & Hash Verification**

---

## 📊 Dashboard Features

### 1. **Status Bar** (Top)
- **ML Accuracy**: Current ML ensemble accuracy
- **Data Count**: Number of data points collected
- **Monitor Status**: Live connection or simulation status
- **Last Value Ticker**: Real-time crash multiplier display
- **Clock**: Current time display

### 2. **Main Charts Area** (Left)
- **Chart 1**: Time series of crash values (green ≥2x, red <2x)
- **Chart 2**: Distribution histogram with mean/median lines
- Live updates every ~100ms

### 3. **Tabbed Control Panel** (Right) - 12 Tabs

#### 🚀 **AI Train Tab**
Train machine learning models on collected data.

**Features:**
- Train Classical ML (sklearn models)
- Train Deep Learning (LSTM, Transformers)
- Check AI API Health
- Show Training Summary
- Feature Importance Analysis
- Progress tracking

**How to Use:**
1. Click "Start Simulation" in Live Feed tab to generate data
2. Click "Train Classical ML" to train sklearn models
3. Click "Deep Learning" to train neural networks
4. Click "📈 Full Report" to see comprehensive analysis

**Expected Results:**
- Classical ML: ~50% accuracy on random data
- Deep Learning: Similar convergence to 50%
- This proves: Random cryptographic RNG cannot be predicted!

---

#### 🎯 **AI Predict Tab**
Get predictions from AI models and APIs.

**Features:**
- Single AI Prediction (next value direction)
- Combined ML+DL+AI mega-prediction
- Auto-Predict Mode (continuous predictions)
- Individual API breakdowns
- Confidence scoring

**How to Use:**
1. Ensure you have ≥20 data points
2. Click "AI Predict Next" for single prediction
3. Click "Combined Predict" for ensemble
4. Click "Start Auto-Predict" for continuous mode

**Output:**
- Direction: UP/DOWN relative to mean
- Confidence: How sure the model is
- Agreement: Consensus among all models

---

#### 📊 **Stats Tab**
Statistical fairness analysis of game data.

**Features:**
- Chi-Squared Test (RNG fairness)
- Runs Test (randomness verification)
- Autocorrelation Analysis
- Fairness Verdict

**How to Use:**
1. Collect ≥30 data points
2. Click "Run Chi-Squared Test"
3. Review statistical conclusions
4. Check "Check Fairness" for quick fairness verdict

---

#### 🔍 **Patterns Tab**
Detect patterns and anomalies in data.

**Features:**
- Pattern Detection (clustering, trends)
- FFT Analysis (frequency analysis)
- Autocorrelation examination
- Reality checks on pattern viability

**How to Use:**
1. Require ≥50 data points
2. Click "Detect Patterns" for pattern analysis
3. Click "FFT Analysis" for frequency analysis

**Expected Findings:**
- Few true patterns (random data)
- Possible clustering by chance
- No predictable trends

---

#### ⚠️ **Risk Tab**
Bankroll management and betting strategies.

**Features:**
- Kelly Criterion optimization
- Monte Carlo simulation (1000s of sessions)
- Optimal bet sizing
- Risk of Ruin calculation
- Expected Value analysis

**How to Use:**
1. Enter bankroll amount (e.g., 1000)
2. Click "Kelly Criterion" for optimal bet size
3. Click "Monte Carlo" for session simulation
4. Review Risk of Ruin percentage

**Formulas Used:**
- **Kelly:** f* = (bp - q) / b
  - Where: b = odds, p = win%, q = loss%
- **Risk of Ruin:** (q/p)^n for ruin on n hands

---

#### ⚖️ **Judge Tab**
Final verdict from all agents combined.

**Features:**
- Aggregates all previous analyses
- Combines ML, stats, patterns, risk analysis
- Overall Score (0-10)
- Recommendations
- Reasoning breakdown

**How to Use:**
1. Complete all other analyses first
2. Click "Get Final Judgment"
3. Review final verdict and recommendations

---

#### 🤖 **AI Brain Tab**
Master AI controller - fused ensemble.

**Features:**
- Transformer neural network predictor
- Accuracy tracking
- Data integrity verification
- Status dashboard
- Adaptive weight adjustment

**How to Use:**
1. Click "🤖 Brain Predict" for fused prediction
2. Click "🧠 Train Transformer" to train neural net
3. Click "📊 Accuracy Report" for performance
4. Click "🔍 Data Integrity" to verify data quality

**Capabilities:**
- Combines: Transformer + MLBrain + APIs + Security
- Adaptive weights based on real accuracy
- Fallback to equal weights if all perform poorly

---

#### 🔒 **Security Tab**
Security analysis and auditing tools.

**Features:**
- Password Strength Analysis
- Code Security Audit (static analysis)
- Port Scanning (localhost only)
- Game Data Integrity Verification
- Security Education
- Encryption tools

**How to Use:**
1. Click "🔐 Password Check" for password analysis
2. Click "🛡️ Audit Code" to scan for vulnerabilities
3. Click "✅ Data Integrity" to verify game data
4. Click "📚 Learn Security" for security tips

---

#### 🌐 **Scraper Tab**
Web scraping for real BC.Game data.

**Features:**
- Multi-method scraping (6 fallback methods)
- Provably Fair verification
- WebSocket monitoring
- API endpoint scraping
- Simulation fallback

**How to Use:**
1. Click "Scrape BC.Game" to attempt live scraping
2. Note: BC.Game has strong anti-scraping
3. Click "Verify Hash" for hash verification

**Note:** Live scraping may be blocked. Use simulation for testing.

---

#### 📡 **Live Feed Tab**
Real-time data collection and monitoring.

**Features:**
- Live BC.Game connection
- Data simulation
- Real-time monitoring
- Anomaly detection
- Session management

**How to Use:**
1. Click "Connect Live" to attempt live feed
2. Click "Start Simulation" for synthetic data
3. Monitor data count and status
4. Click "Stop Simulation" to halt

**For Testing:**
- Use "Start Simulation" for immediate data
- Generates 1-2 crash values per second
- Realistic distribution

---

#### 🔑 **Hash Verify Tab**
BC.Game provably fair verification.

**Features:**
- HMAC-SHA256 verification
- Batch hash verification (last 50)
- Server seed input
- Client seed input
- Nonce tracking
- Crash point computation

**How to Use:**
1. Paste server seed hash (from BC.Game)
2. Paste client seed (from BC.Game)
3. Enter nonce value
4. Click "Verify Hash"
5. Review computed crash point

**Formula Used:**
```
hash = HMAC-SHA256(server_seed, client_seed:nonce)
h = int(hash[:13], 16)
e = 2^52
if h % 33 == 0: crash = 1.0 (house wins!)
else: crash = (100*e - h) / (e - h) / 100
```

---

## 🚀 Getting Started

### 1. **Quick Start (Testing)**
```bash
cd c:\Users\USER\Documents\AI CRACKER\edge_tracker
python launch.py
```

### 2. **Initialize Dashboard**
- Dashboard opens maximized
- Charts display automatically
- Status bar shows real-time data

### 3. **Generate Test Data**
1. Go to "📡 Live Feed" tab
2. Click "Start Simulation"
3. Wait for data points to accumulate
4. Watch ticker update in real-time

### 4. **Train Models**
1. Go to "🚀 AI Train" tab
2. Ensure ≥100 data points
3. Click "Train Classical ML"
4. Wait for training to complete
5. Click "📈 Full Report" to see results

### 5. **Make Predictions**
1. Go to "🎯 AI Predict" tab
2. Ensure ≥20 data points
3. Click "Combined Predict"
4. Review confidence and direction

---

## 📈 Advanced Features

### **Export Results**
- Click "Export Results" in bottom control panel
- Exports:
  - `export_game_TIMESTAMP.json` - Raw data
  - `export_game_TIMESTAMP.csv` - CSV format
  - `export_stats_TIMESTAMP.json` - ML/AI statistics

### **Real-time Charts**
- Green bars: Crash values ≥ 2.0x
- Red bars: Crash values < 2.0x
- Blue line: 2.0x threshold
- Orange line: Last value
- Auto-updates every 100ms

### **Performance Monitoring**
- CPU Usage: Process CPU percentage
- Memory: RAM usage in MB
- Live Status: Connection state
- Data Quality: Assessment of data distribution

### **Comprehensive Reports**
Access via "📈 Full Report" button:
- Data summary (mean, std, min, max)
- ML performance metrics
- AI integration status
- Analysis insights
- System performance
- Data integrity checks

---

## 🎯 Key Principles

### **Fair RNG Cannot Be Predicted**
> **Core Thesis**: Cryptographically secure random number generation (RNG) used in provably fair games cannot be predicted by machine learning.

**Evidence:**
1. ML accuracy should converge to ~50% (coin flip) on truly random data
2. If you observe higher accuracy, either:
   - The RNG is not truly random
   - External factors influence outcomes
   - The model is overfitting to noise

### **Ensemble Strength**
- Multiple models reduce overfitting
- Consensus voting improves robustness
- 10+ ML + 4 AI APIs = 14+ predictions combined
- Weighted ensemble adapts based on real performance

### **Provably Fair Verification**
- Every round is cryptographically verifiable
- Server seed is hashed beforehand
- Client seed is player-provided or random
- Nonce ensures each round is unique
- Formula is deterministic and transparent

---

## ⚡ Performance Tips

1. **For Live Data**: Use simulation instead of live scraping
2. **For Training**: 100-500 data points optimal (more = slower)
3. **For Predictions**: Use combined (ML+DL+AI) for best accuracy
4. **For Analysis**: Statistical tests need ≥50 data points

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| No data showing | Click "Start Simulation" in Live Feed |
| ML models won't train | Need ≥100 data points; run simulation longer |
| AI predictions fail | Check "🔐 Password Check" → "API Health" |
| Charts not updating | Ensure data is being generated; check status bar |
| Slow performance | Close other apps; reduce data window size |
| Live connection fails | Use simulation; live scraping is often blocked |

---

## 📚 Resources

- **Provably Fair**: https://en.wikipedia.org/wiki/Provably_fair
- **Kelly Criterion**: https://en.wikipedia.org/wiki/Kelly_criterion
- **HMAC-SHA256**: https://en.wikipedia.org/wiki/HMAC
- **Random Number Generation**: https://crypto.stackexchange.com/

---

## 🎓 Educational Value

This application demonstrates:
- ✅ Machine Learning ensemble methods
- ✅ Deep Learning time series prediction
- ✅ API integration and consensus
- ✅ Statistical analysis (chi-squared, runs test)
- ✅ Cryptographic verification
- ✅ Real-time data visualization
- ✅ Risk management strategies
- ✅ Security scanning and auditing
- ✅ GUI development (CustomTkinter)
- ✅ Multi-threaded applications

---

## 🚀 Next Steps

1. **Test with simulation data** (complete)
2. **Train classical ML models** (ready)
3. **Train deep learning models** (ready)
4. **Connect to AI APIs** (ready)
5. **Verify provably fair hashes** (ready)
6. **Analyze statistical fairness** (ready)
7. **Export and review results** (ready)
8. **Study results and improve models** (ongoing)

---

## 📝 Version
**Edge Tracker 2026 v1.0** - March 2026
Complete implementation with all features ready for production use.
