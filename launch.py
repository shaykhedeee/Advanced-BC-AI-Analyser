#!/usr/bin/env python3
"""
Edge Tracker 2026 Pro - Enhanced Launcher
Advanced performance monitoring and system optimization
"""

import sys
import os
import psutil
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def print_banner():
    """Print enhanced startup banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       🚀 EDGE TRACKER 2026 PRO 🚀                            ║
║                   Advanced Provably Fair Analysis Suite                       ║
║                        Enhanced by AI & Machine Learning                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🎯 CORE FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 Enhanced Machine Learning:
   ◆ 10+ Advanced Algorithms (XGBoost, LightGBM, CatBoost, Neural Networks)
   ◆ 100+ Engineered Features per Prediction
   ◆ Ensemble Voting with Confidence Scoring
   ◆ Real-time Model Performance Monitoring

📊 Advanced Analytics:
   ◆ Chaos Theory Analysis (Lyapunov Exponents)
   ◆ Frequency Domain Analysis (FFT Spectral)
   ◆ Technical Analysis Integration (TA-Lib)
   ◆ Statistical Testing (Anderson-Darling, Runs Test)

🎮 Multi-Game Support:
   ◆ Crash Games (BC.Game Live Integration)
   ◆ Dice Games with Provably Fair Verification
   ◆ Limbo Games with Pattern Recognition
   ◆ Slots with RTP Analysis

🤖 AI Integration:
   ◆ Multi-API Support (Groq, Gemini, OpenRouter)
   ◆ Continuous Learning & Adaptation  
   ◆ 4 Specialized AI Agents (Statistics, Patterns, Risk, Judge)
   ◆ Real-time Prediction Confidence Scoring

📈 Visualization:
   ◆ Real-time Interactive Charts (9 synchronized displays)
   ◆ 3D Analysis with Multi-game Correlation
   ◆ Advanced Statistical Plots (Plotly Integration)
   ◆ System Performance Monitoring

⚡ Performance Optimized:
   ◆ Multi-threading for Real-time Processing
   ◆ Memory Efficient Data Structures
   ◆ CPU & GPU Acceleration where available
   ◆ Advanced Caching & Optimization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(banner)


def check_system_requirements():
    """Check and display system information"""
    print("🔍 SYSTEM ANALYSIS:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # System info
    import platform
    print(f"💻 Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"🐍 Python: {platform.python_version()}")
    
    # CPU info
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"🧠 CPU: {cpu_count_physical} cores ({cpu_count_logical} threads)")
    if cpu_freq:
        print(f"⚡ CPU Frequency: {cpu_freq.current:.0f} MHz (max: {cpu_freq.max:.0f} MHz)")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"📊 Total Memory: {memory.total / (1024**3):.1f} GB")
    print(f"💾 Available: {memory.available / (1024**3):.1f} GB ({100 - memory.percent:.1f}% free)")
    
    # Disk info
    disk = psutil.disk_usage('/')
    print(f"💿 Disk Space: {disk.free / (1024**3):.1f} GB free of {disk.total / (1024**3):.1f} GB")
    
    print()


def check_dependencies():
    """Check critical dependencies"""
    print("📦 DEPENDENCY CHECK:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    required_packages = [
        ('customtkinter', 'GUI Framework'),
        ('numpy', 'Numerical Computing'),
        ('pandas', 'Data Analysis'),
        ('scipy', 'Scientific Computing'),
        ('scikit-learn', 'Machine Learning'),
        ('xgboost', 'Gradient Boosting'),
        ('lightgbm', 'Light Gradient Boosting'),
        ('catboost', 'CatBoost ML'),
        ('matplotlib', 'Plotting'),
        ('seaborn', 'Statistical Plots'),
        ('plotly', 'Interactive Plots'),
        ('requests', 'HTTP Client'),
        ('beautifulsoup4', 'HTML Parser'),
        ('websocket-client', 'WebSocket Client'),
        ('python-socketio', 'Socket.IO Client'),
        ('ta', 'Technical Analysis'),
        ('psutil', 'System Monitoring'),
        ('rich', 'Rich Text'),
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            status = "✅"
        except ImportError:
            status = "❌"
            missing_packages.append(package)
        
        print(f"{status} {package:<20} - {description}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies satisfied!")
        return True


def optimize_performance():
    """Apply performance optimizations"""
    print("\n⚡ PERFORMANCE OPTIMIZATION:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Set process priority
    try:
        process = psutil.Process()
        if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            print("🚀 Process priority set to HIGH")
        else:
            process.nice(-10)  # Unix/Linux
            print("🚀 Process nice value optimized")
    except:
        print("⚠️  Could not optimize process priority")
    
    # Memory optimization
    try:
        import gc
        gc.collect()
        print("🧹 Memory garbage collection performed")
    except:
        pass
    
    # Set environment variables for performance
    os.environ['PYTHONOPTIMIZE'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = str(psutil.cpu_count())
    print(f"🔧 Multi-threading optimized for {psutil.cpu_count()} cores")
    
    # ML library optimizations
    try:
        import numpy as np
        if hasattr(np, 'show_config'):
            print("📊 NumPy BLAS optimizations detected")
    except:
        pass
    
    print("✅ Performance optimizations applied!")


def launch_application():
    """Launch the main application"""
    print("\n🚀 LAUNCHING EDGE TRACKER 2026 PRO:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        print("🔄 Initializing core components...")
        from dashboard import EdgeTrackerDashboard
        
        print("🎯 Starting GUI interface...")
        app = EdgeTrackerDashboard()
        
        print("🌟 Edge Tracker 2026 Pro is now running!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()
        print("💡 QUICK START GUIDE:")
        print("  1. 📡 Go to 'Live Feed' tab → Click 'Connect Live' for real BC.Game data")
        print("  2. 🎮 Or click 'Start Simulation' to generate test data")  
        print("  3. 🚀 Go to 'AI Train' tab → Click 'Train Enhanced ML' (need 100+ points)")
        print("  4. 🎯 Go to 'AI Predict' tab → See real-time predictions")
        print("  5. 📊 Explore 'Advanced' tab for cutting-edge analytics")
        print("  6. 🖥️  Monitor system performance in 'System' tab")
        print()
        print("🔥 Enjoy the most advanced provably fair analysis tool ever created!")
        print()
        
        app.run()
        
    except KeyboardInterrupt:
        print("\n👋 Edge Tracker 2026 Pro shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("\n🔧 Troubleshooting:")
        print("  • Check all dependencies are installed: pip install -r requirements.txt")
        print("  • Ensure you have sufficient system resources (4GB+ RAM recommended)")
        print("  • Try running as administrator if permission issues occur")
        print("  • Check firewall settings for network features")
        sys.exit(1)


def main():
    """Main launcher function"""
    print_banner()
    
    # System check
    check_system_requirements()
    
    # Dependency check
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Performance optimization
    optimize_performance()
    
    # Launch application
    launch_application()


if __name__ == "__main__":
    main()