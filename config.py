import os

try:
    import importlib
    _dotenv = importlib.import_module("dotenv")
    _load_dotenv = getattr(_dotenv, "load_dotenv", None)
    if callable(_load_dotenv):
        _load_dotenv()
except Exception:
    pass

# Edge Tracker 2026 - Configuration Settings
# This file contains all configuration parameters for the application.


def _env(name: str, placeholder: str = "") -> str:
    """Read environment variable with placeholder fallback."""
    value = os.getenv(name, "").strip()
    return value if value else placeholder

# API Keys for AI Services
API_KEYS = {
    "google_gemini": _env("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE"),
    "groq": _env("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE"),
    "openrouter": _env("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE"),
    "aiml_api": _env("AIML_API_KEY", "YOUR_AIML_API_KEY_HERE"),
    "ollama": _env("OLLAMA_API_KEY", ""),
}

DISABLED_API_KEY_PREFIXES = ("YOUR_", "<")


def has_real_api_key(service_name: str) -> bool:
    """Return True when a key is configured and not an obvious placeholder."""
    key = API_KEYS.get(service_name, "")
    if not key:
        return False
    return not any(key.startswith(prefix) for prefix in DISABLED_API_KEY_PREFIXES)

# Game Settings
GAME_SETTINGS = {
    "crash": {
        "min_crash": 1.00,
        "max_crash": 100.0,
        "house_edge": 0.01,  # 1% house edge
        "provably_fair_formula": True,
    },
    "dice": {
        "min_roll": 1,
        "max_roll": 100,
        "house_edge": 0.01,
    },
    "limbo": {
        "min_multiplier": 1.00,
        "max_multiplier": 1000.0,
        "house_edge": 0.01,
    },
    "slots": {
        "rtp": 0.95,  # 95% RTP
        "symbols": ["🍒", "🍋", "🍊", "🍇", "🔔", "⭐", "💎"],
        "reels": 3,
        "paylines": 1,
    },
}

# ML Settings
ML_SETTINGS = {
    "models": ["RandomForest", "XGBoost", "GradientBoosting", "MLP", "RF_v2"],
    "window_sizes": [5, 10, 20, 50],
    "cv_folds": 5,
    "test_size": 0.2,
    "random_state": 42,
    "feature_count": 50,
}

# Deep Learning Settings
DL_SETTINGS = {
    "lstm_units": [128, 64, 32],
    "sequence_length": 50,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "patience": 15,  # early stopping patience
    "models": ["LSTM", "GRU", "BiLSTM", "AttentionLSTM", "TCN"],
    "ensemble_weights": "auto",  # auto-tune or manual
}

# RT-DETR Vision Analyzer Settings
VISION_SETTINGS = {
    "model_name": "PekingU/rtdetr_r50vd",
    "custom_model_path": None,  # Set to local path after fine-tuning
    "confidence_threshold": 0.3,
    "target_size": (640, 640),
    "device": "auto",  # "auto", "cuda", or "cpu"
    "ocr_engine": "easyocr",  # "easyocr", "tesseract", or None
    "capture_interval": 1.0,  # seconds between live captures
    "monitor_index": 1,  # which monitor to capture
    "game_ui_labels": {
        0: "multiplier_display",
        1: "bet_panel",
        2: "cashout_button",
        3: "graph_area",
        4: "chat_area",
        5: "history_bar",
        6: "balance_display",
        7: "timer_countdown",
        8: "player_list",
        9: "payout_text",
    },
}

# Security & AI Training Settings
TRAINING_SETTINGS = {
    "tiny_transformer": {
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "epochs": 50,
    },
    "qlora_finetune": {
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "max_seq_length": 1024,
    },
    "rtdetr_finetune": {
        "base_model": "PekingU/rtdetr_r50vd",
        "epochs": 20,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "num_labels": 10,
    },
    "security_topics": [
        "python_security",
        "network_security",
        "cryptography",
        "linux_security",
        "malware_defense",
        "rng_probability",
        "web_security",
    ],
    "output_dir": "./training_data",
}

# AI API Prediction Settings
AI_PREDICTION = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.1,
        "max_tokens": 2000,
        "base_url": "https://api.groq.com/openai/v1",
    },
    "google_gemini": {
        "model": "gemini-2.0-flash",
        "temperature": 0.1,
        "max_tokens": 2000,
    },
    "openrouter": {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "temperature": 0.1,
        "max_tokens": 2000,
        "base_url": "https://openrouter.ai/api/v1",
    },
    "aiml_api": {
        "model": "deepseek/deepseek-chat-v3-0324",
        "temperature": 0.1,
        "max_tokens": 2000,
        "base_url": "https://api.aimlapi.com/v1",
    },
    "consensus_threshold": 0.6,  # 60% agreement needed
    "prediction_history_size": 1000,
}

# Scraper Settings
SCRAPER_SETTINGS = {
    "bcgame": {
        "base_url": "https://bc.game",
        "api_endpoints": [
            "/api/crash/history",
            "/api/dice/history",
            "/api/limbo/history",
            "/api/slots/history",
        ],
        "websocket_url": "wss://bc.game/ws",
        "fallback_methods": 6,
        "timeout": 10,
        "enable_live_connector": os.getenv("ENABLE_BCGAME_LIVE", "false").lower() == "true",
    },
    "provably_fair": {
        "hash_algorithm": "HMAC-SHA256",
        "server_seed": "server_seed_placeholder",
        "client_seed": "client_seed_placeholder",
        "nonce": 0,
    },
}

# UI Settings
UI_SETTINGS = {
    "theme": "dark",
    "colors": {
        "background": "#0a0e17",
        "foreground": "#e6edf3",
        "accent": "#238636",
        "secondary": "#58a6ff",
        "warning": "#d29922",
        "error": "#f85149",
        "success": "#56d364",
        "purple": "#a5a2ff",
        "orange": "#ff8c42",
        "cyan": "#39d0d8",
    },
    "fonts": {
        "main": ("Segoe UI", 11),
        "heading": ("Segoe UI", 14, "bold"),
        "subheading": ("Segoe UI", 12, "bold"),
        "mono": ("Fira Code", 10),
        "large": ("Segoe UI", 16, "bold"),
    },
    "chart_colors": ["#56d364", "#58a6ff", "#f85149", "#d29922", "#a5a2ff", "#ff8c42", "#39d0d8"],
    "update_interval": 1500,  # ms for auto-simulation
    "animations": True,
    "transparency": 0.95,
}

# Agent Settings
AGENT_SETTINGS = {
    "statistician": {
        "model": "groq",
        "temperature": 0.1,
        "max_tokens": 1000,
    },
    "pattern": {
        "model": "google_gemini",
        "temperature": 0.3,
        "max_tokens": 800,
    },
    "risk": {
        "model": "openrouter",
        "temperature": 0.2,
        "max_tokens": 1200,
    },
    "judge": {
        "model": "ollama",
        "temperature": 0.0,
        "max_tokens": 1500,
    },
    "ai_predictor": {
        "model": "multi_api",
        "temperature": 0.1,
        "max_tokens": 2000,
    },
}

# Strategy Settings
STRATEGY_SETTINGS = {
    "kelly": {
        "cashouts": list(range(101, 10001)),  # 1.01x to 100x
        "simulations": 1000,
    },
    "optimal_stopping": {
        "strategies": ["conservative", "moderate", "aggressive"],
        "simulations": 1000,
    },
    "session_simulator": {
        "sessions": 10000,
        "max_rounds": 1000,
    },
    "session_manager": {
        "tilt_threshold": 0.1,  # 10% drawdown
        "fatigue_rounds": 50,
    },
    "comparator": {
        "strategies": 7,
        "simulations": 1000,
    },
}

# Pattern Solver Settings (master detection engine)
SOLVER_SETTINGS = {
    "max_history": 10000,
    "regime_windows": [20, 50, 100],
    "confidence_threshold": 0.65,
    "tier1_tests": ["distribution", "autocorrelation", "runs", "entropy", "streaks"],
    "tier2_tests": ["regime", "cusum", "volatility", "clustering"],
    "tier3_tests": ["momentum", "mean_reversion", "spectral"],
}

# Enhanced Transformer Settings
TRANSFORMER_SETTINGS = {
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 4,
    "d_ff": 256,
    "dropout": 0.1,
    "seq_len": 50,
    "quantiles": [0.10, 0.25, 0.50, 0.75, 0.90],
    "regimes": ["normal", "hot", "cold", "volatile", "anomalous"],
}

SECURITY_SETTINGS = {
    "localhost_only_scans": True,
    "max_port_scan_timeout": 1.0,
    "allow_live_remote_connections": SCRAPER_SETTINGS["bcgame"]["enable_live_connector"],
    "strict_placeholder_keys": True,
}