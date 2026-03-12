"""
ml_brain.py — Enhanced ML Brain with 50+ Features, Stacking Ensemble,
Soft Voting, Multi-Window, and Pattern Solver Integration
=================================================================
"""
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
from typing import Optional

SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier,
        BaggingClassifier
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[MLBrain] WARNING: sklearn not available")

XGB_AVAILABLE = False
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    pass

LGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    pass

CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    pass

SCIPY_AVAILABLE = False
try:
    from scipy import stats as sp_stats
    from scipy.fft import fft
    SCIPY_AVAILABLE = True
except ImportError:
    pass

from config import ML_SETTINGS, DL_SETTINGS

DL_AVAILABLE = False
try:
    from training_pipeline import DeepLearningPipeline
    DL_AVAILABLE = True
except ImportError:
    pass

SOLVER_AVAILABLE = False
try:
    from pattern_solver import get_solver
    SOLVER_AVAILABLE = True
except ImportError:
    pass


class MLBrain:
    """
    Enhanced ML Brain — 8+ model stacking ensemble with 50+ features,
    multi-window analysis, soft voting, probability calibration, and
    direct integration with the Pattern Solver engine.

    Architecture:
        Layer 1 (Base): RF, ExtraTrees, GBM, XGB, LightGBM, CatBoost, MLP, Bagging
        Layer 2 (Meta): LogisticRegression stacking meta-learner on calibrated probabilities
        + Deep Learning pipeline (LSTM/GRU/TCN) as auxiliary signal
        + Pattern Solver 50-dim feature vector as additional input
    """

    def __init__(self):
        self.models = {}
        self.stacker = None
        self.master_scaler = None  # Single RobustScaler for all models
        self.feature_importance = {}
        self.prediction_history = []
        self.model_weights = {}     # Dynamic accuracy-based weights
        self.dl_pipeline = None
        self._feature_names = []
        self._is_trained = False
        self._train_time = 0
        self._init_models()

    def _init_models(self):
        """Initialize all ML models — 8 base learners + stacking meta-learner"""
        if not SKLEARN_AVAILABLE:
            print("[MLBrain] sklearn not available - ML models disabled")
            return

        self.models['RandomForest'] = RandomForestClassifier(  # type: ignore[reportPossiblyUnbound]
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        self.models['ExtraTrees'] = ExtraTreesClassifier(  # type: ignore[reportPossiblyUnbound]
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        self.models['GradientBoosting'] = GradientBoostingClassifier(  # type: ignore[reportPossiblyUnbound]
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        if XGB_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(  # type: ignore[reportPossiblyUnbound]
                n_estimators=150, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric='logloss', verbosity=0
            )
        if LGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(  # type: ignore[reportPossiblyUnbound]
                n_estimators=150, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            )
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = CatBoostClassifier(  # type: ignore[reportPossiblyUnbound]
                iterations=150, depth=6, learning_rate=0.05,
                random_seed=42, verbose=0
            )
        self.models['MLP'] = MLPClassifier(  # type: ignore[reportPossiblyUnbound]
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            learning_rate='adaptive', random_state=42
        )
        self.models['AdaBoost'] = AdaBoostClassifier(  # type: ignore[reportPossiblyUnbound]
            n_estimators=100, learning_rate=0.05, random_state=42
        )

        # Initialize equal weights
        for name in self.models:
            self.model_weights[name] = 1.0 / len(self.models)

        # Deep learning pipeline
        if DL_AVAILABLE:
            self.dl_pipeline = DeepLearningPipeline()  # type: ignore[reportPossiblyUnbound]

    # ==================================================================
    # TRAINING
    # ==================================================================
    def train_models(self, data: list, target_threshold: Optional[float] = None) -> bool:
        """Train all models with stacking ensemble on 50+ features"""
        if len(data) < 100 or not SKLEARN_AVAILABLE:
            return False

        start = time.perf_counter()
        X, y = self._prepare_data(data, target_threshold)

        if len(X) < 50 or len(np.unique(y)) < 2:
            return False

        # Single robust scaler (handles outliers better than StandardScaler)
        self.master_scaler = RobustScaler()  # type: ignore[reportPossiblyUnbound]
        X_scaled = self.master_scaler.fit_transform(X)

        # TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=ML_SETTINGS.get('cv_folds', 5))  # type: ignore[reportPossiblyUnbound]

        # Train each base model
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)

                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_

                # Cross-validate for weight assignment
                try:
                    scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')  # type: ignore[reportPossiblyUnbound]
                    self.model_weights[name] = max(0.01, float(np.mean(scores)))
                except Exception:
                    self.model_weights[name] = 0.5

            except Exception as e:
                print(f"[MLBrain] Error training {name}: {e}")

        # Normalize weights
        total_w = sum(self.model_weights.values())
        if total_w > 0:
            self.model_weights = {k: v / total_w for k, v in self.model_weights.items()}

        # Build stacking ensemble (meta-learner)
        try:
            trained_estimators = [
                (name, model) for name, model in self.models.items()
                if hasattr(model, 'classes_')
            ]
            if len(trained_estimators) >= 3:
                self.stacker = StackingClassifier(  # type: ignore[reportPossiblyUnbound]
                    estimators=trained_estimators,
                    final_estimator=LogisticRegression(  # type: ignore[reportPossiblyUnbound]
                        C=1.0, max_iter=1000, random_state=42
                    ),
                    cv=tscv, stack_method='predict_proba', n_jobs=-1
                )
                self.stacker.fit(X_scaled, y)
        except Exception as e:
            print(f"[MLBrain] Stacking failed (using soft vote fallback): {e}")
            self.stacker = None

        self._is_trained = True
        self._train_time = time.perf_counter() - start
        return True

    def train_deep_learning(self, data: list, callback=None) -> dict:
        """Train deep learning models"""
        if not DL_AVAILABLE or len(data) < 100:
            return {"error": "Need DL pipeline + 100 data points"}

        try:
            results = self.dl_pipeline.train_all_models(data, callback=callback)  # type: ignore[union-attr]
            return results
        except Exception as e:
            return {"error": f"Deep learning training failed: {e}"}

    # ==================================================================
    # PREDICTION
    # ==================================================================
    def predict_next(self, data: list) -> dict:
        """
        Get predictions from all models using soft voting + stacking.

        Returns calibrated probability and ensemble prediction.
        """
        if len(data) < 50 or not self._is_trained or self.master_scaler is None:
            return {}

        # Extract features from the full available window
        X = self._extract_features(data[-100:] if len(data) >= 100 else data)
        X_scaled = self.master_scaler.transform([X])

        predictions = {}
        probabilities = {}  # class 1 probability per model

        for name, model in self.models.items():
            try:
                if not hasattr(model, 'classes_'):
                    continue
                pred = model.predict(X_scaled)[0]
                predictions[name] = int(pred)

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    probabilities[name] = float(proba[1]) if len(proba) > 1 else 0.5
                else:
                    probabilities[name] = float(pred)
            except Exception:
                pass

        if not predictions:
            return {}

        # === Stacking prediction (best) ===
        stacked_pred = None
        stacked_proba = None
        if self.stacker is not None:
            try:
                stacked_pred = int(self.stacker.predict(X_scaled)[0])
                sp = self.stacker.predict_proba(X_scaled)[0]
                stacked_proba = float(sp[1]) if len(sp) > 1 else 0.5
            except Exception:
                pass

        # === Weighted soft vote (fallback) ===
        weighted_proba = 0.0
        total_w = 0.0
        for name, prob in probabilities.items():
            w = self.model_weights.get(name, 0.1)
            weighted_proba += prob * w
            total_w += w
        weighted_proba = weighted_proba / total_w if total_w > 0 else 0.5

        # Use stacker if available, else weighted soft vote
        final_proba = stacked_proba if stacked_proba is not None else weighted_proba
        ensemble_pred = stacked_pred if stacked_pred is not None else int(final_proba >= 0.5)

        # Confidence = distance from 0.5 boundary, scaled
        confidence = abs(final_proba - 0.5) * 2  # 0 = no confidence, 1 = max

        # Get Pattern Solver features if available
        solver_verdict = None
        if SOLVER_AVAILABLE:
            try:
                solver = get_solver()  # type: ignore[reportPossiblyUnbound]
                if len(solver.data) > 30:
                    result = solver.analyze()
                    solver_verdict = {
                        'regime': result.get('regime'),
                        'action': result.get('action'),
                        'risk': result.get('risk_level'),
                    }
            except Exception:
                pass

        return {
            'ensemble': ensemble_pred,
            'ensemble_proba': round(float(final_proba), 4),
            'confidence': round(float(confidence), 4),
            'predictions': predictions,
            'probabilities': probabilities,
            'model_weights': dict(self.model_weights),
            'stacking_used': stacked_pred is not None,
            'solver_verdict': solver_verdict,
        }

    def predict_combined(self, data: list) -> dict:
        """Get combined sklearn + deep learning + pattern solver predictions"""
        sklearn_pred = self.predict_next(data)

        dl_pred = None
        if DL_AVAILABLE and self.dl_pipeline:
            try:
                dl_pred = self.dl_pipeline.predict(data)  # type: ignore[union-attr]
            except Exception:
                pass

        combined = {
            'sklearn': sklearn_pred,
            'deep_learning': dl_pred
        }

        if sklearn_pred and dl_pred:
            sk_proba = sklearn_pred.get('ensemble_proba', 0.5)
            dl_prediction = dl_pred.get('prediction', 0.5)

            # Weighted fusion: sklearn 70%, DL 30%
            overall_proba = sk_proba * 0.7 + dl_prediction * 0.3
            overall_conf = (sklearn_pred.get('confidence', 0.5) * 0.7 +
                            dl_pred.get('confidence', 0.5) * 0.3)

            combined['overall'] = {
                'prediction': round(float(overall_proba), 4),
                'confidence': round(float(overall_conf), 4),
                'direction': 'above' if overall_proba > 0.5 else 'below',
            }

        return combined

    # ==================================================================
    # FEATURE EXTRACTION — 50+ features
    # ==================================================================
    def _extract_features(self, window: list) -> list:
        """
        Extract 50+ features from a data window.

        Feature blocks:
          1. Basic stats (10)
          2. Multi-window means & stds (10)
          3. Differences & momentum (10)
          4. Distribution shape (6)
          5. Autocorrelation (4)
          6. FFT spectral features (4)
          7. Streak & run features (4)
          8. Ratio & relative features (4)
        """
        arr = np.array(window, dtype=np.float64)
        n = len(arr)
        features = []

        # ---- Block 1: Basic Statistics (10) ----
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if n > 1 else 1e-10
        features.append(float(mean))
        features.append(float(std))
        features.append(float(np.min(arr)))
        features.append(float(np.max(arr)))
        features.append(float(arr[-1]))
        features.append(float(np.median(arr)))
        features.append(float(np.percentile(arr, 25)))
        features.append(float(np.percentile(arr, 75)))
        features.append(float(np.percentile(arr, 10)))
        features.append(float(np.percentile(arr, 90)))

        # ---- Block 2: Multi-Window Rolling Stats (10) ----
        for w in [5, 10, 20, 30, 50]:
            if n >= w:
                features.append(float(np.mean(arr[-w:])))
                features.append(float(np.std(arr[-w:], ddof=1)))
            else:
                features.append(float(mean))
                features.append(float(std))

        # ---- Block 3: Differences & Momentum (10) ----
        diffs = np.diff(arr) if n > 1 else np.array([0.0])
        features.append(float(np.mean(diffs)))
        features.append(float(np.std(diffs)))
        features.append(float(diffs[-1]))
        features.append(float(np.max(diffs)))
        features.append(float(np.min(diffs)))

        # Slope of linear fit (trend)
        if n >= 5:
            x = np.arange(min(n, 20))
            slope = np.polyfit(x, arr[-len(x):], 1)[0]
            features.append(float(slope))
        else:
            features.append(0.0)

        # Win rate at multiple windows
        for w in [10, 20, 50]:
            if n >= w:
                features.append(float(np.sum(arr[-w:] > mean) / w))
            else:
                features.append(0.5)

        # Rate of change (last vs mean)
        features.append(float(arr[-1] / mean) if mean != 0 else 1.0)

        # ---- Block 4: Distribution Shape (6) ----
        if SCIPY_AVAILABLE and n >= 8:
            features.append(float(sp_stats.skew(arr)))  # type: ignore[reportPossiblyUnbound]
            features.append(float(sp_stats.kurtosis(arr)))  # type: ignore[reportPossiblyUnbound]
        else:
            features.append(0.0)
            features.append(0.0)

        # IQR and coefficient of variation
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        features.append(float(iqr))
        features.append(float(std / mean) if mean != 0 else 0.0)  # CV

        # Entropy (20 bins)
        hist, _ = np.histogram(arr, bins=min(20, n // 2 + 1))
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
        features.append(float(entropy))

        # Range / mean
        features.append(float((np.max(arr) - np.min(arr)) / mean) if mean != 0 else 0)

        # ---- Block 5: Autocorrelation (4) ----
        centered = arr - mean
        var = np.var(arr)
        for lag in [1, 2, 5, 10]:
            if n > lag + 1 and var > 0:
                ac = np.sum(centered[:-lag] * centered[lag:]) / (var * (n - lag))
                features.append(float(ac))
            else:
                features.append(0.0)

        # ---- Block 6: FFT Spectral Features (4) ----
        if n >= 16:
            try:
                detrended = arr - np.polyval(np.polyfit(range(n), arr, 1), range(n))
                yf = np.abs(fft(detrended)) if SCIPY_AVAILABLE else np.abs(np.fft.fft(detrended))  # type: ignore[reportPossiblyUnbound]
                half = yf[1:n // 2]
                if len(half) > 0:
                    features.append(float(np.max(half)))
                    features.append(float(np.mean(half)))
                    peak_idx = int(np.argmax(half))
                    features.append(float(peak_idx + 1))  # dominant period
                    features.append(float(np.sum(half[:3]) / (np.sum(half) + 1e-10)))  # low-freq ratio
                else:
                    features.extend([0, 0, 0, 0])
            except Exception:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])

        # ---- Block 7: Streak & Run Features (4) ----
        above = arr > mean
        runs = 1
        current_run = 1
        max_run = 1
        for i in range(1, n):
            if above[i] != above[i - 1]:
                runs += 1
                max_run = max(max_run, current_run)
                current_run = 1
            else:
                current_run += 1
        max_run = max(max_run, current_run)
        features.append(float(runs))
        features.append(float(max_run))
        features.append(float(current_run))
        features.append(1.0 if above[-1] else 0.0)

        # ---- Block 8: Relative & Ratio Features (2) - pad to 50 ----
        # EWMA (exponentially weighted)
        alpha = 0.1
        ewma = arr[0]
        for v in arr[1:]:
            ewma = alpha * v + (1 - alpha) * ewma
        features.append(float(ewma))
        features.append(float(arr[-1] - ewma))  # deviation from EWMA

        # Pad to exactly 50
        while len(features) < 50:
            features.append(0.0)

        return features[:50]

    def _prepare_data(self, data: list, target_threshold: Optional[float] = None) -> tuple:
        """Prepare multi-window feature matrix with proper target encoding"""
        if target_threshold is None:
            target_threshold = np.median(data)  # median more robust than mean

        X, y = [], []
        window_size = max(ML_SETTINGS.get('window_sizes', [50])[-1], 50)

        for i in range(window_size, len(data)):
            window = data[max(0, i - window_size):i]
            features = self._extract_features(window)
            target = 1 if data[i] > target_threshold else 0
            X.append(features)
            y.append(target)

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int32)

        # Remove any NaN/Inf
        mask = np.all(np.isfinite(X), axis=1)
        return X[mask], y[mask]

    def _prepare_dl_data(self, data: list) -> tuple:
        """Prepare data for deep learning models"""
        data_array = np.array(data)
        std = np.std(data_array)
        normalized = (data_array - np.mean(data_array)) / std if std > 0 else data_array

        seq_len = DL_SETTINGS['sequence_length']
        sequences = []
        targets = []

        for i in range(len(normalized) - seq_len):
            seq = normalized[i:i + seq_len]
            target = normalized[i + seq_len]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    # ==================================================================
    # ACCURACY & STATS
    # ==================================================================
    def get_accuracy_stats(self) -> dict:
        if not self.prediction_history:
            return {
                'accuracy': 0.0, 'total_predictions': 0,
                'recent_accuracy': 0.0, 'model_accuracies': {}
            }

        correct = sum(1 for p in self.prediction_history if p.get('correct', False))
        total = len(self.prediction_history)
        accuracy = correct / total if total > 0 else 0.0

        recent = self.prediction_history[-20:]
        recent_correct = sum(1 for p in recent if p.get('correct', False))
        recent_accuracy = recent_correct / len(recent) if recent else 0.0

        model_accuracies = {}
        for model_name in self.models:
            model_preds = [p for p in self.prediction_history if model_name in p.get('predictions', {})]
            if model_preds:
                mc = sum(1 for p in model_preds if p.get('predictions', {}).get(model_name) == p.get('actual'))
                model_accuracies[model_name] = mc / len(model_preds)

        return {
            'accuracy': accuracy,
            'total_predictions': total,
            'recent_accuracy': recent_accuracy,
            'model_accuracies': model_accuracies,
            'model_weights': dict(self.model_weights),
            'num_models': len(self.models),
            'stacking': self.stacker is not None,
            'train_time': round(self._train_time, 2),
        }

    def get_combined_stats(self) -> dict:
        sklearn_stats = self.get_accuracy_stats()
        dl_stats: dict = {}
        if DL_AVAILABLE and self.dl_pipeline is not None:
            try:
                dl_stats = {
                    'framework': self.dl_pipeline._framework,
                    'best_model': self.dl_pipeline.best_model,
                    'best_accuracy': self.dl_pipeline.best_accuracy,
                }
            except Exception:
                pass

        return {
            'accuracy': sklearn_stats['accuracy'],
            'total_predictions': sklearn_stats['total_predictions'],
            'recent_accuracy': sklearn_stats['recent_accuracy'],
            'sklearn_models': len(self.models),
            'model_weights': sklearn_stats.get('model_weights', {}),
            'stacking_enabled': self.stacker is not None,
            'dl_trained': DL_AVAILABLE and self.dl_pipeline is not None and self.dl_pipeline.is_trained,
            'dl_framework': dl_stats.get('framework'),
            'dl_best_model': dl_stats.get('best_model'),
            'dl_best_accuracy': dl_stats.get('best_accuracy', 0.0),
            'total_models': len(self.models) + (len(getattr(self.dl_pipeline, 'models', {})) if self.dl_pipeline else 0),
            'total_features': 50,
        }

    def cross_validate(self, data: list, cv_folds: int = 5) -> dict:
        """Cross-validate all models using proper TimeSeriesSplit"""
        if len(data) < 100:
            return {}

        X, y = self._prepare_data(data)
        if len(X) < 50:
            return {}

        scaler = RobustScaler()  # type: ignore[reportPossiblyUnbound]
        X_scaled = scaler.fit_transform(X)

        results = {}
        tscv = TimeSeriesSplit(n_splits=cv_folds)  # type: ignore[reportPossiblyUnbound]

        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')  # type: ignore[reportPossiblyUnbound]
                results[name] = {
                    'mean_accuracy': round(float(np.mean(scores)), 4),
                    'std_accuracy': round(float(np.std(scores)), 4),
                    'scores': [round(float(s), 4) for s in scores]
                }
            except Exception as e:
                print(f"[MLBrain] CV error for {name}: {e}")

        return results

    def get_feature_importance(self, model_name: Optional[str] = None) -> dict:
        if model_name:
            return {model_name: self.feature_importance[model_name]} if model_name in self.feature_importance else {}
        return self.feature_importance

    def update_prediction_history(self, actual_value: float, predictions: dict):
        """Update prediction history and dynamically adjust model weights"""
        threshold = np.median([p.get('actual', 0) for p in self.prediction_history[-100:]]) if self.prediction_history else 0
        actual_class = 1 if actual_value > threshold else 0

        history_entry = {
            'timestamp': pd.Timestamp.now(),
            'actual': actual_class,
            'actual_value': actual_value,
            'predictions': predictions,
            'correct': False,
        }

        # Check each model
        if predictions:
            for model_name, pred in predictions.items():
                if isinstance(pred, (int, float)):
                    if int(pred) == actual_class:
                        history_entry['correct'] = True
                    break

            # Dynamic weight update (reward correct models)
            for model_name, pred in predictions.items():
                if isinstance(pred, (int, float)) and model_name in self.model_weights:
                    if int(pred) == actual_class:
                        self.model_weights[model_name] *= 1.02  # 2% boost
                    else:
                        self.model_weights[model_name] *= 0.98  # 2% penalty

            # Re-normalize
            total = sum(self.model_weights.values())
            if total > 0:
                self.model_weights = {k: v / total for k, v in self.model_weights.items()}

        self.prediction_history.append(history_entry)
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)

    def get_model_details(self) -> dict:
        """Get detailed information about all models"""
        details = {}

        for name, model in self.models.items():
            details[name] = {
                'type': type(model).__name__,
                'trained': name in self.scalers,
                'feature_importance': self.feature_importance.get(name, []),
                'parameters': model.get_params() if hasattr(model, 'get_params') else {}
            }

        # Add deep learning details
        if self.dl_pipeline is not None and self.dl_pipeline.is_trained:
            dl_details = {'is_trained': True, 'framework': self.dl_pipeline._framework,
                          'best_model': self.dl_pipeline.best_model,
                          'best_accuracy': self.dl_pipeline.best_accuracy}
            details['deep_learning'] = dl_details

        return details

    def reset_models(self):
        """Reset all models and clear history"""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_history = []
        self.dl_pipeline = DeepLearningPipeline()  # type: ignore[reportPossiblyUnbound]
        self._init_models()

    def save_models(self, filepath: str):
        """Save trained models to file"""
        import joblib
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save sklearn models and scalers
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'prediction_history': self.prediction_history
        }

        joblib.dump(model_data, filepath)

        # Save deep learning models
        if self.dl_pipeline is not None and self.dl_pipeline.is_trained:
            joblib.dump(self.dl_pipeline, filepath.replace('.pkl', '_dl.pkl'))

    def load_models(self, filepath: str):
        """Load trained models from file"""
        import joblib

        try:
            # Load sklearn models
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_importance = model_data['feature_importance']
            self.prediction_history = model_data['prediction_history']

            # Load deep learning models
            dl_filepath = filepath.replace('.pkl', '_dl.pkl')
            import os
            if os.path.exists(dl_filepath):
                self.dl_pipeline = joblib.load(dl_filepath)

            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False