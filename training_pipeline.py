"""
Deep Learning Training Pipeline
LSTM, GRU, BiLSTM, Attention-LSTM, and TCN models for sequence prediction.
Trains on game data to find patterns in random number generators.
"""

import numpy as np
import json
import os
import time
from collections import deque
from datetime import datetime
from config import DL_SETTINGS, ML_SETTINGS
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Any


class DeepLearningPipeline:
    """Deep Learning training pipeline with multiple architectures"""

    def __init__(self):
        self.models = {}
        self.history = {}
        self.scalers = {}
        self.is_trained = False
        self.training_log = []
        self.best_model = None
        self.best_accuracy = 0.0
        self._framework = None
        self._check_framework()

    def _check_framework(self):
        """Check which DL framework is available"""
        try:
            import torch
            import torch.nn as nn
            self._framework = 'pytorch'
            print("[DL] PyTorch detected - using PyTorch backend")
        except ImportError:
            try:
                import tensorflow as tf
                self._framework = 'tensorflow'
                print("[DL] TensorFlow detected - using TF backend")
            except ImportError:
                self._framework = 'numpy'
                print("[DL] No DL framework found - using NumPy neural network")

    def prepare_sequences(self, data, sequence_length=None):
        """Prepare sequential data for training"""
        if sequence_length is None:
            sequence_length = DL_SETTINGS['sequence_length']

        data = np.array(data, dtype=np.float32)

        # Normalize data
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            std = 1.0
        normalized = (data - mean) / std

        self.scalers['mean'] = float(mean)
        self.scalers['std'] = float(std)

        # Create sequences
        X, y = [], []
        for i in range(len(normalized) - sequence_length):
            X.append(normalized[i:i + sequence_length])
            # Binary classification: next value above or below mean
            y.append(1 if data[i + sequence_length] > mean else 0)

        X = np.array(X)
        y = np.array(y)

        # Reshape for LSTM: (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        return X, y

    def train_all_models(self, data, callback=None):
        """Train all deep learning models"""
        if len(data) < DL_SETTINGS['sequence_length'] + 50:
            return {
                'error': f"Need at least {DL_SETTINGS['sequence_length'] + 50} data points",
                'data_available': len(data)
            }

        X, y = self.prepare_sequences(data)

        # Train/test split (time series aware)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        results: Dict[str, Any] = {}
        start_time = time.time()

        if self._framework == 'pytorch':
            results = self._train_pytorch(X_train, y_train, X_test, y_test, callback)
        elif self._framework == 'tensorflow':
            results = self._train_tensorflow(X_train, y_train, X_test, y_test, callback)
        else:
            results = self._train_numpy(X_train, y_train, X_test, y_test, callback)

        total_time = time.time() - start_time
        results['total_training_time'] = round(total_time, 2)
        results['data_points'] = len(data)
        results['sequences_created'] = len(X)
        results['train_size'] = len(X_train)
        results['test_size'] = len(X_test)

        # Find best model
        best_name = None
        best_acc = 0
        for name, info in results.get('models', {}).items():
            acc = info.get('test_accuracy', 0)
            if acc > best_acc:
                best_acc = acc
                best_name = name

        if best_name:
            self.best_model = best_name
            self.best_accuracy = best_acc
            results['best_model'] = best_name
            results['best_accuracy'] = best_acc

        self.is_trained = True
        self.training_log.append({
            'timestamp': datetime.now().isoformat(),
            'results': results,
        })

        return results

    def _train_pytorch(self, X_train, y_train, X_test, y_test, callback=None):
        """Train models using PyTorch"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DL] Training on: {device}")

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.LongTensor(y_test).to(device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=DL_SETTINGS['batch_size'], shuffle=False)

        results = {'models': {}}
        model_configs = self._get_pytorch_models(X_train.shape[1], device)

        for name, model in model_configs.items():
            print(f"[DL] Training {name}...")
            if callback:
                callback(f"Training {name}...")

            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=DL_SETTINGS['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

            best_val_acc = 0
            patience_counter = 0
            train_losses = []

            for epoch in range(DL_SETTINGS['epochs']):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(loader)
                train_losses.append(avg_loss)
                scheduler.step(avg_loss)

                # Evaluate
                model.eval()
                with torch.no_grad():
                    test_out = model(X_test_t)
                    _, predicted = torch.max(test_out, 1)
                    val_acc = (predicted == y_test_t).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model state
                    self.models[name] = {
                        'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        'model_class': type(model),
                    }
                else:
                    patience_counter += 1

                if patience_counter >= DL_SETTINGS['patience']:
                    print(f"[DL] {name} early stopping at epoch {epoch+1}")
                    break

                if callback and epoch % 10 == 0:
                    callback(f"{name}: epoch {epoch+1}, loss={avg_loss:.4f}, val_acc={val_acc:.1%}")

            results['models'][name] = {
                'test_accuracy': round(best_val_acc, 4),
                'final_loss': round(train_losses[-1], 4),
                'epochs_trained': len(train_losses),
                'parameters': sum(p.numel() for p in model.parameters()),
            }
            print(f"[DL] {name}: accuracy={best_val_acc:.1%}, epochs={len(train_losses)}")

        return results

    def _get_pytorch_models(self, seq_length, device):
        """Get all PyTorch model architectures"""
        import torch
        import torch.nn as nn

        models = {}

        # LSTM
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 128, num_layers=2, batch_first=True, dropout=0.2)
                self.fc1 = nn.Linear(128, 64)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(64, 2)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = lstm_out[:, -1, :]
                out = self.dropout(self.relu(self.fc1(out)))
                return self.fc2(out)

        models['LSTM'] = LSTMModel()

        # GRU
        class GRUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(1, 128, num_layers=2, batch_first=True, dropout=0.2)
                self.fc1 = nn.Linear(128, 64)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(64, 2)

            def forward(self, x):
                gru_out, _ = self.gru(x)
                out = gru_out[:, -1, :]
                out = self.dropout(self.relu(self.fc1(out)))
                return self.fc2(out)

        models['GRU'] = GRUModel()

        # BiLSTM
        class BiLSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bilstm = nn.LSTM(1, 64, num_layers=2, batch_first=True,
                                       dropout=0.2, bidirectional=True)
                self.fc1 = nn.Linear(128, 64)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(64, 2)

            def forward(self, x):
                out, _ = self.bilstm(x)
                out = out[:, -1, :]
                out = self.dropout(self.relu(self.fc1(out)))
                return self.fc2(out)

        models['BiLSTM'] = BiLSTMModel()

        # Attention LSTM
        class AttentionLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 128, num_layers=2, batch_first=True, dropout=0.2)
                self.attention = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
                self.fc1 = nn.Linear(128, 64)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(64, 2)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Attention mechanism
                attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context = torch.sum(attn_weights * lstm_out, dim=1)
                out = self.dropout(self.relu(self.fc1(context)))
                return self.fc2(out)

        models['AttentionLSTM'] = AttentionLSTM()

        # TCN (Temporal Convolutional Network)
        class TCNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1, dilation=1)
                self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2)
                self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc1 = nn.Linear(64, 32)
                self.fc2 = nn.Linear(32, 2)

            def forward(self, x):
                # x shape: (batch, seq_len, 1) -> (batch, 1, seq_len)
                x = x.permute(0, 2, 1)
                x = self.dropout(self.relu(self.conv1(x)))
                x = self.dropout(self.relu(self.conv2(x)))
                x = self.dropout(self.relu(self.conv3(x)))
                x = self.pool(x).squeeze(-1)
                x = self.dropout(self.relu(self.fc1(x)))
                return self.fc2(x)

        models['TCN'] = TCNModel()

        return models

    def _train_tensorflow(self, X_train, y_train, X_test, y_test, callback=None):
        """Train models using TensorFlow/Keras"""
        import tensorflow as tf
        from tensorflow.keras import layers, models as tf_models, callbacks

        results = {'models': {}}

        model_configs = {
            'LSTM': self._build_tf_lstm,
            'GRU': self._build_tf_gru,
            'BiLSTM': self._build_tf_bilstm,
            'AttentionLSTM': self._build_tf_attention_lstm,
            'TCN': self._build_tf_tcn,
        }

        for name, build_fn in model_configs.items():
            print(f"[DL] Training {name}...")
            if callback:
                callback(f"Training {name}...")

            model = build_fn(X_train.shape[1])

            early_stop = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=DL_SETTINGS['patience'],
                restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=DL_SETTINGS['epochs'],
                batch_size=DL_SETTINGS['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )

            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            self.models[name] = model

            results['models'][name] = {
                'test_accuracy': round(accuracy, 4),
                'final_loss': round(loss, 4),
                'epochs_trained': len(history.history['loss']),
                'parameters': model.count_params(),
            }
            print(f"[DL] {name}: accuracy={accuracy:.1%}, epochs={len(history.history['loss'])}")

        return results

    def _build_tf_lstm(self, seq_length):
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_tf_gru(self, seq_length):
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(128, return_sequences=True, input_shape=(seq_length, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_tf_bilstm(self, seq_length):
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(seq_length, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_tf_attention_lstm(self, seq_length):
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(seq_length, 1))
        x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        # Simple attention
        attention = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(128)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        x = tf.keras.layers.Multiply()([x, attention])
        x = tf.keras.layers.Lambda(lambda z: tf.keras.backend.sum(z, axis=1))(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_tf_tcn(self, seq_length):
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(seq_length, 1))
        x = tf.keras.layers.Conv1D(64, 3, padding='causal', dilation_rate=1, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv1D(64, 3, padding='causal', dilation_rate=4, activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _train_numpy(self, X_train, y_train, X_test, y_test, callback=None):
        """Pure NumPy neural network fallback (no framework needed)"""
        results = {'models': {}}

        # Flatten sequences for simple NN
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        input_dim = X_train_flat.shape[1]
        architectures = {
            'NumpyNN_128_64': [128, 64],
            'NumpyNN_256_128_64': [256, 128, 64],
            'NumpyNN_64_32': [64, 32],
        }

        for name, hidden_layers in architectures.items():
            print(f"[DL] Training {name}...")
            if callback:
                callback(f"Training {name}...")

            model = NumpyNeuralNetwork(input_dim, hidden_layers, 2)
            history = model.train(
                X_train_flat, y_train,
                X_test_flat, y_test,
                epochs=DL_SETTINGS['epochs'],
                lr=DL_SETTINGS['learning_rate'],
                batch_size=DL_SETTINGS['batch_size'],
                patience=DL_SETTINGS['patience']
            )

            self.models[name] = model
            results['models'][name] = {
                'test_accuracy': round(history['best_val_accuracy'], 4),
                'final_loss': round(history['final_loss'], 4),
                'epochs_trained': history['epochs_trained'],
                'parameters': model.param_count(),
            }
            print(f"[DL] {name}: accuracy={history['best_val_accuracy']:.1%}")

        return results

    def predict(self, data):
        """Make prediction using best trained model"""
        if not self.is_trained or not self.models:
            return None

        seq_length = DL_SETTINGS['sequence_length']
        if len(data) < seq_length:
            return None

        # Prepare input
        recent = np.array(data[-seq_length:], dtype=np.float32)
        mean = self.scalers.get('mean', np.mean(data))
        std = self.scalers.get('std', np.std(data))
        if std == 0:
            std = 1.0
        normalized = (recent - mean) / std

        predictions = {}

        if self._framework == 'pytorch':
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = torch.FloatTensor(normalized.reshape(1, -1, 1)).to(device)

            for name, model_info in self.models.items():
                try:
                    model = model_info['model_class']()
                    model.load_state_dict({k: v.to(device) for k, v in model_info['state_dict'].items()})
                    model.to(device)
                    model.eval()
                    with torch.no_grad():
                        output = model(x)
                        probs = torch.softmax(output, dim=1)
                        pred = int(torch.argmax(probs, dim=1).item())
                        conf = probs[0][pred].item()
                    predictions[name] = {'prediction': pred, 'confidence': conf}
                except Exception as e:
                    print(f"[DL] {name} prediction error: {e}")

        elif self._framework == 'tensorflow':
            x = normalized.reshape(1, seq_length, 1)
            for name, model in self.models.items():
                try:
                    prob = model.predict(x, verbose=0)[0][0]
                    pred = 1 if prob > 0.5 else 0
                    conf = prob if pred == 1 else 1 - prob
                    predictions[name] = {'prediction': pred, 'confidence': float(conf)}
                except Exception as e:
                    print(f"[DL] {name} prediction error: {e}")

        else:  # numpy
            x = normalized.reshape(1, -1)
            for name, model in self.models.items():
                try:
                    pred, conf = model.predict(x)
                    predictions[name] = {'prediction': int(pred[0]), 'confidence': float(conf[0])}
                except Exception as e:
                    print(f"[DL] {name} prediction error: {e}")

        if not predictions:
            return None

        # Ensemble vote
        votes = [p['prediction'] for p in predictions.values()]
        confs = [p['confidence'] for p in predictions.values()]
        ensemble = 1 if sum(votes) > len(votes) / 2 else 0
        avg_conf = np.mean(confs)

        return {
            'models': predictions,
            'ensemble': ensemble,
            'confidence': round(avg_conf, 4),
            'direction': 'above_mean' if ensemble == 1 else 'below_mean',
            'framework': self._framework,
        }

    def get_training_summary(self):
        """Get training summary"""
        if not self.training_log:
            return "No training has been performed yet"

        last = self.training_log[-1]['results']
        summary = f"=== DEEP LEARNING TRAINING SUMMARY ===\n"
        summary += f"Framework: {self._framework}\n"
        summary += f"Data Points: {last.get('data_points', 0)}\n"
        summary += f"Sequences: {last.get('sequences_created', 0)}\n"
        summary += f"Train/Test: {last.get('train_size', 0)}/{last.get('test_size', 0)}\n"
        summary += f"Training Time: {last.get('total_training_time', 0):.1f}s\n\n"

        summary += "=== MODEL RESULTS ===\n"
        for name, info in last.get('models', {}).items():
            marker = " ★ BEST" if name == self.best_model else ""
            summary += f"  {name}: {info['test_accuracy']:.1%} accuracy, "
            summary += f"{info['epochs_trained']} epochs, "
            summary += f"{info.get('parameters', 0):,} params{marker}\n"

        if self.best_model:
            summary += f"\nBest Model: {self.best_model} ({self.best_accuracy:.1%})\n"

        return summary


class NumpyNeuralNetwork:
    """Pure NumPy neural network (no framework dependency)"""

    def __init__(self, input_dim, hidden_layers, output_dim):
        self.layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize weights with He initialization
        dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(dims) - 1):
            weight = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            bias = np.zeros((1, dims[i + 1]))
            self.layers.append({'W': weight, 'b': bias})

    def param_count(self):
        count = 0
        for layer in self.layers:
            count += layer['W'].size + layer['b'].size
        return count

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass"""
        self.activations = [X]
        self.z_values = []

        current = X
        for i, layer in enumerate(self.layers):
            z = current @ layer['W'] + layer['b']
            self.z_values.append(z)

            if i < len(self.layers) - 1:
                current = self._relu(z)
            else:
                current = self._softmax(z)

            self.activations.append(current)

        return current

    def backward(self, y_true, lr):
        """Backward pass with gradient descent"""
        m = y_true.shape[0]
        y_one_hot = np.zeros((m, self.output_dim))
        y_one_hot[np.arange(m), y_true.astype(int)] = 1

        # Output layer gradient
        delta = self.activations[-1] - y_one_hot

        for i in reversed(range(len(self.layers))):
            dW = self.activations[i].T @ delta / m
            db = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.layers[i]['W'].T) * self._relu_deriv(self.z_values[i - 1])

            # Gradient clipping
            np.clip(dW, -1.0, 1.0, out=dW)
            np.clip(db, -1.0, 1.0, out=db)

            self.layers[i]['W'] -= lr * dW
            self.layers[i]['b'] -= lr * db

    def train(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, batch_size=32, patience=15):
        """Train the network"""
        best_val_acc = 0
        patience_counter = 0
        best_weights = None
        final_loss = 0
        epoch = -1

        for epoch in range(epochs):
            # Mini-batch training
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            epoch_loss = 0
            n_batches = 0

            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = indices[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward + backward
                output = self.forward(X_batch)
                self.backward(y_batch, lr)

                # Loss
                m = len(y_batch)
                y_oh = np.zeros((m, self.output_dim))
                y_oh[np.arange(m), y_batch.astype(int)] = 1
                loss = -np.mean(np.sum(y_oh * np.log(output + 1e-8), axis=1))
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            final_loss = avg_loss

            # Validation
            val_output = self.forward(X_val)
            val_pred = np.argmax(val_output, axis=1)
            val_acc = np.mean(val_pred == y_val)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_weights = [(l['W'].copy(), l['b'].copy()) for l in self.layers]
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Restore best weights
        if best_weights:
            for i, (W, b) in enumerate(best_weights):
                self.layers[i]['W'] = W
                self.layers[i]['b'] = b

        return {
            'best_val_accuracy': best_val_acc,
            'final_loss': final_loss,
            'epochs_trained': epoch + 1,
        }

    def predict(self, X):
        """Predict class and confidence"""
        probs = self.forward(X)
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return preds, confidences
