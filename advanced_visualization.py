#!/usr/bin/env python3
"""
Advanced Visualization Engine for Edge Tracker 2026
Enhanced with Plotly, Seaborn, and real-time charts
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
from config import UI_SETTINGS

class AdvancedVisualization:
    """Advanced visualization engine with multiple chart types"""

    def __init__(self):
        self.fig = go.Figure()
        self.traces = []
        self.update_thread = None
        self.running = False
        
        # Set dark theme for all charts
        plt.style.use('dark_background')
        sns.set_theme(style="dark")

    def create_real_time_dashboard(self, data_engine):
        """Create comprehensive real-time dashboard"""
        self.fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                '🎯 Live Game Values', '📊 Distribution Analysis', '🔄 Multiplier Trends',
                '🧠 ML Predictions', '⚡ Performance Metrics', '🎲 Game Statistics',
                '📈 Volatility Analysis', '🎪 Pattern Detection', '💰 Strategy Performance'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "indicator"}, {"type": "pie"}],
                [{"type": "heatmap"}, {"type": "surface"}, {"type": "waterfall"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )

        # Configure layout with enhanced styling
        self.fig.update_layout(
            title={
                'text': "🚀 Edge Tracker 2026 - Advanced Analytics Dashboard",
                'x': 0.5,
                'font': {'size': 24, 'color': '#56d364'}
            },
            template="plotly_dark",
            height=1000,
            showlegend=True,
            paper_bgcolor='#0a0e17',
            plot_bgcolor='#0a0e17',
            font=dict(color='#e6edf3', size=12),
            margin=dict(l=50, r=50, t=80, b=50)
        )

        return self.fig

    def update_live_charts(self, data_engine, ml_brain):
        """Update all charts with live data"""
        try:
            # Get data for current game
            current_game = 'crash'  # Default
            df = data_engine.get_dataframe(current_game, n_points=200)
            
            if len(df) == 0:
                return

            # Clear existing traces
            self.fig.data = []

            # Chart 1: Live Game Values with predictions
            self._add_live_values_chart(df, ml_brain, row=1, col=1)
            
            # Chart 2: Distribution Analysis
            self._add_distribution_chart(df, row=1, col=2)
            
            # Chart 3: Trend Analysis
            self._add_trend_chart(df, row=1, col=3)
            
            # Chart 4: ML Predictions
            self._add_ml_predictions_chart(df, ml_brain, row=2, col=1)
            
            # Chart 5: Performance Metrics
            self._add_performance_chart(ml_brain, row=2, col=2)
            
            # Chart 6: Game Statistics
            self._add_statistics_chart(df, row=2, col=3)
            
            # Chart 7: Volatility Analysis
            self._add_volatility_chart(df, row=3, col=1)
            
            # Chart 8: Pattern Detection
            self._add_pattern_chart(df, row=3, col=2)
            
            # Chart 9: Strategy Performance
            self._add_strategy_chart(df, row=3, col=3)

        except Exception as e:
            print(f"Chart update error: {e}")

    def _add_live_values_chart(self, df, ml_brain, row, col):
        """Add live values with ML predictions"""
        values = df['value'].values
        timestamps = df.index
        
        # Actual values
        self.fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name='Live Values',
                line=dict(color='#56d364', width=2),
                marker=dict(size=4),
                hovertemplate='Value: %{y:.2f}<br>Time: %{x}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # ML predictions if available
        try:
            predictions = ml_brain.get_recent_predictions()
            if len(predictions) > 0:
                pred_x = [p['timestamp'] for p in predictions]
                pred_y = [p['prediction'] for p in predictions]
                
                self.fig.add_trace(
                    go.Scatter(
                        x=pred_x,
                        y=pred_y,
                        mode='markers',
                        name='ML Predictions',
                        marker=dict(color='#f85149', size=8, symbol='diamond'),
                        hovertemplate='Prediction: %{y:.2f}<br>Time: %{x}<extra></extra>'
                    ),
                    row=row, col=col
                )
        except:
            pass

    def _add_distribution_chart(self, df, row, col):
        """Add distribution histogram"""
        values = df['value'].values
        
        self.fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=30,
                name='Distribution',
                marker=dict(color='#58a6ff', opacity=0.7),
                hovertemplate='Value Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col
        )

    def _add_trend_chart(self, df, row, col):
        """Add trend analysis"""
        values = df['value'].values
        if len(values) > 20:
            # Calculate moving averages
            ma_5 = pd.Series(values).rolling(5).mean()
            ma_20 = pd.Series(values).rolling(20).mean()
            
            self.fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma_5,
                    mode='lines',
                    name='MA-5',
                    line=dict(color='#d29922', width=2)
                ),
                row=row, col=col
            )
            
            self.fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma_20,
                    mode='lines',
                    name='MA-20',
                    line=dict(color='#a5a2ff', width=2)
                ),
                row=row, col=col
            )

    def _add_ml_predictions_chart(self, df, ml_brain, row, col):
        """Add ML prediction accuracy chart"""
        try:
            accuracy_history = ml_brain.accuracy_history
            if len(accuracy_history) > 0:
                models = list(accuracy_history[0].keys())
                x_vals = list(range(len(accuracy_history)))
                
                for model in models:
                    y_vals = [acc.get(model, 0) for acc in accuracy_history]
                    self.fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='lines+markers',
                            name=f'{model} Accuracy',
                            line=dict(width=2),
                            marker=dict(size=4)
                        ),
                        row=row, col=col
                    )
        except:
            # Add placeholder
            self.fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0.5, 0.5],
                    mode='lines',
                    name='No ML Data',
                    line=dict(color='gray', dash='dash')
                ),
                row=row, col=col
            )

    def _add_performance_chart(self, ml_brain, row, col):
        """Add performance indicator"""
        try:
            stats = ml_brain.get_accuracy_stats()
            accuracy = stats.get('accuracy', 0)
            
            self.fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=accuracy * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "ML Accuracy %"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#56d364"},
                        'steps': [
                            {'range': [0, 50], 'color': "#f85149"},
                            {'range': [50, 75], 'color': "#d29922"},
                            {'range': [75, 100], 'color': "#56d364"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=row, col=col
            )
        except:
            self.fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=0,
                    title={'text': "ML Accuracy %"},
                    gauge={'bar': {'color': "gray"}}
                ),
                row=row, col=col
            )

    def _add_statistics_chart(self, df, row, col):
        """Add statistics pie chart"""
        values = df['value'].values
        
        # Categorize values
        categories = ['Low (1-2x)', 'Medium (2-5x)', 'High (5-10x)', 'Extreme (>10x)']
        counts = [
            np.sum((values >= 1) & (values < 2)),
            np.sum((values >= 2) & (values < 5)),
            np.sum((values >= 5) & (values < 10)),
            np.sum(values >= 10)
        ]
        
        self.fig.add_trace(
            go.Pie(
                labels=categories,
                values=counts,
                hole=0.3,
                marker=dict(colors=['#56d364', '#58a6ff', '#d29922', '#f85149'])
            ),
            row=row, col=col
        )

    def _add_volatility_chart(self, df, row, col):
        """Add volatility heatmap"""
        values = df['value'].values
        if len(values) > 10:
            # Calculate rolling volatility
            volatility = pd.Series(values).rolling(10).std().values
            volatility = volatility[~np.isnan(volatility)]
            
            if len(volatility) > 0:
                # Create heatmap data
                heatmap_data = volatility.reshape(-1, 1) if len(volatility) > 0 else [[0]]
                
                self.fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    row=row, col=col
                )

    def _add_pattern_chart(self, df, row, col):
        """Add pattern detection chart"""
        values = df['value'].values
        if len(values) > 5:
            # Simple pattern: consecutive increases/decreases
            patterns = []
            for i in range(1, len(values)):
                if values[i] > values[i-1]:
                    patterns.append(1)  # Increase
                elif values[i] < values[i-1]:
                    patterns.append(-1)  # Decrease
                else:
                    patterns.append(0)  # Same
            
            if patterns:
                self.fig.add_trace(
                    go.Bar(
                        x=list(range(len(patterns))),
                        y=patterns,
                        marker=dict(
                            color=patterns,
                            colorscale='RdYlGn',
                            cmin=-1,
                            cmax=1
                        ),
                        name='Patterns'
                    ),
                    row=row, col=col
                )

    def _add_strategy_chart(self, df, row, col):
        """Add strategy performance waterfall"""
        # Mock strategy performance data
        strategies = ['Kelly', 'Martingale', 'Conservative', 'Aggressive']
        values = [10, -5, 15, 8]  # Mock performance
        
        self.fig.add_trace(
            go.Waterfall(
                name="Strategy Performance",
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=strategies,
                textposition="outside",
                text=["+10%", "-5%", "+15%", "+28%"],
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ),
            row=row, col=col
        )

    def create_3d_analysis(self, data_engine):
        """Create 3D analysis visualization"""
        fig_3d = go.Figure()
        
        # Get multi-game data
        games = ['crash', 'dice', 'limbo']
        
        for i, game in enumerate(games):
            df = data_engine.get_dataframe(game, n_points=100)
            if len(df) > 0:
                values = df['value'].values
                timestamps = np.arange(len(values))
                
                fig_3d.add_trace(go.Scatter3d(
                    x=timestamps,
                    y=[i] * len(values),  # Game type axis
                    z=values,
                    mode='markers+lines',
                    name=f'{game.title()} Game',
                    marker=dict(
                        size=3,
                        color=values,
                        colorscale='Viridis',
                        showscale=True if i == 0 else False
                    ),
                    line=dict(width=2)
                ))
        
        fig_3d.update_layout(
            title="🎮 Multi-Game 3D Analysis",
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Game Type",
                zaxis_title="Multiplier",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template="plotly_dark",
            height=600
        )
        
        return fig_3d

    def start_real_time_updates(self, data_engine, ml_brain, update_callback):
        """Start real-time chart updates"""
        self.running = True
        
        def update_loop():
            while self.running:
                try:
                    self.update_live_charts(data_engine, ml_brain)
                    if update_callback:
                        update_callback(self.fig)
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    print(f"Real-time update error: {e}")
                    time.sleep(5)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()

    def stop_real_time_updates(self):
        """Stop real-time updates"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)