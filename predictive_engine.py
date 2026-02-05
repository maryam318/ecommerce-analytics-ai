import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from prophet import Prophet  # Optional

class EcomPredictor:
    def __init__(self):
        self.forecast_model = None
        self.anomaly_model = IsolationForest(contamination='auto')
        self.detected_freq = None

    def _detect_column(self, df, possible_names):
        """Helper to find columns with flexible matching"""
        for name in possible_names:
            if name in df.columns:
                return name
            for col in df.columns:
                if name in col.lower():
                    return col
        return None

    def preprocess(self, df):
        """Robust preprocessing with automatic column detection"""
        # Auto-detect date column
        date_col = self._detect_column(df, ['date', 'time', 'day', 'order_date', 'timestamp'])
        if date_col is None:
            date_col = df.select_dtypes(include=['datetime']).columns[0] if \
                      len(df.select_dtypes(include=['datetime'])) > 0 else None
        
        # Auto-detect value column
        value_col = self._detect_column(df, ['sales', 'amount', 'revenue', 'value', 'total'])
        if value_col is None:
            value_col = df.select_dtypes(include=['number']).columns[0] if \
                       len(df.select_dtypes(include=['number'])) > 0 else None

        if not date_col or not value_col:
            raise ValueError(
                f"Could not detect required columns. "
                f"Date candidates: {df.select_dtypes(include=['datetime']).columns.tolist()}, "
                f"Value candidates: {df.select_dtypes(include=['number']).columns.tolist()}"
            )

        # Standardize and clean
        clean_df = df[[date_col, value_col]].copy()
        clean_df.columns = ['date', 'value']
        clean_df['date'] = pd.to_datetime(clean_df['date'])
        clean_df = clean_df.dropna(subset=['date', 'value'])
        
        # Detect frequency
        self.detected_freq = pd.infer_freq(clean_df['date']) or 'D'
        return clean_df

    def train_forecaster(self, df, model_type='rf'):
        """Train forecasting model"""
        clean_df = self.preprocess(df)
        
        # Feature engineering
        features = pd.DataFrame({
            'day_of_week': clean_df['date'].dt.dayofweek,
            'month': clean_df['date'].dt.month,
            'day_of_year': clean_df['date'].dt.dayofyear
        })
        
        if model_type == 'rf':
            self.forecast_model = RandomForestRegressor()
            self.forecast_model.fit(features, clean_df['value'])
        elif model_type == 'prophet':
            try:
                model = Prophet()
                train_df = clean_df.rename(columns={'date': 'ds', 'value': 'y'})
                model.fit(train_df)
                self.forecast_model = model
            except ImportError:
                raise ImportError("Prophet not available. Install with: pip install prophet")
        
        return self.forecast_model

    def detect_anomalies(self, df):
        """Detect anomalies in the data"""
        clean_df = self.preprocess(df)
        values = clean_df['value'].values.reshape(-1, 1)
        return self.anomaly_model.fit_predict(values)

    def get_forecast(self, df, periods=7):
        """Generate forecast"""
        if not self.forecast_model:
            self.train_forecaster(df)
        
        last_date = df['date'].max()
        future_dates = pd.date_range(
            start=last_date,
            periods=periods + 1,
            freq=self.detected_freq
        )[1:]
        
        if isinstance(self.forecast_model, RandomForestRegressor):
            future_features = pd.DataFrame({
                'day_of_week': future_dates.dayofweek,
                'month': future_dates.month,
                'day_of_year': future_dates.dayofyear
            })
            predictions = self.forecast_model.predict(future_features)
            return pd.Series(predictions, index=future_dates)
        
        elif hasattr(self.forecast_model, 'make_future_dataframe'):
            future = self.forecast_model.make_future_dataframe(periods=periods)
            forecast = self.forecast_model.predict(future)
            return forecast.set_index('ds')['yhat'].iloc[-periods:]
        
        raise ValueError("No valid forecasting model trained")