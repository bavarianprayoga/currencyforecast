import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('currency_forecaster.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CurrencyForecaster:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.volatility_lookback = 30
        
    def _create_cyclic_features(self, df):
        """Create cyclic features for time variables"""
        # Day of week: map to [0, 2π]
        day_rad = 2 * np.pi * df.index.dayofweek / 7
        return pd.DataFrame({
            'day_sin': np.sin(day_rad),
            'day_cos': np.cos(day_rad)
        }, index=df.index)
        
    def _create_features(self, df):
        """Create simple, robust features using percentage changes"""
        try:
            logger.info("Creating technical features...")
            features = pd.DataFrame(index=df.index)
            
            # Percentage changes instead of absolute values
            features['pct_change'] = df['rate'].pct_change()
            
            # Simple moving averages of percentage changes
            for window in [3, 7, 14]:
                features[f'ma_{window}d'] = df['rate'].pct_change().rolling(
                    window=window, 
                    min_periods=1
                ).mean()
                
            # Add volatility features
            features['volatility'] = features['pct_change'].rolling(
                window=5,
                min_periods=1
            ).std()
            
            # Add cyclic time features
            cyclic_features = self._create_cyclic_features(df)
            features = pd.concat([features, cyclic_features], axis=1)
            
            # Fill missing values
            features = features.ffill().bfill()
            
            logger.info(f"Created {len(features.columns)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            raise
            
    def fit(self, df):
        """Fit model with simplified validation"""
        try:
            logger.info("Starting model fitting...")
            
            # Create features
            features = self._create_features(df)
            self.feature_columns = features.columns
            
            # Prepare target (next day's percentage change)
            target = df['rate'].pct_change().shift(-1)
            
            # Remove last row since we won't have target for it
            features = features[:-1]
            target = target[:-1]
            
            # Simple train-test split preserving time order
            train_size = int(len(features) * 0.8)
            X_train = features[:train_size]
            y_train = target[:train_size]
            
            # Initialize LightGBM with conservative parameters
            self.model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                num_leaves=15,
                min_child_samples=3,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42
            )
            
            # Fit model
            self.model.fit(X_train, y_train)
            
            logger.info("Model fitting completed")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise
            
    def predict(self, df, forecast_days=7):
        """Generate forecasts with sanity checks"""
        try:
            if self.model is None:
                raise ValueError("Model not fitted. Call fit() first.")
                
            logger.info(f"Generating {forecast_days}-day forecast...")
            
            # Create features
            features = self._create_features(df)
            
            # Make recursive predictions
            predictions = []
            last_rate = df['rate'].iloc[-1]
            last_data = df.copy()
            
            for i in range(forecast_days):
                # Predict next day's percentage change
                next_day_features = self._create_features(last_data).iloc[-1:]
                pct_change = self.model.predict(next_day_features[self.feature_columns])[0]
                
                # Sanity check - limit extreme predictions
                pct_change = np.clip(pct_change, -0.1, 0.1)  # Max 10% change per day
                
                # Convert to next day's rate
                next_rate = last_rate * (1 + pct_change)
                predictions.append(next_rate)
                
                # Update for next iteration
                new_idx = last_data.index[-1] + pd.Timedelta(days=1)
                last_data.loc[new_idx, 'rate'] = next_rate
                last_rate = next_rate
            
            logger.info("Forecast generated successfully")
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise