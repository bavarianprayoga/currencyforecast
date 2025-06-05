import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress specific LightGBM and sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

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
    """sklearn Pipeline-based currency forecaster using LightGBM"""
    
    def __init__(self):
        self.pipeline = None
        self.feature_columns = None
        self.volatility_lookback = 30
        self.cv_results_ = None  # Store cross-validation results
        self.best_score_ = None  # Store best CV score
        self.best_params_ = None  # Store best parameters
        self._grid_search = None  # Store the GridSearchCV object
        
    def _create_cyclic_features(self, df):
        """Create cyclic features for time variables"""
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
            
            # Percentage changes
            features['pct_change'] = df['rate'].pct_change()
            
            # Moving averages of percentage changes
            for window in [3, 7, 14]:
                features[f'ma_{window}d'] = df['rate'].pct_change().rolling(
                    window=window, 
                    min_periods=1
                ).mean()
                
            # Volatility features
            features['volatility'] = features['pct_change'].rolling(
                window=self.volatility_lookback,
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
        """Fit model with walk-forward validation and hyperparameter tuning using sklearn Pipeline"""
        try:
            logger.info("Starting model fitting with sklearn Pipeline approach...")
            
            # Create features
            features = self._create_features(df)
            
            # Prepare target (next day's percentage change)
            target = df['rate'].pct_change().shift(-1)
            
            # Align features and target
            combined = pd.concat([features, target.rename('target')], axis=1)
            combined = combined.dropna(subset=['target'])
            combined = combined.dropna(subset=features.columns)
            
            if combined.empty:
                logger.error("No data available for training after NA handling.")
                raise ValueError("No data available for training after NA handling.")
            
            X = combined[features.columns]
            y = combined['target']
            self.feature_columns = X.columns
            
            if len(X) < 10:
                # Fallback to simple model fitting if too few samples
                logger.warning(f"Very few samples ({len(X)}) available for training.")
                self.pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                                    ('model', LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    num_leaves=15,
                    min_child_samples=3,
                    colsample_bytree=0.8,
                    subsample=0.8,
                    random_state=42,
                    verbose=-1
                ))
                ])
                self.pipeline.fit(X, y)
                
                # Set default values for compatibility
                self.best_score_ = None
                self.best_params_ = {'model__n_estimators': 100, 'model__learning_rate': 0.05, 'model__max_depth': 3}
                self.cv_results_ = {'params': [self.best_params_], 'mean_test_score': [0], 'std_test_score': [0]}
                
                logger.info("Fitted with default parameters due to insufficient data for CV.")
                return self
            
            # Create pipeline
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Standardize features
                ('model', LGBMRegressor(
                    num_leaves=15,
                    min_child_samples=3,
                    colsample_bytree=0.8,
                    subsample=0.8,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1
                ))
            ])
            
            # Configure TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Define parameter grid
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 4, 5, 7]
            }
            
            # Setup GridSearchCV
            self._grid_search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=tscv,
                verbose=1,
                n_jobs=-1
            )
            
            # Fit GridSearchCV
            logger.info(f"Starting GridSearchCV with {param_grid} across {tscv.get_n_splits(X)} splits...")
            self._grid_search.fit(X, y)
            
            # Store results
            self.pipeline = self._grid_search.best_estimator_
            self.cv_results_ = self._grid_search.cv_results_
            self.best_score_ = self._grid_search.best_score_
            self.best_params_ = self._grid_search.best_params_
            
            logger.info(f"GridSearchCV completed. Best parameters found: {self.best_params_}")
            logger.info(f"Best score (neg_mean_squared_error): {self.best_score_}")
            logger.info("Model fitting completed.")
            
            return self
            
        except ValueError as ve:
            logger.error(f"ValueError during model fitting: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise
            
    def predict(self, df, forecast_days=7):
        """Generate forecasts with sanity checks"""
        try:
            if self.pipeline is None:
                raise ValueError("Model not fitted. Call fit() first.")
                
            logger.info(f"Generating {forecast_days}-day forecast...")
            
            # Make recursive predictions
            predictions = []
            last_rate = df['rate'].iloc[-1]
            current_data_for_prediction = df.copy()
            
            # Minimum rows needed for feature creation
            min_rows_for_features = self.volatility_lookback + 5
            
            for i in range(forecast_days):
                # Prepare data slice for features
                if len(current_data_for_prediction) > min_rows_for_features:
                    data_slice_for_features = current_data_for_prediction.iloc[-min_rows_for_features:]
                else:
                    data_slice_for_features = current_data_for_prediction
                
                # Get features for the last available data point
                next_day_features_df = self._create_features(data_slice_for_features)
                
                # Ensure features are valid
                if next_day_features_df.empty or not all(col in next_day_features_df.columns for col in self.feature_columns):
                    logger.error("Feature creation for prediction loop resulted in empty or incomplete DataFrame.")
                    logger.warning(f"Predicting no change (0.0% change) for forecast step {i+1} due to feature creation error.")
                    pct_change = 0.0
                else:
                    # Select features for the last row
                    next_day_feature_values = next_day_features_df[self.feature_columns].iloc[-1:]
                    pct_change = self.pipeline.predict(next_day_feature_values)[0]
                
                # Sanity check - limit extreme predictions
                pct_change = np.clip(pct_change, -0.1, 0.1)  # Max 10% change per day
                
                # Convert to next day's rate
                next_rate = last_rate * (1 + pct_change)
                predictions.append(next_rate)
                
                # Update for next iteration
                new_idx = current_data_for_prediction.index[-1] + pd.Timedelta(days=1)
                current_data_for_prediction.loc[new_idx, 'rate'] = next_rate
                last_rate = next_rate
            
            logger.info("Forecast generated successfully")
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def evaluate(self, df, test_size=0.2):
        """Evaluate model performance"""
        try:
            if self.pipeline is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            logger.info(f"Evaluating model on {test_size*100:.0f}% hold-out test set...")
            
            # Create features
            features = self._create_features(df)
            target = df['rate'].pct_change().shift(-1)
            
            # Align features and target
            combined = pd.concat([features, target.rename('target')], axis=1)
            combined = combined.dropna()
            
            X = combined[self.feature_columns]
            y = combined['target']
            
            # Split into train and test
            split_index = int(len(X) * (1 - test_size))
            X_train = X[:split_index]
            X_test = X[split_index:]
            y_test = y[split_index:]
            
            if len(X_test) < 1:
                logger.warning("Test set too small for evaluation.")
                return None
            
            # Make predictions on test set
            y_pred = self.pipeline.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # MAPE calculation
            mape_mask = y_test != 0
            if mape_mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mape_mask] - y_pred[mape_mask]) / y_test[mape_mask])) * 100
            else:
                mape = np.nan
            
            # Directional Accuracy
            directional_accuracy = ((y_test * y_pred) > 0).mean() * 100
            
            # No change accuracy
            threshold = 0.0001
            no_change_mask = np.abs(y_test) < threshold
            if no_change_mask.sum() > 0:
                no_change_accuracy = (np.abs(y_pred[no_change_mask]) < threshold).mean() * 100
            else:
                no_change_accuracy = np.nan
            
            evaluation_results = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'no_change_accuracy': no_change_accuracy,
                'test_size': len(X_test),
                'train_size': len(X_train),
                'y_test': y_test,
                'y_pred': y_pred,
                'test_dates': X_test.index
            }
            
            logger.info(f"Evaluation completed. MAE: {mae:.6f}, RMSE: {rmse:.6f}, "
                        f"MAPE: {mape:.2f}%, Directional Accuracy: {directional_accuracy:.2f}%")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise