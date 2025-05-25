import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

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
                
            # Add volatility features using the class attribute for lookback period
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
        """Fit model with walk-forward validation and hyperparameter tuning."""
        try:
            logger.info("Starting model fitting with walk-forward validation and hyperparameter tuning...")
            
            # Create features
            features = self._create_features(df)
            
            # Prepare target (next day's percentage change)
            # Ensure target aligns with features after potential NA rows from feature creation are handled.
            target = df['rate'].pct_change().shift(-1)

            # Align features and target: drop rows where target is NaN (last row)
            # Also, features might have NaNs at the beginning due to rolling windows
            # df.index should be the same for features and target before this alignment step.
            
            # Concatenate features and target to handle NA dropping consistently
            combined = pd.concat([features, target.rename('target')], axis=1)
            combined = combined.dropna(subset=['target']) # Drop rows where target is NaN (last row of original data)
            # Further drop rows if features resulted in NaNs at the beginning
            # (though ffill().bfill() in _create_features should have handled most internal NaNs in features)
            combined = combined.dropna(subset=features.columns)


            if combined.empty:
                logger.error("No data available for training after NA handling. Check input data and feature creation.")
                raise ValueError("No data available for training after NA handling.")

            X = combined[features.columns]
            y = combined['target']
            self.feature_columns = X.columns # Store feature columns after potential NA row drops

            if len(X) < 10: # Arbitrary small number, ensure enough data for CV
                logger.warning(f"Very few samples ({len(X)}) available for training. Model may not be reliable.")
                # Fallback to simple model fitting if too few samples for CV
                # This is similar to the old method but ensures it runs.
                if not X.empty:
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
                    self.model.fit(X,y)
                    logger.info("Fitted with default parameters due to insufficient data for CV.")
                    return self
                else:
                    raise ValueError("Training data is empty even for fallback.")


            # --- Old simple train-test split and model fitting (commented out) ---
            # features = features[:-1]
            # target = target[:-1]
            # train_size = int(len(features) * 0.8)
            # X_train = features[:train_size]
            # y_train = target[:train_size]
            # self.model = LGBMRegressor(
            #     n_estimators=100,
            #     learning_rate=0.05,
            #     max_depth=3,
            #     num_leaves=15,
            #     min_child_samples=3,
            #     colsample_bytree=0.8,
            #     subsample=0.8,
            #     random_state=42
            # )
            # self.model.fit(X_train, y_train)
            # --- End of old method ---

            # Configure TimeSeriesSplit
            # n_splits chosen to be 3 for faster execution in a project context. 5 is also common.
            tscv = TimeSeriesSplit(n_splits=3)

            # Define the parameter grid for GridSearchCV
            # Keeping the grid small for manageable computation time.
            param_grid = {
                'n_estimators': [50, 100],         # Number of boosting rounds
                'learning_rate': [0.05, 0.1],     # Step size shrinkage
                'max_depth': [3, 4],                # Max depth of individual trees
                # num_leaves is often 2^max_depth or slightly less. We'll fix it or use a small range.
                # For simplicity, other params from your original model are used as fixed values.
            }

            # Initialize LightGBM with other fixed parameters
            # Note: 'num_leaves' could also be in param_grid.
            # If max_depth is small, num_leaves is also constrained.
            # Default num_leaves for LGBM is 31. Let's use a common value.
            lgbm = LGBMRegressor(
                num_leaves=15, # Max leaves for base learners
                min_child_samples=3,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42,
                # Muting LightGBM verbosity during grid search; remove if you want to see it
                verbose=-1, 
                n_jobs=-1 # Use all available cores
            )

            # Setup GridSearchCV
            # Using 'neg_mean_squared_error' as scoring because GridSearchCV maximizes scores.
            grid_search = GridSearchCV(
                estimator=lgbm,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=tscv,
                verbose=1 # Set to 0 for less output, 1 or 2 for more
            )

            # Fit GridSearchCV
            logger.info(f"Starting GridSearchCV with {param_grid} across {tscv.get_n_splits(X)} splits...")
            grid_search.fit(X, y)

            # Store the best estimator
            self.model = grid_search.best_estimator_
            logger.info(f"GridSearchCV completed. Best parameters found: {grid_search.best_params_}")
            logger.info(f"Best score (neg_mean_squared_error): {grid_search.best_score_}")
            logger.info("Model fitting completed.")
            return self
            
        except ValueError as ve: # Catch specific errors we raise
            logger.error(f"ValueError during model fitting: {str(ve)}")
            raise
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
            # features = self._create_features(df) # Original features for the input df
            
            # Make recursive predictions
            predictions = []
            last_rate = df['rate'].iloc[-1]
            # last_data starts as a copy of the original historical data
            current_data_for_prediction = df.copy() 
            
            # Determine the minimum number of rows needed for _create_features to correctly calculate features for the last row.
            # This depends on the largest window size used in feature creation (self.volatility_lookback)
            # plus a small buffer for pct_change and safety.
            min_rows_for_features = self.volatility_lookback + 5 # Buffer of 5 for safety and pct_change

            for i in range(forecast_days):
                # Prepare the input data for _create_features for the next prediction step.
                # We only need the tail of current_data_for_prediction long enough to compute features for its last row.
                if len(current_data_for_prediction) > min_rows_for_features:
                    data_slice_for_features = current_data_for_prediction.iloc[-min_rows_for_features:]
                else:
                    data_slice_for_features = current_data_for_prediction
                
                # Get features for the last available data point in the slice
                next_day_features_df = self._create_features(data_slice_for_features)
                
                # Ensure next_day_features_df is not empty and contains the required feature columns
                if next_day_features_df.empty or not all(col in next_day_features_df.columns for col in self.feature_columns):
                    logger.error("Feature creation for prediction loop resulted in empty or incomplete DataFrame.")
                    # Fallback or error handling: e.g., append NaN or last prediction, or raise error
                    # For now, let's append NaN and log error, this might break plotting if not handled there.
                    predictions.append(np.nan) 
                    # To continue with the last known rate or a zero change prediction:
                    # next_rate = last_rate 
                    # predictions.append(next_rate)
                    # Or simply break / raise an error depending on desired robustness for this edge case.
                    logger.warning(f"Appending NaN for prediction step {i+1} due to feature creation error.")
                    # Update last_rate and current_data_for_prediction to allow loop to continue if possible
                    # This part depends on how one wants to handle such errors.
                    # For simplicity, if we append NaN, we can't easily compute the next 'last_rate'.
                    # Let's assume for now that if features fail, we might have to stop or use a naive forecast.
                    # A simple approach: if features fail, predict no change from last_rate.
                    pct_change = 0.0 
                else:
                    # Select the features for the very last row of the processed slice
                    next_day_feature_values = next_day_features_df[self.feature_columns].iloc[-1:]
                    pct_change = self.model.predict(next_day_feature_values)[0]
                
                # Sanity check - limit extreme predictions
                pct_change = np.clip(pct_change, -0.1, 0.1)  # Max 10% change per day
                
                # Convert to next day's rate
                next_rate = last_rate * (1 + pct_change)
                predictions.append(next_rate)
                
                # Update for next iteration
                new_idx = current_data_for_prediction.index[-1] + pd.Timedelta(days=1)
                # Use .loc to append the new row to current_data_for_prediction
                current_data_for_prediction.loc[new_idx, 'rate'] = next_rate 
                last_rate = next_rate
            
            logger.info("Forecast generated successfully")
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise