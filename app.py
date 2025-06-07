import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from data_fetcher import DataFetcher
from model import CurrencyForecaster
import numpy as np
import requests

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

# Initialize session state
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

def format_value(num):
    """Format numbers for display"""
    if pd.isna(num):
        return "N/A"
    
    abs_num = abs(num)
    if abs_num == 0:
        return "0.00"
        
    if abs_num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"{num/1e3:.2f}K"
    elif abs_num >= 0.01: # Show 2 to 4 decimal places depending on magnitude if it's not an integer
        if abs_num >= 1:
            return f"{num:.2f}"
        else: # 0.50, 0.0123
            return f"{num:.4f}"
    elif abs_num < 1e-9 and abs_num > 0: # Extremely small, switch to scientific to avoid too many zeros
        return f"{num:.2e}"
    else: # For very small numbers like 0.00006, this will show up to 8 decimal places for small numbers
        return f"{num:.8f}".rstrip('0').rstrip('.')

@st.cache_data(ttl=3600)
def get_currency_list():
    """Fetch and cache currency list"""
    try:
        # Fetch currency list and names
        fetcher = DataFetcher()

        # CDN URL
        response = requests.get("https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies.min.json")
        if response.status_code == 200:
            currencies = response.json()
            # Proper formatting
            formatted_currencies = {
                code: f"{code.upper()} - {name.title()}"
                for code, name in currencies.items()
            }
            return formatted_currencies
            
        # Fallback URL
        response = requests.get("https://latest.currency-api.pages.dev/v1/currencies.min.json")
        if response.status_code == 200:
            currencies = response.json()
            formatted_currencies = {
                code: f"{code.upper()} - {name.title()}"
                for code, name in currencies.items()
            }
            return formatted_currencies
            
        logger.error(f"Failed to fetch currency list. Status code: {response.status_code}")
        return {}
    
    except Exception as e:
        logger.error(f"Error fetching currency list: {str(e)}")
        return {}

def create_forecast_plot(historical_data, forecast_data, base_currency, target_currency):
    """Create forecast visualization"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['rate'],
        name='Historical',
        line=dict(color='aqua', width=2)
    ))
    
    # Calculate historical daily percentage change volatility, represents the typical one-day percentage fluctuation.
    daily_pct_change_volatility = historical_data['rate'].pct_change().std()
    
    # If daily_pct_change_volatility is NaN (if historical data is too short or constant),
    # use a small default to avoid errors.
    if pd.isna(daily_pct_change_volatility):
        daily_pct_change_volatility = 0.001 # Default to 0.1% volatility
        logger.warning("Could not calculate historical volatility for CI, using a small default.")

    # Project this daily volatility into the future.
    # The standard deviation of a sum of N independent random variables scales with sqrt(N). Apply this to the percentage change.
    # np.arange starts from 1 up to len(forecast_data).
    forecast_horizon_steps = np.arange(1, len(forecast_data) + 1)
    projected_pct_std_dev = daily_pct_change_volatility * np.sqrt(forecast_horizon_steps)
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['forecast'],
        name='Forecast',
        line=dict(color='red', width=2)
    ))
    
    # Calculate Upper and Lower bounds based on the forecast and projected percentage standard deviation
    # This means the absolute size of the interval grows with the forecast value.
    upper_bound = forecast_data['forecast'] * (1 + 2 * projected_pct_std_dev)
    lower_bound = forecast_data['forecast'] * (1 - 2 * projected_pct_std_dev)
    
    # Ensure lower bound doesn't go below zero for exchange rates
    lower_bound = np.maximum(lower_bound, 0.00001) 

    # Add confidence intervals (shaded area)
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=upper_bound,
        name='Upper Bound (95% CI)',
        line=dict(color='maroon', width=1, dash='dash'),
        showlegend=False,
        opacity=0.3
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=lower_bound,
        name='Lower Bound (95% CI)',
        line=dict(color='maroon', width=1, dash='dash'),
        fill='tonexty', # Fill the area between this trace and the previous one (upper_bound)
        showlegend=False,
        opacity=0.3
    ))
    
    # Layout
    fig.update_layout(
        title=f'{base_currency.upper()}/{target_currency.upper()} Exchange Rate Forecast',
        xaxis_title='Date',
        yaxis_title='Exchange Rate',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def main():
    st.set_page_config(page_title="Currency Forecaster", page_icon="📈", layout="wide")
    
    st.title('📈 Exchange Rate Forecaster using LightGBM')
    
    try:
        # Get currency list
        currencies = get_currency_list()
        if not currencies:
            st.error("Failed to fetch currency list.")
            return
            
        # Currency selection in main panel
        col1, col2 = st.columns(2)
        with col1:
            base_currency = st.selectbox(
                'Base Currency',
                options=sorted(currencies.keys()),
                format_func=lambda x: currencies[x],
                index=list(currencies.keys()).index('usd') if 'usd' in currencies else 0 # default to USD
            ).lower()
            
        with col2:
            target_currency = st.selectbox(
                'Target Currency',
                options=[c for c in sorted(currencies.keys()) if c != base_currency],
                format_func=lambda x: currencies[x],
                index=list(currencies.keys()).index('idr') if 'idr' in currencies else 1 # default to IDR
            ).lower()
        
        # Settings sidebar
        st.sidebar.title('⚙️ Settings')
        
        # Option to use yesterday's data (to make the app directly uses yesterday's data and won't even try to use today's data)
        use_yesterday_data = st.sidebar.checkbox(
            'Use yesterday\'s data', 
            value=False,
            help='Use this if API hasn\'t updated yet'
        )
        
        # Historical data range
        days = st.sidebar.number_input(
            'Historical Data Range (days)',
            min_value=14,
            max_value=1000,
            value=30,
            help='Enter the number of days of historical data to analyze'
        )       
        
        # Forecast horizon
        forecast_days = st.sidebar.slider(
            'Forecast Horizon (days)',
            min_value=1,
            max_value=14,
            value=7,
            help='Select the number of days to forecast'
        )
            
        if st.button('Generate Forecast', key='forecast_button'):
            with st.spinner('Fetching data and generating forecast...'):
                # Fetch historical data
                fetcher = DataFetcher()
                historical_rates_collected = []
                dates_collected = []
                
                progress_bar = st.progress(0)

                # Determine the end date for the historical period, the most recent date
                end_date_dt = datetime.now()
                if use_yesterday_data:
                    end_date_dt = datetime.now() - timedelta(days=1)

                for i in range(days):
                    progress = (i + 1) / days
                    progress_bar.progress(progress)
                    
                    # Calculate the actual date to fetch, going backwards from end_date_dt
                    current_fetch_date_dt = end_date_dt - timedelta(days=i)
                    date_str_to_fetch = current_fetch_date_dt.strftime('%Y-%m-%d')

                    # Fetch data for the determined date
                    # data_fetcher.fetch_rates can return None if data is not available for the specific date
                    api_response_data = fetcher.fetch_rates(base_currency, date_str_to_fetch)

                    if api_response_data and base_currency in api_response_data and \
                        isinstance(api_response_data[base_currency], dict) and \
                        target_currency in api_response_data[base_currency]:
                        rate = api_response_data[base_currency].get(target_currency)
                        if rate is not None: # Ensure rate itself is not None
                            historical_rates_collected.append(rate)
                            dates_collected.append(date_str_to_fetch) # Store the date string for which data was found
                        else:
                            logger.warning(f"Rate for {target_currency} was None on {date_str_to_fetch} for base {base_currency}.")
                    else:
                        logger.warning(f"Could not fetch or parse data for {base_currency} to {target_currency} on {date_str_to_fetch}.")
                
                progress_bar.empty() # Clear progress bar

                if not historical_rates_collected:
                    st.error("Failed to fetch sufficient historical data. Please check logs or try different parameters.", icon="❌")
                    return
                    
                # Create DataFrame from successfully fetched rates and their corresponding dates and ensure data is sorted by date
                historical_data = pd.DataFrame(
                    {'rate': historical_rates_collected},
                    index=pd.to_datetime(dates_collected)
                ).sort_index()

                if historical_data.empty:
                    st.error("Historical data is empty after filtering. Please try again.", icon="❌")
                    return
                
                # Check if automatic fallback occurred (or if data is just older than 'today')
                latest_date_in_data = historical_data.index.max()
                today_dt = pd.Timestamp(datetime.now().date())
                yesterday_dt = today_dt - pd.Timedelta(days=1)
                
                # Print info reflecting the actual latest data point obtained
                if not use_yesterday_data and latest_date_in_data.date() < today_dt.date():
                    if latest_date_in_data.date() == yesterday_dt.date():
                        st.info(f"The most recent data point obtained is from {latest_date_in_data.strftime('%Y-%m-%d')}, as today's data might not be available yet. The API updates once daily.", icon="ℹ️")
                    else:
                        st.warning(f"The most recent data obtained is from {latest_date_in_data.strftime('%Y-%m-%d')}, which is older than yesterday. Check data source or API status.", icon="⚠️")
                elif use_yesterday_data and latest_date_in_data.date() < yesterday_dt.date():
                    st.warning(f"Expected yesterday's data, but the most recent obtained is from {latest_date_in_data.strftime('%Y-%m-%d')}. Check data source or API status.", icon="⚠️")

                # ========== EXPLORATORY DATA ANALYSIS (EDA) ==========
                st.subheader('📊 Exploratory Data Analysis')
                
                # Calculate percentage changes for analysis
                historical_data['pct_change'] = historical_data['rate'].pct_change()
                
                # Create tabs for different EDA sections
                eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Distribution Analysis", "Descriptive Statistics", "Volatility Analysis"])
                
                with eda_tab1:
                    st.markdown("### Distribution Analysis")
                    
                    # Two columns for histograms
                    hist_col1, hist_col2 = st.columns(2)
                    
                    with hist_col1:
                        # Histogram of exchange rates
                        fig_hist_rates = go.Figure()
                        fig_hist_rates.add_trace(go.Histogram(
                            x=historical_data['rate'],
                            nbinsx=30, # Number of bins for the histogram
                            name='Exchange Rates',
                            marker_color='aqua'
                        ))
                        fig_hist_rates.update_layout(
                            title=f'{base_currency.upper()}/{target_currency.upper()} Exchange Rate Distribution',
                            xaxis_title='Exchange Rate',
                            yaxis_title='Frequency',
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_hist_rates, use_container_width=True)
                        
                        # Interpretation
                        st.caption("""
                        This shows how often different exchange rate values occurred. 
                        A narrow distribution suggests stable rates, while a wide distribution indicates high variability.
                        """)
                    
                    with hist_col2:
                        # Histogram of percentage changes
                        fig_hist_pct = go.Figure()
                        fig_hist_pct.add_trace(go.Histogram(
                            x=historical_data['pct_change'].dropna(),
                            nbinsx=30,
                            name='Daily % Changes',
                            marker_color='red'
                        ))
                        fig_hist_pct.update_layout(
                            title='Daily Percentage Change Distribution',
                            xaxis_title='Daily % Change',
                            yaxis_title='Frequency',
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_hist_pct, use_container_width=True)
                        
                        # Interpretation
                        st.caption("""
                        This shows the distribution of daily percentage changes. 
                        A bell-shaped curve centered at 0 suggests random walk behavior. 
                        Fat tails indicate occasional large movements (high risk).
                        """)
                
                with eda_tab2:
                    st.markdown("### Descriptive Statistics")
                    
                    # Calculate comprehensive statistics
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        st.markdown("**Exchange Rate Statistics**")
                        rate_stats = {
                            'Count': len(historical_data),
                            'Mean': historical_data['rate'].mean(),
                            'Std Dev': historical_data['rate'].std(),
                            'Min': historical_data['rate'].min(),
                            '25%': historical_data['rate'].quantile(0.25),
                            'Median': historical_data['rate'].quantile(0.50),
                            '75%': historical_data['rate'].quantile(0.75),
                            'Max': historical_data['rate'].max(),
                            'Skewness': historical_data['rate'].skew(),
                            'Kurtosis': historical_data['rate'].kurtosis()
                        }
                        
                        # Format the statistics for display
                        stats_df = pd.DataFrame(rate_stats.items(), columns=['Metric', 'Value'])
                        st.dataframe(stats_df, hide_index=True, use_container_width=True)
                    
                    with stats_col2:
                        st.markdown("**Daily % Change Statistics**")
                        pct_stats = {
                            'Count': len(historical_data['pct_change'].dropna()),
                            'Mean (%)': historical_data['pct_change'].mean() * 100,
                            'Std Dev (%)': historical_data['pct_change'].std() * 100,
                            'Min (%)': historical_data['pct_change'].min() * 100,
                            '25% (%)': historical_data['pct_change'].quantile(0.25) * 100,
                            'Median (%)': historical_data['pct_change'].quantile(0.50) * 100,
                            '75% (%)': historical_data['pct_change'].quantile(0.75) * 100,
                            'Max (%)': historical_data['pct_change'].max() * 100,
                            'Skewness': historical_data['pct_change'].skew(),
                            'Kurtosis': historical_data['pct_change'].kurtosis()
                        }
                        
                        pct_stats_df = pd.DataFrame(pct_stats.items(), columns=['Metric', 'Value'])
                        st.dataframe(pct_stats_df, hide_index=True, use_container_width=True)
                
                with eda_tab3:
                    st.markdown("### Volatility Analysis")
                    
                    # Calculate rolling volatility (same window as used in model)
                    volatility_window = 30  # Same as model's volatility_lookback
                    historical_data['rolling_volatility'] = historical_data['pct_change'].rolling(
                        window=volatility_window, min_periods=1
                    ).std()

                    fig_volatility = go.Figure()

                    fig_volatility.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data['rolling_volatility'] * 100,
                        name=f'{volatility_window}-Day Rolling Volatility',
                        line=dict(color='purple', width=2)
                    ))
                    
                    # Add average volatility line
                    avg_volatility = historical_data['rolling_volatility'].mean() * 100
                    fig_volatility.add_hline(
                        y=avg_volatility, 
                        line_dash="dash", 
                        line_color="gray",
                        annotation_text=f"Average: {avg_volatility:.2f}%"
                    )
                    
                    fig_volatility.update_layout(
                        title=f'{volatility_window}-Day Rolling Volatility',
                        xaxis_title='Date',
                        yaxis_title='Volatility (%)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig_volatility, use_container_width=True)
                    
                    # Current volatility metric
                    current_volatility = historical_data['rolling_volatility'].iloc[-1] * 100
                    volatility_percentile = (historical_data['rolling_volatility'] <= historical_data['rolling_volatility'].iloc[-1]).mean() * 100
                    # creates boolean array of all values less than or equal to the current volatility
                    
                    vol_col1, vol_col2, vol_col3 = st.columns(3)
                    with vol_col1:
                        st.metric(
                            "Current Volatility",
                            f"{current_volatility:.2f}%",
                            f"{current_volatility - avg_volatility:.2f}% vs avg"
                        )
                    with vol_col2:
                        st.metric(
                            "Average Volatility",
                            f"{avg_volatility:.2f}%"
                        )
                    with vol_col3:
                        st.metric(
                            "Volatility Percentile",
                            f"{volatility_percentile:.2f}%",
                            help="Current volatility is higher than this % of historical values"
                        )
                
                # ========== MODEL TRAINING & FORECAST ==========
                # Clean up temporary column
                historical_data = historical_data.drop(columns=['pct_change', 'rolling_volatility'])
                
                # Train model and generate forecast
                forecaster = CurrencyForecaster()
                forecaster.fit(historical_data)
                
                # Generate forecast
                forecast = forecaster.predict(historical_data, forecast_days=forecast_days)
                
                # Create forecast DataFrame
                forecast_dates = pd.date_range(
                    start=historical_data.index[-1] + pd.Timedelta(days=1),
                    periods=forecast_days
                )
                forecast_data = pd.DataFrame(
                    {'forecast': forecast},
                    index=forecast_dates
                )
                
                # Display current rate and forecasts
                st.subheader('📈 Exchange Rate Analysis')
                
                current_rate = historical_data['rate'].iloc[-1]
                display_threshold = 0.01 # Threshold to decide if inversion is needed
                display_inverted = current_rate < display_threshold and current_rate != 0 # Avoid division by zero if rate is exactly 0

                display_base = base_currency
                display_target = target_currency
                rate_to_display = current_rate
                forecast_to_display = forecast.copy() # Make a copy to potentially invert

                if display_inverted:
                    display_base = target_currency # Flip for display
                    display_target = base_currency   # Flip for display
                    
                    if current_rate != 0:
                        rate_to_display = 1 / current_rate
                    else:
                        rate_to_display = np.inf # Or handle as error/unavailable
                    
                    # Invert forecast values, handle zeros in forecast if any
                    forecast_to_display = np.array([1/f if f != 0 else np.inf for f in forecast])
                    
                    st.info(f"Displaying rates as 1 {display_base.upper()} = X {display_target.upper()} for better readability.", icon="ℹ️")

                st.metric(
                    f"Current Rate (1 {display_base.upper()})",
                    f"{format_value(rate_to_display)} {display_target.upper()}"
                )
                
                # Calculate percentage changes based on the original forecast direction
                # The meaning of pct_change remains the same.
                pct_changes = (forecast - current_rate) / current_rate * 100
                
                # Display forecasts with context
                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric(
                        "Tomorrow",
                        f"{format_value(forecast_to_display[0])}",
                        f"{pct_changes[0]:.2f}%"
                    )
                with result_col2:
                    st.metric(
                        "3 Days",
                        f"{format_value(forecast_to_display[min(2, len(forecast_to_display)-1)])}",
                        f"{pct_changes[min(2, len(forecast_to_display)-1)]:.2f}%"
                    )
                with result_col3:
                    st.metric(
                        "End of Forecast",
                        f"{format_value(forecast_to_display[-1])}",
                        f"{pct_changes[-1]:.2f}%"
                    )
                
                # Plot
                st.subheader('📊 Forecast Visualization')

                fig = create_forecast_plot(
                    historical_data, # Original orientation
                    forecast_data,   # Original orientation
                    base_currency,   # Original base for title
                    target_currency  # Original target for title
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add summary statistics
                with st.expander("📊 Additional Statistics"):
                    summary_col1, summary_col2 = st.columns(2)
                    with summary_col1:
                        st.write("Historical Statistics (original scale):")
                        st.write(f"- Average Rate: {format_value(historical_data['rate'].mean())}")
                        st.write(f"- Volatility: {historical_data['rate'].pct_change().std()*100:.2f}%")
                        
                    with summary_col2:
                        st.write("Forecast Statistics (original scale):")
                        st.write(f"- Average Forecast: {format_value(forecast.mean())}")
                        st.write(f"- Forecast Range: {format_value(forecast.min())} - {format_value(forecast.max())}")
            
                # Add warning for volatile pairs
                if historical_data['rate'].pct_change().std() > 0.02:  # More than 2% daily volatility
                    st.warning("This currency pair shows high volatility. Forecasts may be less reliable.", icon="⚠️")

                # ========== MODEL EVALUATION ==========
                st.subheader('🎯 Model Evaluation')
                
                evaluation_results = forecaster.evaluate(historical_data, test_size=0.2)
                
                if evaluation_results:
                    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["Cross-Validation Results", "Test Set Metrics", "Prediction vs Actual"])
                    
                    with eval_tab1:
                        st.markdown("### Cross-Validation Results")

                        # Best CV Scores and params
                        st.metric(
                            "Best CV Score (Neg MSE)",
                            f"{forecaster.best_score_:.6f}",
                            help="Negative Mean Squared Error on percentage changes"
                        )
                        st.markdown("**Best Parameters:**")
                        for param, value in forecaster.best_params_.items():
                            st.write(f"- {param}: {value}")


                        cv_scores = []
                        for i in range(len(forecaster.cv_results_['params'])):
                            cv_scores.append({
                                'Parameters': str(forecaster.cv_results_['params'][i]),
                                'Mean CV Score': forecaster.cv_results_['mean_test_score'][i],
                                'Std CV Score': forecaster.cv_results_['std_test_score'][i]
                            })
                        cv_df = pd.DataFrame(cv_scores)
                        cv_df = cv_df.sort_values('Mean CV Score', ascending=False)
                        st.markdown("**Top 5 CV Results:**")
                        st.dataframe(
                            cv_df.head(), 
                            hide_index=True, 
                            use_container_width=True,     
                            column_config={
                                "Mean CV Score": st.column_config.NumberColumn(
                                    format="scientific"
                                ),
                                "Std CV Score": st.column_config.NumberColumn(
                                    format="scientific"
                                )
                            }
                        )
                    
                    with eval_tab2:
                        st.markdown("### Hold-Out Test Set Metrics")
                        st.info(f"""
                        The model was evaluated on the last {evaluation_results['test_size']} days ({evaluation_results['test_size']/len(historical_data)*100:.1f}%) 
                        of data that were not used during training.
                        """)

                        row1_col1, row1_col2, row1_col3 = st.columns(3)
                        row2_col1, row2_col2, row2_col3 = st.columns(3)
                        
                        with row1_col1:
                            st.metric(
                                "MAE",
                                f"{evaluation_results['mae']:.6f}",
                                help="Mean Absolute Error of percentage changes. Lower is better."
                            )
                        
                        with row1_col2:
                            st.metric(
                                "RMSE", 
                                f"{evaluation_results['rmse']:.6f}",
                                help="Root Mean Square Error of percentage changes."
                            )
                        
                        with row1_col3:
                            st.metric(
                                "Directional Accuracy (%)",
                                f"{evaluation_results['directional_accuracy']:.1f}%",
                                help="Percentage of times the model correctly predicted whether the rate would go up or down."
                            )
                        
                        with row2_col1:
                            st.metric(
                                "MAPE (%)",
                                f"{evaluation_results['mape']:.2f}%" if not pd.isna(evaluation_results['mape']) else "N/A",
                                help="Mean Absolute Percentage Error."
                            )

                        with row2_col2:
                            st.metric(
                                "sMAPE (%)",
                                f"{evaluation_results['smape']:.2f}%" if not pd.isna(evaluation_results['smape']) else "N/A",
                                help="Symmetric MAPE."
                            )
                        
                        with row2_col3:
                            st.metric(
                                "MASE",
                                f"{evaluation_results['mase']:.2f}" if not pd.isna(evaluation_results['mase']) else "N/A",
                                help="Mean Absolute Scaled Error. < 1 means the model is better than a naive forecast."
                            )

                        # Interpretation guide
                        st.markdown("""
                        **📊 Metrics Interpretation:**
                        - **MAE & RMSE**: Lower is better. These measure the average prediction error in percentage change units.
                        - **Directional Accuracy**: > 50% means the model predicts the direction (up/down) better than random guessing.
                        - **MAPE**: Shows the average percentage error.
                        - **sMAPE**: A percentage error metric that is more reliable than MAPE. Lower is better.
                        - **MASE**: Compares the model to a naive forecast. A value < 1 is good, indicating the model is outperforming the baseline.
                        """)
                    
                    with eval_tab3:
                        st.markdown("### Prediction vs Actual on Test Set")
                        
                        test_dates = evaluation_results['test_dates']
                        y_test = evaluation_results['y_test'] * 100
                        y_pred = evaluation_results['y_pred'] * 100
                        
                        fig_comparison = go.Figure()
                        
                        fig_comparison.add_trace(go.Scatter(
                            x=test_dates,
                            y=y_test,
                            name='Actual %',
                            line=dict(color='blue', width=2),
                            mode='lines+markers'
                        ))
                        
                        fig_comparison.add_trace(go.Scatter(
                            x=test_dates,
                            y=y_pred,
                            name='Predicted %',
                            line=dict(color='red', width=2, dash='dash'),
                            mode='lines+markers'
                        ))
                        
                        # Add zero line
                        fig_comparison.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        
                        fig_comparison.update_layout(
                            title='Predicted vs Actual Daily Percentage Changes (Test Set)',
                            xaxis_title='Date',
                            yaxis_title='Daily % Change',
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Residual analysis
                        residuals = y_test - y_pred
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # Residual distribution
                            fig_residuals = go.Figure()
                            fig_residuals.add_trace(go.Histogram(
                                x=residuals,
                                nbinsx=20,
                                name='Prediction Errors',
                                marker_color='purple'
                            ))
                            fig_residuals.update_layout(
                                title='Distribution of Prediction Errors',
                                xaxis_title='Error (Actual - Predicted %)',
                                yaxis_title='Frequency',
                                height=300
                            )
                            st.plotly_chart(fig_residuals, use_container_width=True)
                        
                        with col2:
                            # Error statistics
                            st.markdown("**Error Analysis:**")
                            error_stats = {
                                'Mean Error': f"{residuals.mean():.4f}%",
                                'Std Error': f"{residuals.std():.4f}%", 
                                'Max Overestimate': f"{residuals.min():.4f}%",
                                'Max Underestimate': f"{residuals.max():.4f}%"
                            }
                            for metric, value in error_stats.items():
                                st.write(f"- {metric}: {value}")
                            
                            if abs(residuals.mean()) < 0.01:
                                st.success("✅ Model shows no significant bias (mean error near zero)")
                            else:
                                direction = "overestimates" if residuals.mean() < 0 else "underestimates"
                                st.warning(f"⚠️ Model slightly {direction} on average")
                
                else:
                    st.warning("Unable to perform evaluation - insufficient data for test set.")
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please try again with different parameters.", icon="❌")
        
if __name__ == "__main__":
    main()