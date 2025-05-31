import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import logging
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
    """Format numbers for display, handling large, small, and typical values."""
    if pd.isna(num):
        return "N/A"
    
    abs_num = abs(num)
    if abs_num == 0: # Handle zero explicitly
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

                # Train model and generate forecast
                forecaster = CurrencyForecaster()
                forecaster.fit(historical_data)
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
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Tomorrow",
                        f"{format_value(forecast_to_display[0])}",
                        f"{pct_changes[0]:.2f}%"
                    )
                with col2:
                    st.metric(
                        "3 Days",
                        f"{format_value(forecast_to_display[min(2, len(forecast_to_display)-1)])}",
                        f"{pct_changes[min(2, len(forecast_to_display)-1)]:.2f}%"
                    )
                with col3:
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
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Historical Statistics (original scale):")
                        st.write(f"- Average Rate: {format_value(historical_data['rate'].mean())}")
                        st.write(f"- Volatility: {historical_data['rate'].pct_change().std()*100:.2f}%")
                        
                    with col2:
                        st.write("Forecast Statistics (original scale):")
                        st.write(f"- Average Forecast: {format_value(forecast.mean())}")
                        st.write(f"- Forecast Range: {format_value(forecast.min())} - {format_value(forecast.max())}")
                
                # Add warning for volatile pairs
                if historical_data['rate'].pct_change().std() > 0.02:  # More than 2% daily volatility
                    st.warning("This currency pair shows high volatility. Forecasts may be less reliable.", icon="⚠️")
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please try again with different parameters.", icon="❌")
        
if __name__ == "__main__":
    main()