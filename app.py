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

def format_large_number(num):
    """Format large numbers for display"""
    if abs(num) >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

@st.cache_data(ttl=3600)
def get_currency_list():
    """Fetch and cache currency list"""
    try:
        # Fetch currency list and names
        fetcher = DataFetcher()
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
    
    # Calculate simple volatility
    volatility = historical_data['rate'].pct_change().std()
    
    # Forecast with confidence intervals
    forecast_std = volatility * np.sqrt(np.arange(1, len(forecast_data) + 1))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['forecast'] + 2*forecast_std,
        name='Upper Bound',
        line=dict(color='maroon', width=1, dash='dash'),
        opacity=0.3
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['forecast'] - 2*forecast_std,
        name='Lower Bound',
        line=dict(color='maroon', width=1, dash='dash'),
        fill='tonexty',
        opacity=0.3
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['forecast'],
        name='Forecast',
        line=dict(color='red', width=2)
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
    
    st.title('📈 Currency Exchange Rate Forecaster')
    
    try:
        # Get currency list
        currencies = get_currency_list()
        if not currencies:
            st.error("Failed to fetch currency list. Please try again later.")
            return
            
        # Currency selection in main panel
        col1, col2 = st.columns(2)
        with col1:
            base_currency = st.selectbox(
                'Base Currency',
                options=sorted(currencies.keys()),
                format_func=lambda x: currencies[x],
                index=list(currencies.keys()).index('usd') if 'usd' in currencies else 0
            ).lower()
            
        with col2:
            target_currency = st.selectbox(
                'Target Currency',
                options=[c for c in sorted(currencies.keys()) if c != base_currency],
                format_func=lambda x: currencies[x],
                index=0
            ).lower()
        
        # Settings sidebar
        st.sidebar.title('⚙️ Settings')
        
        # Option to use yesterday's data (useful around midnight)
        use_yesterday_data = st.sidebar.checkbox(
            'Use yesterday\'s data', 
            value=False,
            help='Use this if API hasn\'t updated yet (e.g., around midnight)'
        )
        
        # Historical data range with text input and warning
        days = st.sidebar.number_input(
            'Historical Data Range (days)',
            min_value=14,
            max_value=1000,
            value=30,
            help='Enter the number of days of historical data to analyze'
        )
        
        if days > 150:
            st.sidebar.warning('⚠️ Using more than 90 days of historical data may impact performance')
        
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
                historical_rates = []
                dates = []
                
                progress_bar = st.progress(0)
                for i in range(days):
                    progress = (i + 1) / days
                    progress_bar.progress(progress)
                    
                    # Determine which date to fetch
                    if i == 0 and use_yesterday_data:
                        # If using yesterday's data for the most recent point
                        date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
                    else:
                        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                    
                    data = fetcher.fetch_rates(base_currency, date)
                    if data and base_currency in data:
                        historical_rates.append(data[base_currency].get(target_currency))
                        dates.append(date)
                
                if not historical_rates:
                    st.error("❌ Failed to fetch historical data. Please try again.")
                    return
                    
                # Create DataFrame
                historical_data = pd.DataFrame(
                    {'rate': historical_rates},
                    index=pd.to_datetime(dates)
                ).sort_index()
                
                # Check if automatic fallback occurred
                latest_date = historical_data.index.max()
                today = pd.Timestamp(datetime.now().date())
                yesterday = today - pd.Timedelta(days=1)
                
                if not use_yesterday_data and latest_date.date() < today.date() and latest_date.date() == yesterday.date():
                    st.info("ℹ️ The latest data wasn't available, so yesterday's data was used automatically. The API typically updates once daily.")
                
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
                st.metric(
                    f"Current Rate (1 {base_currency.upper()})",
                    f"{format_large_number(current_rate)} {target_currency.upper()}"
                )
                
                # Calculate percentage changes
                pct_changes = (forecast - current_rate) / current_rate * 100
                
                # Display forecasts with context
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Tomorrow",
                        f"{format_large_number(forecast[0])}",
                        f"{pct_changes[0]:.2f}%"
                    )
                with col2:
                    st.metric(
                        "3 Days",
                        f"{format_large_number(forecast[min(2, len(forecast)-1)])}",
                        f"{pct_changes[min(2, len(forecast)-1)]:.2f}%"
                    )
                with col3:
                    st.metric(
                        "End of Forecast",
                        f"{format_large_number(forecast[-1])}",
                        f"{pct_changes[-1]:.2f}%"
                    )
                
                # Plot
                st.subheader('📊 Forecast Visualization')
                fig = create_forecast_plot(
                    historical_data,
                    forecast_data,
                    base_currency,
                    target_currency
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add summary statistics
                with st.expander("📊 Additional Statistics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Historical Statistics:")
                        st.write(f"- Average Rate: {format_large_number(historical_data['rate'].mean())}")
                        st.write(f"- Volatility: {historical_data['rate'].pct_change().std()*100:.2f}%")
                        
                    with col2:
                        st.write("Forecast Statistics:")
                        st.write(f"- Average Forecast: {format_large_number(forecast.mean())}")
                        st.write(f"- Forecast Range: {format_large_number(forecast.min())} - {format_large_number(forecast.max())}")
                
                # Add warning for volatile pairs
                if historical_data['rate'].pct_change().std() > 0.02:  # More than 2% daily volatility
                    st.warning("⚠️ This currency pair shows high volatility. Forecasts may be less reliable.")
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please try again with different parameters or contact support if the issue persists.")
        
if __name__ == "__main__":
    main()