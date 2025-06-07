import json
import logging
from datetime import datetime, timedelta
import requests
from pathlib import Path
import pandas as pd
import time
from functools import wraps
import random

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

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if x == retries:
                        raise ValueError(f"Failed after {retries} attempts: {str(e)}")
                    wait = (backoff_in_seconds * 2 ** x + random.uniform(0, 1))  # Add jitter
                    logger.warning(f"Attempt {x + 1} failed: {str(e)}. Retrying in {wait:0.1f} seconds")
                    time.sleep(wait)
                    x += 1
        return wrapper
    return decorator

class DataFetcher:
    def __init__(self, cache_ttl=3600):
        """Initialize with cache ttl in seconds"""
        self.base_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date}/v1/{endpoint}"
        self.fallback_url = "https://{date}.currency-api.pages.dev/v1/{endpoint}"
        self.rates_dir = Path("rates")
        self.cache_ttl = cache_ttl
        
    def _ensure_directory(self, currency_code):
        """Ensure currency directory exists"""
        currency_dir = self.rates_dir / currency_code.lower()
        currency_dir.mkdir(parents=True, exist_ok=True)
        return currency_dir

    def _is_cache_valid(self, cache_file, date):
        """Check if cached data is still valid"""
        if not cache_file.exists():
            return False
            
        if date != 'latest':
            return True
            
        # Only check TTL for latest rates
        if date == 'latest':
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > self.cache_ttl:
                logger.info(f"Latest rates cache expired")
                return False
                
        return True

    @retry_with_backoff(retries=3)
    def _fetch_url(self, url):
        """Fetch data from URL with retries and backoff"""
        response = requests.get(url)
        response.raise_for_status()
        logger.info(f"Successfully fetched data from {url}")
        return response.json()

    def _validate_rate_data(self, data, base_currency):
        """Validate fetched rate data"""
        if not data or not isinstance(data, dict):
            raise ValueError("Invalid data format")
            
        if base_currency not in data:
            raise ValueError(f"Base currency {base_currency} not found in response")
            
        rates = data[base_currency]
        if not isinstance(rates, dict):
            raise ValueError("Invalid rates format")
            
        return data

    def fetch_rates(self, base_currency, date=None):
        """Fetch rates with cache handling"""
        if date is None:
            date = 'latest'
            
        # Check cache first
        currency_dir = self._ensure_directory(base_currency)
        cache_file = currency_dir / f"{date}.json"
        
        if self._is_cache_valid(cache_file, date):
            logger.info(f"Reading cached data for {base_currency} on {date}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Cache not valid, need to fetch from API
        try:
            # Try primary URL
            url = self.base_url.format(date=date, endpoint=f"currencies/{base_currency}.json")
            data = self._fetch_url(url)
            
            # Validate and save to cache
            data = self._validate_rate_data(data, base_currency)
            with open(cache_file, 'w') as f:
                json.dump(data, f)
                
            logger.info(f"Successfully fetched and cached data for {base_currency} on {date}")
            return data
            
        except (requests.exceptions.RequestException, ValueError) as e:
            # Try fallback URL
            try:
                url = self.fallback_url.format(date=date, endpoint=f"currencies/{base_currency}.json")
                data = self._fetch_url(url)
                
                # Validate and save to cache
                data = self._validate_rate_data(data, base_currency)
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                    
                logger.info(f"Successfully fetched and cached data from fallback for {base_currency} on {date}")
                return data
                
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"Failed to fetch data for {base_currency} on {date}: {str(e)}")
                
                # If we were trying to get the latest data, try yesterday's data as fallback
                if date == 'latest':
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                    logger.info(f"Trying yesterday's data ({yesterday}) as fallback")
                    return self.fetch_rates(base_currency, yesterday)
                    
                return None

    def fetch_historical_rates(self, base_currency, days=14):
        """Fetch historical rates"""
        rates_data = []
        current_date = datetime.now()
        
        days_checked = 0
        days_fetched = 0
        
        while days_fetched < days and days_checked < days * 2:  # Allow checking up to 2x days to handle potential gaps
            date_to_fetch = (current_date - timedelta(days=days_checked)).strftime('%Y-%m-%d')
            
            # Directly attempt to fetch data for the date_to_fetch
            data = self.fetch_rates(base_currency, date_to_fetch)
            if data:
                rates_data.append(data)
                days_fetched += 1
                    
            days_checked += 1
            
        logger.info(f"Fetched {len(rates_data)} days of historical data for {base_currency} after checking {days_checked} potential days.")
        
        # Handle missing data
        if len(rates_data) < days:
            logger.warning(f"Only found {len(rates_data)} trading days out of {days} requested")
            
        return rates_data

# Example usage
if __name__ == "__main__":
    fetcher = DataFetcher()
    
    # Test fetch latest EUR rates
    # try:
    #     latest_rates = fetcher.fetch_rates('eur')
    #     if latest_rates:
    #         logger.info("Successfully fetched latest EUR rates")
    # except Exception as e:
    #     logger.error(f"Error fetching latest rates: {str(e)}")
    
    # Test fetch historical USD rates
    # try:
    #     historical_rates = fetcher.fetch_historical_rates('usd', days=7)
    #     if historical_rates:
    #         logger.info(f"Successfully fetched {len(historical_rates)} days of USD historical rates")
    # except Exception as e:
    #     logger.error(f"Error fetching historical rates: {str(e)}")