import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from ta.trend import SMAIndicator, MACD, cci
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import logging
import threading
from datetime import datetime, timedelta
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
import os
import pickle
import sys
import msvcrt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# --- Initialize MT5 Connection ---
if not mt5.initialize():
    logger.error("Failed to initialize MT5 connection")
    sys.exit(1)

# --- Configuration ---
symbol = input("Enter currency pair (e.g., EURUSD): ").strip().upper()
timeframes = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1
}
num_candles = 1000
capital = 50
risk_per_trade = 0.02
min_risk_reward_ratio = 3

# --- News Configuration ---
NEWS_API_KEY = 'YOUR_API_KEY_HERE'
NEWS_API_URL = 'https://newsapi.org/v2/everything'
FOREX_FACTORY_URL = 'https://www.forexfactory.com/calendar'
TRADINGVIEW_URL = 'https://www.tradingview.com/news/'

# --- ML Configuration ---
LSTM_MODEL = None
LSTM_TRAINING_LOCK = threading.Lock()
ML_START_TIME = datetime.now()
ML_CORRECT_DECISIONS = 0
ML_TOTAL_DECISIONS = 0
ML_MEMORY_FILE = 'ml_memory.pkl'
ML_STATUS = "Idle"
LAST_SAVE_TIME = datetime.now()

# --- Rich Console Setup ---
console = Console()

# --- Helper Functions ---
def save_ml_memory():
    global LSTM_MODEL, ML_CORRECT_DECISIONS, ML_TOTAL_DECISIONS, LAST_SAVE_TIME
    try:
        with open(ML_MEMORY_FILE, 'wb') as f:
            pickle.dump({
                'model': LSTM_MODEL,
                'correct_decisions': ML_CORRECT_DECISIONS,
                'total_decisions': ML_TOTAL_DECISIONS,
                'start_time': ML_START_TIME
            }, f)
        LAST_SAVE_TIME = datetime.now()
        logger.info("ML memory saved successfully")
    except Exception as e:
        logger.error(f"Failed to save ML memory: {e}")

def load_ml_memory():
    global LSTM_MODEL, ML_CORRECT_DECISIONS, ML_TOTAL_DECISIONS, ML_START_TIME
    if os.path.exists(ML_MEMORY_FILE):
        try:
            with open(ML_MEMORY_FILE, 'rb') as f:
                memory = pickle.load(f)
                LSTM_MODEL = memory['model']
                ML_CORRECT_DECISIONS = memory['correct_decisions']
                ML_TOTAL_DECISIONS = memory['total_decisions']
                ML_START_TIME = memory['start_time']
            logger.info("ML memory loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ML memory: {e}")
            initialize_ml_memory()
    else:
        initialize_ml_memory()

def initialize_ml_memory():
    global LSTM_MODEL, ML_CORRECT_DECISIONS, ML_TOTAL_DECISIONS, ML_START_TIME
    LSTM_MODEL = None
    ML_CORRECT_DECISIONS = 0
    ML_TOTAL_DECISIONS = 0
    ML_START_TIME = datetime.now()
    save_ml_memory()
    logger.info("Initialized new ML memory")

def change_currency(new_symbol):
    global symbol
    symbol = new_symbol.upper()
    logger.info(f"Currency pair changed to {symbol}")

def check_for_currency_change():
    if msvcrt.kbhit():
        user_input = input("\nEnter new currency pair (or press Enter to continue): ").strip().upper()
        if user_input:
            change_currency(user_input)
            return True
    return False

def fetch_historical_data(symbol, timeframe, num_candles):
    try:
        data = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
        if data is None or len(data) == 0:
            raise ValueError("No data returned from MT5")
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol} on {timeframe}: {e}")
        return pd.DataFrame()

def add_indicators(df):
    # Moving Averages
    df['MA_10'] = SMAIndicator(df['close'], window=10).sma_indicator()
    df['MA_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    
    # MACD
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Other indicators
    df['RSI_10'] = RSIIndicator(df['close'], window=10).rsi()
    df['ROC_2'] = ROCIndicator(df['close'], window=2).roc()
    df['Momentum_4'] = df['close'].diff(4)
    
    # Bollinger Bands
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()
    
    # CCI
    df['CCI_20'] = cci(df['high'], df['low'], df['close'], window=20)
    
    # Signal columns
    df['MAC_Signal'] = np.where(df['MA_10'] > df['MA_50'], 1, -1)
    df['BBMA_Signal'] = np.where(df['close'] > df['BB_Middle'], 1, -1)
    
    return df.dropna()

def train_lstm_model(df, symbol):
    global LSTM_MODEL, ML_STATUS, ML_CORRECT_DECISIONS, ML_TOTAL_DECISIONS
    
    if df.empty or len(df) < 100:
        logger.warning(f"Insufficient data for {symbol}. Skipping training.")
        return None
    
    ML_STATUS = "Training"
    logger.info(f"Starting LSTM training for {symbol}")
    
    try:
        df = add_indicators(df)
        features = ['open', 'high', 'low', 'close', 'MA_10', 'MACD', 'MACD_Signal', 
                   'MACD_Hist', 'ROC_2', 'Momentum_4', 'RSI_10', 'BB_Upper', 
                   'BB_Lower', 'CCI_20', 'MA_50', 'MAC_Signal', 'BBMA_Signal']
        
        X = df[features].values
        y = df['close'].shift(-1).dropna().values
        X = X[:len(y)]
        
        if len(X) != len(y):
            logger.warning("Feature-target length mismatch. Adjusting...")
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"LSTM training complete for {symbol}. MSE: {mse:.6f}")
        
        with LSTM_TRAINING_LOCK:
            LSTM_MODEL = model
        
        return model
    except Exception as e:
        logger.error(f"Error during LSTM training: {e}")
        return None
    finally:
        ML_STATUS = "Idle"

def predict_next_price(model, df):
    try:
        if model is None or df.empty:
            return np.nan
            
        df = add_indicators(df)
        features = ['open', 'high', 'low', 'close', 'MA_10', 'MACD', 'MACD_Signal',
                   'MACD_Hist', 'ROC_2', 'Momentum_4', 'RSI_10', 'BB_Upper',
                   'BB_Lower', 'CCI_20', 'MA_50', 'MAC_Signal', 'BBMA_Signal']
        
        latest_data = df[features].tail(1).values
        latest_data = latest_data.reshape((1, 1, latest_data.shape[1]))
        return model.predict(latest_data)[0][0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return np.nan

def analyze_mac_rsi_bbma(df):
    if df.empty:
        return df
        
    df = add_indicators(df)
    
    # Generate signals
    df['Signal'] = 'Hold'
    buy_condition = (df['MA_10'] > df['MA_50']) & (df['RSI_10'] > 50) & (df['close'] > df['BB_Middle'])
    sell_condition = (df['MA_10'] < df['MA_50']) & (df['RSI_10'] < 50) & (df['close'] < df['BB_Middle'])
    
    df.loc[buy_condition, 'Signal'] = 'Buy'
    df.loc[sell_condition, 'Signal'] = 'Sell'
    
    # Calculate TP/SL
    df['Take_Profit'] = np.nan
    df['Stop_Loss'] = np.nan
    
    df.loc[df['Signal'] == 'Buy', 'Take_Profit'] = df['close'] + (df['close'] - df['BB_Lower']) * 3
    df.loc[df['Signal'] == 'Buy', 'Stop_Loss'] = df['BB_Lower']
    df.loc[df['Signal'] == 'Sell', 'Take_Profit'] = df['close'] - (df['BB_Upper'] - df['close']) * 3
    df.loc[df['Signal'] == 'Sell', 'Stop_Loss'] = df['BB_Upper']
    
    df['Volatility'] = df['high'] - df['low']
    return df

def fetch_news():
    news = []
    try:
        # NewsAPI
        params = {'q': symbol, 'apiKey': NEWS_API_KEY, 'language': 'en', 'pageSize': 3}
        newsapi = requests.get(NEWS_API_URL, params=params).json().get('articles', [])
        news.extend([{'source': 'NewsAPI', 'title': a['title'], 'time': a['publishedAt']} for a in newsapi])
        
        # Forex Factory
        ff = requests.get(FOREX_FACTORY_URL).text
        soup = BeautifulSoup(ff, 'html.parser')
        for item in soup.select('tr.calendar_row'):
            title = item.select_one('.calendar__event').text.strip()
            time = item.select_one('.calendar__time').text.strip()
            news.append({'source': 'ForexFactory', 'title': title, 'time': time})
            
        # TradingView
        tv = requests.get(TRADINGVIEW_URL).text
        soup = BeautifulSoup(tv, 'html.parser')
        for item in soup.select('.news-item'):
            title = item.select_one('.news-item__title').text.strip()
            time = item.select_one('.news-item__date').text.strip()
            news.append({'source': 'TradingView', 'title': title, 'time': time})
            
    except Exception as e:
        logger.error(f"News fetch error: {e}")
    
    return news

def analyze_news_sentiment(news):
    positive = ['positive', 'bullish', 'strong', 'growth', 'rise', 'increase']
    negative = ['negative', 'bearish', 'weak', 'decline', 'fall', 'drop']
    score = 0
    
    for item in news:
        text = item['title'].lower()
        score += sum(1 for word in positive if word in text)
        score -= sum(1 for word in negative if word in text)
    
    if score > 0: return 'Positive'
    if score < 0: return 'Negative'
    return 'Neutral'

def calculate_ml_stats():
    runtime = datetime.now() - ML_START_TIME
    days = runtime.days
    hours, remainder = divmod(runtime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    lifetime = f"{days}d {hours}h {minutes}m {seconds}s"
    
    accuracy = "0.00%" if ML_TOTAL_DECISIONS == 0 else f"{(ML_CORRECT_DECISIONS/ML_TOTAL_DECISIONS)*100:.2f}%"
    return lifetime, accuracy

def create_display(symbol, signals, final_signal, news_sentiment, tp, sl, cooldown):
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3)
    )
    
    # Header with ML status
    ml_lifetime, ml_accuracy = calculate_ml_stats()
    header = Panel(
        f"ML Status: {ML_STATUS}\n"
        f"Uptime: {ml_lifetime}\n"
        f"Accuracy: {ml_accuracy}",
        title="Machine Learning"
    )
    layout["header"].update(header)
    
    # Body with analysis
    body_lines = [
        f"ANALYSIS FOR {symbol}",
        f"Final Signal: {final_signal}",
        f"News Sentiment: {news_sentiment}",
        f"Take Profit: {tp:.5f}" if not np.isnan(tp) else "Take Profit: N/A",
        f"Stop Loss: {sl:.5f}" if not np.isnan(sl) else "Stop Loss: N/A",
        "",
        "TIMEFRAME SIGNALS:"
    ]
    
    for tf in ['M5', 'M15', 'H1', 'H4', 'D1']:
        if tf in signals:
            body_lines.append(
                f"{tf}: {signals[tf]['signal']} "
                f"(TP: {signals[tf]['tp']:.5f}, "
                f"SL: {signals[tf]['sl']:.5f}, "
                f"Risk: {signals[tf]['risk']})"
            )
    
    layout["body"].update(Panel("\n".join(body_lines), title="Market Analysis"))
    
    # Footer with cooldown
    layout["footer"].update(Panel(f"Next analysis in: {cooldown}s", title="Status"))
    
    return layout

def real_time_analysis():
    global LAST_SAVE_TIME
    
    # Load initial ML memory
    load_ml_memory()
    
    # Start save thread
    def save_thread():
        while True:
            time.sleep(1800)  # 30 minutes
            save_ml_memory()
    
    threading.Thread(target=save_thread, daemon=True).start()
    
    with Live(auto_refresh=False) as live:
        while True:
            start_time = time.time()
            
            # Check for currency change
            if check_for_currency_change():
                continue
                
            # Check if we need to save
            if (datetime.now() - LAST_SAVE_TIME).total_seconds() >= 1800:
                save_ml_memory()
            
            # Fetch and analyze data
            signals = {}
            for tf_name, tf in timeframes.items():
                df = fetch_historical_data(symbol, tf, num_candles)
                if not df.empty:
                    df = analyze_mac_rsi_bbma(df)
                    last_row = df.iloc[-1]
                    signals[tf_name] = {
                        'signal': last_row['Signal'],
                        'tp': last_row['Take_Profit'],
                        'sl': last_row['Stop_Loss'],
                        'risk': 'High' if last_row['Volatility'] > 0.02 else 
                               'Medium' if last_row['Volatility'] > 0.01 else 'Low'
                    }
            
            # Combine signals (H1 and H4 must agree)
            final_signal = 'Hold'
            final_tp = final_sl = np.nan
            if 'H1' in signals and 'H4' in signals:
                if signals['H1']['signal'] == signals['H4']['signal'] and signals['H1']['signal'] != 'Hold':
                    final_signal = signals['H1']['signal']
                    final_tp = np.nanmean([signals['H1']['tp'], signals['H4']['tp']])
                    final_sl = np.nanmean([signals['H1']['sl'], signals['H4']['sl']])
            
            # Incorporate LSTM prediction if available
            if LSTM_MODEL is not None and 'H1' in signals:
                df_h1 = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, num_candles)
                predicted = predict_next_price(LSTM_MODEL, df_h1)
                current = df_h1['close'].iloc[-1]
                
                if not np.isnan(predicted):
                    ML_TOTAL_DECISIONS += 1
                    if ((final_signal == 'Buy' and predicted > current) or 
                        (final_signal == 'Sell' and predicted < current)):
                        ML_CORRECT_DECISIONS += 1
                    else:
                        final_signal = 'Hold'
            
            # Fetch and analyze news
            news = fetch_news()
            news_sentiment = analyze_news_sentiment(news)
            
            # High impact news check
            high_impact = any(word in article['title'].lower() 
                            for article in news 
                            for word in ['interest rate', 'nfp', 'cpi', 'fomc', 'fed'])
            
            if high_impact or news_sentiment == 'Negative':
                final_signal = 'Hold'
            
            # Display results with cooldown
            cooldown = 20
            for remaining in range(cooldown, 0, -1):
                display = create_display(
                    symbol=symbol,
                    signals=signals,
                    final_signal=final_signal,
                    news_sentiment=news_sentiment,
                    tp=final_tp,
                    sl=final_sl,
                    cooldown=remaining
                )
                live.update(display)
                live.refresh()
                
                # Check for currency change during cooldown
                if check_for_currency_change():
                    break
                    
                time.sleep(1)
            
            # Train model periodically (every 6 hours)
            if (datetime.now() - ML_START_TIME).total_seconds() % 21600 < 20:
                df_train = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, num_candles)
                threading.Thread(target=train_lstm_model, args=(df_train, symbol), daemon=True).start()

if __name__ == "__main__":
    try:
        real_time_analysis()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        save_ml_memory()
    finally:
        mt5.shutdown()
