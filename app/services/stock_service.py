"""
Stock service: fetch data via yfinance and calculate indicators
"""
import yfinance as yf
import pandas as pd
from typing import Optional, List
from app.services.indicator import calculate_rsi, calculate_kd, calculate_ma


# Taiwan stock suffix mapping (yfinance uses .TW for Taiwan)
TW_SUFFIXES = ['.TW', '.TWO']  # TW = TAIEX, TWO = OTC


def _get_ticker(code: str):
    """Get yfinance ticker, trying .TW then .TWO"""
    for suffix in TW_SUFFIXES:
        ticker = yf.Ticker(f"{code}{suffix}")
        try:
            # Test if this ticker has data
            info = ticker.fast_info
            return ticker
        except Exception:
            continue
    # Fallback: just use .TW
    return yf.Ticker(f"{code}.TW")


def fetch_stock_data(code: str, period: str = "3mo") -> Optional[dict]:
    """
    Fetch stock data from yfinance.
    Returns dict with price info + raw DataFrame for indicator calculation.
    Returns None if stock not found.
    """
    ticker = _get_ticker(code)
    
    try:
        hist = ticker.history(period=period, auto_adjust=True)
    except Exception:
        return None
    
    if hist.empty or len(hist) < 30:
        return None
    
    info = {}
    try:
        info = ticker.fast_info
    except Exception:
        pass
    
    # Current price
    current_price = float(hist['Close'].iloc[-1])
    
    # Stock name
    try:
        name = ticker.info.get('longName') or ticker.info.get('shortName', code)
    except Exception:
        name = code
    
    # Price change
    prev_price = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price * 100) if prev_price else 0
    
    return {
        'code': code,
        'name': name,
        'price': current_price,
        'change': f"{'+' if change >= 0 else ''}{change:.2f} ({change_pct:+.2f}%)",
        'hist': hist,  # DataFrame for indicator calc
    }


def calculate_indicators(hist: pd.DataFrame) -> dict:
    """Calculate all technical indicators from price history."""
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    
    rsi = calculate_rsi(close, period=14)
    k, d = calculate_kd(close, high, low, n=9, m1=3, m2=3)
    ma5 = calculate_ma(close, 5)
    ma20 = calculate_ma(close, 20)
    ma60 = calculate_ma(close, 60) if len(close) >= 60 else None
    
    return {
        'rsi': rsi,
        'kd_k': k,
        'kd_d': d,
        'ma5': ma5,
        'ma20': ma20,
        'ma60': ma60,
    }


def determine_light(price: float, indicators: dict, hist: pd.DataFrame) -> dict:
    """
    Determine traffic light based on rules:
    
    🟢 GREEN (buy): 
      - price < MA20 * 0.90 (10% below MA20) OR
      - RSI < 30 OR
      - (KD golden cross AND K < 30)
    
    🔴 RED (sell):
      - price > MA20 * 1.15 (15% above MA20) OR
      - RSI > 70 OR
      - (KD death cross AND K > 70)
    
    🟡 YELLOW (watch): everything else
    """
    rsi = indicators.get('rsi')
    k = indicators.get('kd_k')
    d = indicators.get('kd_d')
    ma20 = indicators.get('ma20')
    
    reasons = []
    light = 'yellow'
    signal_type = 'watch'
    
    # Compute KD golden/death cross
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    
    # Get recent K and D series for cross detection
    kd_k_series, kd_d_series = None, None
    if k is not None and d is not None and len(hist) >= 11:
        # Reconstruct full K series for cross detection
        k_vals, d_vals = [], []
        n = 9
        for i in range(n - 1, len(hist)):
            window_high = high.iloc[i - n + 1:i + 1].max()
            window_low = low.iloc[i - n + 1:i + 1].min()
            close_val = close.iloc[i]
            if window_high == window_low:
                rsv = 50
            else:
                rsv = (close_val - window_low) / (window_high - window_low) * 100
            k_vals.append(rsv)

        k_s = pd.Series(k_vals).ewm(com=3-1, min_periods=3).mean()
        d_s = k_s.ewm(com=3-1, min_periods=3).mean()
        kd_k_series, kd_d_series = k_s, d_s
    
    # Check conditions (order matters for priority)
    # GREEN conditions
    green_conditions = []
    
    if ma20 is not None and price < ma20 * 0.90:
        green_conditions.append(f'股價(${price:.1f})低於月線(MA20=${ma20:.1f})10%以上，屬超跌區')
    
    if rsi is not None and rsi < 30:
        green_conditions.append(f'RSI(14)={rsi:.1f} < 30，屬超賣區')
    
    if green_conditions:
        light = 'green'
        signal_type = 'buy'
        reasons = green_conditions
    
    # RED conditions
    if light == 'yellow':  # Only check if not already green
        red_conditions = []
        
        if ma20 is not None and price > ma20 * 1.15:
            red_conditions.append(f'股價(${price:.1f})高於月線(MA20=${ma20:.1f})15%以上，屬超漲區')
        
        if rsi is not None and rsi > 70:
            red_conditions.append(f'RSI(14)={rsi:.1f} > 70，屬超買區')
        
        if red_conditions:
            light = 'red'
            signal_type = 'sell'
            reasons = red_conditions
    
    # Check KD cross separately
    if kd_k_series is not None and kd_d_series is not None and k is not None:
        if len(kd_k_series) >= 2:
            k_prev = kd_k_series.iloc[-2]
            k_curr = kd_k_series.iloc[-1]
            d_prev = kd_d_series.iloc[-2]
            d_curr = kd_d_series.iloc[-1]
            
            # Golden cross: K crosses above D while K < 30
            if k_prev <= d_prev and k_curr > d_curr and k_curr < 30:
                if light == 'green':
                    reasons.append('KD 低檔黃金交叉，買進確認')
                else:
                    light = 'green'
                    signal_type = 'buy'
                    reasons = [f'KD 低檔黃金交叉 (K={k_curr:.1f}<30)，買進訊號']
            
            # Death cross: K crosses below D while K > 70
            if k_prev >= d_prev and k_curr < d_curr and k_curr > 70:
                if light == 'red':
                    reasons.append('KD 高檔死亡交叉，賣出確認')
                else:
                    light = 'red'
                    signal_type = 'sell'
                    reasons = [f'KD 高檔死亡交叉 (K={k_curr:.1f}>70)，賣出訊號']
    
    if not reasons:
        reasons = ['目前無明確方向，請持續追蹤']
    
    light_labels = {
        'green': ('買進', '超跌或超賣區，適合考慮進場'),
        'yellow': ('觀望', '目前方向不明，建議等待'),
        'red': ('賣出', '超漲或超買區，考慮分批獲利了結'),
    }
    label, desc = light_labels[light]
    
    return {
        'light': light,
        'label': label,
        'description': desc,
        'signal_description': reasons[0] if reasons else label,
    }