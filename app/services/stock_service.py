"""
Stock service: fetch data via yfinance and calculate indicators
"""
import yfinance as yf
import pandas as pd
from typing import Optional

from app.services.indicator import (
    calculate_rsi, calculate_kd, calculate_ma, calculate_ema,
    calculate_macd, calculate_bollinger_bands, calculate_atr,
    calculate_williams_r, calculate_cci, calculate_obv, calculate_mfi,
    is_ma_golden_cross, is_ma_death_cross,
)


def fetch_stock_data(code: str, period: str = "3mo") -> Optional[dict]:
    """
    Fetch stock data from yfinance.
    Tries .TW (TAIEX) first, then .TWO (OTC).
    Handles letter-suffix ETF codes like 00981A → 00981A.TW (not 00981.TW).
    Returns dict with price info + raw DataFrame for indicator calculation.
    Returns None if stock not found.
    """
    import re
    # Check if code contains letters (mixed alphanumeric like 00981A)
    has_letters = bool(re.search(r'[A-Za-z]', code))

    # Try .TW first, then .TWO
    suffixes = ['.TW', '.TWO']
    ticker = None
    used_suffix = None

    for suffix in suffixes:
        ticker_symbol = f"{code}{suffix}"
        t = yf.Ticker(ticker_symbol)
        try:
            h = t.history(period=period, auto_adjust=True, raise_errors=True)
            if h is not None and not h.empty and len(h) >= 5:
                ticker = t
                used_suffix = suffix
                hist = h
                break
        except Exception:
            continue

    if ticker is None:
        return None
    
    # Current price
    current_price = round(float(hist['Close'].iloc[-1]), 2)

    # Stock name
    try:
        name = ticker.info.get('longName') or ticker.info.get('shortName', code)
    except Exception:
        name = code

    # Price change
    prev_price = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else current_price
    change = round(current_price - prev_price, 2)
    change_pct = round((change / prev_price * 100), 2) if prev_price else 0

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
    volume = hist['Volume']

    rsi = calculate_rsi(close, period=14)
    k, d = calculate_kd(close, high, low, n=9, m1=3, m2=3)
    ma5 = calculate_ma(close, 5)
    ma20 = calculate_ma(close, 20)
    ma60 = calculate_ma(close, 60) if len(close) >= 60 else None
    ma120 = calculate_ma(close, 120) if len(close) >= 120 else None

    # MACD (12, 26, 9)
    macd_dif, macd_dea, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)

    # Bollinger Bands (20, 2)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, k=2)

    # ATR (14)
    atr = calculate_atr(high, low, close, period=14)

    # Williams %R (14)
    williams_r = calculate_williams_r(high, low, close, period=14)

    # CCI (14)
    cci = calculate_cci(high, low, close, period=14)

    # OBV
    obv = calculate_obv(close, volume)

    # MFI (14)
    mfi = calculate_mfi(high, low, close, volume, period=14)

    # MA Golden/Death Cross (20 vs 60)
    ma_cross_20_60_golden = is_ma_golden_cross(close, 20, 60)
    ma_cross_20_60_death = is_ma_death_cross(close, 20, 60)

    return {
        'rsi': round(rsi, 2) if rsi is not None else None,
        'kd_k': round(k, 2) if k is not None else None,
        'kd_d': round(d, 2) if d is not None else None,
        'ma5': round(ma5, 2) if ma5 is not None else None,
        'ma20': round(ma20, 2) if ma20 is not None else None,
        'ma60': round(ma60, 2) if ma60 is not None else None,
        'ma120': round(ma120, 2) if ma120 is not None else None,
        'macd_dif': macd_dif,
        'macd_dea': macd_dea,
        'macd_hist': macd_hist,
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
        'atr': atr,
        'williams_r': williams_r,
        'cci': cci,
        'obv': obv,
        'mfi': mfi,
        'ma_cross_20_60_golden': ma_cross_20_60_golden,
        'ma_cross_20_60_death': ma_cross_20_60_death,
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