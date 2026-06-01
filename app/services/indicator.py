"""
Technical indicators: RSI, KD, MA
All calculations are pure Python (no ta-lib required)
"""
import pandas as pd
import numpy as np
from typing import Tuple


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI(14) from close prices. Returns float or None."""
    if len(prices) < period + 1:
        return None
    
    deltas = prices.diff()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    
    # Use EMA for smoothing
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1])


def calculate_kd(prices: pd.Series, high: pd.Series, low: pd.Series, 
                 n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[float, float]:
    """
    Calculate KD (RSV->K->D). Returns (K, D) or (None, None).
    Uses the fast stochastic formula.
    """
    if len(prices) < n:
        return None, None
    
    # RSV for each window
    rsv_list = []
    for i in range(n - 1, len(prices)):
        window_high = high.iloc[i - n + 1:i + 1].max()
        window_low = low.iloc[i - n + 1:i + 1].min()
        close = prices.iloc[i]
        
        if window_high == window_low:
            rsv = 50
        else:
            rsv = (close - window_low) / (window_high - window_low) * 100
        rsv_list.append(rsv)
    
    rsv_series = pd.Series(rsv_list, index=prices.index[n - 1:])
    
    # K = EMA(RSV, m1)
    k = rsv_series.ewm(com=m1 - 1, min_periods=m1).mean()
    # D = EMA(K, m2)
    d = k.ewm(com=m2 - 1, min_periods=m2).mean()
    
    return float(k.iloc[-1]), float(d.iloc[-1])


def calculate_ma(prices: pd.Series, period: int) -> float:
    """Calculate simple moving average. Returns float or None."""
    if len(prices) < period:
        return None
    return float(prices.tail(period).mean())


def is_kd_golden_cross(k_series: pd.Series, d_series: pd.Series, n: int = 9) -> bool:
    """Check if KD is in golden cross (K crosses above D while K < 30)."""
    if len(k_series) < n + 2:
        return False
    
    k_vals = k_series.tail(n + 1).values
    d_vals = d_series.tail(n + 1).values
    
    # Need at least n+1 points (n for KD calc + 1 for cross check)
    k_prev = k_vals[-2]
    k_curr = k_vals[-1]
    d_prev = d_vals[-2]
    d_curr = d_vals[-1]
    
    # Golden cross: K was below D, now above D
    was_below = k_prev <= d_prev
    is_above = k_curr > d_curr
    k_is_low = k_curr < 30
    
    return was_below and is_above and k_is_low


def is_kd_death_cross(k_series: pd.Series, d_series: pd.Series, n: int = 9) -> bool:
    """Check if KD is in death cross (K crosses below D while K > 70)."""
    if len(k_series) < n + 2:
        return False
    
    k_vals = k_series.tail(n + 1).values
    d_vals = d_series.tail(n + 1).values
    
    k_prev = k_vals[-2]
    k_curr = k_vals[-1]
    d_prev = d_vals[-2]
    d_curr = d_vals[-1]
    
    # Death cross: K was above D, now below D
    was_above = k_prev >= d_prev
    is_below = k_curr < d_curr
    k_is_high = k_curr > 70
    
    return was_above and is_below and k_is_high