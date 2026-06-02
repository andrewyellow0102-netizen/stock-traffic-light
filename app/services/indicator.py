"""
Technical indicators: RSI, KD, MA, MACD, Williams%R, CCI, Bollinger, ATR, OBV, MFI
All calculations are pure Python (no ta-lib required)
"""
import pandas as pd
import numpy as np
from typing import Tuple


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return None
    deltas = prices.diff()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def calculate_kd(prices: pd.Series, high: pd.Series, low: pd.Series,
                 n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[float, float]:
    if len(prices) < n:
        return None, None
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
    k = rsv_series.ewm(com=m1 - 1, min_periods=m1).mean()
    d = k.ewm(com=m2 - 1, min_periods=m2).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])


def calculate_ma(prices: pd.Series, period: int) -> float:
    if len(prices) < period:
        return None
    return float(prices.tail(period).mean())


def calculate_ema(prices: pd.Series, period: int) -> float:
    if len(prices) < period:
        return None
    ema = prices.ewm(span=period, adjust=False, min_periods=period).mean()
    return float(ema.iloc[-1])


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
                  ) -> Tuple[float, float, float]:
    """Returns (DIF, DEA, HISTOGRAM) or (None, None, None)."""
    if len(prices) < slow + signal:
        return None, None, None
    ema_fast = prices.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, adjust=False, min_periods=slow).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = dif - dea
    return round(float(dif.iloc[-1]), 4), round(float(dea.iloc[-1]), 4), round(float(histogram.iloc[-1]), 4)


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, k: float = 2.0
                             ) -> Tuple[float, float, float]:
    """Returns (upper, middle, lower) or (None, None, None)."""
    if len(prices) < period:
        return None, None, None
    middle = prices.tail(period).mean()
    std = prices.tail(period).std()
    upper = middle + k * std
    lower = middle - k * std
    return round(upper, 2), round(middle, 2), round(lower, 2)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Average True Range."""
    if len(high) < period + 1:
        return None
    tr_list = []
    for i in range(1, len(high)):
        h, l, c = high.iloc[i], low.iloc[i], close.iloc[i]
        tr = max(h - l, abs(h - close.iloc[i - 1]), abs(l - close.iloc[i - 1]))
        tr_list.append(tr)
    if len(tr_list) < period:
        return None
    atr = pd.Series(tr_list).ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    return round(float(atr), 2)


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 14) -> float:
    """Williams %R. Returns float 0 to -100."""
    if len(high) < period:
        return None
    highest_high = high.tail(period).max()
    lowest_low = low.tail(period).min()
    if highest_high == lowest_low:
        return -50.0
    wr = (highest_high - close.iloc[-1]) / (highest_high - lowest_low) * -100
    return round(float(wr), 2)


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> float:
    """Commodity Channel Index."""
    if len(high) < period:
        return None
    typical = (high + low + close) / 3
    tp = typical.tail(period)
    ma = tp.mean()
    mean_dev = (tp - ma).abs().mean()
    if mean_dev == 0:
        return 0.0
    cci = (typical.iloc[-1] - ma) / (0.015 * mean_dev)
    return round(float(cci), 2)


def calculate_obv(close: pd.Series, volume: pd.Series) -> float:
    """On-Balance Volume. Returns cumulative OBV value."""
    if len(close) < 2 or len(volume) < 2:
        return None
    obv = 0.0
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv += float(volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv -= float(volume.iloc[i])
    return round(obv, 2)


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series, period: int = 14) -> float:
    """Money Flow Index (volume-weighted RSI)."""
    if len(high) < period + 1:
        return None
    typical = (high + low + close) / 3
    raw_money = typical * volume
    money_flow = []
    for i in range(1, len(typical)):
        if typical.iloc[i] > typical.iloc[i - 1]:
            money_flow.append(float(raw_money.iloc[i]))
        else:
            money_flow.append(-float(raw_money.iloc[i]))
    if len(money_flow) < period:
        return None
    positive = pd.Series(money_flow).clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    negative = pd.Series(money_flow).clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    mfi = 100 - (100 / (1 + positive / negative.abs()))
    return round(float(mfi.iloc[-1]), 2)


def is_ma_golden_cross(close: pd.Series, short_ma_period: int, long_ma_period: int) -> dict:
    """
    Detect MA golden cross: short MA crosses above long MA.
    Returns dict with signal info. All values are native Python types.
    """
    if len(close) < long_ma_period + 2:
        return {"cross": False, "short_ma": None, "long_ma": None}
    short_ma = close.ewm(span=short_ma_period, adjust=False).mean()
    long_ma = close.ewm(span=long_ma_period, adjust=False).mean()
    short_prev = float(short_ma.iloc[-2])
    short_curr = float(short_ma.iloc[-1])
    long_prev = float(long_ma.iloc[-2])
    long_curr = float(long_ma.iloc[-1])
    cross = bool(short_prev <= long_prev and short_curr > long_curr)
    return {"cross": cross, "short_ma": round(short_curr, 2), "long_ma": round(long_curr, 2)}


def is_ma_death_cross(close: pd.Series, short_ma_period: int, long_ma_period: int) -> dict:
    """Detect MA death cross: short MA crosses below long MA. All native Python types."""
    if len(close) < long_ma_period + 2:
        return {"cross": False, "short_ma": None, "long_ma": None}
    short_ma = close.ewm(span=short_ma_period, adjust=False).mean()
    long_ma = close.ewm(span=long_ma_period, adjust=False).mean()
    short_prev = float(short_ma.iloc[-2])
    short_curr = float(short_ma.iloc[-1])
    long_prev = float(long_ma.iloc[-2])
    long_curr = float(long_ma.iloc[-1])
    cross = bool(short_prev >= long_prev and short_curr < long_curr)
    return {"cross": cross, "short_ma": round(short_curr, 2), "long_ma": round(long_curr, 2)}