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


def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> float:
    """
    Volume ratio: today's volume vs 20-day average volume.
    > 2.0 = 異常放量（可能是突破或出貨）
    > 1.5 = 溫和放量
    < 0.5 = 量能萎縮
    """
    if len(volume) < period:
        return None
    avg_vol = volume.tail(period).mean()
    if avg_vol == 0:
        return None
    ratio = float(volume.iloc[-1]) / float(avg_vol)
    return round(ratio, 2)


def calculate_ma_deviation_pct(close: pd.Series, ma_period: int = 20) -> float:
    """
    Price deviation from MA as percentage.
    > 0 = 價格在均線之上（正偏離 = 可能超漲）
    < 0 = 價格在均線之下（負偏離 = 可能超跌）
    林穎老師用 ±10~15% 作為超買超賣輔助判斷
    """
    if len(close) < ma_period:
        return None
    ma = close.tail(ma_period).mean()
    if ma == 0:
        return None
    deviation = (float(close.iloc[-1]) - ma) / ma * 100
    return round(deviation, 2)


def calculate_trend_position(close: pd.Series, ma_period: int = 60) -> dict:
    """
    Trend position: price vs MA60 and how far.
    多頭：價格在 MA60 之上
    空頭：價格在 MA60 之下
    Returns native Python types.
    """
    if len(close) < ma_period:
        return {"above": None, "deviation_pct": None, "trend": "unknown"}

    ma = float(close.tail(ma_period).mean())
    price = float(close.iloc[-1])
    above = price > ma
    dev_pct = round((price - ma) / ma * 100, 2)

    if above:
        if dev_pct > 10:
            trend = "strong_uptrend"   # 強勢多頭
        elif dev_pct > 3:
            trend = "uptrend"         # 正常多頭
        else:
            trend = "weak_uptrend"     # 微弱多頭
    else:
        if dev_pct < -10:
            trend = "strong_downtrend" # 強勢空頭
        elif dev_pct < -3:
            trend = "downtrend"       # 正常空頭
        else:
            trend = "weak_downtrend"   # 微弱空頭

    return {
        "above": bool(above),
        "deviation_pct": dev_pct,
        "trend": trend,
    }


def calculate_entry_quality(price: float, indicators: dict,
                             volume_ratio: float, ma_deviation: float) -> dict:
    """
    Entry quality score (進場品質評分):
    - 綜合 RSI、MA偏離度、成交量缺口判斷進場時機好壞
    - score: 0~100，越高代表進場時機越好
    - verdict: 極佳/良好/尚可/不佳
    - reason: 具體原因陣列
    林穎老師核心：訊號完整才進場，提前進場是最大虧損來源
    """
    score = 50  # 基準分
    reasons = []  # [{type: 'positive'|'negative'|'neutral', reason: str}, ...]

    rsi = indicators.get('rsi')
    kd_k = indicators.get('kd_k')
    macd_hist = indicators.get('macd_hist')
    trend = indicators.get('trend_position', {}).get('trend', 'unknown')
    trend_above = indicators.get('trend_position', {}).get('above')

    # ── RSI 貢獻（權重 25%）─────────────────────────
    if rsi is not None:
        if rsi < 25:  # 極超賣
            score += 20
            reasons.append({'type': 'positive', 'reason': f'RSI={rsi:.0f} 極超賣 (+20)'})
        elif rsi < 35:  # 超賣區
            score += 15
            reasons.append({'type': 'positive', 'reason': f'RSI={rsi:.0f} 超賣區 (+15)'})
        elif rsi < 45:
            score += 5
            reasons.append({'type': 'neutral', 'reason': f'RSI={rsi:.0f} 偏低但不超賣 (+5)'})
        elif rsi > 80:
            score -= 20
            reasons.append({'type': 'negative', 'reason': f'RSI={rsi:.0f} 極超買 (-20)'})
        elif rsi > 65:
            score -= 10
            reasons.append({'type': 'negative', 'reason': f'RSI={rsi:.0f} 偏高 (-10)'})

    # ── MA 偏離度貢獻（權重 20%）────────────────────
    if ma_deviation is not None:
        if ma_deviation < -15:  # 低於 MA20 超過 15%
            score += 15
            reasons.append({'type': 'positive', 'reason': f'MA偏離={ma_deviation:+.1f}% 低檔超跌 (+15)'})
        elif ma_deviation < -8:
            score += 10
            reasons.append({'type': 'positive', 'reason': f'MA偏離={ma_deviation:+.1f}% 低於均線 (+10)'})
        elif ma_deviation > 15:  # 高於 MA20 超過 15%
            score -= 15
            reasons.append({'type': 'negative', 'reason': f'MA偏離={ma_deviation:+.1f}% 高檔超漲 (-15)'})
        elif ma_deviation > 8:
            score -= 8
            reasons.append({'type': 'negative', 'reason': f'MA偏離={ma_deviation:+.1f}% 高於均線 (-8)'})

    # ── 成交量貢獻（權重 20%）───────────────────────
    if volume_ratio is not None:
        if volume_ratio >= 2.0:  # 異常放量
            macd_val = macd_hist or 0
            if macd_val > 0:
                score += 15
                reasons.append({'type': 'positive', 'reason': f'放量{volume_ratio:.1f}x + MACD偏多 (+15)'})
            else:
                score -= 10
                reasons.append({'type': 'negative', 'reason': f'放量{volume_ratio:.1f}x + MACD偏空 (-10)'})
        elif volume_ratio >= 1.5:
            score += 8
            reasons.append({'type': 'positive', 'reason': f'溫和放量{volume_ratio:.1f}x (+8)'})
        elif volume_ratio < 0.5:
            score -= 5
            reasons.append({'type': 'negative', 'reason': f'量能萎縮{volume_ratio:.1f}x (-5)'})

    # ── 趨勢方向貢獻（權重 20%）─────────────────────
    if trend_above is not None:
        if trend in ("strong_uptrend", "uptrend"):
            score += 12
            reasons.append({'type': 'positive', 'reason': f'趨勢={trend}，順勢進場 (+12)'})
        elif trend in ("strong_downtrend", "downtrend"):
            score -= 12
            reasons.append({'type': 'negative', 'reason': f'趨勢={trend}，逆勢風險 (-12)'})
        else:
            score -= 3
            reasons.append({'type': 'neutral', 'reason': f'趨勢={trend}，方向不明 (-3)'})

    # ── KD 位置輔助（權重 15%）─────────────────────
    if kd_k is not None:
        if kd_k < 20:  # 低檔
            score += 10
            reasons.append({'type': 'positive', 'reason': f'KD K={kd_k:.0f} 低檔 (+10)'})
        elif kd_k > 80:  # 高檔
            score -= 10
            reasons.append({'type': 'negative', 'reason': f'KD K={kd_k:.0f} 高檔 (-10)'})

    # 限制範圍
    score = max(0, min(100, score))

    # 評語
    if score >= 80:
        verdict = "🌟 極佳進場點"
        verdict_color = "text-green"  # 台股綠漲
    elif score >= 60:
        verdict = "✅ 良好進場點"
        verdict_color = "text-green"
    elif score >= 40:
        verdict = "⚠️ 尚可，等待更好時機"
        verdict_color = "text-yellow"
    else:
        verdict = "❌ 不佳，觀望為宜"
        verdict_color = "text-red"

    return {
        "score": score,
        "reasons": reasons[:6],  # 最多6條
    }


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