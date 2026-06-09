from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import asyncio

from app.services.stock_service import fetch_stock_data, calculate_indicators, determine_light, calculate_entry_quality

app = FastAPI(
    title="股票紅綠燈",
    description="台股紅綠燈系統：輸入股號查燈號，搜尋最佳投資標的",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "stock-traffic-light"}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    template_path = Path(__file__).parent / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/stock")
async def get_stock(code: str = Query(..., description="股票代碼（如 2330）")):
    """取得個股燈號與技術指標"""
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, fetch_stock_data, code)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"查無此股票：{code}")
    
    indicators = calculate_indicators(data['hist'])
    light_result = determine_light(data['price'], indicators, data['hist'])

    # 進場品質評分（林穎老師方法論）
    entry_quality = calculate_entry_quality(
        price=data['price'],
        indicators=indicators,
        volume_ratio=indicators.get('volume_ratio'),
        ma_deviation=indicators.get('ma_deviation'),
    )

    return JSONResponse({
        "code": data['code'],
        "name": data['name'],
        "price": data['price'],
        "change": data['change'],
        "light": light_result['light'],
        "light_label": light_result['label'],
        "light_description": light_result['description'],
        "signal_description": light_result['signal_description'],
        "entry_quality": entry_quality,
        "indicators": {
            "rsi": indicators.get('rsi'),
            "kd_k": indicators.get('kd_k'),
            "kd_d": indicators.get('kd_d'),
            "ma5": indicators.get('ma5'),
            "ma20": indicators.get('ma20'),
            "ma60": indicators.get('ma60'),
            "ma120": indicators.get('ma120'),
            "macd_dif": indicators.get('macd_dif'),
            "macd_dea": indicators.get('macd_dea'),
            "macd_hist": indicators.get('macd_hist'),
            "bb_upper": indicators.get('bb_upper'),
            "bb_middle": indicators.get('bb_middle'),
            "bb_lower": indicators.get('bb_lower'),
            "atr": indicators.get('atr'),
            "williams_r": indicators.get('williams_r'),
            "cci": indicators.get('cci'),
            "obv": indicators.get('obv'),
            "mfi": indicators.get('mfi'),
            "ma_cross_20_60_golden": indicators.get('ma_cross_20_60_golden'),
            "ma_cross_20_60_death": indicators.get('ma_cross_20_60_death'),
            # 林穎老師方法論新增
            "volume_ratio": indicators.get('volume_ratio'),
            "ma_deviation": indicators.get('ma_deviation'),
            "trend_position": indicators.get('trend_position'),
        },
    })


# 熱門股票清單
WATCHLIST_CODES = [
    "0050", "0056", "2330", "2317", "2884", "2885",
    "2615", "2603", "2002", "1101", "1216", "1304",
    "1326", "1707", "2006", "2207", "2227", "2303",
    "2308", "2327", "2337", "2352", "2353", "2377",
]

# 產業類股對應代碼
CATEGORY_STOCKS = {
    "半導體": ["2330", "2303", "2454", "3034", "5347", "3711", "4966", "6230", "6770", "6415", "6488", "2458", "3535"],
    "IC設計": ["2337", "2377", "2408", "2449", "2451", "3535", "3686", "4953", "6139", "6525", "6666", "6741"],
    "電子代工": ["2324", "2352", "2382", "2474", "3023", "4958", "3711", "5344", "5522", "5871"],
    "顯示器": ["2382", "2468", "6176", "8299", "3481", "3692", "3653", "6180", "6451", "6120"],
    "網通": ["2345", "3231", "3680", "4904", "6426", "6558", "3701", "8046", "3596", "6768"],
    "IPC": ["3231", "6166", "8478", "6576", "5203", "6409", "2480", "5609", "5457", "6257"],
    "金融": ["2880", "2881", "2882", "2883", "2884", "2885", "2886", "2887", "2888", "2889", "5871", "5820"],
    "壽險": ["2823", "2834", "2845", "2849", "2867", "2825"],
    "鋼鐵": ["2002", "2006", "2014", "2027", "2031", "2105", "2201", "2204", "2227", "2231"],
    "塑化": ["1301", "1303", "1304", "1326", "1707", "1710", "1720", "2105", "2227", "6505"],
    "紡織": ["1402", "1414", "1417", "1434", "1442", "1451", "1470", "1476", "1477", "1504"],
    "電動車": ["2207", "2231", "3701", "3665", "4551", "4502", "6203", "1512", "6741", "6598"],
    "生技": ["1760", "1795", "3171", "4108", "4119", "4133", "4148", "4162", "4763", "6576"],
    "AI": ["2382", "2454", "3034", "3529", "4930", "6230", "6568", "6668", "6716", "6770"],
}

# 買進排序：綠燈優先 > RSI超賣 > MA偏離超跌
CATEGORY_CODES_FLAT = set(c for codes in CATEGORY_STOCKS.values() for c in codes)


@app.get("/api/category/{category}")
async def get_category_stocks(
    category: str,
    limit: int = Query(5, ge=1, le=20),
):
    """取得指定產業類股中燈號為綠燈的股票（最多 limit 檔）"""
    codes = CATEGORY_STOCKS.get(category, [])
    if not codes:
        raise HTTPException(status_code=404, detail=f"未知類別：{category}")

    loop = asyncio.get_event_loop()
    scored = []

    for code in codes:
        try:
            data = await loop.run_in_executor(None, fetch_stock_data, code)
            if not data:
                continue
            indicators = calculate_indicators(data['hist'])
            light_result = determine_light(data['price'], indicators, data['hist'])

            item = {
                "code": data['code'],
                "name": data['name'],
                "price": data['price'],
                "change": data['change'],
                "light": light_result['light'],
                "light_label": light_result['label'],
                "indicators": {
                    "rsi": indicators.get('rsi'),
                    "kd_k": indicators.get('kd_k'),
                },
            }

            # 評分：綠燈最高分，其次依 RSI 超賣程度
            score = 0
            if light_result['light'] == 'green':
                score = 100 + (30 - (indicators.get('rsi') or 50))  # RSI 越低分越高
            elif light_result['light'] == 'yellow':
                rsi = indicators.get('rsi')
                if rsi and rsi < 40:
                    score = 50 + (40 - rsi)
                else:
                    score = 30
            else:
                score = -100 + (indicators.get('kd_k') or 50)

            scored.append((score, item))
        except Exception:
            continue

    # 分數高的排前面（綠燈優先）
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [item for _, item in scored[:limit]]

    return JSONResponse({
        "category": category,
        "stocks": results,
        "count": len(results),
    })


@app.get("/api/watchlist")
async def get_watchlist(
    light: str = Query(None, description="燈號過濾：green/yellow/red")
):
    """取得股票清單，可依燈號篩選"""
    results = []
    
    loop = asyncio.get_event_loop()
    
    for code in WATCHLIST_CODES:
        try:
            data = await loop.run_in_executor(None, fetch_stock_data, code)
            if not data:
                continue
            
            indicators = calculate_indicators(data['hist'])
            light_result = determine_light(data['price'], indicators, data['hist'])
            
            item = {
                "code": data['code'],
                "name": data['name'],
                "price": data['price'],
                "change": data['change'],
                "light": light_result['light'],
                "light_label": light_result['label'],
                "indicators": {
                    "rsi": indicators.get('rsi'),
                    "kd_k": indicators.get('kd_k'),
                    "ma20": indicators.get('ma20'),
                },
            }
            
            # Filter by light
            if light and light_result['light'] != light:
                continue
            
            results.append(item)
        except Exception:
            continue
    
    return JSONResponse({"stocks": results, "count": len(results)})