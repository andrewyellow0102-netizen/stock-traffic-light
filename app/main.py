from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import asyncio

from app.services.stock_service import fetch_stock_data, calculate_indicators, determine_light

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
    
    return JSONResponse({
        "code": data['code'],
        "name": data['name'],
        "price": data['price'],
        "change": data['change'],
        "light": light_result['light'],
        "light_label": light_result['label'],
        "light_description": light_result['description'],
        "signal_description": light_result['signal_description'],
        "indicators": {
            "rsi": indicators.get('rsi'),
            "kd_k": indicators.get('kd_k'),
            "kd_d": indicators.get('kd_d'),
            "ma5": indicators.get('ma5'),
            "ma20": indicators.get('ma20'),
            "ma60": indicators.get('ma60'),
        },
    })


# 熱門股票清單
WATCHLIST_CODES = [
    "0050", "0056", "2330", "2317", "2884", "2885",
    "2615", "2603", "2002", "1101", "1216", "1304",
    "1326", "1707", "2006", "2207", "2227", "2303",
    "2308", "2327", "2337", "2352", "2353", "2377",
]


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