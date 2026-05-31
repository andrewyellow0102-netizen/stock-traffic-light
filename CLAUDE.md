# Stock Traffic Light — AI Agent 開發說明書

## 專案背景
台股紅綠燈系統：輸入股號查燈號 + 搜尋最佳燈號股票。使用 FastAPI + yfinance + pandas + Chart.js。

## 技術約定
- Python 3.12（venv 用 `uv venv --python /usr/bin/python3.12`）
- 啟動：`uv run uvicorn app.main:app --host 0.0.0.0 --port 8000`
- yfinance 資料格式：`yfinance 1.4.0`，`range=` → `period=`，`interval='1d'`
- 垂直切片：每個 Slice 完整從 API 到 UI，可獨立運作

## 目錄結構
```
app/
  main.py          # FastAPI 路由
  templates/
    index.html     # 主頁（Jinja2 + Tailwind CDN）
  services/
    stock_service.py   # yfinance 查詢 + 燈號計算
    indicator.py       # RSI / KD / 均線計算
  models/
    stock.py       # Pydantic models
```

## 紅綠燈判斷規則（請嚴格遵守）
```
🟢 綠燈（買進）：
  - 條件A：股價比月線(20日)低於-10%以上（超跌）
  - 條件B：RSI(14) < 30
  - 條件C：KD黃金交叉 且 K值 < 30
  - 顯示原因文字

🟡 黃燈（觀望）：不符合綠也不符合紅

🔴 紅燈（賣出）：
  - 條件A：股價比月線高於+15%以上（超漲）
  - 條件B：RSI(14) > 70
  - 條件C：KD死亡交叉 且 K值 > 70
  - 顯示原因文字
```

## API 端點
- `GET /` → 首頁
- `GET /api/stock?code=2330` → 個股燈號詳情
- `GET /api/watchlist?light=green` → 燈號股票列表

## 關鍵依賴
```
fastapi
uvicorn
jinja2
yfinance
pandas
httpx
ta-lib  # 或純 Python RSI/KD 實作
```

## 熱門股票預設清單（for 燈號列表）
0050, 0056, 2330, 2317, 2884, 2885, 2615, 2603, 2002, 1101

## 開發流程
1. 每個 Slice 是獨立 PR，需要人類 merge 才能前進
2. 合併後進入下一個 Slice
3. 開發模式：Agent-first，人類純管理