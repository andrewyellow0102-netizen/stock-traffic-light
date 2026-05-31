from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)