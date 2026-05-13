# 金融商品回測分析工具

本專案是一個本機 Streamlit 回測分析工具，支援 Yahoo Finance 代號與 CSV 上傳，並以互動式圖表呈現策略績效。

## 功能

- 支援股票、ETF、加密貨幣資料，可選 5 分鐘、15 分鐘、30 分鐘、1 小時、4 小時、日線、週線、月線等週期。
- 資料來源可選 Yahoo Finance 或 CSV。
- 內建買入持有、均線交叉、RSI 反轉、布林通道策略。
- 顯示 K 線買賣點、權益曲線、回撤、月報酬熱力圖、績效指標與交易紀錄。

## 安裝與啟動

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

若不使用虛擬環境，也可以直接執行：

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

若建立虛擬環境時卡在 `ensurepip`，可改用每位使用者安裝：

```powershell
py -m pip install --user -r requirements.txt
py -m streamlit run app.py
```

## CSV 格式

CSV 需要包含以下欄位。CSV 模式會將資料視為介面上選定的 K 線週期：

```csv
Date,Open,High,Low,Close,Volume
2024-01-02,100,102,99,101,1000000
```

`Adj Close` 可選。日期會自動解析並排序。

## 注意

Yahoo Finance 資料僅供個人研究與教育用途。本工具不提供投資建議，回測結果不代表未來績效。
