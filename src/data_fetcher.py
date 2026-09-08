import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pandas as pd
from vnstock import Market, Reference
import config

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def fetch_vn30_symbols() -> list[str]:
    ref = Reference()
    symbols = ref.equity.list_by_group("VN30").tolist()
    return symbols


def fetch_stock_ohlcv(symbol: str,
                      start_date: str = config.START_DATE,
                      end_date: str = config.END_DATE) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    for attempt in range(5):
        try:
            df = Market().equity(symbol=symbol).ohlcv(
                start=start_date, end=end_date, count=config.DATA_COUNT
            )
            if df is None or df.empty:
                return pd.DataFrame()
            return df.sort_values("time").reset_index(drop=True)
        except Exception:
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
            else:
                return pd.DataFrame()
    return pd.DataFrame()


def fetch_market_index(index_code: str = "VNINDEX",
                       start_date: str = config.START_DATE,
                       end_date: str = config.END_DATE) -> pd.DataFrame:
    for attempt in range(5):
        try:
            df = Market().index(symbol=index_code).ohlcv(
                start=start_date, end=end_date, count=config.DATA_COUNT
            )
            if df is None or df.empty:
                return pd.DataFrame()
            return df.sort_values("time").reset_index(drop=True)
        except Exception:
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
            else:
                return pd.DataFrame()
    return pd.DataFrame()


def save_data(symbols: list[str], df_index: pd.DataFrame) -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "stocks"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "index"), exist_ok=True)

    with open(os.path.join(DATA_DIR, "symbols.json"), "w") as f:
        json.dump(symbols, f, indent=2, ensure_ascii=False)

    df_index.to_csv(os.path.join(DATA_DIR, "index", "VNINDEX.csv"), index=False)

    summary = {"VN30_count": len(symbols), "symbols": symbols}
    for i, sym in enumerate(symbols):
        csv_path = os.path.join(DATA_DIR, "stocks", f"{sym}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            summary[sym] = {"rows": len(df), "columns": list(df.columns), "cached": True}
            print(f"  ⏭️ {sym}: {len(df)} rows (cached)")
            continue

        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = fetch_stock_ohlcv(sym)
                if not df.empty:
                    break
            except Exception:
                pass
            time.sleep(8)

        if not df.empty:
            df.to_csv(csv_path, index=False)
            summary[sym] = {"rows": len(df), "columns": list(df.columns), "cached": False}
            print(f"  ✅ {sym}: {len(df)} rows → {csv_path}")
        else:
            summary[sym] = {"rows": 0, "error": "no data"}
            print(f"  ❌ {sym}: no data")
        time.sleep(6)

    return summary


if __name__ == "__main__":
    symbols = fetch_vn30_symbols()
    print(f"\n📋 VN30: {len(symbols)} mã")

    df_index = fetch_market_index()
    print(f"📈 VNINDEX: {len(df_index)} bản ghi\n")

    summary = save_data(symbols, df_index)
    print(f"\n✅ Đã lưu vào {DATA_DIR}")
    print(f"   - symbols.json")
    print(f"   - index/VNINDEX.csv")
    print(f"   - stocks/*.csv (30 file)")