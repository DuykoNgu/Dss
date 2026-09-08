# 📈 DSS — Hệ thống Hỗ trợ Quyết định Mua Bán Cổ Phiếu (VN30)

> **Decision Support System** kết hợp Phân tích Kỹ thuật (Technical Analysis) và Học máy (Machine Learning) để đưa ra khuyến nghị giao dịch ngắn hạn (T+5) cho rổ cổ phiếu VN30.

---

## 🎯 Giới hạn Phạm vi (Scope)

| Tiêu chí | Giới hạn |
|----------|----------|
| **Dữ liệu** | Chỉ 30 mã thuộc rổ **VN30** (tự động quét từ `vnstock`) |
| **Thời gian** | Giao dịch ngắn hạn **T+5** (5 phiên giao dịch) |
| **Phương pháp** | Thuần **Phân tích kỹ thuật + Machine Learning** — Không sử dụng Phân tích cơ bản |
| **Dữ liệu lịch sử** | 3 năm gần nhất (~750 phiên/mã) |

---

## 🧠 Kiến trúc Hệ thống

```
[Tự động quét rổ VN30 từ vnstock]
         │
         ▼
[Phase 1: Lấy Data OHLCV] ──> 30 mã × 750 phiên + VNINDEX
         │
         ▼
[Phase 2: Data Cleaning] ───> Xử lý Missing, Outlier (>6.8%)
         │
         ▼
[Phase 3: Indicators] ─────> SMA, EMA, MACD, RSI, BB, ATR, OBV (20+ chỉ báo)
         │
         ▼
[Phase 4: Features] ───────> Crossover, Positions, Returns + Gán nhãn T+5
         │
         ├──► [Phase 5: ML] ───> Random Forest + XGBoost ──► ML Score (40%)
         │                                                        │
         └──► [Phase 6: Rules] ──> Chấm điểm 5 nhóm TA ─► Rule Score (60%)
                                                                  │
                                                        Total = 60% Rule + 40% ML
                                                                  │
                                                     ┌────────────┴────────────┐
                                                     ▼                         ▼
                                            [Bảng Khuyến Nghị]     [Phase 7: Backtest]
                                            [🟢🟡⚪🟠🔴]          [So sánh vs Buy&Hold]
```

### Thuật toán Machine Learning

| Thuật toán | Vai trò | Cơ chế |
|-----------|---------|--------|
| **Random Forest** | "Phòng thủ" — Ổn định, chống nhiễu | 200 cây quyết định song song (Bagging), bỏ phiếu đa số |
| **XGBoost** | "Tấn công" — Nhạy bén, bắt pattern tinh vi | 300 cây tuần tự (Boosting), mỗi cây sửa lỗi cây trước |
| **Ensemble** | Trung bình xác suất 2 model | `ML Score = P(MUA) × 100` |
| **Rule-Based** | "Trọng tài" — Kiểm tra logic TA cơ bản | Chấm điểm 5 nhóm: Trend 30%, Momentum 25%, Volume 20%, Volatility 15%, Market 10% |

### Tín hiệu đầu ra

| Signal | Điểm | Ý nghĩa |
|--------|------|---------|
| 🟢 **MUA MẠNH** | ≥ 75 | Cả ML và Rules đều tích cực mạnh |
| 🟡 **MUA** | 60–74 | Phần lớn chỉ báo tích cực |
| ⚪ **GIỮ** | 40–59 | Tín hiệu trái chiều, chưa rõ xu hướng |
| 🟠 **BÁN** | 25–39 | Phần lớn chỉ báo tiêu cực |
| 🔴 **BÁN MẠNH** | < 25 | Cả ML và Rules đều cảnh báo rủi ro |

---

## 📂 Cấu trúc Dự án

```
DSS/
├── config.py                    # Tham số hệ thống (thời gian, ngưỡng, trọng số)
├── requirements.txt             # Thư viện Python
├── run.sh                       # Script chạy nhanh (setup + fetch + pipeline)
├── main.py                      # Chạy khuyến nghị hôm nay
├── backtest_runner.py           # Kiểm chứng lợi nhuận lịch sử
│
├── src/
│   ├── data_fetcher.py          # Phase 1: Quét VN30, gọi vnstock API
│   ├── data_cleaner.py          # Phase 2: Fill missing, flag outlier
│   ├── indicators.py            # Phase 3: Tính 20+ chỉ báo kỹ thuật
│   ├── features.py              # Phase 4: Tạo features + gán nhãn T+5
│   ├── ml_models.py             # Phase 5: Train RF + XGBoost
│   ├── scoring.py               # Phase 6A: Chấm điểm Rule-based
│   ├── decision.py              # Phase 6B: Tổng hợp + in bảng Terminal
│   └── backtester.py            # Phase 7: Giả lập giao dịch
│
├── data/                        # Dữ liệu CSV (gitignored)
│   ├── symbols.json
│   ├── stocks/*.csv             # 30 file OHLCV
│   └── index/VNINDEX.csv
│
├── models/                      # Model .pkl đã train (gitignored)
│
└── docs/
    ├── implementation_plan.md   # Kế hoạch 7 Phase chi tiết
    ├── dss_algorithm_analysis.md # Phân tích thuật toán (DT, RF, XGB, Ensemble)
    └── dss_labels_reference.md  # Tra cứu toàn bộ labels & ý nghĩa
```

---

## 🚀 Cài đặt & Chạy

### Cách 1: Dùng `run.sh` (Khuyến nghị)

```bash
# Lần đầu: setup môi trường + cài thư viện + tải data VN30
./run.sh setup

# Chạy khuyến nghị hôm nay
./run.sh dss

# Chạy backtest 6 tháng
./run.sh backtest

# Xem tất cả lệnh
./run.sh help
```

### Cách 2: Chạy thủ công

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Tải dữ liệu VN30
python src/data_fetcher.py

# 3. Chạy khuyến nghị
python main.py

# 4. Chạy backtest
python backtest_runner.py
```

---

## ⚙️ Cấu hình (`config.py`)

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `LOOKBACK_YEARS` | `3` | Số năm dữ liệu lịch sử |
| `ML_FORWARD_DAYS` | `5` | Dự đoán lợi nhuận T+5 |
| `ML_PROFIT_THRESHOLD` | `0.03` | Ngưỡng ±3% để gán nhãn MUA/BÁN |
| `TEST_SIZE_RATIO` | `0.2` | 20% data cuối làm tập Test (Walk-forward) |
| `WEIGHT_RULE_BASED` | `0.60` | Trọng số Rule-based trong tổng điểm |
| `WEIGHT_ML_MODEL` | `0.40` | Trọng số ML trong tổng điểm |
| `BACKTEST_MONTHS` | `6` | Khoảng thời gian backtest |

---

## 📖 Tài liệu Chi tiết

| Tài liệu | Nội dung |
|-----------|---------|
| [DSS_FULL_CODE_GUIDE.md](DSS_FULL_CODE_GUIDE.md) | Mã nguồn đầy đủ 7 Phase kèm giải thích |
| [implementation_plan.md](implementation_plan.md) | Kế hoạch triển khai 7 Phase + sơ đồ Mermaid |
| [dss_algorithm_analysis.md](dss_algorithm_analysis.md) | Phân tích Decision Tree, Random Forest, XGBoost, Ensemble, Walk-Forward |
| [dss_labels_reference.md](dss_labels_reference.md) | Tra cứu toàn bộ labels, ngưỡng điểm, ý nghĩa tài chính |

---

## 📊 Output mẫu

```
╔══════════════════════════════════════════════════════════════╗
║        DSS Khuyến Nghị Cổ Phiếu VN30 — 2026-09-08          ║
╠══════╦════════╦═══════╦════════╦════════════════════════════╣
║  Mã  ║ Signal ║ Điểm  ║  Giá   ║ Lý do chính              ║
╠══════╬════════╬═══════╬════════╬════════════════════════════╣
║ FPT  ║ 🟢 MUA ║ 72/100║ 72,200 ║ RSI phục hồi, MACD+      ║
║ TCB  ║ ⚪ GIỮ ║ 51/100║ 48,600 ║ Sideway, chờ breakout     ║
║ HPG  ║ 🟠 BÁN ║ 32/100║ 26,100 ║ Death cross, volume giảm  ║
╚══════╩════════╩═══════╩════════╩════════════════════════════╝
```

---

## ⚠️ Disclaimer

Đây là **Hệ thống Hỗ trợ Quyết định (DSS)**, không phải Bot Giao dịch tự động.
Quyết định cuối cùng và quản trị vốn (cắt lỗ, đi lệnh) vẫn thuộc về người dùng.
Kết quả trong quá khứ không đảm bảo lợi nhuận trong tương lai.
