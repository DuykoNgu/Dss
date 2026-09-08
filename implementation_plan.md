# DSS Mua Bán Cổ Phiếu — Implementation Plan (Scope VN30)

Chia thành **7 phase nhỏ**, mỗi phase có deliverable rõ ràng, chạy được độc lập. 
**Giới hạn (Scope):** Tự động lấy **VN30** (`vnstock`), Giao dịch **Ngắn hạn (T+5)**, Tích hợp **Backtest**.

---

## 1. Kiến Trúc Tổng Quan Toàn Hệ Thống

```mermaid
flowchart TD
    START["▶ Chạy DSS"] --> P1

    P1["📡 Phase 1: Lấy Data<br>(Tự động quét rổ VN30)"] --> P2
    P2["🧹 Phase 2: Cleaning<br>(Xử lý Missing, Outliers)"] --> P3
    P3["📐 Phase 3: Indicators<br>(Trend, Momentum, Volatility)"] --> P4
    P4["🔧 Phase 4: Features<br>(Kéo VNINDEX, Gán nhãn MUA/BÁN)"] --> P5

    P5["🤖 Phase 5: Machine Learning<br>(Random Forest + XGBoost)"] --> P6
    P4 -.->|"Dữ liệu thuần TA"| P6

    P6["📢 Phase 6: Scoring Engine<br>(60% Rules + 40% ML)"] --> P7
    
    P7["📊 Phase 7: Backtesting<br>(Đánh giá Lãi/Lỗ lịch sử)"] --> OUTPUT

    OUTPUT["Xuất báo cáo Terminal<br>Bảng Khuyến Nghị & Hiệu Suất"]
```

---

## 2. Chi tiết luồng xử lý từng Phase

### Phase 2: Data Cleaning (Làm sạch dữ liệu)

```mermaid
flowchart TD
    RAW["Raw Data từ vnstock"] --> CHECK{"Kiểm tra"}

    CHECK --> M["Missing Values?"]
    CHECK --> O["Outliers?"]
    CHECK --> T["Kiểu dữ liệu?"]

    M -->|"Giá bị null"| M1["Forward fill<br>(dùng giá phiên trước)"]
    M -->|"Đầu mút bị null"| M2["Backward fill"]

    O -->|"Giá biến động > ±6.8%"| O1["Đánh flag cảnh báo<br>(Trần/Sàn HOSE)"]

    T --> T1["close, open... → float"]
    T --> T2["tradingDate → datetime"]
    T --> T3["Sắp xếp theo ngày tăng dần"]

    M1 --> CLEAN["✅ Clean DataFrame"]
    M2 --> CLEAN
    O1 --> CLEAN
    T1 --> CLEAN
    T2 --> CLEAN
    T3 --> CLEAN
```

### Phase 3: Technical Indicators (Tính chỉ báo)

```mermaid
flowchart TD
    CLEAN["Clean Data (OHLCV)"] --> TREND & MOMENTUM & VOLATILITY & VOLUME_IND

    subgraph TREND["📈 Xu Hướng (Trend)"]
        T1["SMA (10, 20, 50, 200)"]
        T2["EMA (12, 26)"]
        T3["MACD (line, signal, hist)"]
    end

    subgraph MOMENTUM["⚡ Động Lượng (Momentum)"]
        M1["RSI (14)"]
        M2["Stochastic (%K, %D)"]
        M3["Williams %R"]
    end

    subgraph VOLATILITY["🌊 Biến Động (Volatility)"]
        V1["Bollinger Bands (upper, mid, lower)"]
        V2["ATR (14)"]
    end

    subgraph VOLUME_IND["📊 Khối Lượng (Volume)"]
        VL1["OBV"]
        VL2["Volume Ratio (vol / avg20)"]
    end
```

### Phase 4: Feature Engineering & Labeling (Gán nhãn T+5)

```mermaid
flowchart LR
    TODAY["Giá đóng cửa hôm nay"] --> FUTURE["Giá sau 5 phiên (T+5)"]
    FUTURE --> CALC["Tính % thay đổi (Lợi nhuận)"]

    CALC -->|"> +3%"| BUY["Label = 1 (MUA)"]
    CALC -->|"-3% đến +3%"| HOLD["Label = 0 (GIỮ)"]
    CALC -->|"< -3%"| SELL["Label = -1 (BÁN)"]
```

### Phase 5: Machine Learning (Huấn luyện & Walk-Forward)

```mermaid
flowchart LR
    subgraph DATA["Dữ liệu 3 năm (~750 phiên)"]
        direction LR
        TRAIN["🟦 Train (80%)<br>Dữ liệu cũ"] --> TEST["🟧 Test (20%)<br>Dữ liệu mới nhất"]
    end
    
    TRAIN -->|"fit()"| RF["Random Forest<br>(Chống nhiễu)"]
    TRAIN -->|"fit()"| XGB["XGBoost<br>(Bắt trend tinh vi)"]
    
    TEST -->|"predict()"| EVAL["Đánh giá Accuracy, F1<br>Feature Importance"]
    
    RF --> ENSEMBLE["Ensemble<br>Trung bình xác suất"]
    XGB --> ENSEMBLE
```

### Phase 6: Scoring Engine (Bộ Não Ra Quyết Định)

```mermaid
flowchart TD
    subgraph SCORING["Tính Điểm Phiên Hiện Tại"]
        S1["Xu Hướng (30%)<br>SMA, MACD"]
        S2["Động Lượng (25%)<br>RSI, Stochastic"]
        S3["Khối Lượng (20%)<br>OBV, Vol Ratio"]
        S4["Biến Động (15%)<br>BB position, ATR"]
        S5["Thị Trường VNINDEX (10%)<br>Trend chung"]
    end

    S1 & S2 & S3 & S4 & S5 --> RULE["📏 Rule-based Score (60%)"]
    
    ML_MODELS["🤖 Mô hình ML (Phase 5)"] --> ML["P(MUA) x 100<br>ML Score (40%)"]

    RULE --> TOTAL["⚖️ Tổng Điểm = (Rule × 0.6) + (ML × 0.4)"]
    ML --> TOTAL

    TOTAL -->|"≥ 75"| A["🟢 MUA MẠNH"]
    TOTAL -->|"60-74"| B["🟡 MUA"]
    TOTAL -->|"40-59"| C["⚪ GIỮ"]
    TOTAL -->|"25-39"| D["🟠 BÁN"]
    TOTAL -->|"< 25"| E["🔴 BÁN MẠNH"]
```

---

## 3. Tổng hợp Files & Deliverables

| Phase | File thực thi | Chức năng chính |
|-------|---------------|-----------------|
| **Phase 1** | `src/data_fetcher.py` | Quét tự động danh sách VN30 mới nhất, gọi API vnstock. |
| **Phase 2** | `src/data_cleaner.py` | Lọc nhiễu, điền missing values, flag outlier trần/sàn. |
| **Phase 3** | `src/indicators.py` | Chạy thư viện `ta` tính toán ~25 cột chỉ báo. |
| **Phase 4** | `src/features.py` | Chuyển đổi thành giá trị tương đối, merge VNINDEX, gán nhãn T+5. |
| **Phase 5** | `src/ml_models.py` | Train Random Forest & XGBoost, lưu 60 model file (30 mã x 2). |
| **Phase 6** | `src/scoring.py`, `src/decision.py` | Cân quyền trọng số 60/40, in bảng màu Terminal. |
| **Phase 7** | `src/backtester.py`, `backtest_runner.py` | Giả lập giao dịch 6 tháng, so sánh PnL với Buy & Hold. |
| **Main** | `config.py`, `main.py` | Tùy chỉnh tham số và điểm neo chạy toàn bộ dự án. |

---

## User Review Required

> [!IMPORTANT]
> Toàn bộ các sơ đồ trực quan (Kiến trúc, Cleaning, Cấu trúc Chỉ báo, Labeling, ML Walk-forward, Scoring) đã được phục hồi đầy đủ. Bạn xác nhận plan này đã trực quan và chuẩn xác để đưa vào báo cáo đồ án chưa?
