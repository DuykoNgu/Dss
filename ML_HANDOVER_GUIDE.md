# DSS ML Handover Guide

Tài liệu này mô tả quy trình Machine Learning của DSS từ dữ liệu đầu vào đến tín
hiệu cuối cùng. Mục tiêu là giúp người tiếp nhận hiểu hệ thống đang làm gì, vì
sao từng bước tồn tại, cách chạy, cách đánh giá và các điểm cần cẩn thận khi
thay đổi.

## 1. Mục tiêu và phạm vi

DSS là hệ thống hỗ trợ phân tích cổ phiếu ngắn hạn trong rổ VN30. Hệ thống sử
dụng dữ liệu OHLCV theo ngày, chỉ báo kỹ thuật, dữ liệu VNINDEX và hai mô hình
phân loại:

- Random Forest.
- XGBoost.

Mỗi phiên của một mã được phân loại thành `SELL`, `HOLD` hoặc `BUY`. ML không
trực tiếp đặt lệnh. Kết quả ML được kết hợp với điểm rule-based để tạo điểm DSS
từ `0` đến `100`, sau đó chuyển thành tín hiệu `MUA MẠNH`, `MUA`, `GIỮ`, `BÁN`
hoặc `BÁN MẠNH`.

Đây là hệ thống nghiên cứu và hỗ trợ quyết định. Kết quả backtest không đảm bảo
lợi nhuận trong tương lai.

## 2. Toàn bộ luồng xử lý

```mermaid
flowchart TD
    A[CSV cache hoặc vnstock API] --> B[Clean OHLCV và VNINDEX]
    B --> C[Tính technical indicators]
    C --> D[Tạo features quá khứ hiện tại]
    D --> E[Tạo future return và label]
    E --> F[Drop dòng thiếu feature label]
    F --> G[Train toàn bộ hoặc walk-forward split]
    G --> H[Train Random Forest]
    G --> I[Train XGBoost]
    H --> J[Predict probability]
    I --> J
    J --> K[Prior correction, ML score = 50 + 50 x P_BUY - P_SELL]
    D --> L[Rule-based score]
    K --> M[Rule 60% + ML 40%]
    L --> M
    M --> N[Tín hiệu DSS]
    N --> O[Backtest blend/rule/ml với phí thuế slippage T+2]
```

| Phase | Nội dung | Source chính |
|---|---|---|
| 1 | Fetch và cache dữ liệu | `backend/src/data/data_fetcher.py` |
| 1–4 | Nạp dữ liệu dùng chung cho mọi entrypoint | `backend/src/pipeline.py` |
| 2 | Clean dữ liệu | `backend/src/data/data_cleaner.py` |
| 3 | Technical indicators | `backend/src/features/indicators.py` |
| 4 | Feature engineering và label | `backend/src/features/features.py` |
| 5 | Train và validation ML | `backend/src/models/` |
| 6 | Rule score và ensemble decision | `backend/src/scoring/` |
| 7 | Backtest | `backend/src/backtest/backtester.py` |

## 3. Dữ liệu đầu vào

### 3.1. Dữ liệu cổ phiếu

Dữ liệu cổ phiếu được lưu tại `backend/data/stocks/<SYMBOL>.csv` với các cột bắt buộc:

```text
time, open, high, low, close, volume
```

Mặc định hệ thống lấy 8 năm (giới hạn nến ngày của vnstock bản miễn phí).
Quy tắc cache:

- Mã chưa có CSV hoặc lịch sử ngắn hơn 8 năm: tải full và ghi đè.
- Mã đã có CSV: tải chồng 10 ngày cuối rồi gộp, trùng `time` thì lấy bản mới.
- Không lưu nến của hôm nay trước 15:00 giờ Việt Nam (nến chưa chốt).
- Giá đóng cửa trùng ngày lệch > 0,5% giữa cache và API nghĩa là giá đã được
  điều chỉnh (cổ tức, chia tách): tải full lại để cả chuỗi cùng cơ sở giá.
- Mọi lần gọi API giãn cách 7 giây vì gói Guest giới hạn 20 đơn vị quota/phút
  và mỗi lần tải tốn 2 đơn vị.

### 3.2. Dữ liệu VNINDEX

File `backend/data/index/VNINDEX.csv` được chuẩn hóa thành:

```text
time, indexValue
```

Các feature thị trường được merge vào từng mã cổ phiếu theo ngày.

### 3.3. Kiểm tra dữ liệu

`clean_ohlcv_data()` thực hiện:

1. Chuẩn hóa tên cột và kiểm tra schema OHLCV.
2. Chuyển thời gian sang datetime và sort tăng dần.
3. Chuyển OHLC, volume sang numeric.
4. Chỉ forward-fill dữ liệu có giá trị quá khứ.
5. Đánh dấu `volume_missing`, `is_extreme` và `is_ohlc_invalid`.

Không được dùng backward-fill cho giá vì giá trị tương lai có thể đi ngược về
quá khứ và tạo leakage.

## 4. Technical indicators

`calculate_technical_indicators()` thêm các nhóm chỉ báo:

| Nhóm | Chỉ báo |
|---|---|
| Trend | SMA 10/20/50/200, EMA 12/26 |
| MACD | MACD, signal, histogram |
| Momentum | RSI, Stochastic K/D, Williams %R |
| Volatility | Bollinger Bands, ATR, ATR% |
| Volume | OBV, volume SMA 20, volume ratio |

Các chỉ báo được tính tuần tự theo thời gian. Dòng đầu không đủ window sẽ có
NaN và bị loại khi xây tập train. `SMA_200` khiến dữ liệu thực tế cần khoảng
200 phiên trước khi có đầy đủ feature baseline.

## 5. Feature engineering

### 5.1. Baseline feature set

19 feature được train production hiện tại:

```text
price_vs_sma50, price_vs_sma200, sma50_vs_sma200
macd_hist, macd_hist_slope
rsi, stoch_k, stoch_d, williams_r
bb_position, bb_width, atr_pct
vol_ratio, obv_slope
return_1d, return_5d, return_20d
vnindex_vs_sma50, vnindex_return_5d
```

Các feature tương đối như `price_vs_sma50` giảm phụ thuộc vào mức giá tuyệt đối
giữa các mã khác nhau.

### 5.2. Extended feature set

Extended thêm:

```text
return_3d, return_10d, return_60d, rsi_slope
volume_zscore_20, vnindex_return_20d
vnindex_volatility_20d, relative_strength_5d
```

Extended được dùng trong lệnh evaluate khi chọn `--feature-set extended`. Model
production hiện dùng `FEATURE_COLUMNS` baseline, vì vậy phải cập nhật đồng bộ
train, predict, model cache và backtest nếu muốn đưa extended vào production.

### 5.3. Nguyên tắc leakage

Feature tại thời điểm `t` chỉ được phép dùng OHLCV, indicator và VNINDEX đến
thời điểm `t` hoặc thời điểm gần nhất trong quá khứ.

Không được đưa `future_return`, `label`, giá tương lai hoặc kết quả backtest vào
feature. Khi thêm feature mới, phải kiểm tra lại nguồn dữ liệu và hướng thời gian.

## 6. Label và supervised learning

Mỗi dòng tại thời điểm `t` là một mẫu. `X_t` gồm các feature quan sát được đến
`t`; `y_t` là kết quả tương lai dùng làm label. Model học quan hệ:

```text
X_t -> P(SELL), P(HOLD), P(BUY)
```

`future_return` chỉ dùng để tạo `y_t`, không được dùng làm `X_t`.

### 6.1. Fixed label

Với `forward_days = 5`:

```text
future_return[t] = close[t+5] / close[t] - 1
```

```text
future_return >= +(3% + chi phí vòng đi-về) -> BUY
future_return <= -(3% + chi phí vòng đi-về) -> SELL
còn lại                                      -> HOLD
```

Chi phí label hiện tại là `ML_LABEL_COST_RATE = 2 * fee + tax + 2 * slippage
= 0,6%`. Mục đích là tránh gọi một giao dịch gross +3% là thành công khi lợi
nhuận ròng sau chi phí không còn đạt mục tiêu.

### 6.2. Volatility label

```text
dynamic_threshold = max(3%, 1.5 * ATR% / 100) + chi phí vòng đi-về
```

Cổ phiếu biến động mạnh cần đạt mức lợi nhuận lớn hơn mới được gán BUY hoặc SELL.

### 6.2b. Excess label (vượt VNINDEX)

```text
excess = future_return − vnindex_future_return   (cùng N phiên)
excess >= +(3% + chi phí) -> BUY
excess <= −(3% + chi phí) -> SELL
còn lại                   -> HOLD
```

Nhãn này loại phần biến động chung của thị trường, để model học chọn mã tốt
hơn mặt bằng thay vì đoán hướng thị trường. `vnindex_future_return` nhìn tương
lai nên chỉ dùng tạo nhãn, không nằm trong feature.

Mọi nhãn nhận tầm nhìn `N` qua `--horizon` (mặc định 5); purge gap của
walk-forward và backtest dùng cùng `N`.

### 6.3. Triple barrier label

Mỗi điểm vào có upper barrier, lower barrier và cửa sổ quan sát 5 phiên. Barrier
đầu tiên bị chạm quyết định label; nếu không chạm barrier nào thì HOLD. Nếu high
và low cùng chạm barrier trong cùng thời điểm mà không xác định được thứ tự trong
ngày, dùng chính sách bảo thủ và tránh giả định thứ tự.

### 6.4. Chọn label

Không chọn label chỉ dựa trên Macro F1. Cần xem đồng thời phân phối label,
BaselineHold, BaselineMomentum, precision BUY/SELL, độ ổn định qua mã/fold và
backtest sau chi phí.

`fixed` là label production vì dễ giải thích. `volatility` và `triple_barrier`
là ứng viên nghiên cứu, chưa thay thế production. Kết quả so sánh mới nhất nằm
trong README.

## 7. Chia dữ liệu theo thời gian

### 7.1. Vì sao không random shuffle

Dữ liệu tài chính là chuỗi thời gian. Random shuffle có thể đưa thông tin tương
lai vào train, làm metric cao giả tạo và không phản ánh cách model vận hành thật.
Mọi split phải giữ thứ tự thời gian.

### 7.2. Production training

`train_ml_models()` train model production trên **toàn bộ** dòng có label để
model dùng cả giai đoạn gần nhất. Khi `verbose=True`, hàm train thêm một model
phụ trên 80% thời gian đầu (bỏ 5 phiên ở ranh giới) và in metric trên 20% cuối
để tham khảo; model phụ không được lưu. Không shuffle.

### 7.3. Walk-forward validation

`walk_forward_validate()` dùng:

```text
initial train: 50% usable rows
validation mỗi fold: 10%
purge gap: 5 phiên
```

```text
[train 50%] [gap 5] [validation 10%]
[train 60%] [gap 5] [validation 10%]
[train 70%] [gap 5] [validation 10%]
```

Train set mở rộng dần và validation luôn nằm sau train. Purge gap loại bỏ các
mẫu sát ranh giới có label còn nhìn vào vùng validation.

## 8. Baseline trước khi đánh giá model

Một model chỉ có ý nghĩa khi vượt được chiến lược đơn giản.

### 8.1. BaselineHold

Luôn dự đoán HOLD. Baseline này cho biết accuracy có thể đạt chỉ bằng cách chọn
class phổ biến nhất. Nó không tạo tín hiệu hành động.

### 8.2. BaselineMomentum

```text
return_5d >= +1% -> BUY
return_5d <= -1% -> SELL
còn lại           -> HOLD
```

Đây là baseline gần với một quy tắc giao dịch thực tế hơn BaselineHold. Ensemble
cần vượt baseline này trên nhiều mã và nhiều fold.

## 9. Mô hình ML

### 9.1. Random Forest

Random Forest là ensemble của nhiều decision tree độc lập. Mỗi tree học trên một
mẫu bootstrap và một phần feature, sau đó kết quả được tổng hợp.

Ưu điểm là bắt quan hệ phi tuyến, ít nhạy với scale và có class balancing. Hạn
chế là xác suất chưa chắc đã calibrated và tree quá sâu có thể học nhiễu.

Cấu hình hiện tại:

```python
{
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
```

### 9.2. XGBoost

XGBoost xây các tree tuần tự; tree sau sửa lỗi của tree trước. Learning rate,
depth và subsampling kiểm soát mức overfit.

Cấu hình hiện tại:

```python
{
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": 42,
}
```

XGBoost mạnh trên dữ liệu bảng nhưng nhạy với hyperparameter và dễ overfit dữ
liệu tài chính nhiễu.

### 9.3. Class imbalance

HOLD thường chiếm nhiều hơn BUY và SELL. RF dùng `class_weight="balanced"`,
XGBoost dùng `sample_weight`, validation báo cáo metric riêng cho từng class và
BaselineHold được tính để phát hiện accuracy giả.

Nếu training window thiếu class, model được bỏ qua an toàn và ML score trở về
neutral thay vì crash hoặc tạo model không đầy đủ.

## 10. Training và xác suất

Các dòng thiếu feature hoặc label bị loại. Model cần tối thiểu
`MIN_TRAIN_ROWS = 100` dòng usable.

RF và XGBoost trả về `P(SELL), P(HOLD), P(BUY)`. Production lấy trung bình:

```text
P_ensemble = (P_RF + P_XGB) / 2
P_corrected(c) ∝ P_ensemble(c) × prior_train(c) / (1/3)
ML score = 50 + 50 × (P_corrected(BUY) − P_corrected(SELL))
```

Do train có class balancing, prior của train được lưu vào model và dùng để hiệu
chỉnh xác suất. ML score cùng thang với Rule score: `50` là trung tính. Không
dùng `P(BUY) × 100` vì BUY chỉ chiếm ~20% nhãn, điểm đó gần như luôn thấp và kéo
tổng điểm xuống. Nếu thiếu model hoặc feature hiện tại có NaN, ML score là `50`.
Đây là score phục vụ decision, không phải xác suất lợi nhuận đã calibration đầy đủ.

Model lưu kèm `class_priors_`, `feature_columns_` và `data_end_` (ngày cuối của
dữ liệu train). `backend/main.py` tự train lại khi model thiếu các thuộc tính này, khác
feature schema hoặc cũ hơn dữ liệu quá `MODEL_MAX_AGE_DAYS = 7` ngày.

`ML_POOLED = True` (hoặc `--pooled`) train một cặp model chung trên dữ liệu mọi
mã thay vì mỗi mã một cặp; so sánh hai cách bằng backtest.

## 11. Rule score và decision

Rule-based score bắt đầu từ `50` và cộng/trừ theo trend, SMA, golden/death
cross, MACD, RSI, Stochastic, volume, OBV, Bollinger Bands, ATR và VNINDEX.
Điểm được clip trong khoảng `0` đến `100`.

```text
total_score = rule_score * 0.60 + ml_score * 0.40
```

| Total score | Tín hiệu |
|---:|---|
| `>= 75` | MUA MẠNH |
| `60 - 74.99` | MUA |
| `40 - 59.99` | GIỮ |
| `25 - 39.99` | BÁN |
| `< 25` | BÁN MẠNH |

Rule score và ML score có thể bất đồng. Khi đó total score thường kéo về vùng
trung tính; cần xem cả hai thành phần trong output.

## 12. Validation metrics

### 12.1. Accuracy

```text
accuracy = số dự đoán đúng / tổng số mẫu
```

Chỉ dùng để tham khảo vì bị chi phối bởi class HOLD.

### 12.2. Precision và recall

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

BUY precision trả lời: trong các lần model báo BUY, bao nhiêu lần đúng BUY.
BUY recall trả lời: trong các cơ hội BUY thực tế, model bắt được bao nhiêu.
SELL có cách đọc tương tự.

### 12.3. F1 và balanced accuracy

```text
F1 = 2 * precision * recall / (precision + recall)
```

Macro F1 là trung bình F1 của SELL, HOLD và BUY. Balanced accuracy là trung bình
recall của các class. Mức balanced accuracy khoảng `0,333` trong bài toán 3
class thường gần mức ngẫu nhiên.

## 13. Backtest

Backtest nằm ở `backend/src/backtest/backtester.py`, chạy qua `backend/backtest_runner.py`:

1. `score_history` chấm Rule/ML/Total cho mọi mã, từng phiên trong 12 tháng cuối.
   Model retrain mỗi 20 phiên, mỗi lần chỉ dùng các dòng có label đã biết
   (bỏ 6 dòng cuối tính đến ngày retrain).
2. Với mỗi nguồn điểm `blend` (60/40), `rule`, `ml`:
   - `simulate_symbol`: mỗi mã giao dịch riêng với toàn bộ vốn.
   - `simulate_portfolio`: danh mục chung vốn chia 5 slot, mỗi phiên lấp slot
     trống bằng các mã điểm cao nhất (>= 60).
   - `rank_ic`: tương quan hạng theo ngày giữa điểm và lợi nhuận T+5 thực tế.
3. Tín hiệu sinh tại close ngày `i`, lệnh khớp tại open ngày `i+1`.
4. Thoát tại close khi đủ T+5, hoặc bán tại open khi điểm < 25. Lệnh bán tại
   open chỉ được phép từ phiên T+3 (cổ phiếu về tài khoản chiều T+2).
5. Tính phí mua, phí bán, thuế bán và slippage.

So sánh `blend` với `rule` và `ml` cho biết từng thành phần có thêm giá trị hay
không; `rank_ic` đo khả năng xếp hạng mã, phù hợp với cách dùng bảng khuyến nghị.

Chi phí cấu hình:

```text
fee      = 0,15% mỗi chiều
tax      = 0,10% khi bán
slippage = 0,10% mỗi chiều
```

Buy & Hold cũng được tính với chi phí để so sánh công bằng hơn.

Metric chính:

- `rank_ic`: > 0 nghĩa là mã điểm cao thực sự tăng tốt hơn mã điểm thấp.
- `portfolio_return`: lợi nhuận danh mục chung vốn.
- `dss_return`: lợi nhuận cuối kỳ của DSS trên từng mã.
- `buy_hold_return`: lợi nhuận mua và giữ.
- `vnindex_return`: biến động VNINDEX.
- `win_rate`: tỷ lệ lệnh thắng.
- `profit_factor`: tổng lãi chia tổng lỗ tuyệt đối.
- `sharpe`: lợi nhuận điều chỉnh theo biến động, annualized theo 252 phiên.
- `max_drawdown`: mức giảm lớn nhất từ đỉnh equity.
- `exposure`: tỷ lệ thời gian đang giữ vị thế.

Không kết luận model tốt nếu chỉ nhìn một mã, một giai đoạn hoặc một trade có
lợi nhuận rất lớn.

## 14. Cấu trúc report

| File | Mục đích |
|---|---|
| `label_distribution.csv` | Phân phối label và số dòng trainable |
| `walk_forward_metrics.csv` | Metric phân loại theo fold và aggregate |
| `confusion_matrix.csv` | Chi tiết actual/predicted theo class |
| `backtest_<cấu hình>/backtest_summary.csv` | So sánh blend/rule/ml: rank IC, từng mã, danh mục |
| `backtest_<cấu hình>/backtest_ic_by_year.csv` | Rank IC và lợi nhuận danh mục theo năm |
| `backtest_<cấu hình>/backtest_symbols.csv` | Hiệu quả theo mã và nguồn điểm |
| `backtest_<cấu hình>/backtest_trades.csv` | Chi tiết từng giao dịch (từng mã và danh mục) |
| `backtest_<cấu hình>/backtest_scores.csv` | Điểm Rule/ML/Total từng phiên dùng cho backtest |
| `tuning_results.csv` | So sánh candidate hyperparameter |

Chi tiết schema nằm trong [REPORT_METRICS.md](REPORT_METRICS.md).

## 15. Quy trình vận hành

```bash
# Cài môi trường
./run.sh setup

# Cập nhật dữ liệu
./run.sh fetch

# Chạy khuyến nghị offline
./run.sh dss --no-fetch --symbols FPT,ACB

# Train lại model
./run.sh dss --no-fetch --symbols FPT,ACB --retrain

# Test
./run.sh test
./run.sh smoke

# Đánh giá label/model
./run.sh evaluate --label-strategy fixed --feature-set baseline
./run.sh evaluate --label-strategy volatility --feature-set baseline
./run.sh evaluate --label-strategy triple_barrier --feature-set baseline

# Backtest (model từng mã, rồi model chung)
./run.sh backtest --months 12 --retrain-every 20
./run.sh backtest --pooled
```

## 16. Khi thay đổi hệ thống

### Thay đổi label

Phải chạy lại label distribution, BaselineHold, BaselineMomentum, RF/XGB/Ensemble
walk-forward và backtest cùng phí. Không thay label rồi giữ nguyên kết luận của
model cũ.

### Thay đổi feature

Phải kiểm tra feature chỉ dùng quá khứ, không tạo NaN ngoài dự kiến, được dùng
giống nhau ở train và predict, không dùng nhầm model cache cũ và có vượt baseline.

### Thay đổi model

Phải ghi lại hyperparameter, random seed, feature set, label strategy, khoảng
thời gian dữ liệu, kết quả từng fold và backtest sau chi phí. Không chọn model chỉ
vì accuracy cao hoặc vì một mã có kết quả nổi bật.

## 17. Checklist bàn giao

- [ ] `./run.sh test` đạt.
- [ ] `python3 -m compileall -q backend` đạt.
- [ ] `bash -n run.sh` đạt.
- [ ] Smoke test offline đạt.
- [ ] Dữ liệu đúng thời gian và không trùng phiên.
- [ ] Label distribution không lệch bất thường.
- [ ] Model được so sánh với baseline trên nhiều fold và nhiều mã.
- [ ] Backtest có phí, thuế và slippage.
- [ ] Report ghi rõ label, feature set và thời gian đánh giá.
- [ ] Không có leakage từ `future_return`, `label` hoặc dữ liệu tương lai.
- [ ] Model artifact và source code dùng cùng feature schema.
- [ ] Thay đổi đã được ghi vào Git.

## 18. Giới hạn hiện tại

Hệ thống chỉ dùng OHLCV và VNINDEX. Chưa có tin tức, sentiment, báo cáo tài
chính, position sizing theo rủi ro hoặc probability calibration đầy đủ.
Thành phần VN30 theo từng kỳ chỉ có từ 08/2020 (`backend/reference/vn30_changes.csv`)
và mới được dùng trong backtest `--universe history`; `evaluate` và `backend/main.py`
vẫn dùng rổ hiện tại. Backtest danh mục chia vốn đều, chưa tính thanh khoản.

Kết quả nên được dùng làm baseline nghiên cứu. Bất kỳ thay đổi nào nhằm cải thiện
Macro F1 vẫn phải được xác nhận bằng backtest net return, drawdown, turnover và
độ ổn định qua nhiều giai đoạn.

## 19. Source of truth

Khi tài liệu khác với code, ưu tiên kiểm tra theo thứ tự:

1. `backend/config.py` cho tham số.
2. `backend/src/pipeline.py` cho thứ tự nạp dữ liệu; `backend/src/features/features.py` cho label và feature.
3. `backend/src/models/ml_models.py` cho production training/predict.
4. `backend/src/models/validation.py` cho walk-forward và baseline.
5. `backend/src/backtest/backtester.py` cho execution và chi phí.
6. `REPORT_METRICS.md` cho schema report.

Các file report chỉ là snapshot của một lần chạy, không phải nguồn định nghĩa
logic.
