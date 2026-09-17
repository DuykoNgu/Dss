# Hướng dẫn đọc các file report của DSS

Tài liệu này mô tả các CSV được tạo bởi pipeline đánh giá ML và backtest. Các
file trong `reports/` là artifact sinh ra ở máy local và có thể thay đổi khi dữ
liệu, label hoặc cấu hình model thay đổi.

## 1. Tạo report

Đánh giá walk-forward cho một hoặc nhiều mã:

```bash
./run.sh evaluate --symbols FPT,ACB
./run.sh evaluate --label-strategy fixed --feature-set baseline
./run.sh evaluate --label-strategy volatility --feature-set baseline
./run.sh evaluate --label-strategy triple_barrier --feature-set baseline
```

Các thư mục label chính là:

```text
reports/fixed_baseline/
reports/volatility_baseline/
reports/triple_barrier_baseline/
```

Chạy backtest:

```bash
./run.sh backtest --months 12 --retrain-every 20   # model từng mã
./run.sh backtest --pooled                          # model chung mọi mã
```

Lệnh này tạo 5 file trong
`reports/backtest_<pooled|per_symbol>_<nhãn>_h<N>_<exit>_<universe>_<tháng>m/`:
`backtest_summary.csv`, `backtest_ic_by_year.csv`, `backtest_symbols.csv`,
`backtest_trades.csv`, `backtest_scores.csv`.

Walk-forward với tầm nhìn khác 5 phiên ghi vào `reports/<nhãn>_<feature_set>_h<N>/`.

## 2. Quy ước chung

Label được ánh xạ trong report ML như sau:

| Giá trị report | Ý nghĩa |
|---:|---|
| `0` | SELL |
| `1` | HOLD |
| `2` | BUY |

Trong dữ liệu feature gốc, label vẫn dùng mapping `-1 = SELL`, `0 = HOLD`,
`1 = BUY`; chỉ report validation mới chuyển sang `0/1/2` để dùng với scikit-learn.

Các model xuất hiện trong report:

| Tên | Ý nghĩa |
|---|---|
| `RF` | Random Forest |
| `XGB` | XGBoost |
| `Ensemble50` | Trung bình xác suất của RF và XGBoost |
| `BaselineHold` | Luôn dự đoán HOLD |
| `BaselineMomentum` | BUY khi `return_5d >= 1%`, SELL khi `return_5d <= -1%`, còn lại HOLD |

Model chỉ được xem là có tiến bộ khi vượt baseline trên nhiều mã và nhiều fold.

## 3. `label_distribution.csv`

File này cho biết dữ liệu label của từng mã trước khi train.

| Cột | Ý nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu |
| `rows` | Số dòng sau khi làm sạch, tính indicator và tạo label |
| `trainable_rows` | Số dòng còn đủ feature và label để train |
| `sell` | Số mẫu SELL |
| `hold` | Số mẫu HOLD |
| `buy` | Số mẫu BUY |

Nên kiểm tra file này trước khi đọc metric model. Nếu `hold` chiếm gần như toàn
bộ dữ liệu, accuracy cao có thể chỉ phản ánh việc đoán HOLD. Nếu một mã có quá
ít `trainable_rows` hoặc thiếu hẳn một class, kết quả model của mã đó không ổn
định và có thể không tạo được fold.

## 4. `walk_forward_metrics.csv`

Đây là report chính để đánh giá phân loại theo thời gian. Mỗi dòng là một model
trên một fold hoặc dòng aggregate.

### Metadata của fold

| Cột | Ý nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu |
| `label_strategy` | `fixed`, `volatility` hoặc `triple_barrier` |
| `feature_set` | `baseline` hoặc `extended` |
| `fold` | Số thứ tự fold; `aggregate` là gộp các fold của mã |
| `model` | RF, XGB, Ensemble hoặc baseline |

Fold được tạo theo expanding walk-forward: train trên quá khứ, bỏ qua purge gap
T+5, rồi đánh giá trên đoạn thời gian kế tiếp. Không shuffle dữ liệu.

### Metric tổng quát

| Cột | Công thức / ý nghĩa |
|---|---|
| `accuracy` | Tỷ lệ dự đoán đúng trên toàn bộ mẫu |
| `balanced_accuracy` | Trung bình recall của các class xuất hiện trong tập thực tế |
| `macro_f1` | Trung bình F1 của SELL, HOLD và BUY; mỗi class có trọng số ngang nhau |

`balanced_accuracy` và `macro_f1` phù hợp hơn accuracy khi HOLD chiếm đa số.
Mức balanced accuracy quanh `0,333` trong bài toán 3 class thường gần mức ngẫu
nhiên.

### Metric theo class

Các cột có tiền tố `sell_`, `hold_` và `buy_` có cùng cách hiểu:

| Hậu tố | Ý nghĩa |
|---|---|
| `precision` | Trong các lần model dự đoán class đó, tỷ lệ dự đoán đúng |
| `recall` | Trong các mẫu thực sự thuộc class đó, tỷ lệ model bắt được |
| `f1` | Trung bình điều hòa của precision và recall |

Ví dụ:

```text
buy_precision = số BUY dự đoán đúng / tổng số BUY model dự đoán
buy_recall    = số BUY dự đoán đúng / tổng số BUY thực tế
```

Khi đọc tín hiệu giao dịch, `buy_precision` và `sell_precision` cho biết chất
lượng tín hiệu hành động. `buy_recall` và `sell_recall` cho biết model bỏ sót
bao nhiêu cơ hội hoặc rủi ro.

## 5. `confusion_matrix.csv`

File này lưu ma trận nhầm lẫn của các dòng `aggregate`.

| Cột | Ý nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu |
| `model` | Model hoặc baseline |
| `actual` | Class thực tế: `0 SELL`, `1 HOLD`, `2 BUY` |
| `predicted` | Class được dự đoán theo cùng mapping |
| `count` | Số mẫu thuộc cặp actual/predicted |

Cách đọc một dòng:

```text
actual=2, predicted=1, count=30
```

Nghĩa là có 30 mẫu BUY thực tế nhưng model dự đoán thành HOLD. Đây là lỗi bỏ
sót cơ hội BUY. Ngược lại, `actual=1, predicted=2` là tín hiệu BUY phát ra khi
thực tế chỉ là HOLD.

Khi đánh giá model, nên chú ý các ô:

- SELL bị dự đoán thành BUY: rủi ro nguy hiểm nhất.
- BUY bị dự đoán thành HOLD: bỏ lỡ cơ hội.
- HOLD bị dự đoán thành BUY/SELL: tạo giao dịch thừa.

## 6. `backtest_summary.csv`

Mỗi dòng là một nguồn điểm. Đây là file đọc đầu tiên sau khi backtest.

| Cột | Ý nghĩa |
|---|---|
| `mode` | `blend` (Rule 60% + ML 40%), `rule` (chỉ Rule Score), `ml` (chỉ ML Score) |
| `run` | Tên cấu hình, trùng tên thư mục report |
| `rank_ic` | Trung bình theo ngày của tương quan hạng (Spearman) giữa điểm và lợi nhuận T+N thực tế giữa các mã thuộc rổ |
| `rank_ic_positive_years` | Số năm có Rank IC trung bình > 0 trên tổng số năm |
| `symbol_avg_trade_return` | Lợi nhuận ròng trung bình mỗi lệnh khi giao dịch từng mã |
| `symbol_avg_return` | Lợi nhuận trung bình khi mỗi mã giao dịch riêng |
| `symbol_beat_buy_hold` | Số mã có lợi nhuận DSS lớn hơn Buy & Hold của chính mã đó |
| `symbols` | Số mã được backtest |
| `symbol_trades`, `symbol_win_rate` | Tổng số lệnh và tỷ lệ thắng khi giao dịch từng mã |
| `portfolio_return` | Lợi nhuận danh mục chung vốn chia `BACKTEST_PORTFOLIO_SLOTS` phần |
| `portfolio_trades`, `portfolio_win_rate` | Số lệnh và tỷ lệ thắng của danh mục |
| `portfolio_avg_trade_return` | Lợi nhuận ròng trung bình mỗi lệnh của danh mục |
| `portfolio_sharpe`, `portfolio_max_drawdown` | Sharpe và drawdown lớn nhất của danh mục |
| `portfolio_exposure` | Tỷ lệ vốn trung bình đang nằm trong cổ phiếu |
| `buy_hold_avg_return` | Buy & Hold trung bình các mã trên cả cửa sổ (không xét mã có thuộc rổ hay không) |
| `equal_weight_return` | Nắm đều các mã thuộc rổ, tái cân bằng mỗi ngày, không phí — benchmark chính của danh mục |
| `vnindex_return` | Biến động VNINDEX trong cùng cửa sổ |

Cách đọc:

- `rank_ic` quanh `0` nghĩa là điểm không xếp hạng được mã; `0,02–0,05` ổn định
  qua thời gian đã là tín hiệu có ích trong thực tế. Lợi nhuận T+5 chồng lấn
  nhau giữa các ngày nên không coi IC trung bình là kiểm định thống kê.
- Nếu `blend` không tốt hơn `rule`, phần ML chưa thêm giá trị; nếu `ml` không
  tốt hơn `rule`, không nên tăng trọng số ML.
- `portfolio_return` so với `equal_weight_return` và `vnindex_return`.

## 6b. `backtest_ic_by_year.csv`

| Cột | Ý nghĩa |
|---|---|
| `mode` | Nguồn điểm |
| `year` | Năm |
| `rank_ic` | Rank IC trung bình các ngày trong năm |
| `portfolio_return` | Lợi nhuận danh mục trong năm (năm đầu/cuối có thể không đủ 12 tháng) |

Một cấu hình chỉ đáng tin khi Rank IC dương ở đa số các năm, không chỉ trung
bình toàn kỳ dương nhờ một năm đột biến.

## 7. `backtest_symbols.csv`

Mỗi dòng là kết quả backtest của một mã với một nguồn điểm trong cửa sổ được chọn.

| Cột | Ý nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu |
| `mode` | Nguồn điểm: `blend`, `rule` hoặc `ml` |
| `start`, `end` | Ngày bắt đầu và kết thúc backtest |
| `num_trades` | Số giao dịch đã đóng |
| `win_rate` | Tỷ lệ giao dịch có `net_return > 0` |
| `avg_win` | Lợi nhuận trung bình của các giao dịch thắng |
| `avg_loss` | Lợi nhuận trung bình của các giao dịch thua |
| `avg_return` | Lợi nhuận trung bình trên mỗi giao dịch |
| `profit_factor` | Tổng lợi nhuận giao dịch thắng chia cho trị tuyệt đối tổng lỗ |
| `dss_return` | Lợi nhuận cuối kỳ của chiến lược DSS |
| `buy_hold_return` | Lợi nhuận mua đầu kỳ và giữ đến cuối kỳ, có chi phí |
| `vnindex_return` | Biến động VNINDEX trong cùng giai đoạn |
| `sharpe` | Sharpe annualized từ chuỗi lợi nhuận theo ngày |
| `max_drawdown` | Mức sụt giảm lớn nhất từ đỉnh equity trước đó |
| `exposure` | Tỷ lệ số ngày danh mục đang giữ vị thế |

### Cách diễn giải

- `dss_return` nên được so với cả `buy_hold_return` và `vnindex_return`.
- `profit_factor > 1` nghĩa là tổng lãi lớn hơn tổng lỗ trong các giao dịch đã
  đóng.
- `sharpe` dương cao hơn thường tốt hơn, nhưng không có ý nghĩa khi số giao
  dịch quá ít.
- `max_drawdown` càng gần 0 càng ít sụt giảm; giá trị thường là số âm.
- `exposure` thấp có thể làm DSS có lợi nhuận thấp nhưng drawdown cũng thấp.
- Mã không có giao dịch sẽ có nhiều metric bằng 0; không được xem đó là model
  tốt nếu Buy & Hold tăng mạnh trong cùng giai đoạn.

Chi phí hiện được tính gồm phí mua/bán, thuế bán và slippage. Tín hiệu tính tại
giá đóng cửa ngày `i`, lệnh được khớp tại giá mở cửa ngày `i+1`. Lệnh bán tại
giá mở cửa chỉ được phép từ phiên T+3 (quy tắc thanh toán T+2).

## 8. `backtest_trades.csv`

File này có một dòng cho mỗi giao dịch đã đóng.

| Cột | Ý nghĩa |
|---|---|
| `entry_date` | Ngày vào lệnh |
| `exit_date` | Ngày thoát lệnh |
| `entry_price` | Giá vào lệnh trước chi phí |
| `exit_price` | Giá thoát lệnh trước chi phí |
| `hold_days` | Số phiên nắm giữ |
| `net_return` | Lợi nhuận sau chi phí mua và bán |
| `exit_reason` | `stop-loss` (điểm < 25), `T+N` (hết tầm nhìn), `barrier-profit` hoặc `barrier-stop` (với `--exit barrier`) |
| `symbol` | Mã cổ phiếu |
| `mode` | Nguồn điểm: `blend`, `rule` hoặc `ml` |
| `scope` | `symbol` (giao dịch từng mã) hoặc `portfolio` (danh mục chung vốn) |

Nên dùng file này để kiểm tra từng giao dịch bất thường, đặc biệt khi
`profit_factor` rất cao nhưng số lệnh ít.

## 9. `backtest_scores.csv`

Mỗi dòng là một mã trong một phiên của cửa sổ backtest: `time`, `symbol`,
`open`, `high`, `low`, `close`, `in_universe` (mã có thuộc rổ ngày đó không),
`rule_score`, `ml_score`, `total_score` và `future_return` (lợi nhuận T+N thực tế,
chỉ dùng để đánh giá). Dùng file này để tự phân tích
thêm mà không phải chạy lại model.

## 10. `tuning_results.csv`

File này so sánh các cấu hình model trong `src/models/tune.py`.

| Cột | Ý nghĩa |
|---|---|
| `symbol` | Mã được tuning |
| `candidate` | `baseline`, `regularized` hoặc `responsive` |
| `fold` | Fold hoặc `aggregate` |
| `model` | RF, XGB hoặc Ensemble50 |
| Các cột metric | Giống `walk_forward_metrics.csv` |
| `confusion_matrix` | Ma trận nhầm lẫn của dòng metric |

Không chọn candidate chỉ vì một fold hoặc một mã có điểm cao. Candidate nên ổn
định trên nhiều fold, nhiều mã và vẫn phải được kiểm tra lại bằng backtest.

## 11. Nguyên tắc kết luận

Một model hoặc label strategy chỉ nên được xem là ứng viên tốt khi đồng thời:

1. Vượt `BaselineHold` và `BaselineMomentum` trên Macro F1 hoặc balanced
   accuracy.
2. BUY/SELL precision đủ cao cho mục tiêu giao dịch.
3. Kết quả ổn định qua nhiều mã và nhiều fold.
4. Backtest có lợi nhuận ròng hợp lý sau chi phí.
5. Không đánh đổi lợi nhuận bằng drawdown quá lớn.

Metric phân loại chỉ đo khả năng dự đoán label. Nó không tự chứng minh chiến
lược có lợi nhuận. Kết luận cuối cần đối chiếu với backtest, số lệnh, exposure,
chi phí và giai đoạn thị trường được đánh giá.
