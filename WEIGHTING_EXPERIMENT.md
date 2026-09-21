# Đối chứng balancing và trọng số thời gian

Thử nghiệm trên dữ liệu cache đến 17/09/2026, dùng cùng dữ liệu cho cả bốn
cấu hình. Đây là nghiên cứu trên lịch sử đã từng xem, không phải một tập test
mới hoàn toàn và không tự động thay cấu hình website.

## Kết quả

Bốn lượt dùng đúng cùng 63.880 dòng chấm điểm, trong đó 44.450 dòng thuộc rổ
có đủ feature và nhãn để đánh giá phân loại. Số nhãn SELL/HOLD/BUY tương ứng
13.326/19.470/11.654. Thời gian bốn lượt khoảng 10,3 phút.

| Cấu hình | Lợi nhuận ròng cả kỳ | Max drawdown | Sharpe | Lệnh đã đóng | Win rate |
|---|---:|---:|---:|---:|---:|
| Balanced, không decay | +109,66% | −49,77% | 0,661 | 254 | 55,51% |
| Không balanced, không decay | +205,25% | −49,59% | 0,946 | 250 | 58,80% |
| Balanced, decay 2 năm | +170,81% | −42,93% | 0,850 | 270 | 55,56% |
| Không balanced, decay 2 năm | **+233,09%** | **−41,47%** | **0,989** | 264 | 57,58% |

Cùng kỳ, benchmark equal-weight không phí đạt +139,92%, VNINDEX +108,53%.
Lợi nhuận trên là toàn giai đoạn, không phải lợi nhuận mỗi năm.

| Cấu hình | BUY precision tại điểm ≥60 | BUY recall tại điểm ≥60 | Tín hiệu MUA | Tín hiệu BÁN <25 | Rank IC |
|---|---:|---:|---:|---:|---:|
| Balanced, không decay | 36,54% | 10,82% | 3.463 | 219 | +0,0701 |
| Không balanced, không decay | 36,77% | 10,67% | 3.396 | 242 | **+0,0733** |
| Balanced, decay 2 năm | 36,06% | 11,16% | 3.621 | 233 | +0,0659 |
| Không balanced, decay 2 năm | 36,95% | 11,32% | 3.580 | 288 | +0,0640 |

Precision dùng các tín hiệu đã có nhãn; số tín hiệu MUA ở bảng gồm cả đuôi chưa
có nhãn. Số tín hiệu MUA có nhãn lần lượt là 3.451, 3.383, 3.608 và 3.570.
Cả bốn cấu hình có IC trung bình dương ở 7/7 năm; 2020 và 2026 là năm không đủ.

| Cấu hình | Balanced accuracy | Macro F1 | Class BUY P/R | Class SELL P/R |
|---|---:|---:|---:|---:|
| Balanced, không decay | 0,3675 | 0,3308 | 37,35% / 11,65% | 35,10% / 16,23% |
| Không balanced, không decay | 0,3718 | 0,3432 | 36,95% / 13,39% | 35,25% / 18,86% |
| Balanced, decay 2 năm | 0,3680 | 0,3332 | 37,63% / 12,37% | 34,67% / 16,36% |
| Không balanced, decay 2 năm | 0,3719 | 0,3455 | 36,71% / 14,04% | 35,03% / 19,27% |

### Độ ổn định qua thời gian

Lợi nhuận từng năm, nối tiếp cùng một chuỗi vốn và vị thế:

| Năm | Balanced, đều | Không balanced, đều | Balanced, decay | Không balanced, decay |
|---|---:|---:|---:|---:|
| 2020 (từ 25/08) | +38,40% | +40,82% | +39,07% | +39,03% |
| 2021 | +83,28% | +66,84% | +79,55% | +77,45% |
| 2022 | −38,08% | −34,84% | −30,06% | −30,97% |
| 2023 | −2,99% | −1,62% | +7,87% | +5,82% |
| 2024 | +3,99% | +14,56% | −0,11% | +7,81% |
| 2025 | +46,15% | +72,27% | +68,08% | +73,44% |
| 2026 (đến 17/09) | −9,47% | +2,70% | −14,37% | −1,15% |

| Giai đoạn | Balanced, đều | Không balanced, đều | Balanced, decay | Không balanced, decay |
|---|---:|---:|---:|---:|
| 2020–2023 | +52,37% | +50,62% | +88,38% | +80,21% |
| 2024–17/09/2026 | +37,60% | **+102,66%** | +43,76% | +84,83% |

Các giai đoạn này chỉ để phân tích độ ổn định, không gọi là holdout chưa từng
được xem. Số liệu chi tiết ở `by_year.csv` và `by_period.csv` trong thư mục run.

### Nhận định

1. Bỏ balancing tăng lợi nhuận trong cả hai cặp đối chứng trên toàn kỳ, nhưng
   BUY precision gần như không thay đổi. Chưa có bằng chứng cho nhận định
   balancing là nguyên nhân duy nhất gây nhiều tín hiệu MUA sai hoặc thua lỗ.
2. Decay 2 năm cải thiện lợi nhuận và drawdown toàn kỳ ở cả hai cặp. Tuy vậy,
   IC giảm ở cả hai và không balanced/không decay thắng ở giai đoạn 2024–2026.
   Không có một cấu hình dẫn đầu mọi tiêu chí và mọi giai đoạn.
3. Không balanced + decay là ứng viên tốt nhất về lợi nhuận/Sharpe/drawdown
   toàn kỳ trong bốn cấu hình, nhưng drawdown −41,47% vẫn lớn. Giữ cả hai
   cấu hình không balancing để kiểm chứng tiếp bằng dự đoán tương lai, chưa
   tự động chuyển website sang cấu hình thắng backtest.
4. Không đối chiếu trực tiếp con số BUY precision 16,9% cũ với bảng này:
   khác nhãn, horizon, pooled, độ phủ, phiên bản dữ liệu và có hiệu chỉnh điểm.
   Lợi nhuận +257,9% cũ cũng không phải baseline của đối chứng mới.

Kiểm tra: 51 unit test đạt; kiểm tra dữ liệu thật xác nhận hai đầu mút nhãn
đúng ngày; bốn lượt có cùng giá/nhãn/thành phần rổ/ngày chấm điểm, equity cuối
khớp lợi nhuận báo cáo và hash source khớp manifest.

## Thiết kế

- Pooled RF + XGBoost, 19 feature baseline, nhãn `excess`, horizon 20.
- Thành phần VN30 theo lịch sử: 44 mã có dữ liệu, chỉ train và đánh giá những
  dòng thuộc rổ ở ngày tương ứng. Rổ trước 03/08/2020 vẫn dùng xấp xỉ rổ gốc.
- Cửa sổ backtest 72 × 21 = 1.512 phiên, 25/08/2020–17/09/2026.
- Retrain mỗi 60 phiên; purge bằng `label_end < ngày retrain`.
- Giữ nguyên luật giao dịch: mua phiên sau khi điểm ≥60, tối đa 5 vị thế,
  thoát theo T+20 hoặc điểm <25 khi cổ phiếu đủ thời gian về tài khoản.
- Phí 0,15% mỗi chiều, thuế bán 0,1%, slippage 0,1% mỗi chiều.
- RF/XGB, seed, ngưỡng nhãn, điểm và cách phân bổ vốn giống nhau ở cả bốn lượt.

| Cấu hình | Balancing | Trọng số thời gian |
|---|---|---|
| `balanced_uniform` | Có | Bằng nhau |
| `unbalanced_uniform` | Không | Bằng nhau |
| `balanced_decay_2y` | Có | Giảm một nửa sau 730,5 ngày |
| `unbalanced_decay_2y` | Không | Giảm một nửa sau 730,5 ngày |

Trọng số thời gian `2 ** (-age_days / 730.5)` chỉ dùng ngày trong tập train
của từng lượt. Khi balancing, tần suất lớp được tính có trọng số thời gian.
Hai model nhận cùng `sample_weight`, chuẩn hóa trung bình 1; RF không nhân
thêm `class_weight`. Khi dự đoán, chỉ đảo phần trọng số lớp, giữ tác động của
recency. Cấu hình không balancing không áp dụng hiệu chỉnh prior cũ.

## Sửa lỗi trước khi đo

1. Nhãn excess so sánh cổ phiếu và VNINDEX tại đúng cùng hai ngày. Ngày cuối
   là ngày sau N nến hợp lệ của cổ phiếu; thiếu giá chỉ số tại một đầu mút
   thì nhãn là NaN. Không forward-fill nhãn hoặc giá chỉ số dùng cho nhãn.
2. Walk-forward giữ các mã cùng ngày trong cùng fold, purge theo ngày kết
   thúc nhãn và nhận cả khối cuối chưa đủ kích thước.
3. Metric phân loại có thêm số mẫu thật và số dự đoán từng lớp để nhận biết
   trường hợp thiếu lớp. Quy ước metric không xác định vẫn là 0.

## Cách đọc báo cáo

- `class_*`: phân loại ba lớp bằng argmax của ensemble sau hiệu chỉnh, trên
  các dòng có nhãn, feature hợp lệ và thuộc rổ. Khác `Ensemble50` chưa hiệu
  chỉnh trong lệnh evaluate cũ.
- `signal_buy_precision/recall`: đánh giá điều kiện điểm ML ≥60 đối với nhãn
  BUY excess. BUY nghĩa là vượt VNINDEX ít nhất 3,6%, không đồng nghĩa có lãi.
- `buy_signals`: số cặp mã–ngày đạt ngưỡng, kể cả những ngày cuối chưa có nhãn;
  `labelled_buy_signals` là mẫu số dùng cho precision. Tín hiệu lặp nhiều ngày
  không đồng nghĩa nhiều giao dịch được khớp.
- `num_trades`, `win_rate`: chỉ các giao dịch đã đóng trong danh mục.
- `rank_ic`: Spearman giữa điểm và lợi nhuận tương lai theo ngày, chỉ trên
  mã thuộc rổ có dự đoán hợp lệ. Các cửa sổ T+20 chồng lấn nên không coi từng
  ngày IC là một quan sát độc lập để kết luận ý nghĩa thống kê.
- `portfolio_return`, `max_drawdown`: tính từ equity sau chi phí đã phát sinh;
  vị thế còn mở cuối kỳ được định giá theo close, chưa trừ phí thanh lý giả định.
- Benchmark equal-weight tái cân bằng hàng ngày không phí; VNINDEX là thay
  đổi chỉ số. Chúng không có cấu trúc chi phí/exposure giống danh mục.

Nhãn vẫn là close-to-close trên N nến hợp lệ; giao dịch vào next-open và danh
mục giữ theo lịch chung. Sự khác nhau này được giữ cố định trong cả bốn lượt,
chưa được thử nghiệm thay đổi ở đây. Không bổ sung giả định thanh khoản mới.

## Chạy lại

```bash
./run.sh test
python3 backend/weighting_experiment.py
```

Mặc định mỗi lần tạo một thư mục mới trong `backend/reports/weighting_ablation_*`.
Có thể chỉ định `--output`, nhưng thư mục phải chưa tồn tại để tránh ghi đè.

Thư mục của lần chạy này: `backend/reports/weighting_ablation_20260922/`.
`manifest.json` lưu cấu hình, phiên bản thư viện, hash source và hash dữ liệu
featured. `data_coverage.csv` ghi độ phủ từng mã; mỗi cấu hình lưu `scores.csv`,
`daily_rank_ic.csv`, `equity.csv`, `trades.csv`, `by_year.csv`. `summary.csv`
được ghi sau mỗi lượt. Các artifact này là local, không được đưa vào Git.

Đánh giá bổ sung bằng `write_reports(['FPT', 'ACB'], 'excess', 'baseline', 20)`
với `config.REPORT_DIR` trỏ vào `evaluation_checks/` trong thư mục trên: cả hai
mã có 5 fold, lần lượt 1.769 và 1.765 mẫu usable; fold cuối đến 17/08/2026
(nhãn T+20 cuối cùng đã biết tại ngày dữ liệu 17/09/2026). Đây là kiểm tra luồng
evaluate per-symbol, không phải kết quả pooled trong bảng đối chứng.
