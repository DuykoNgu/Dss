# AGENTS.md — DSS VN30

Hướng dẫn cho người/AI sửa code trong repo này. Chi tiết nghiệp vụ xem
`README.md`, `ML_HANDOVER_GUIDE.md`, `REPORT_METRICS.md`.

## Lệnh

```bash
./run.sh test                                  # unit test (bắt buộc chạy sau khi sửa)
./run.sh fetch                                 # đồng bộ dữ liệu (gọi API vnstock)
python3 backend/main.py --no-fetch --symbols FPT,ACB   # khuyến nghị offline
./run.sh backtest [--pooled]                   # so sánh blend/rule/ml, danh mục, rank IC
./run.sh backtest --pooled --universe history --months 72 --retrain-every 60 \
    [--horizon 20] [--label-strategy excess|triple_barrier] [--exit barrier]
./run.sh evaluate --label-strategy fixed       # walk-forward metrics
```

## Cấu trúc

- `backend/`: toàn bộ Python gồm `main.py`, `api/`, `src/`, `tests/`, `config.py` và `requirements.txt`.
- `frontend/`: React, Vite, CSS và test giao diện; không chứa mã Python.
- `backend/src/pipeline.py`: nạp SQLite -> clean -> indicators -> features, dùng chung cho mọi entrypoint.
- `backend/src/data/`: fetch (import vnstock muộn, chỉ khi gọi API), SQLite store, clean, `universe.py` (VN30 theo kỳ).
- `backend/reference/vn30_changes.csv`: thành phần VN30 từng kỳ kèm nguồn; cập nhật mỗi kỳ xét duyệt (tháng 1, 7) và khi có thay thế bất thường.
- `backend/src/features/`: indicators, 19 feature baseline, nhãn T+5.
- `backend/src/models/`: train/predict (`ml_models.py`), walk-forward (`validation.py`), evaluate, tune.
- `backend/src/scoring/`: điểm luật + tổng hợp khuyến nghị.
- `backend/src/backtest/`: chấm điểm lịch sử rồi mô phỏng từng mã, danh mục và rank IC.
- `backend/data/`, `backend/models/`, `backend/reports/`: artifact local, đã gitignore.

## Quy tắc không được phá

- Không dùng dữ liệu tương lai làm feature; không `bfill` giá. `future_return`/`label` chỉ để train và đánh giá.
- Mọi split theo thời gian, không shuffle; train trong backtest phải bỏ các dòng có label chạm tương lai.
- Rule Score và ML Score cùng thang 0–100, 50 là trung tính.
- Không cache nến của phiên chưa đóng cửa.
- Ghi VNINDEX và toàn bộ mã hiện tại trong một transaction; nến OHLC sai được đánh dấu và không dùng cho ML.
- Backtest `--universe history`: chỉ train, vào lệnh, xếp hạng trên mã thuộc rổ tại ngày đó.
- Import vnstock chỉ qua `backend/src/data/data_fetcher._vnstock()` (tắt việc vnstock tự ghi file chỉ dẫn AI vào project).
- Đổi feature/label/model thì chạy lại evaluate + backtest và cập nhật số liệu trong README.
