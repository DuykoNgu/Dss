# 🏷️ Tài liệu Labels & Ý Nghĩa — Hệ thống DSS VN30

Tổng hợp toàn bộ **nhãn (labels)** được sử dụng trong hệ thống, kèm **giải thích ý nghĩa tài chính** của từng label.

---

## 1. Label Machine Learning (Phase 4 — Nhãn huấn luyện)

### 1.1. Ba nhãn phân loại

| Label | Giá trị | Điều kiện | Ý nghĩa |
|-------|---------|-----------|---------|
| **MUA** | `1` | Lợi nhuận T+5 ≥ +3% | Giá sẽ tăng mạnh trong 5 phiên tới |
| **GIỮ** | `0` | -3% < Lợi nhuận < +3% | Giá dao động nhẹ, không đủ biên lợi nhuận |
| **BÁN** | `-1` | Lợi nhuận T+5 ≤ -3% | Giá sẽ giảm mạnh trong 5 phiên tới |

### 1.2. Giải thích ý nghĩa

**Tại sao là 3 nhãn (không phải 2)?**
- Nếu chỉ có MUA/BÁN (2 nhãn), hệ thống sẽ bị ép phải chọn 1 trong 2 ngay cả khi thị trường đi ngang (sideway). Điều này dẫn đến nhiều tín hiệu sai.
- Nhãn **GIỮ (0)** cho phép hệ thống nói rằng: *"Tôi chưa thấy tín hiệu rõ ràng, tốt nhất là chờ đợi."* Đây là điều rất quan trọng trong giao dịch thực tế — đôi khi **không làm gì** là quyết định tốt nhất.

**Tại sao ngưỡng ±3%?**
- Trong thị trường Việt Nam (HOSE), biên độ trần/sàn là ±7%/ngày. Mức 3% trong 5 phiên nghĩa là trung bình mỗi ngày cần tăng/giảm ~0.6%, đủ để bù chi phí giao dịch (phí mua bán ~ 0.15-0.35%) và vẫn có lãi ròng.
- Nếu đặt ngưỡng quá thấp (1%): Quá nhiều tín hiệu → nhiều tín hiệu sai (nhiễu).
- Nếu đặt quá cao (5%): Quá ít tín hiệu → bỏ lỡ cơ hội.
- **3% là điểm cân bằng** giữa "đủ tín hiệu" và "đủ chính xác".

**Tại sao T+5 (5 phiên)?**
- T+5 phù hợp với **giao dịch ngắn hạn (swing trading)** — giữ cổ phiếu khoảng 1 tuần.
- Quy tắc thanh toán trên sàn HOSE là **T+2** (mua hôm nay, T+2 mới nhận cổ phiếu). Vậy T+5 cho phép: mua → nhận hàng (T+2) → giữ 3 ngày → bán → nhận tiền (T+2 tiếp).
- Khoảng thời gian ngắn đủ để các chỉ báo kỹ thuật (MACD, RSI) có ý nghĩa, nhưng không quá dài để bị ảnh hưởng bởi tin tức vĩ mô khó dự đoán.

### 1.3. Mapping sang XGBoost

XGBoost yêu cầu nhãn bắt đầu từ 0:

| Label gốc | Giá trị gốc | Giá trị XGBoost | Ý nghĩa trong output `predict_proba()` |
|-----------|-------------|-----------------|----------------------------------------|
| BÁN | `-1` | `0` | `proba[0]` = Xác suất giá giảm > 3% |
| GIỮ | `0` | `1` | `proba[1]` = Xác suất giá đi ngang |
| MUA | `1` | `2` | `proba[2]` = Xác suất giá tăng > 3% |

**ML Score = `proba[2]` (đã hiệu chỉnh prior) × 100** — Chỉ lấy xác suất MUA vì đó là thứ nhà đầu tư quan tâm nhất: *"Khả năng cổ phiếu này tăng giá là bao nhiêu?"*

> Vì sao phải hiệu chỉnh prior: cả RF và XGBoost train với trọng số class **cân bằng** (để không "lười" đoán GIỮ — nhãn chiếm đa số). Điều này làm xác suất output bị kéo về prior đều 1/3, nên hệ thống nhân ngược lại theo tỷ lệ nhãn thật của tập train trước khi lấy `proba[2]`. Kết quả ML Score là ước lượng trung thực của P(MUA), không bị thổi phồng bởi trọng số cân bằng.

---

## 2. Label Tín Hiệu Đầu Ra (Phase 6 — Khuyến nghị cuối cùng)

### 2.1. Năm mức khuyến nghị

| Label | Icon | Khoảng điểm | Ý nghĩa chi tiết |
|-------|------|-------------|-------------------|
| **MUA MẠNH** | 🟢 | ≥ 75 | Cả Rule-based VÀ ML đều đồng thuận rằng cổ phiếu này rất tích cực. Nhiều chỉ báo kỹ thuật cùng xác nhận xu hướng tăng. Xác suất ML cho nhãn MUA rất cao. **Đây là tín hiệu mạnh nhất, độ tin cậy cao nhất.** |
| **MUA** | 🟡 | 60 – 74 | Phần lớn chỉ báo tích cực nhưng chưa hoàn toàn đồng thuận. Có thể 1-2 yếu tố còn trung tính. **Cân nhắc mua nhưng nên đặt Stop-loss phòng thủ.** |
| **GIỮ** | ⚪ | 40 – 59 | Các tín hiệu trái chiều hoặc trung tính. Thị trường đang phân vân, chưa rõ xu hướng. **Nếu đang giữ cổ phiếu thì tiếp tục giữ. Nếu chưa có thì chờ đợi.** |
| **BÁN** | 🟠 | 25 – 39 | Phần lớn chỉ báo tiêu cực. Xu hướng giảm bắt đầu hình thành. **Cân nhắc bán bớt hoặc giảm tỷ trọng.** |
| **BÁN MẠNH** | 🔴 | < 25 | Cả Rule-based VÀ ML đều cảnh báo rủi ro cao. Nhiều chỉ báo đồng loạt tiêu cực (Death Cross, RSI quá bán, Volume xả hàng). **Nên thoát hàng sớm để bảo toàn vốn.** |

### 2.2. Tại sao điểm trung tính là 50?

Hệ thống bắt đầu chấm điểm từ **50 điểm** (trung tính). Từ đó:
- Các yếu tố tích cực **cộng điểm** → đẩy lên trên 50 → hướng MUA.
- Các yếu tố tiêu cực **trừ điểm** → kéo xuống dưới 50 → hướng BÁN.

Điều này đảm bảo: nếu tất cả chỉ báo đều "trung tính" (không tốt không xấu), thì hệ thống sẽ ra tín hiệu **GIỮ** (40-59), đúng với logic "khi không chắc chắn thì không nên hành động".

### 2.3. Tại sao tỷ lệ 60% Rule / 40% ML?

| Thành phần | Trọng số | Lý do |
|-----------|----------|-------|
| **Rule-based** | 60% | Dựa trên kinh nghiệm phân tích kỹ thuật đã được kiểm chứng hàng chục năm. Ổn định, ít bị "ảo tưởng" (overfit). Đóng vai trò **phanh hãm rủi ro**. |
| **ML** | 40% | Bắt được các patterns phi tuyến tính mà rules không nhìn thấy. Nhưng có thể bị overfit trên dữ liệu quá khứ → không nên tin tưởng 100%. Đóng vai trò **bổ sung góc nhìn**. |

---

## 3. Label Rule-Based (Phase 6A — Chấm điểm chuyên gia)

### 3.1. Nhóm Xu Hướng (Trend) — Trả lời: *"Giá đang đi lên hay đi xuống?"*

| Điều kiện | Label | Điểm | Ý nghĩa tài chính |
|-----------|-------|------|--------------------|
| Giá > SMA50 > SMA200 | **Uptrend mạnh** | +10 | Xu hướng tăng rõ ràng ở cả trung hạn (SMA50) và dài hạn (SMA200). Đây là trạng thái lý tưởng nhất để mua. |
| Golden Cross gần đây | **Đảo chiều tăng** | +8 | SMA50 cắt lên trên SMA200 — tín hiệu cổ điển cho thấy xu hướng dài hạn chuyển từ giảm sang tăng. Rất được giới phân tích chú ý. |
| MACD > Signal & Histogram mở rộng | **Momentum tăng** | +7 | Không chỉ MACD trên Signal (tích cực) mà Histogram còn đang mở rộng, nghĩa là **tốc độ tăng giá đang gia tăng**. |
| Giá > SMA50 | **Trên trung hạn** | +5 | Giá nằm trên đường trung bình 50 phiên — cổ phiếu đang trong vùng "an toàn" của xu hướng trung hạn. |
| Giá < SMA50 < SMA200 | **Downtrend** | -8 | Ngược lại hoàn toàn: xu hướng giảm rõ ở cả 2 tầng thời gian. Rủi ro cao nếu mua vào. |
| Death Cross gần đây | **Đảo chiều giảm** | -10 | SMA50 cắt xuống dưới SMA200 — tín hiệu cảnh báo thị trường gấu (bear market). Điểm trừ rất nặng. |
| MACD < Signal | **Momentum giảm** | -10 | MACD nằm dưới Signal cho thấy lực bán đang mạnh hơn lực mua. |

### 3.2. Nhóm Động Lượng (Momentum) — Trả lời: *"Giá có bị đẩy quá xa không?"*

| Chỉ báo | Vùng | Label | Điểm | Ý nghĩa tài chính |
|---------|------|-------|------|--------------------|
| RSI | 30–50, đang tăng | **Phục hồi Oversold** | +10 | RSI vừa thoát vùng quá bán (dưới 30) và đang hồi phục. Đây là thời điểm "mua đáy" lý tưởng — người bán đã kiệt sức, người mua bắt đầu vào. |
| RSI | 50–60 | **Tích cực** | +7 | RSI nằm ở vùng cân bằng thiên tích cực. Cổ phiếu đang tăng nhưng chưa quá nóng. |
| RSI | 70–80 | **Quá mua** | -8 | Giá đã tăng quá nhanh, quá nhiều người mua đuổi giá. Rủi ro giá điều chỉnh (pullback) tăng cao. |
| RSI | > 80 | **Quá mua mạnh** | -12 | Cực kỳ nguy hiểm. Giá gần như chắc chắn sẽ điều chỉnh. Không nên mua mới. |
| Stochastic | %K cắt lên %D, vùng < 20 | **Mua Oversold** | +8 | Stochastic giao cắt tăng ở vùng quá bán — xác nhận thêm rằng đáy đã hình thành. Kết hợp với RSI oversold sẽ cho tín hiệu rất mạnh. |
| Stochastic | %K cắt xuống %D, vùng > 80 | **Bán Overbought** | -8 | Ngược lại — giao cắt giảm ở vùng quá mua, cảnh báo đỉnh ngắn hạn. |

### 3.3. Nhóm Khối Lượng (Volume) — Trả lời: *"Dòng tiền đang chảy vào hay ra?"*

| Điều kiện | Label | Điểm | Ý nghĩa tài chính |
|-----------|-------|------|--------------------|
| Vol ≥ 1.5x TB20 & Giá tăng | **Volume bùng nổ tích cực** | +10 | Khối lượng giao dịch đột biến (gấp 1.5 lần trung bình 20 phiên) VÀ giá tăng: cho thấy **dòng tiền lớn đang đổ vào**. Đây là tín hiệu xác nhận xu hướng tăng rất mạnh. |
| OBV tăng 5 phiên | **Tích lũy** | +5 | On-Balance Volume tăng liên tục: tổng khối lượng mua > tổng khối lượng bán trong 5 phiên. "Tiền thông minh" (smart money) đang gom hàng âm thầm. |
| Vol ≥ 1.5x TB20 & Giá giảm > 2% | **Bán tháo** | -10 | Volume đột biến nhưng giá lại giảm: **dòng tiền lớn đang tháo chạy**. Đây là dấu hiệu xả hàng (distribution) rất nguy hiểm. |
| OBV giảm 5 phiên | **Rút vốn** | -5 | Dòng tiền đang rời khỏi cổ phiếu một cách âm thầm, ngay cả khi giá chưa giảm nhiều. |

### 3.4. Nhóm Biến Động (Volatility) — Trả lời: *"Giá đang ổn định hay bất ổn?"*

| Điều kiện | Label | Điểm | Ý nghĩa tài chính |
|-----------|-------|------|--------------------|
| BB position < 0.2 & Giá tăng | **Bật tăng từ dải dưới** | +8 | Giá chạm dải dưới Bollinger Bands (vùng "rẻ" theo thống kê) rồi bật lên: cho thấy lực mua xuất hiện ở vùng hỗ trợ. Nếu kết hợp RSI oversold → tín hiệu mua rất đáng tin cậy. |
| BB position 0.2–0.5 | **Vùng tích lũy** | +4 | Giá nằm ở nửa dưới dải BB nhưng đang đi lên: giai đoạn tích lũy trước khi bứt phá. |
| ATR/Price < 3% | **Biến động thấp** | +3 | Biên độ dao động hàng ngày nhỏ so với giá: cổ phiếu ổn định, rủi ro thấp, phù hợp cho giao dịch có kỷ luật. |
| BB position ≥ 0.95 | **Chạm dải trên** | -5 | Giá chạm dải trên BB (vùng "đắt" theo thống kê): có khả năng đảo chiều giảm ngắn hạn (mean reversion). |
| ATR/Price > 5% | **Biến động cao** | -5 | Cổ phiếu đang "lắc" rất mạnh mỗi ngày: rủi ro cao, khó đặt Stop-loss hợp lý, không phù hợp cho giao dịch ngắn hạn. |

### 3.5. Nhóm Thị Trường VNINDEX — Trả lời: *"Thị trường chung đang tốt hay xấu?"*

| Điều kiện | Label | Điểm | Ý nghĩa tài chính |
|-----------|-------|------|--------------------|
| VNINDEX > SMA50 | **Thị trường tốt** | +5 | VNINDEX nằm trên đường trung bình 50 ngày: thị trường chung đang ở xu hướng tăng. Câu nói kinh điển: *"A rising tide lifts all boats"* — khi thị trường tốt, đa số cổ phiếu sẽ tăng theo. |
| VNINDEX return 5d > 1% | **Đang tăng tốt** | +3 | Thị trường tăng > 1% trong tuần qua: sentiment tích cực, dòng tiền đang vào. |
| VNINDEX < SMA50 | **Thị trường xấu** | -5 | Thị trường chung downtrend: dù cổ phiếu riêng lẻ có tốt, vẫn bị kéo giảm theo thị trường. *"Don't fight the market."* |
| VNINDEX return 5d < -3% | **Sụt giảm mạnh** | -5 | Thị trường giảm sâu > 3% trong 1 tuần: tâm lý hoảng loạn, rủi ro lan tỏa. Hầu hết mọi tín hiệu mua riêng lẻ đều nên tạm dừng. |

---

## 4. Label Dữ Liệu (Phase 2 — Data Cleaning)

| Cột | Giá trị | Ý nghĩa |
|-----|---------|---------|
| `is_extreme = True` | Biến động > ±6.8% (nến ngày) | Phiên giao dịch chạm hoặc gần trần/sàn HOSE. Đây là biến động **bất thường** — có thể do tin tức đột biến, thao túng giá, hoặc sự kiện vĩ mô. Hệ thống **đánh flag để cảnh báo nhưng không loại bỏ** vì đây vẫn là dữ liệu thật. Ngưỡng này chỉ đúng với nến ngày. |
| `is_extreme = False` | Biến động < ±6.8% | Phiên giao dịch bình thường, biến động trong biên độ cho phép. |
| `period_return` | Số thực | % thay đổi giá đóng cửa so với kỳ trước (timeframe-neutral, hiện tại = nến ngày). Ví dụ: `+0.025` nghĩa là kỳ này giá tăng 2.5% so với kỳ trước. |
| `volume_missing = True` | Khối lượng NaN gốc | Phiên **không lấy được dữ liệu volume** (khác với volume = 0 là không có giao dịch). Volume được điền 0 để tính toán nhưng cờ này giữ lại thông tin thiếu cho ML. |
| `is_ohlc_invalid = True` | Nến sai logic | Nến vi phạm `high >= max(open, close)` hoặc `low <= min(open, close)`. Flag để phase sau biết, không tự xóa. |

> Quy tắc fill: Phase 2 **chỉ forward-fill** (lấy quá khứ đắp vào lỗ hổng). Cấm backward-fill vì đó là lấy dữ liệu tương lai điền về quá khứ (leakage cho ML). NaN đầu file không có quá khứ để fill sẽ bị loại bỏ.

---

## 5. Label ML Score (Phase 5 — Xác suất từ mô hình)

### Output từ Ensemble (RF + XGBoost)

| Index | Class | Ý nghĩa | Ví dụ |
|-------|-------|---------|-------|
| `proba[0]` | BÁN | Xác suất giá sẽ giảm > 3% trong 5 phiên | 0.10 (10%) |
| `proba[1]` | GIỮ | Xác suất giá dao động trong ±3% | 0.22 (22%) |
| `proba[2]` | MUA | Xác suất giá sẽ tăng > 3% trong 5 phiên | 0.68 (68%) |

**ML Score = `proba[2]` (hiệu chỉnh prior) × 100**

Ví dụ: model trả `[0.10, 0.22, 0.68]` dưới prior cân bằng; nếu nhãn thật của tập train là BÁN 21.5% / GIỮ 61.3% / MUA 17.1%, xác suất được nhân tỷ lệ `prior_thật / (1/3)` rồi chuẩn hóa lại trước khi lấy MUA.

Hai trường hợp đặc biệt trả về **50 điểm** (trung tính, để Rule-based 60% quyết định):
- Mã **không đủ dữ liệu train** (dưới `MIN_TRAIN_ROWS=100` dòng — VD: TCX chỉ ~14 dòng trainable) → không có model.
- Dòng dự đoán có **feature NaN** → không nên bịa số.

Tại sao chỉ lấy `P(MUA)`? Vì đối với nhà đầu tư, câu hỏi quan trọng nhất là *"Khả năng cổ phiếu này TĂNG GIÁ là bao nhiêu?"*. ML Score cao (> 60) nghĩa là mô hình tự tin rằng cổ phiếu sẽ tăng. ML Score thấp (< 30) nghĩa là mô hình nghiêng về phía GIỮ hoặc BÁN.

### Feature Importance — Biến nào ảnh hưởng quyết định ML nhất?

Sau khi train, hệ thống in ra **Feature Importance** (độ quan trọng của từng feature). Thông thường các feature quan trọng nhất sẽ là:
- `rsi` — RSI phản ánh trạng thái quá mua/quá bán rất rõ
- `macd_hist` — MACD Histogram cho thấy động lượng tăng/giảm
- `price_vs_sma50` — Khoảng cách giá so với SMA50 cho thấy xu hướng
- `vol_ratio` — Khối lượng bất thường thường đi trước biến động giá lớn

---

## 6. Label Backtest (Phase 7 — Kiểm chứng)

### 6.1. Điều kiện vào/ra lệnh

| Hành động | Điều kiện | Ý nghĩa |
|-----------|-----------|---------|
| **MUA vào** | Total Score ≥ 60 | Chỉ mua khi hệ thống đủ tự tin (Signal = MUA hoặc MUA MẠNH). Không mua khi Signal là GIỮ để tránh giao dịch không có lợi thế. |
| **BÁN ra (chốt)** | Đã giữ ≥ 5 phiên (T+5) | Tuân thủ kỷ luật: mua T+5 thì bán T+5, bất kể lãi hay lỗ. Tránh tham lam giữ quá lâu. |
| **BÁN ra (cắt lỗ)** | Total Score < 25 trong khi đang giữ | Tín hiệu BÁN MẠNH xuất hiện trong khi đang nắm giữ: **cắt lỗ sớm** để bảo toàn vốn, không chờ đủ T+5. |

### 6.2. Đánh giá kết quả

| Label | Điều kiện | Ý nghĩa |
|-------|-----------|---------|
| **Win** | `pnl_pct > 0` | Giao dịch có lãi — hệ thống dự đoán đúng xu hướng tăng. |
| **Loss** | `pnl_pct ≤ 0` | Giao dịch lỗ — hệ thống dự đoán sai hoặc thị trường biến động ngoài dự kiến. |
| **Win Rate** | `số_win / tổng_lệnh × 100%` | Tỷ lệ thắng. DSS tốt nên đạt Win Rate > 50%. Nếu < 50% nhưng lãi TB/lệnh win > lỗ TB/lệnh loss thì vẫn có thể lãi tổng. |
| **DSS vượt trội** | `dss_return > buy_hold_return` | Hệ thống DSS mang lại lợi nhuận **cao hơn** chiến lược đơn giản "mua rồi giữ luôn". Đây là thước đo quan trọng nhất: nếu DSS không đánh bại được Buy & Hold thì chẳng cần hệ thống làm gì. |
| **Buy&Hold tốt hơn** | `dss_return ≤ buy_hold_return` | Mua giữ im hiệu quả hơn. Có thể do thị trường uptrend mạnh liên tục (mọi tín hiệu bán đều sai) hoặc ML bị overfit. |

---

## 7. Bảng Tổng Hợp Toàn Bộ Labels

| Phase | Tên Label | Giá trị | Mục đích | Ý nghĩa chính |
|-------|-----------|---------|----------|----------------|
| **2** | `is_extreme` | True/False | Cảnh báo phiên bất thường | Biến động > 6.8%, có thể là trần/sàn |
| **4** | `label` (ML) | -1, 0, 1 | Huấn luyện ML phân loại | BÁN/GIỮ/MUA dựa trên lợi nhuận T+5 |
| **4** | `golden_cross` | True/False | Phát hiện đảo chiều tăng | SMA50 cắt lên SMA200 |
| **4** | `death_cross` | True/False | Phát hiện đảo chiều giảm | SMA50 cắt xuống SMA200 |
| **4** | `macd_cross_up` | True/False | Xác nhận momentum tăng | MACD cắt lên Signal Line |
| **4** | `macd_cross_down` | True/False | Cảnh báo momentum giảm | MACD cắt xuống Signal Line |
| **5** | `ml_score` | 0–100 | Độ tin cậy của ML | P(MUA) × 100 — ML nghĩ khả năng tăng giá là bao nhiêu% |
| **6** | `rule_score` | 0–100 | Đánh giá chuyên gia | Tổng hợp 5 nhóm chỉ báo kỹ thuật |
| **6** | `total_score` | 0–100 | Điểm quyết định cuối | 60% Rule + 40% ML |
| **6** | Signal | 🟢🟡⚪🟠🔴 | Khuyến nghị người dùng | MUA MẠNH → BÁN MẠNH |
| **7** | Win/Loss | pnl > 0 / ≤ 0 | Đánh giá từng giao dịch | Lệnh này lãi hay lỗ |
| **7** | DSS vs B&H | So sánh % | Đánh giá toàn hệ thống | DSS có đáng dùng hơn Mua & Giữ không |
