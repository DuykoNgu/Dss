# 🧠 Phân Tích Thuật Toán — Hệ thống DSS VN30

Tài liệu giải thích **lý thuyết** và **cách dùng** từng thuật toán trong hệ thống.

> Đây là tài liệu lý thuyết, không phải bằng chứng hiệu quả. Kết quả đo thực tế
> (walk-forward, backtest so sánh blend/rule/ml, rank IC) nằm trong `README.md`.
> Khi tài liệu khác code, tin `backend/config.py` và `backend/src/`.

---

## 1. Tổng Quan Bài Toán

### 1.1. Đây là bài toán gì?

**Bài toán Phân loại (Classification)** — Không phải Hồi quy (Regression).

| | Classification (Đang dùng) | Regression (Không dùng) |
|---|---|---|
| **Câu hỏi** | "Cổ phiếu sẽ TĂNG, GIẢM hay ĐI NGANG?" | "Giá cổ phiếu ngày mai chính xác là bao nhiêu?" |
| **Output** | Nhãn: MUA / GIỮ / BÁN + Xác suất | Một con số: 48.650 VND |
| **Ưu điểm** | Loại bỏ nhiễu, tập trung vào xu hướng | Cho con số chính xác |
| **Nhược điểm** | Mất thông tin chi tiết về mức giá | Sai số cực lớn do thị trường nhiễu |

**Tại sao chọn Classification?**
- Thị trường chứng khoán chứa **vô vàn nhiễu ngẫu nhiên** (tin tức, tâm lý, thao túng...). Việc dự đoán chính xác giá tuyệt đối gần như bất khả thi.
- Nhà đầu tư không cần biết *"giá chính xác là 48.650"*. Họ cần biết *"nên MUA hay BÁN?"*.
- Classification biến bài toán phức tạp (đoán 1 con số chính xác) thành bài toán đơn giản hơn (chọn 1 trong 3 nhóm), giúp mô hình **học được patterns hiệu quả hơn**.

### 1.2. Pipeline thuật toán

```
Dữ liệu OHLCV nến ngày (~8 năm)
    │
    ▼
Tính 20+ Features (Chỉ báo kỹ thuật biến đổi)
    │
    ▼
Gán nhãn T+5 (MUA / GIỮ / BÁN)
    │
    ├──► Random Forest ──► P(MUA), P(GIỮ), P(BÁN)
    │                                │
    ├──► XGBoost ────────► P(MUA), P(GIỮ), P(BÁN)
    │                                │
    │                    Trung bình xác suất (Ensemble)
    │                                │
    │              ML Score = 50 + 50 × (P(MUA) − P(BÁN)), đã hiệu chỉnh prior
    │                                │
    └──► Rule-Based ──────────► Rule Score (0-100)
                                     │
                              Total = 60% Rule + 40% ML
                                     │
                              🟢🟡⚪🟠🔴 Signal
```

---

### 1.3. Nền tảng cơ bản: Cây Quyết Định (Decision Tree)

Cả hai thuật toán Machine Learning sử dụng trong hệ thống (Random Forest và XGBoost) đều được xây dựng dựa trên một "viên gạch" cơ bản là **Cây Quyết Định (Decision Tree)**. Để hiểu hệ thống, ta cần hiểu viên gạch này trước.

**Cây Quyết Định là gì?**
Thuật toán này mô phỏng quá trình tư duy của con người bằng cách đặt ra các câu hỏi dạng Yes/No liên tiếp để chia nhỏ dữ liệu, cho đến khi ra được kết luận cuối cùng (nhãn MUA/GIỮ/BÁN).

**Ví dụ trực quan về 1 cây quyết định trong Trading:**
```
[Nút gốc: RSI có nhỏ hơn 30 không?]
     │
     ├── (Có) ──► [Nút nhánh: Khối lượng có lớn hơn 1.5 lần trung bình không?]
     │                  │
     │                  ├── (Có) ──► 🟢 Kết luận: MUA (Bắt đáy thành công có dòng tiền)
     │                  └── (Không) ─► ⚪ Kết luận: GIỮ (Cạn cung nhưng chưa có cầu)
     │
     └── (Không) ─► [Nút nhánh: Giá có đang chạm dải trên Bollinger Band không?]
                        │
                        ├── (Có) ──► 🔴 Kết luận: BÁN (Rủi ro đảo chiều cao)
                        └── (Không) ─► ⚪ Kết luận: GIỮ (Xu hướng chưa rõ ràng)
```

**Tại sao KHÔNG dùng 1 Cây Quyết Định duy nhất cho hệ thống DSS?**
Mặc dù rất dễ hiểu và giống với các "Rules" của phân tích kỹ thuật, 1 Cây Quyết Định đơn lẻ có điểm yếu chí mạng trong tài chính:
- **Overfitting (Học vẹt):** Nếu không bị giới hạn, cây sẽ tiếp tục chẻ nhánh tạo ra những quy tắc siêu phức tạp và vô lý (ví dụ: *"Nếu RSI = 42.1 và Giá = 24.500 và OBV giảm nhẹ thì MUA"*). Nó học thuộc lòng các trường hợp nhiễu ngẫu nhiên trong quá khứ nhưng sẽ dự đoán sai bét trong tương lai.
- **Tính bất ổn cao:** Chỉ cần dữ liệu đầu vào thay đổi một chút, cấu trúc toàn bộ cây có thể bị lật ngược hoàn toàn.

#### Cây quyết định "chọn câu hỏi" như thế nào? — Entropy & Information Gain

Khi cây phải chọn giữa việc hỏi "RSI < 30?" hay "Volume > 1.5x?" trước, nó dựa vào các thước đo toán học để tìm ra **câu hỏi nào chia dữ liệu ra thành các nhóm "sạch" nhất** (mỗi nhóm gần như chỉ chứa 1 loại nhãn).

**1. Entropy (Độ hỗn loạn)**

Entropy đo mức độ "lộn xộn" của một tập dữ liệu. Nếu một tập dữ liệu chứa đều đặn cả MUA, GIỮ, BÁN thì Entropy cao (rất hỗn loạn, khó đoán). Nếu tập dữ liệu gần như toàn MUA thì Entropy thấp (gần 0, rất "sạch").

```
Công thức:
  Entropy(S) = -Σ pᵢ × log₂(pᵢ)

  Trong đó pᵢ = tỷ lệ của từng nhãn (MUA, GIỮ, BÁN) trong tập S
```

Ví dụ tính Entropy trên dữ liệu DSS:

```
Tập dữ liệu gốc (600 phiên training):
  MUA: 150 phiên (25%)
  GIỮ: 300 phiên (50%)
  BÁN: 150 phiên (25%)

  Entropy = -(0.25 × log₂(0.25)) - (0.50 × log₂(0.50)) - (0.25 × log₂(0.25))
          = -(0.25 × -2) - (0.50 × -1) - (0.25 × -2)
          = 0.5 + 0.5 + 0.5
          = 1.5 (Khá hỗn loạn — thang 0 đến log₂(3) ≈ 1.585)
```

```
Sau khi chia theo "RSI < 30?":
  Nhánh "Có" (RSI < 30): 80 phiên → MUA: 50 (62.5%), GIỮ: 20 (25%), BÁN: 10 (12.5%)
    Entropy = 1.27 (giảm — nhánh này thiên về MUA rõ ràng hơn!)

  Nhánh "Không" (RSI ≥ 30): 520 phiên → MUA: 100 (19%), GIỮ: 280 (54%), BÁN: 140 (27%)
    Entropy = 1.46 (vẫn hỗn loạn — cần chia thêm)
```

**2. Information Gain (IG — Lượng Thông Tin Thu Được)**

IG đo **mức giảm Entropy** sau khi chia dữ liệu theo một feature. Feature nào cho IG cao nhất → Cây chọn feature đó để hỏi trước.

```
Công thức:
  IG(S, Feature) = Entropy(S) - Σ (|Sᵥ| / |S|) × Entropy(Sᵥ)

  Trong đó Sᵥ = tập con sau khi chia theo giá trị v của Feature
```

Ví dụ so sánh IG của 2 features:

```
IG(RSI < 30) = 1.5 - (80/600 × 1.27 + 520/600 × 1.46)
             = 1.5 - (0.169 + 1.264)
             = 1.5 - 1.433
             = 0.067

IG(Volume > 1.5x) = 1.5 - (120/600 × 1.35 + 480/600 × 1.48)
                   = 1.5 - (0.270 + 1.184)
                   = 1.5 - 1.454
                   = 0.046

→ IG(RSI) = 0.067 > IG(Volume) = 0.046
→ Cây chọn hỏi RSI trước vì nó chia dữ liệu "sạch" hơn!
```

**3. Gini Impurity (Độ không tinh khiết Gini)**

Đây là thước đo thay thế cho Entropy, được `scikit-learn` dùng **mặc định** trong Random Forest vì tính toán nhanh hơn (không cần logarithm).

```
Công thức:
  Gini(S) = 1 - Σ pᵢ²

Ví dụ:
  Tập "hoàn toàn sạch" (100% MUA): Gini = 1 - 1² = 0 (hoàn hảo)
  Tập "hỗn loạn" (33% MUA, 33% GIỮ, 33% BÁN): Gini = 1 - 3×(0.33²) = 0.667 (tệ nhất)
```

| Thước đo | Công thức | Phạm vi | Khi nào dùng | Dùng trong DSS |
|----------|-----------|---------|-------------|---------------|
| **Entropy** | -Σ pᵢ log₂(pᵢ) | 0 → log₂(K) | Khi cần đo lượng thông tin chính xác | XGBoost (qua log loss) |
| **Gini Impurity** | 1 - Σ pᵢ² | 0 → 1-1/K | Khi cần tốc độ, kết quả gần giống Entropy | Random Forest (mặc định `criterion='gini'`) |
| **Information Gain** | Entropy trước - Entropy sau khi chia | ≥ 0 | Chọn feature tốt nhất để phân nhánh | Cả RF lẫn XGBoost đều dùng nguyên lý này |

**4. Các tham số kiểm soát cây quyết định — Chống Overfitting**

Nếu để cây tự do phát triển, nó sẽ phân nhánh cho đến khi mỗi lá chỉ chứa đúng 1 phiên giao dịch (Entropy = 0 ở mọi lá, nhưng hoàn toàn overfitting). Vì vậy, các tham số sau đóng vai trò "cắt tỉa" cây:

| Tham số | Ý nghĩa | Giá trị trong DSS | Tác động |
|---------|---------|-------------------|----------|
| `max_depth` | Số tầng câu hỏi tối đa | RF: **10**, XGB: **6** | Giới hạn độ phức tạp. Nếu = None → Cây mọc vô hạn → Overfit. Tại sao XGB nhỏ hơn? Vì Boosting xây 300 cây tuần tự, mỗi cây chỉ cần "yếu" (nông) là đủ. |
| `min_samples_split` | Số mẫu tối thiểu để 1 nút được phép tiếp tục chia nhánh | Mặc định: **2** | Nếu 1 nút chỉ chứa 1 phiên giao dịch thì không chia nữa. |
| `min_samples_leaf` | Số mẫu tối thiểu ở mỗi lá (kết luận cuối) | RF: **20** | Mỗi quyết định MUA/GIỮ/BÁN phải dựa trên ít nhất 20 phiên giao dịch. Ngăn cây tạo ra quy tắc chỉ dựa trên 1-2 trường hợp may mắn. |
| `max_features` | Số features được xem xét mỗi lần phân nhánh | RF: **√N ≈ 4** features/nhánh | Buộc các cây trong Random Forest phải "sáng tạo" — không phải cây nào cũng bắt đầu bằng RSI. Đây chính là yếu tố "Random" trong Random Forest. |
| `criterion` | Thước đo dùng để chọn feature | RF: **gini**, XGB: dùng gradient của log loss | Gini nhanh hơn Entropy, kết quả gần như giống nhau trong thực tế. |

**Minh họa tác động của `max_depth`:**

```
max_depth = 2 (cây nông — Underfitting):
  [RSI < 30?]
    ├── Có → MUA
    └── Không → GIỮ
  → Quá đơn giản! Bỏ lỡ rất nhiều pattern.

max_depth = 20 (cây sâu — Overfitting):
  [RSI < 30?]
    ├── Có → [Volume > 1.5x?]
    │         ├── Có → [MACD > 0.003?]
    │         │         ├── Có → [BB pos < 0.187?]
    │         │         │         ├── Có → [OBV slope = 0.00234?]
    │         │         │         │         └── ... (tầng 15) → MUA
    │         │         │         └── ... (tầng 16) → GIỮ
    │         │         └── ... → BÁN
    │         └── ... → GIỮ
    └── ... (nhánh khổng lồ)
  → Quá phức tạp! Nhớ thuộc data cũ, dự đoán sai data mới.

max_depth = 10 (vừa phải — Đang dùng cho RF):
  → Đủ sâu để bắt patterns có ý nghĩa (RSI + Volume + MACD kết hợp)
  → Đủ nông để không nhớ thuộc nhiễu ngẫu nhiên
```

**Giải pháp:** Để loại bỏ nhược điểm "học vẹt" và "bất ổn" của 1 cây đơn lẻ, hệ thống sử dụng các thuật toán **Ensemble** (Học tập tập hợp) bằng cách kết hợp hàng trăm cây quyết định lại với nhau:
1. **Xây song song nhiều cây (Bagging)** → Sinh ra **Random Forest**
2. **Xây tuần tự, cây sau sửa lỗi cây trước (Boosting)** → Sinh ra **XGBoost**

---

## 2. Random Forest — Thuật Toán Chính

### 2.1. Random Forest là gì?

Random Forest (Rừng ngẫu nhiên) là thuật toán học máy hoạt động theo nguyên lý **"Trí tuệ đám đông"**: Thay vì dựa vào 1 cây quyết định (Decision Tree) duy nhất, nó xây dựng **hàng trăm cây quyết định khác nhau** rồi lấy kết quả **bỏ phiếu đa số**.

### 2.2. Cách hoạt động trong DSS

```
Dữ liệu Training (600 phiên × 19 features)
    │
    ├──► Cây 1: Lấy ngẫu nhiên 70% dữ liệu + 13 features → Dự đoán: MUA
    ├──► Cây 2: Lấy ngẫu nhiên 70% dữ liệu + 13 features → Dự đoán: MUA
    ├──► Cây 3: Lấy ngẫu nhiên 70% dữ liệu + 13 features → Dự đoán: GIỮ
    ├──► ...
    └──► Cây 200: Lấy ngẫu nhiên 70% dữ liệu + 13 features → Dự đoán: MUA
                    │
                    ▼
              Bỏ phiếu: MUA chiếm 65%, GIỮ chiếm 25%, BÁN chiếm 10%
                    │
                    ▼
              Kết quả: MUA (xác suất 65%)
```

**Hai yếu tố ngẫu nhiên (Random) tạo nên sức mạnh:**
1. **Bagging (Bootstrap Aggregating):** Mỗi cây được train trên một tập con dữ liệu khác nhau (lấy mẫu có hoàn lại). Nhờ vậy, mỗi cây "nhìn" dữ liệu từ một góc khác.
2. **Feature Randomness:** Mỗi lần phân nhánh, cây chỉ được chọn từ một tập con features ngẫu nhiên (thường là √N features). Điều này buộc các cây phải "sáng tạo" — không phải cây nào cũng dùng RSI làm tiêu chí đầu tiên.

### 2.3. Hyperparameters đã chọn và lý do

```python
RandomForestClassifier(
    n_estimators=200,          # Số cây trong rừng
    max_depth=10,              # Độ sâu tối đa mỗi cây
    min_samples_leaf=20,       # Số mẫu tối thiểu ở lá
    class_weight='balanced',   # Cân bằng trọng số 3 class
    random_state=42            # Seed cố định để kết quả lặp lại
)
```

| Tham số | Giá trị | Tại sao chọn giá trị này? |
|---------|---------|--------------------------|
| `n_estimators=200` | 200 cây | Đủ để giảm phương sai (variance). Tăng lên 500 gần như không cải thiện accuracy nhưng chậm gấp 2.5 lần. 200 là điểm cân bằng hiệu suất/tốc độ. |
| `max_depth=10` | Tối đa 10 tầng | Giới hạn độ phức tạp mỗi cây. Nếu không giới hạn (None), cây sẽ phân nhánh cho đến khi "nhớ thuộc lòng" dữ liệu training → **overfitting nghiêm trọng** với dữ liệu tài chính. 10 tầng đủ để bắt patterns nhưng không quá chi tiết. |
| `min_samples_leaf=20` | Tối thiểu 20 mẫu/lá | Mỗi nút lá (quyết định cuối) phải có ít nhất 20 phiên giao dịch. Điều này ngăn cây tạo ra các quy tắc quá cụ thể kiểu *"Nếu RSI=47.3 VÀ MACD=0.0012 thì MUA"* — những quy tắc này chỉ đúng trên data training và vô nghĩa trong thực tế. |
| `class_weight='balanced'` | Tự cân bằng | Trong dữ liệu chứng khoán, nhãn GIỮ thường chiếm đa số (~50-60%), MUA và BÁN ít hơn (~20% mỗi loại). Nếu không cân bằng, model sẽ "lười" — luôn đoán GIỮ để đạt accuracy cao mà không thực sự học được gì. `balanced` tự động tăng trọng số cho class thiểu số. |

### 2.4. Ưu nhược điểm của Random Forest trong bài toán này

| Ưu điểm | Nhược điểm |
|----------|-----------|
| **Chống nhiễu tốt** — Trung bình 200 cây triệt tiêu nhiễu ngẫu nhiên | **Không bắt được xu hướng mới** — Chỉ học từ data cũ, không ngoại suy được |
| **Không cần chuẩn hóa dữ liệu** — Hoạt động tốt với dữ liệu thô (RSI 0-100, giá 10.000-200.000) | **Chậm hơn XGBoost** khi predict (phải duyệt 200 cây) |
| **Feature Importance** — Cho biết feature nào quan trọng nhất, rất hữu ích để viết báo cáo | **Khó bắt tương tác phức tạp** giữa các features |
| **Ít bị overfit** hơn 1 Decision Tree đơn lẻ | **Kết quả "an toàn"** — Có xu hướng cho xác suất gần 50% (không dám quyết liệt) |

---

## 3. XGBoost — Thuật Toán Bổ Sung

### 3.1. XGBoost là gì?

XGBoost (eXtreme Gradient Boosting) là thuật toán học máy hoạt động theo nguyên lý **"Học từ sai lầm"**: Thay vì xây nhiều cây độc lập (như RF), XGBoost xây các cây **tuần tự**, mỗi cây mới tập trung sửa lỗi của cây trước.

### 3.2. Sự khác biệt cốt lõi với Random Forest

| | Random Forest (Bagging) | XGBoost (Boosting) |
|---|---|---|
| **Cách xây cây** | 200 cây xây **song song**, độc lập nhau | 300 cây xây **tuần tự**, cây sau sửa lỗi cây trước |
| **Mỗi cây học gì?** | Học từ 1 bản sao ngẫu nhiên của data gốc | Học từ **phần sai** (residuals) của cây trước |
| **Ẩn dụ** | 200 chuyên gia độc lập cùng bỏ phiếu | 1 học sinh làm bài → Thầy chữa lỗi → Học sinh sửa → Lặp lại 300 lần |
| **Thế mạnh** | Ổn định, ít overfit | Chính xác cao, bắt patterns tinh vi |
| **Rủi ro** | Có thể bỏ lỡ patterns nhỏ | Dễ overfit nếu tham số sai |

### 3.3. Cách hoạt động trong DSS

```
Cây 1: Dự đoán ban đầu → Sai ở 35% dữ liệu
    │
    ▼
Cây 2: Tập trung vào 35% dữ liệu sai → Sửa được 20%, còn sai 15%
    │
    ▼
Cây 3: Tập trung vào 15% còn sai → Sửa được 8%, còn sai 7%
    │
    ▼
... (lặp 300 lần) ...
    │
    ▼
Cây 300: Tổng hợp tất cả sai sót đã sửa → Mô hình cuối cùng
```

### 3.4. Hyperparameters đã chọn và lý do

```python
XGBClassifier(
    n_estimators=300,          # Số vòng boosting
    max_depth=6,               # Độ sâu mỗi cây
    learning_rate=0.05,        # Tốc độ học
    subsample=0.8,             # Tỷ lệ mẫu dùng mỗi cây
    colsample_bytree=0.8,      # Tỷ lệ features dùng mỗi cây
    eval_metric='mlogloss',    # Hàm đánh giá
    random_state=42
)
```

| Tham số | Giá trị | Tại sao? |
|---------|---------|----------|
| `n_estimators=300` | 300 vòng | Boosting cần nhiều vòng hơn Bagging vì mỗi cây nhỏ và yếu. 300 vòng × learning_rate 0.05 = vừa đủ để hội tụ. |
| `max_depth=6` | 6 tầng | **Nhỏ hơn RF (10)** vì trong Boosting, mỗi cây nên là "weak learner" (cây yếu). Cây quá sâu + Boosting = overfit rất nhanh. |
| `learning_rate=0.05` | 0.05 | **Tốc độ học chậm** — Mỗi cây mới chỉ sửa 5% lỗi của cây trước. Chậm nhưng ổn định hơn. Nếu đặt 0.3 (mặc định), model sẽ hội tụ nhanh nhưng dễ bị "lao qua" điểm tối ưu. |
| `subsample=0.8` | 80% backend/data/cây | Mỗi cây chỉ dùng 80% dữ liệu → tạo sự đa dạng, giảm overfitting (tương tự Bagging nhưng nhẹ hơn). |
| `colsample_bytree=0.8` | 80% features/cây | Mỗi cây chỉ dùng 80% features → buộc model khám phá nhiều tổ hợp features khác nhau. |
| `eval_metric='mlogloss'` | Multi-class Log Loss | Hàm mất mát chuẩn cho bài toán phân loại 3 lớp. Phạt nặng khi model tự tin sai (ví dụ: đoán MUA 90% nhưng thực tế là BÁN). |

### 3.5. Ưu nhược điểm của XGBoost trong bài toán này

| Ưu điểm | Nhược điểm |
|----------|-----------|
| **Accuracy thường cao hơn RF** — Boosting tinh chỉnh liên tục | **Dễ overfit** — Nếu tham số không cẩn thận, model "nhớ thuộc" data cũ |
| **Bắt patterns phi tuyến phức tạp** — VD: "RSI < 35 VÀ Volume > 2x VÀ VNINDEX tăng thì 80% MUA" | **Nhạy cảm với nhiễu** — Boosting tập trung vào dữ liệu sai, nếu data sai là do nhiễu thì model sẽ học nhiễu |
| **Nhanh** — Thư viện tối ưu hóa cực mạnh bằng C++ | **Khó giải thích** hơn RF — 300 cây tuần tự phức tạp hơn 200 cây độc lập |
| **Xử lý tốt dữ liệu mất cân bằng** | **Cần tinh chỉnh kỹ** — Nhiều hyperparameter hơn RF |

---

## 4. Ensemble — Tại Sao Kết Hợp 2 Mô Hình?

### 4.1. Nguyên lý

Mỗi model có **thế mạnh và điểm mù** riêng:
- **Random Forest** → Ổn định, ít bị "giật" bởi nhiễu, nhưng có thể bỏ lỡ tín hiệu yếu.
- **XGBoost** → Nhạy bén, bắt được tín hiệu tinh vi, nhưng đôi khi "ảo tưởng" (overfit).

Khi kết hợp: **Sai lầm của model này được model kia bù đắp.**

### 4.2. Phương pháp: Trung bình xác suất (Probability Averaging)

```python
# Random Forest đoán:  P = [0.10, 0.25, 0.65]  →  65% MUA
# XGBoost đoán:        P = [0.08, 0.20, 0.72]  →  72% MUA
# ─────────────────────────────────────────────
# Ensemble (Trung bình): P = [0.09, 0.225, 0.685] → 68.5% MUA
```

**Tại sao trung bình xác suất thay vì bỏ phiếu đa số (Majority Voting)?**
- Bỏ phiếu đa số chỉ cho biết "MUA hay BÁN" — mất thông tin về **mức độ tự tin**.
- Trung bình xác suất giữ lại **toàn bộ thông tin**: RF tự tin 65% MUA, XGBoost tự tin 72% MUA → Ensemble tự tin 68.5% MUA.

**Bước hiệu chỉnh prior (quan trọng, đừng bỏ qua):**
Cả 2 model train với trọng số class **cân bằng** (chống "lười" đoán GIỮ), nên xác suất output bị kéo về prior đều 1/3. Trước khi lấy P(MUA), hệ thống nhân ngược theo tỷ lệ nhãn thật của tập train:

```
P_hiệu_chỉnh(c) = P_trung_bình(c) × (prior_thật(c) / ⅓)   →   chuẩn hóa lại tổng = 1
```

Sau hiệu chỉnh, ML Score được đưa về cùng thang với Rule Score (50 = trung tính):

```
ML Score = 50 + 50 × (P(MUA) − P(BÁN))
```

Không dùng thẳng `P(MUA) × 100`: vì MUA chỉ chiếm ~20% nhãn, xác suất này hiếm
khi vượt 0,5, nên điểm ML sẽ luôn thấp và kéo tổng điểm xuống như một lá phiếu
phủ quyết thay vì góp 40% ý kiến.

### 4.3. Khi nào Ensemble đặc biệt hữu ích?

| Tình huống | RF đoán | XGBoost đoán | Ensemble | Hành động |
|-----------|---------|-------------|----------|-----------|
| Cả hai đồng thuận mạnh | MUA (80%) | MUA (85%) | MUA (82.5%) | ✅ Tín hiệu rất đáng tin |
| Cả hai đồng thuận yếu | MUA (55%) | MUA (58%) | MUA (56.5%) | ⚠️ Tín hiệu yếu, nên thận trọng |
| **Bất đồng** | MUA (70%) | GIỮ (45% MUA) | MUA (57.5%) | 🔍 RF thấy tín hiệu nhưng XGBoost không → Giảm confidence, đúng hướng thận trọng |
| **Bất đồng ngược** | GIỮ (40%) | MUA (75%) | MUA (57.5%) | 🔍 XGBoost bắt pattern tinh vi mà RF bỏ lỡ → Vẫn là MUA nhưng yếu |

Trường hợp **bất đồng** là lúc Ensemble phát huy giá trị nhất: Nó **tự động hạ confidence** khi 2 model không thống nhất, giúp hệ thống cẩn trọng hơn.

---

## 5. Walk-Forward Validation — Tại Sao Không Random Split?

### 5.1. Vấn đề của Random Split trong Time Series

```
❌ SAI (Random Split):
[MUA] [BÁN] [GIỮ] [MUA] [BÁN] [GIỮ] [MUA] [BÁN]
  ↑Train ↑Test ↑Train ↑Test ↑Train ↑Test ↑Train ↑Test

→ Model nhìn thấy ngày 15/03 (test) nhưng đã được "dạy" bằng ngày 20/03 (train)
→ Model đang nhìn vào TƯƠNG LAI để dự đoán QUÁ KHỨ → Kết quả ảo!
```

```
✅ ĐÚNG (Walk-Forward mở rộng dần, bỏ 5 phiên ở ranh giới):
[train 50%] [gap 5] [val 10%]
[train 60%] [gap 5] [val 10%]
[train 70%] [gap 5] [val 10%] ...

→ Model chỉ học từ quá khứ, kiểm tra trên tương lai
→ Gap 5 phiên: nhãn T+5 của các dòng cuối train không được chạm vào vùng validation
```

### 5.2. Cách thực hiện trong DSS

- `backend/src/models/validation.py`: walk-forward như trên, báo cáo RF/XGB/Ensemble và 2 baseline.
- `backend/src/backtest/backtester.py`: retrain mỗi 20 phiên, mỗi lần chỉ dùng dữ liệu có nhãn đã biết.
- Model production (`train_ml_models`) train trên **toàn bộ** dữ liệu có nhãn để dùng cả
  giai đoạn gần nhất; metric holdout 20% cuối chỉ in ra để tham khảo.

### 5.3. Ý nghĩa thực tế

Walk-Forward trả lời câu hỏi: *"Nếu tôi train model vào cuối năm 2025 và dùng nó để giao dịch từ đầu năm 2026, kết quả sẽ như thế nào?"*

Metric walk-forward trung thực hơn nhiều so với random split, nhưng vẫn chỉ đo độ đúng
của nhãn. Balanced accuracy ~0,333 (3 lớp) nghĩa là ngang đoán ngẫu nhiên; chất lượng
quyết định phải xem thêm backtest sau phí.

---

## 6. Rule-Based Scoring — Tại Sao Vẫn Cần Khi Đã Có ML?

### 6.1. Lý do tồn tại

| Vấn đề của ML | Cách Rule-Based giải quyết |
|------------|--------------------------|
| **ML có thể overfit** — Học thuộc patterns trong quá khứ mà không lặp lại | Rules dựa trên logic phân tích kỹ thuật đã kiểm chứng hàng chục năm, ổn định hơn |
| **ML là hộp đen** — Khó giải thích tại sao đoán MUA | Rules cho ra **lý do rõ ràng**: "RSI phục hồi từ oversold", "MACD cắt lên Signal" |
| **ML cần data nhiều** — 100+ mẫu mới train được | Rules hoạt động ngay với bất kỳ lượng data nào |
| **ML nhạy cảm với data mới** — Thị trường thay đổi cấu trúc thì ML bối rối | Rules vẫn áp dụng được vì logic cơ bản (RSI quá mua → rủi ro) không thay đổi |

### 6.2. Cách chấm điểm hoạt động

```
Điểm khởi đầu: 50 (Trung tính)
    │
    ├── Kiểm tra Xu hướng ──► +30 đến -30 điểm
    ├── Kiểm tra Động lượng ─► +25 đến -25 điểm
    ├── Kiểm tra Khối lượng ─► +20 đến -20 điểm
    ├── Kiểm tra Biến động ──► +15 đến -15 điểm
    └── Kiểm tra VNINDEX ───► +10 đến -10 điểm
                                    │
                              Giới hạn [0, 100]
                                    │
                              Rule Score cuối cùng
```

### 6.3. Ví dụ cụ thể

**Tình huống: Cổ phiếu FPT ngày 03/09/2026**

```
Điểm khởi đầu:                       50.0

[Xu hướng]
  ✅ Giá > SMA50 > SMA200 (Uptrend)   +10.0  →  60.0
  ✅ MACD > Signal, Histogram tăng     +7.0   →  67.0
     (không cộng thêm "Giá > SMA50" vì uptrend đã bao gồm điều kiện này)

[Động lượng]
  ✅ RSI = 55 (Vùng tích cực)          +7.0   →  74.0

[Khối lượng]
  ✅ OBV tăng 5 phiên                  +5.0   →  79.0

[Biến động]
  ✅ BB position = 0.35 (Tích lũy)     +4.0   →  83.0
  ✅ ATR < 3%                          +3.0   →  86.0

[Thị trường]
  ✅ VNINDEX > SMA50                   +5.0   →  91.0

→ Rule Score = min(91.0, 100) = 91.0 điểm
→ Lý do: "Uptrend mạnh; MACD tích cực"
```

---

## 7. Tổng Hợp So Sánh Thuật Toán

| Tiêu chí | Random Forest | XGBoost | Rule-Based |
|----------|---------------|---------|-----------|
| **Loại** | Ensemble (Bagging) | Ensemble (Boosting) | Heuristic (Kinh nghiệm) |
| **Cách học** | 200 cây độc lập, bỏ phiếu | 300 cây tuần tự, sửa lỗi | Không học — dùng quy tắc cố định |
| **Thế mạnh** | Ổn định, chống nhiễu | Chính xác, bắt pattern tinh vi | Dễ hiểu, luôn có lý do |
| **Điểm yếu** | Bảo thủ, có thể bỏ lỡ | Dễ overfit | Cứng nhắc, bỏ lỡ pattern phức tạp |
| **Vai trò trong DSS** | Ensemble 50% trong ML Score | Ensemble 50% trong ML Score | Trọng số 60% trong tổng điểm |
| **Đã kiểm chứng?** | Walk-forward + backtest (xem README) | Walk-forward + backtest | Chỉ qua backtest nguồn điểm `rule` |

---

## 8. Hạn Chế & Cảnh Báo Quan Trọng

### 8.1. Overfitting (Quá khớp)

**Vấn đề:** Model có thể "nhớ thuộc" dữ liệu quá khứ thay vì học được quy luật tổng quát.

**Dấu hiệu nhận biết:**
- Accuracy trên Train rất cao (> 80%) nhưng trên Test thấp hơn nhiều (< 50%).
- Feature Importance bất thường (ví dụ: `return_20d` quan trọng nhất → model chỉ đang nhìn momentum quá khứ).

**Cách phòng tránh trong DSS:**
- `max_depth` giới hạn (10 cho RF, 6 cho XGBoost) → Ngăn cây quá phức tạp
- `min_samples_leaf=20` → Mỗi quy tắc phải dựa trên ít nhất 20 phiên
- Walk-Forward validation → Kiểm tra trên dữ liệu "chưa từng thấy"
- So sánh với baseline (luôn GIỮ, momentum) và backtest từng nguồn điểm → phát hiện khi ML không thêm giá trị

### 8.2. Data Snooping (Nhìn trộm tương lai)

**Vấn đề:** Vô tình sử dụng thông tin tương lai khi xây features hoặc train model.

**Ví dụ sai:**
```python
# ❌ SAI: Dùng Return tương lai làm feature
df['future_5d_return'] = df['close'].shift(-5) / df['close'] - 1
# Feature này chứa thông tin tương lai → Model đạt accuracy 95% nhưng vô nghĩa
```

**Cách phòng tránh trong DSS:**
- Tất cả features chỉ dùng data **quá khứ và hiện tại** (RSI, SMA, MACD... đều tính từ data đã có)
- Label (`future_return`) chỉ dùng để **gán nhãn training**, KHÔNG được dùng làm feature
- Walk-Forward split đảm bảo tập test luôn nằm SAU tập train theo thời gian

### 8.3. Thị trường thay đổi cấu trúc (Regime Change)

**Vấn đề:** Thị trường chứng khoán không cố định. Quy luật năm 2023 có thể không đúng năm 2026. Ví dụ:
- Giai đoạn 2021: Tiền rẻ, mọi thứ tăng → RSI > 70 vẫn tiếp tục tăng
- Giai đoạn 2022: Lãi suất tăng, mọi thứ giảm → RSI < 30 vẫn giảm tiếp

**Cách giảm thiểu:**
- Model cache tự train lại khi dữ liệu mới hơn model quá `MODEL_MAX_AGE_DAYS` ngày; backtest retrain mỗi 20 phiên
- Dữ liệu 8 năm bao gồm nhiều chu kỳ (2018, 2020, 2021, 2022) thay vì chỉ một giai đoạn
- Đánh giá lại trên nhiều cửa sổ thời gian thay vì tin một kết quả backtest duy nhất
