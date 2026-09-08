# 🧠 Phân Tích Thuật Toán — Hệ thống DSS VN30

Tài liệu giải thích chi tiết **tại sao chọn**, **cách hoạt động**, và **cách sử dụng** từng thuật toán trong hệ thống.

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
Dữ liệu OHLCV (750 phiên)
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
    │                         ML Score = P(MUA) × 100
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
| `subsample=0.8` | 80% data/cây | Mỗi cây chỉ dùng 80% dữ liệu → tạo sự đa dạng, giảm overfitting (tương tự Bagging nhưng nhẹ hơn). |
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
- Trung bình xác suất giữ lại **toàn bộ thông tin**: RF tự tin 65% MUA, XGBoost tự tin 72% MUA → Ensemble tự tin 68.5% MUA. Con số 68.5 này sau đó được dùng trực tiếp làm ML Score.

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
✅ ĐÚNG (Walk-Forward):
[──────── Train (80% đầu) ────────] [── Test (20% cuối) ──]
 2023-09 ──────────────── 2025-12    2026-01 ────── 2026-08

→ Model chỉ học từ quá khứ, kiểm tra trên tương lai
→ Đúng với logic thực tế: bạn không thể dùng data ngày mai để quyết định hôm nay
```

### 5.2. Cách thực hiện trong DSS

```python
# KHÔNG shuffle, KHÔNG random
split_idx = int(len(X) * 0.80)  # 80% đầu tiên

X_train = X.iloc[:split_idx]     # Phiên 1 → 600 (quá khứ)
X_test  = X.iloc[split_idx:]     # Phiên 601 → 750 (gần đây nhất)
```

### 5.3. Ý nghĩa thực tế

Walk-Forward trả lời câu hỏi: *"Nếu tôi train model vào cuối năm 2025 và dùng nó để giao dịch từ đầu năm 2026, kết quả sẽ như thế nào?"*

Đây chính xác là cách hệ thống sẽ được sử dụng trong thực tế. Vì vậy, Accuracy trên tập Walk-Forward test **đáng tin cậy hơn nhiều** so với Random Split.

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
  ✅ Giá > SMA50                       +5.0   →  72.0

[Động lượng]
  ✅ RSI = 55 (Vùng tích cực)          +7.0   →  79.0

[Khối lượng]
  ✅ OBV tăng 5 phiên                  +5.0   →  84.0

[Biến động]
  ✅ BB position = 0.35 (Tích lũy)     +4.0   →  88.0
  ✅ ATR < 3%                          +3.0   →  91.0

[Thị trường]
  ✅ VNINDEX > SMA50                   +5.0   →  96.0

→ Rule Score = min(96.0, 100) = 96.0 điểm
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
| **Vai trò trong DSS** | Model chính (Ensemble 50%) | Model bổ sung (Ensemble 50%) | Nền tảng an toàn (Trọng số 60%) |
| **Khi nào tỏa sáng?** | Thị trường nhiễu, sideway | Thị trường có xu hướng rõ | Mọi lúc — là "phanh hãm" rủi ro |

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
- Rule-Based chiếm 60% → Dù ML overfit thì hệ thống vẫn có 60% dựa trên logic vững

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
- Train lại model định kỳ (mỗi lần chạy DSS đều train lại từ đầu với data mới nhất)
- Rule-Based đóng vai trò "ổn định" — RSI quá mua vẫn là cảnh báo bất kể market regime
- Ensemble giảm rủi ro — nếu 1 model bối rối, model kia có thể vẫn đúng

---

## 9. Tính Kết Hợp Bổ Trợ — Tại Sao Pipeline Mạnh Hơn Từng Phần Riêng Lẻ?

Đây là phần **cốt lõi** của thiết kế hệ thống: 3 thuật toán (Random Forest, XGBoost, Rule-Based) không hoạt động riêng lẻ mà **bổ trợ nhau theo chuỗi pipeline**, mỗi thuật toán đóng một vai trò khác nhau để bù đắp điểm mù của thuật toán còn lại.

### 9.1. Ba "Lớp Phòng Thủ" của hệ thống

Hãy hình dung hệ thống như một đội **3 chuyên gia** cùng đánh giá một cổ phiếu:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DỮ LIỆU CỔ PHIẾU (19 FEATURES)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐
   │ RANDOM      │  │ XGBOOST     │  │ RULE-BASED      │
   │ FOREST      │  │             │  │ (Chuyên gia TA) │
   │             │  │             │  │                 │
   │ Vai trò:    │  │ Vai trò:    │  │ Vai trò:        │
   │ "PHÒNG THỦ" │  │ "TẤN CÔNG"  │  │ "TRỌNG TÀI"    │
   │             │  │             │  │                 │
   │ Ổn định,    │  │ Nhạy bén,   │  │ Kinh nghiệm,   │
   │ ít sai lầm  │  │ phát hiện   │  │ giải thích      │
   │ lớn         │  │ cơ hội sớm  │  │ được, an toàn   │
   └──────┬──────┘  └──────┬──────┘  └────────┬────────┘
          │                │                   │
          ▼                ▼                   │
   ┌──────────────────────────┐                │
   │   ENSEMBLE (50/50)       │                │
   │   ML Score = P(MUA)×100  │                │
   └────────────┬─────────────┘                │
                │                              │
                ▼ (40%)                        ▼ (60%)
        ┌───────────────────────────────────────────┐
        │     TỔNG ĐIỂM = Rule × 60% + ML × 40%    │
        │                                           │
        │  → Quyết định cuối cùng phải đi qua       │
        │    CẢ 3 THUẬT TOÁN mới được ra tín hiệu   │
        └───────────────────────────────────────────┘
```

### 9.2. Mỗi thuật toán giải quyết MỘT LOẠI VẤN ĐỀ khác nhau

| Vấn đề cần giải quyết | Thuật toán phụ trách | Tại sao thuật toán này? |
|------------------------|---------------------|------------------------|
| **"Đa số trường hợp bình thường, pattern cơ bản"** | Random Forest | RF là "bình quân" của 200 cây → Rất giỏi ở những tình huống phổ biến, lặp đi lặp lại. Ví dụ: RSI < 30 + Volume tăng → Thường tăng giá. RF bắt tốt pattern này vì nó xuất hiện nhiều lần trong dữ liệu. |
| **"Trường hợp hiếm, pattern phức tạp ẩn"** | XGBoost | XGBoost tập trung vào những điểm dữ liệu mà các cây trước **đoán sai**. Nghĩa là nó chuyên bắt những pattern mà RF bỏ lỡ — ví dụ: "RSI = 45 (trung tính) NHƯNG MACD histogram đang tăng tốc VÀ volume đột biến ở phiên trước" → Đây là combo tinh vi mà RF không nhìn ra. |
| **"Đảm bảo quyết định có logic, không phi lý"** | Rule-Based | Dù ML nói gì, Rule-Based kiểm tra lại bằng logic phân tích kỹ thuật cơ bản. Nếu RSI = 90 (quá mua cực mạnh) mà ML vẫn bảo MUA → Rule-Based sẽ trừ điểm nặng, kéo tổng điểm xuống → Ngăn chặn quyết định phi lý. |

### 9.3. Bốn tình huống thực tế minh họa sự bổ trợ

#### Tình huống A: CẢ 3 ĐỒNG THUẬN → Tín hiệu cực mạnh

```
Bối cảnh: Cổ phiếu HPG — Giá vừa breakout khỏi kháng cự, volume x2

Random Forest:  P(MUA) = 72%  → ML Score RF = 72
XGBoost:        P(MUA) = 78%  → ML Score XGB = 78
─────────────────────────────────────────────────
Ensemble ML Score:              (72 + 78) / 2 = 75.0

Rule-Based:     SMA uptrend (+10), MACD cắt lên (+7), Volume bùng nổ (+10)
                RSI phục hồi (+10), BB bật từ dưới (+8), VNINDEX tốt (+5)
                Rule Score = 50 + 10 + 7 + 10 + 10 + 8 + 5 = 100 → Cap: 100

TOTAL = 100 × 0.6 + 75 × 0.4 = 60 + 30 = 90.0
→ 🟢 MUA MẠNH (90/100)
→ Tất cả thuật toán cùng đồng ý → Tín hiệu đáng tin cậy nhất!
```

**Ý nghĩa pipeline:** Khi cả 3 "chuyên gia" đều gật đầu, xác suất đúng là cao nhất. Đây là lúc nhà đầu tư nên tự tin nhất.

---

#### Tình huống B: ML MUA nhưng RULES BÁN → Pipeline "phanh" lại

```
Bối cảnh: Cổ phiếu TCB — ML phát hiện pattern ẩn nhưng chỉ báo TA tiêu cực

Random Forest:  P(MUA) = 68%  → ML Score RF = 68
XGBoost:        P(MUA) = 74%  → ML Score XGB = 74
─────────────────────────────────────────────────
Ensemble ML Score:              (68 + 74) / 2 = 71.0

Rule-Based:     SMA downtrend (-8), Death Cross (-10), MACD < Signal (-10)
                RSI quá mua (-8), Volume xả hàng (-10)
                Rule Score = 50 - 8 - 10 - 10 - 8 - 10 = 4.0

TOTAL = 4 × 0.6 + 71 × 0.4 = 2.4 + 28.4 = 30.8
→ 🟠 BÁN (31/100) — dù ML bảo MUA!
```

**Ý nghĩa pipeline:** ML có thể bị **overfit** — nó "thấy" một pattern trong quá khứ (ví dụ: mỗi lần RSI = 45 kết hợp một feature nào đó thì giá tăng), nhưng lần này bối cảnh hoàn toàn khác (Death Cross, downtrend nặng). Rule-Based đóng vai trò **phanh hãm khẩn cấp**: "Không, tất cả chỉ báo cơ bản đều nói GIẢM. ML có thể sai!" Trọng số 60% cho Rules đảm bảo hệ thống **ưu tiên an toàn** hơn mạo hiểm.

---

#### Tình huống C: RULES TRUNG TÍNH nhưng ML phát hiện tín hiệu sớm

```
Bối cảnh: Cổ phiếu FPT — Sideway, chỉ báo TA đều ở vùng trung tính, 
          nhưng ML phát hiện tổ hợp features ẩn (OBV slope + MACD hist slope + vol_ratio)

Random Forest:  P(MUA) = 52%  → ML Score RF = 52 (RF bảo thủ, chưa chắc)
XGBoost:        P(MUA) = 78%  → ML Score XGB = 78 (XGB nhạy, bắt pattern sớm)
─────────────────────────────────────────────────
Ensemble ML Score:              (52 + 78) / 2 = 65.0

Rule-Based:     Mọi chỉ báo đều ở vùng trung tính (±0 điểm)
                Rule Score = 50.0 (điểm khởi đầu, không cộng không trừ)

TOTAL = 50 × 0.6 + 65 × 0.4 = 30 + 26 = 56.0
→ ⚪ GIỮ (56/100) — chưa đủ tự tin để MUA
```

**Ý nghĩa pipeline:** Đây là trường hợp thú vị nhất!
- **XGBoost** bắt được pattern sớm mà RF và Rules không thấy (vì Boosting chuyên tìm pattern ẩn trong dữ liệu mà Bagging bỏ lỡ).
- **Random Forest** thận trọng hơn (52% — gần như đồng xu), đóng vai trò "kiểm tra lại" — nếu RF không thấy thì pattern đó có thể chưa đủ mạnh.
- **Rule-Based** trung tính (50) — chưa có tín hiệu rõ ràng.
- **Kết quả:** GIỮ (56) — Pipeline **không vội vàng MUA** chỉ vì 1 trong 3 thuật toán nói MUA. Nó chờ thêm xác nhận. Nếu ngày hôm sau XGBoost vẫn mạnh VÀ RF bắt đầu tăng VÀ Rules bắt đầu dương → Lúc đó mới chuyển sang MUA.

---

#### Tình huống D: RF và XGB BẤT ĐỒNG → Ensemble tự giảm confidence

```
Bối cảnh: Cổ phiếu VNM — Thị trường biến động, tín hiệu lẫn lộn

Random Forest:  P(MUA) = 70%  → "Tôi thấy pattern uptrend cơ bản"
XGBoost:        P(BÁN) = 60%, P(MUA) = 25%  → "Tôi thấy pattern đảo chiều tinh vi"
─────────────────────────────────────────────────
Ensemble ML Score:              (70 + 25) / 2 = 47.5 → GIỮ vùng

Rule-Based:     SMA trung tính (+5), RSI = 62 (tích cực nhẹ, +7)
                Rule Score = 50 + 5 + 7 = 62

TOTAL = 62 × 0.6 + 47.5 × 0.4 = 37.2 + 19.0 = 56.2
→ ⚪ GIỮ (56/100)
```

**Ý nghĩa pipeline:**
- RF thấy pattern đơn giản (uptrend) → MUA 70%.
- XGBoost thấy pattern phức tạp hơn (dấu hiệu đảo chiều mà RF bỏ lỡ) → BÁN 60%.
- Ensemble **tự động hạ confidence** về 47.5 khi 2 model bất đồng. Đây là cơ chế **tự bảo vệ tuyệt vời** — thay vì chọn theo 1 bên, nó nói *"2 chuyên gia không đồng ý → Tốt nhất là CHỜ"*.

### 9.4. Bảng tổng hợp: Khi nào thuật toán nào "cứu" hệ thống?

| Tình huống thị trường | RF một mình | XGB một mình | Rules một mình | **Pipeline kết hợp** |
|----------------------|-------------|-------------|---------------|---------------------|
| **Uptrend rõ ràng** (mọi chỉ báo tốt) | ✅ MUA đúng | ✅ MUA đúng | ✅ MUA đúng | ✅ MUA MẠNH (cả 3 đồng thuận) |
| **Sideway nhiễu** (tín hiệu lẫn lộn) | ⚠️ Hay đoán sai (RF bảo thủ → GIỮ khi nên MUA) | ❌ Hay đoán sai (XGB quá nhạy → MUA khi nên GIỮ) | ⚠️ Trung tính mãi (không bao giờ phát hiện cơ hội sớm) | ✅ GIỮ chờ (an toàn, không mất tiền) |
| **Đảo chiều bất ngờ** (tin xấu, crash) | ❌ Phản ứng chậm (200 cây "bỏ phiếu" mất thời gian cập nhật) | ✅ Phát hiện sớm (Boosting nhạy với dữ liệu mới) | ✅ Death Cross, RSI lao dốc → BÁN ngay | ✅ BÁN (XGB + Rules cùng cảnh báo, dù RF chậm) |
| **ML bị overfit** (pattern quá khứ không lặp lại) | ❌ Overfit | ❌ Overfit | ✅ Vẫn đúng (logic cơ bản không thay đổi) | ✅ Rules chiếm 60% → kéo hệ thống về đúng hướng |
| **Pattern phức tạp ẩn** (tổ hợp 5+ features) | ⚠️ Bắt được một phần | ✅ Bắt tốt nhất | ❌ Không bắt được (Rules chỉ kiểm tra từng chỉ báo) | ✅ ML Score tăng, đẩy tổng điểm lên (dù Rules không thấy) |
| **Thị trường thay đổi cấu trúc** (Regime change) | ❌ Bối rối | ❌ Bối rối | ⚠️ Vẫn hoạt động nhưng thiếu context | ✅ Rules giữ nền (60%), ML re-train với data mới |

### 9.5. Nguyên lý thiết kế: "Không ai hoàn hảo, nhưng kết hợp lại thì gần hoàn hảo"

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Random Forest     XGBoost         Rule-Based                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                     │
│   │ Điểm mù: │    │ Điểm mù: │    │ Điểm mù: │                    │
│   │ Pattern   │    │ Overfit   │    │ Pattern   │                    │
│   │ tinh vi   │───▶│           │    │ phức tạp  │                    │
│   │           │    │ XGB BÙ    │    │           │                    │
│   └──────────┘    └──────────┘    └──────────┘                     │
│        │                │               │                           │
│        │           ┌────┘               │                           │
│        │           ▼                    │                           │
│        │    ┌──────────┐               │                           │
│        │    │ Điểm mù: │               │                           │
│        └───▶│ Overfit   │◀──────────────┘                           │
│             │           │                                           │
│             │ RF BÙ:    │  Rule BÙ:                                 │
│             │ Trung bình│  60% trọng số                             │
│             │ 200 cây   │  giữ an toàn                              │
│             │ giảm      │                                           │
│             │ variance  │                                           │
│             └──────────┘                                            │
│                                                                     │
│   KẾT QUẢ: Mỗi điểm mù của thuật toán A được B hoặc C bù đắp     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Tóm lại:** Pipeline không phải là 3 thuật toán chạy riêng rồi cộng lại. Nó là một **hệ thống kiểm tra chéo (cross-validation)** trong đó:

1. **RF** cho baseline ổn định → *"Đa số 200 cây tôi nghĩ là MUA"*
2. **XGBoost** tinh chỉnh thêm → *"Tôi đồng ý/không đồng ý, vì tôi thấy pattern mà RF không thấy"*
3. **Ensemble** trung bình 2 ý kiến → *"Khi 2 chuyên gia bất đồng, tôi tự giảm confidence"*
4. **Rule-Based** kiểm tra lần cuối → *"ML bảo MUA nhưng RSI 90 + Death Cross? Tôi trừ 30 điểm. Không được MUA."*
5. **Tỷ trọng 60/40** ưu tiên an toàn → *"Trong tài chính, tránh lỗ quan trọng hơn tìm lãi"*

Kết quả: **Không có thuật toán đơn lẻ nào trong 3 thuật toán trên có thể đạt được sự cân bằng giữa nhạy bén (phát hiện cơ hội) và an toàn (tránh bẫy) như khi kết hợp cả 3 trong pipeline.**

