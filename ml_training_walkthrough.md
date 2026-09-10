# Giai thich tham so train ML cua DSS

Tai lieu nay la nguon tham khao cho tham so train va validation cua DSS. Cac
gia tri duoi day duoc doi chieu voi `config.py`, `src/models/ml_models.py`,
`src/models/validation.py`, `src/features/features.py` va report hien co.

## 1. Hai quy trinh khac nhau

DSS co hai quy trinh can tach biet:

| Quy trinh | Lenh | Muc dich | Cach chia du lieu |
|---|---|---|---|
| Production train | `python main.py ... --retrain` | Tao model de tinh khuyen nghi hien tai | Single chronological holdout 80/20 |
| Evaluation | `./run.sh evaluate ...` | Do kha nang tong quat hoa theo thoi gian | Expanding walk-forward, gap 5 phien |

`TEST_SIZE_RATIO=0.2` chi dieu khien production train. No khong thay the cho
walk-forward evaluation. Ket qua trong `reports/` duoc tao boi quy trinh
evaluation.

## 2. Bai toan va du lieu

### 2.1. Du lieu dau vao

Moi ma su dung nen ngay OHLCV:

```text
time, open, high, low, close, volume
```

Mac dinh:

| Tham so | Gia tri | File | Y nghia |
|---|---:|---|---|
| `LOOKBACK_YEARS` | `3` | `config.py` | Khoang lich su du kien |
| `DATA_COUNT` | `750` | `config.py` | So dong toi da gui cho API |
| `END_DATE` | Ngay hien tai | `config.py` | Ngay ket thuc fetch |
| `START_DATE` | Hien tai - 3 nam | `config.py` | Ngay bat dau fetch |
| `RANDOM_STATE` | `42` | `config.py` | Lap lai ket qua ngau nhien cua mo hinh |

`DATA_COUNT=750` khop xap xi 3 nam giao dich, nhung so dong thuc te co the
thap hon do ngay nghi, ma moi niem yet, du lieu thieu hoac cache khong day du.

### 2.2. Loi nhuan muc tieu

```text
future_return[t] = close[t + 5] / close[t] - 1
```

Voi label fixed:

```text
future_return >= +0.03  -> BUY  (1)
future_return <= -0.03  -> SELL (-1)
con lai                  -> HOLD (0)
```

| Tham so | Gia tri | Anh huong |
|---|---:|---|
| `ML_FORWARD_DAYS` | `5` | Mo hinh nhin truoc 5 phien |
| `ML_PROFIT_THRESHOLD` | `0.03` | Bien BUY/SELL ±3% |

`ML_FORWARD_DAYS` khong phai so ngay lich ma la so phien co du lieu. Nam dong
cuoi khong co gia tuong lai nen label la NaN va bi loai khoi tap train.

### 2.3. Cac chien luoc label

`build_features_and_labels()` ho tro ba strategy:

#### `fixed`

Dung mot nguong co dinh ±3% trong 5 phien. Day la strategy production mac dinh,
de giai thich va de so sanh.

#### `volatility`

Nguong moi mau duoc tinh:

```text
dynamic_threshold = max(0.03, 1.5 x ATR% / 100)
```

Co phieu bien dong manh can muc dich loi nhuan lon hon moi duoc gan BUY/SELL.

#### `triple_barrier`

Voi moi diem vao:

- Upper barrier = gia vao x `1.03`.
- Lower barrier = gia vao x `0.97`.
- Quan sat high/low cua 5 phien tiep theo.
- Barrier nao cham truoc se quyet dinh BUY hoac SELL.
- Neu khong cham barrier nao thi gan HOLD.

Triple barrier co the phu hop hon voi logic giao dich, nhung khong duoc tu
dong dung cho production chi vi metrics phan loai cao hon; can kiem tra bang
backtest hop le sau nay.

## 3. Feature dung de train

### 3.1. Baseline: 19 feature

```python
FEATURE_COLUMNS = [
    "price_vs_sma50", "price_vs_sma200", "sma50_vs_sma200",
    "macd_hist", "macd_hist_slope",
    "rsi", "stoch_k", "stoch_d", "williams_r",
    "bb_position", "bb_width", "atr_pct",
    "vol_ratio", "obv_slope",
    "return_1d", "return_5d", "return_20d",
    "vnindex_vs_sma50", "vnindex_return_5d",
]
```

Tat ca la feature tuong doi hoac chi bao, giup mo hinh it phu thuoc vao muc
gia tuyet doi giua cac ma.

### 3.2. Extended: 27 feature

Them 8 feature:

```text
return_3d
return_10d
return_60d
rsi_slope
volume_zscore_20
vnindex_return_20d
vnindex_volatility_20d
relative_strength_5d
```

`feature_set` duoc dung boi `evaluate.py` de chon tap cot danh gia. Ham
`build_features_and_labels()` van tao cac cot extended trong ca hai truong hop;
production train hien tai truyen `FEATURE_COLUMNS` baseline tu
`src.models.ml_models`.

### 3.3. Xu ly NaN

Chi bao co window lon se tao NaN o dau chuoi, vi du `sma_200`. Truoc khi train,
code dung:

```python
df.dropna(subset=FEATURE_COLUMNS + ["label"])
```

Khong duoc dien label cuoi chuoi bang gia tri gia tao. Khi du doan hien tai,
neu row co feature NaN hoac khong co cap model, `predict_ml_score()` tra ve
`50.0` thay vi tu y thay NaN bang 0.

## 4. Tham so chia tap du lieu

### 4.1. Production train: 80/20 theo thoi gian

Trong `train_ml_models()`:

```python
split = int(len(X) * (1 - TEST_SIZE_RATIO))
X_train = X.iloc[:split]
X_test = X.iloc[split:]
```

| Tham so | Gia tri | Y nghia |
|---|---:|---|
| `TEST_SIZE_RATIO` | `0.2` | 20% dong cuoi lam test |
| `MIN_TRAIN_ROWS` | `100` | Duoi 100 dong thi bo qua train |
| `RANDOM_STATE` | `42` | Seed cho RF/XGBoost |

Day la holdout theo thoi gian, khong phai random split. Model chi hoc phan
qua khu va test tren phan moi hon.

### 4.2. Evaluation: expanding walk-forward

Trong `walk_forward_validate()`:

| Tham so | Gia tri mac dinh | Y nghia |
|---|---:|---|
| `initial_fraction` | `0.5` | Tap train ban dau bang 50% usable rows |
| `validation_fraction` | `0.1` | Moi fold validation bang 10% rows |
| `gap` | `ML_FORWARD_DAYS = 5` | Bo qua 5 rows giua train va validation |
| `shuffle` | Khong co | Giu nguyen thu tu thoi gian |

Vi du:

```text
[train 50%] [gap 5] [validation 10%]
[train 60%] [gap 5] [validation 10%]
[train 70%] [gap 5] [validation 10%]
...
```

Tap train mo rong dan sau moi fold. `gap=5` la purge gap: label cua mot diem
cuoi tap train co the nhin toi 5 phien tuong lai, nen can cach validation de
tranh chong lan thong tin.

## 5. Random Forest

```python
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": 42,
}
```

| Tham so | Gia tri | Tac dong thuc te |
|---|---:|---|
| `n_estimators` | `200` | So cay doc lap. Tang so cay thuong giam variance nhung tang thoi gian train/predict |
| `max_depth` | `10` | Gioi han do sau. Thap hon giam overfit nhung co the underfit |
| `min_samples_leaf` | `20` | Moi la phai co it nhat 20 mau. Day la regularization manh, tranh quy tac tu 1-2 mau |
| `class_weight` | `balanced` | Tang trong so lop SELL/BUY khi HOLD chiem da so |
| `random_state` | `42` | Ket qua co the lap lai |

### Vi sao dung `class_weight="balanced"`?

Voi fixed label hien tai, HOLD chiem phan lon. Neu khong can bang lop, model co
the dat accuracy kha bang cach doan HOLD qua nhieu nhung khong bat duoc tin
hieu BUY/SELL. `balanced` dat trong so xap xi nghich dao tan suat lop trong
tap train.

### Tac dong khi thay doi

- Giam `max_depth` hoac tang `min_samples_leaf`: mo hinh on dinh hon, it nhay
  voi nhiem nhung co nguy co bo sot pattern.
- Tang `max_depth` hoac giam `min_samples_leaf`: bat pattern chi tiet hon,
  nhung rui ro overfit cao hon.
- Tang `n_estimators`: thuong khong thay doi bias nhieu; chu yeu giam variance
  va tang chi phi tinh toan.

## 6. XGBoost

```python
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": 42,
}
```

| Tham so | Gia tri | Tac dong thuc te |
|---|---:|---|
| `n_estimators` | `300` | So vong boosting/cay. Nhieu vong hon cho phep hoc dan dan |
| `max_depth` | `6` | Do sau moi cay. Cay nong han che viec hoc vet |
| `learning_rate` | `0.05` | Muc dong gop cua moi cay. Nho thi cham hon nhung thuong an toan hon |
| `subsample` | `0.8` | Moi cay dung 80% mau, tao ngau nhien va giam overfit |
| `colsample_bytree` | `0.8` | Moi cay dung 80% feature, giam phu thuoc vao mot nhom cot |
| `eval_metric` | `mlogloss` | Log loss cho bai toan 3 lop, phat nang khi tu tin sai |
| `random_state` | `42` | Lap lai ket qua ngau nhien |

### Class balancing cua XGBoost

XGBoost khong nhan `class_weight="balanced"` trong cau hinh nay. Code tu tinh
class weight bang `compute_class_weight()` tren `y_train`, sau do truyen vao
`fit(..., sample_weight=sample_weight)`. Walk-forward validation dung cung co
che nay.

### Moi quan he `n_estimators` va `learning_rate`

Hai tham so nay co trade-off:

- Learning rate nho: moi cay sua it, can nhieu cay hon.
- Learning rate lon: hoc nhanh, nhung de vuot qua diem tot nhat va overfit.
- Khong nen tang dong thoi ca `learning_rate` va `n_estimators` ma khong kiem
  tra lai validation theo thoi gian.

## 7. Ensemble va prior calibration

### 7.1. Mapping lop

Model sklearn/XGBoost dung lop bat dau tu 0:

```text
label goc -1 (SELL) -> class 0
label goc  0 (HOLD) -> class 1
label goc  1 (BUY)  -> class 2
```

### 7.2. Trung binh xac suat

Voi mot row:

```text
avg_proba = (RF.predict_proba(row) + XGB.predict_proba(row)) / 2
```

Day la ensemble 50/50. No giu lai muc do tu tin, khac voi majority vote chi
tra ve mot lop.

### 7.3. Hieu chinh prior

Model duoc train voi class balancing, nhung phan bo lop that cua tap train
khong nhat thiet la 1/3. Code luu `class_priors_` vao ca hai model va hieu
chinh:

```text
corrected[c] = avg_proba[c] x (true_prior[c] / (1/3))
corrected = corrected / sum(corrected)
ML Score = corrected[BUY] x 100
```

Muc dich la tranh dien giai truc tiep xac suat cua model da duoc can bang lop
nhu xac suat BUY trong phan bo that.

### 7.4. Fallback

`ML Score=50` neu:

- Khong co ca hai file model.
- So dong trainable nho hon `MIN_TRAIN_ROWS`.
- Dong hien tai con feature NaN.

Gia tri 50 la trung tinh, de Rule-based tiep tuc dong gop thay vi tao tin hieu
ML gia.

## 8. Chi so danh gia

`src/models/metrics.py` tinh cac chi so cho ba lop `SELL/HOLD/BUY`:

| Metric | Cach hieu |
|---|---|
| Accuracy | Ty le du doan dung tren tong mau; de bi HOLD ap dao danh lua |
| Balanced Accuracy | Trung binh recall cua cac lop |
| Precision | Trong cac du doan mot lop, bao nhieu mau dung |
| Recall | Trong cac mau that cua mot lop, bat duoc bao nhieu |
| F1 | Trung binh dieu hoa giua precision va recall |
| Macro F1 | Trung binh F1 cua ba lop, moi lop co trong so nhu nhau |
| Confusion matrix | Ma tran actual/predicted de xem lop bi nham sang dau |

Trong bai toan nay, can uu tien `Macro F1`, `Balanced Accuracy`, BUY/SELL
precision va recall thay vi chi nhin Accuracy.

### 8.1. Dinh nghia va cach danh gia tot/xau

Khong co mot nguong co dinh nao dam bao mo hinh giao dich tot. Cac nguong
duoi day la moc tham khao de doc nhanh; quyet dinh cuoi cung phai dua tren
ket qua out-of-sample, do on dinh giua cac fold va backtest co phi/slippage.

#### Accuracy

```text
Accuracy = so du doan dung / tong so mau
```

Accuracy tra loi: "Mo hinh doan dung bao nhieu phan tram mau?". Tuy nhien,
HOLD dang chiem khoang 62,2% trong fixed label. Mo hinh luon doan HOLD co the
dat Accuracy xap xi 62,2% ma khong bat duoc bat ky tin hieu BUY/SELL nao.

| Muc tham khao | Dien giai |
|---|---|
| Thap hon baseline HOLD | Yeu, neu cung khong cai thien BUY/SELL |
| Xap xi baseline HOLD | Co the chi dang doan lop HOLD |
| Cao hon baseline va BUY/SELL van co recall | Co dau hieu cai thien, can xem them Macro F1 |
| Cao nhung BUY/SELL recall gan 0 | Khong tot cho muc tieu giao dich |

Accuracy chi nen dung de tham khao, khong dung lam metric chon model chinh.

#### Balanced Accuracy

Balanced Accuracy la trung binh recall cua cac lop:

```text
Balanced Accuracy = (Recall_SELL + Recall_HOLD + Recall_BUY) / 3
```

Chi so nay khong de lop HOLD dong mau lan ap ket qua. Voi 3 lop, muc ngau
nhien gan `0,333` neu ca ba lop xuat hien day du.

| Gia tri tham khao | Dien giai |
|---:|---|
| `< 0,333` | Yeu hon muc ngau nhien can bang |
| `0,333` | Gan muc ngau nhien; chua cho thay kha nang tach lop |
| `0,40 - 0,50` | Cai thien nhe den vua; can kiem tra do on dinh |
| `0,50 - 0,60` | Co tin hieu phan loai dang chu y |
| `> 0,60` | Kha tot neu lap lai tren nhieu fold/ma, khong chi mot giai doan |

#### Precision

Precision tra loi: "Trong cac lan model bao BUY/SELL, bao nhieu lan dung?"

```text
Precision_BUY = TP_BUY / (TP_BUY + FP_BUY)
Precision_SELL = TP_SELL / (TP_SELL + FP_SELL)
```

- BUY precision cao: it mua nham hon.
- SELL precision cao: it ban/canh bao nham hon.
- Precision thap: nhieu false positive, tin hieu phat ra nhieu nhung khong
  dang tin.

| Gia tri tham khao | Dien giai |
|---:|---|
| `< 0,25` | Yeu; phan lon tin hieu hanh dong la bao nham |
| `0,25 - 0,40` | Han che; chi co y nghia neu loi trung binh lon hon lo trung binh |
| `0,40 - 0,60` | Dang chu y, can xac nhan bang backtest |
| `> 0,60` | Tot ve mat phan loai neu support du lon va on dinh |

Khong nen chi toi da precision bang cach dat nguong qua chat, vi khi do recall
co the giam ve gan 0 va bo lo hau het co hoi.

#### Recall

Recall tra loi: "Trong cac mau that su la BUY/SELL, model bat duoc bao nhieu?"

```text
Recall_BUY = TP_BUY / (TP_BUY + FN_BUY)
Recall_SELL = TP_SELL / (TP_SELL + FN_SELL)
```

- BUY recall cao: bat duoc nhieu co hoi tang gia hon.
- SELL recall cao: bat duoc nhieu dot giam/canh bao rui ro hon.
- Recall thap: model bo sot nhieu su kien cua lop do.

| Gia tri tham khao | Dien giai |
|---:|---|
| `< 0,20` | Rat yeu, gan nhu khong bat duoc lop hanh dong |
| `0,20 - 0,40` | Bat duoc mot phan nhung con bo sot nhieu |
| `0,40 - 0,60` | Co the su dung de sang loc, can xem precision |
| `> 0,60` | Tot neu precision khong sut qua thap |

Mot model tot khong nhat thiet co precision va recall cung cuc dai. Can chon
trade-off phu hop: uu tien precision neu muon it tin hieu nham, uu tien recall
neu muon khong bo lo nhieu co hoi/canh bao.

#### F1 va Macro F1

F1 la trung binh dieu hoa cua precision va recall:

```text
F1 = 2 x Precision x Recall / (Precision + Recall)
```

`Macro F1` la trung binh F1 cua SELL, HOLD va BUY. No quan trong hon F1 hoac
Accuracy tong khi du lieu mat can bang, vi moi lop co cung trong so.

| Muc tham khao | Dien giai |
|---:|---|
| Thap hon baseline cua mo hinh nguyen | Khong cai thien so voi chien luoc don gian |
| Gan baseline | Mo hinh chua tach lop hanh dong ro |
| Cao hon baseline ro rang va on dinh | Co cai thien dang tin hon |
| `> 0,40` | Muc tham khao kha, can xac nhan nhieu fold/ma |
| `> 0,50` | Tot ve phan loai neu khong do mot fold bat thuong tao ra |

Voi phan bo fixed label tong hien tai, chien luoc luon doan HOLD co Macro F1
xap xi `0,256`. Tuy nhien baseline phai tinh lai theo tung ma/tung fold khi
so sanh chi tiet.

#### Confusion matrix

Ma tran co dang:

```text
                    Du doan SELL  Du doan HOLD  Du doan BUY
Thuc te SELL              TP             FN            FN
Thuc te HOLD              FP             TP            FP
Thuc te BUY               FN             FN            TP
```

Doc theo hang `actual`:

- Duong SELL: xem cac dot giam bi nham thanh HOLD hay BUY.
- Duong BUY: xem co hoi tang bi bo sot hay bi nham thanh HOLD/SELL.
- Duong HOLD: xem model co phat qua nhieu tin hieu hanh dong trong thi truong
  di ngang hay khong.

Mot confusion matrix tot khong chi co nhieu o duong cheo; cac o ngoai duong
cheo cua SELL/BUY phai duoc giam, dac biet la SELL bi doan BUY va BUY bi doan
SELL. Neu gan nhu toan bo du lieu nam o cot HOLD, Accuracy co the cao nhung
model khong co gia tri hanh dong.

#### Tieu chi ket luan mo hinh

Co the xem mot cau hinh la **co tiem nang** khi dong thoi thoa cac dieu kien:

1. Balanced Accuracy cao hon baseline ngau nhien va baseline luon doan HOLD.
2. Macro F1 cao hon baseline ro rang.
3. BUY/SELL precision va recall khong gan 0.
4. Ket qua khong dao dong qua manh giua cac fold va cac ma.
5. Loi nhuan ky vong sau phi/slippage duong trong backtest out-of-sample.

Co the xem la **chua dat** neu:

- Accuracy cao nhung Balanced Accuracy gan `0,333`.
- BUY hoac SELL recall bang 0 tren nhieu fold.
- Macro F1 khong vuot baseline HOLD.
- Mot fold rat cao nhung cac fold con lai rat thap.
- Metrics phan loai tot nhung backtest am sau chi phi.

Trong giao dich, metric cuoi cung khong phai Accuracy ma la loi nhuan ky vong:

```text
Expected value = ty le thang x loi trung binh
                 - ty le thua x lo trung binh
                 - chi phi giao dich/slippage
```

Expected value phai duong tren du lieu chua dung de chon mo hinh. Can xem them
maximum drawdown, Sharpe/Sortino, turnover va so luong lenh truoc khi su dung
thuc te.

## 9. Ket qua report hien co

Snapshot `reports/fixed_baseline/` gom 30 ma trong label distribution. Co 29 ma
co aggregate walk-forward metrics; cac ma co qua it du lieu co the khong tao
du fold hop le.

### 9.1. Phan bo fixed label

| Lop | So mau | Ty le xap xi |
|---|---:|---:|
| SELL | 3.586 | 16,9% |
| HOLD | 13.211 | 62,2% |
| BUY | 4.449 | 20,9% |
| Tong | 21.246 | 100% |

### 9.2. Trung binh theo ma cua aggregate metrics

| Model | Balanced Accuracy | Macro F1 | BUY P | BUY R | SELL P | SELL R |
|---|---:|---:|---:|---:|---:|---:|
| RF | 0,341 | 0,295 | 0,221 | 0,367 | 0,200 | 0,343 |
| XGB | 0,333 | 0,317 | 0,231 | 0,250 | 0,219 | 0,154 |
| Ensemble 50/50 | 0,333 | 0,318 | 0,242 | 0,267 | 0,218 | 0,168 |

Cac so tren la macro-average giua cac ma, khong phai pooled metric co trong so
theo so mau. Chung cho thay:

- Ensemble co Macro F1 nhinh hon RF/XGB trong snapshot nay.
- BUY/SELL recall con thap va khong on dinh.
- Accuracy cao o mot so ma co the chi phan anh viec doan HOLD.
- Chua the ket luan model tao loi nhuan neu chua co backtest giao dich.

### 9.3. Ket luan snapshot hien tai

Voi Ensemble 50/50 trong report nay:

- Balanced Accuracy `0,333`: gan muc ngau nhien 3 lop, chua cho thay kha nang
  tach SELL/HOLD/BUY tot.
- Macro F1 `0,318`: cao hon baseline luon doan HOLD uoc tinh `0,256`, nhung
  muc cai thien con nho.
- BUY Precision `0,242` va Recall `0,267`: tin hieu BUY con yeu, vua bao nham
  nhieu vua bo sot nhieu co hoi.
- SELL Precision `0,218` va Recall `0,168`: kha nang canh bao giam gia con yeu,
  dac biet bo sot phan lon mau SELL.

Vi vay, ML hien tai **chua du tot de dung doc lap** nhu mot bo du bao BUY/SELL.
No chi nen duoc xem la mot thanh phan bo sung cho Rule-based cho den khi co
backtest out-of-sample xac nhan loi nhuan ky vong sau chi phi.

## 10. Tuning hien tai

`src/models/tune.py` so sanh:

| Ung vien | RF | XGBoost |
|---|---|---|
| `baseline` | Config hien tai | Config hien tai |
| `regularized` | depth 6, leaf 30 | depth 3, lr 0,03, 400 cay |
| `responsive` | depth 12, leaf 10 | depth 4, lr 0,05, 250 cay |

Tuning dung cung expanding walk-forward va purge gap. Ket qua hien tai chi la
thu nghiem tren tap ma nho, khong du co so de thay doi production tren toan bo
VN30. Muon chon cau hinh moi can:

1. Danh gia tren nhieu ma va nhieu giai doan.
2. Theo doi Macro F1 va BUY/SELL metrics, khong chi Accuracy.
3. Kiem tra do on dinh giua cac fold.
4. Sau cung phai kiem tra bang backtest co phi va slippage.

## 11. Tham so khong phai tham so train nhung anh huong dau ra

| Tham so | Gia tri | Anh huong |
|---|---:|---|
| `WEIGHT_RULE_BASED` | `0.60` | Ty trong Rule score trong Total score |
| `WEIGHT_ML_MODEL` | `0.40` | Ty trong ML score trong Total score |
| `SCORE_STRONG_BUY` | `75` | Nguong MUA MANH |
| `SCORE_BUY` | `60` | Nguong MUA |
| `SCORE_HOLD` | `40` | Nguong GIU |
| `SCORE_SELL` | `25` | Nguong BAN; duoi nguong la BAN MANH |

Day la tham so decision engine, khong duoc hoc tu `y_train`. Thay doi chung co
the doi tin hieu ma khong lam model phan loai tot hon.

## 12. Cach thay doi tham so an toan

1. Chi thay doi mot nhom tham so moi lan.
2. Ghi lai commit/config va tap ma da dung.
3. Chay `evaluate` voi cung label strategy va feature set.
4. So sanh theo tung fold, tung ma va aggregate.
5. Kiem tra confusion matrix de phat hien model chi doan HOLD.
6. Khong chon model dua tren mot ma co ket qua dep.
7. Khong dung ket qua train de khang dinh loi nhuan giao dich.

Khong co gia tri `max_depth=10`, `learning_rate=0.05` hay threshold `3%` nao
duoc dam bao la toi uu vinh vien. Chung la cau hinh hien tai, can duoc xac
nhan lai khi du lieu, regime thi truong hoac muc tieu giao dich thay doi.

## 13. Han che quan trong

- Du lieu chuoi thoi gian co regime change; quy luat qua khu co the khong lap
  lai.
- Fixed label bi mat can bang lop, HOLD chiem da so.
- Class balancing giup model khong bo qua lop nho nhung co the tang false
  positive.
- Prior calibration chi dieu chinh cach doc xac suat, khong tao them thong
  tin du bao.
- Chua co calibration day du cho probability, chi co prior correction.
- Chua co backtest production voi phi giao dich, slippage, position sizing,
  Sharpe hay maximum drawdown.
- Report CSV la snapshot tai thoi diem chay, khong tu dong cap nhat neu data
  thay doi.

## 14. Lenh tham khao

```bash
# Train lai hai ma tu cache
python main.py --no-fetch --symbols FPT,ACB --retrain

# Danh gia fixed baseline
./run.sh evaluate --symbols FPT,ACB 

# Thu triple barrier
./run.sh evaluate --symbols FPT,ACB --label-strategy triple_barrier

# Thu extended features
./run.sh evaluate --symbols FPT,ACB --feature-set extended

# So sanh cau hinh
./run.sh tune --symbols FPT,ACB
```
