**Tổng quan ML từ dữ liệu đến kết quả giao dịch — kiểm tra ngày 22/09/2026**

Phạm vi: pooled Random Forest + XGBoost, 19 feature baseline, nhãn excess
T+20, rổ VN30 theo lịch sử. Dữ liệu kết thúc ngày 17/09/2026. Số liệu dựa trên
đối chứng `weighting_ablation_20260922`, không phải một lần kiểm chứng tương
lai mới. Cấu hình không balancing + decay 2 năm được dùng để minh họa chi tiết
vì có lợi nhuận cả kỳ cao nhất trong bốn cấu hình; website chưa tự động chuyển
sang cấu hình này.

Nhận định: hệ thống có bằng chứng về khả năng chọn lọc và xếp hạng cổ phiếu
trong dữ liệu lịch sử, nhưng năng lực phân loại còn hạn chế. Lợi nhuận mô phỏng
khá cao đi cùng drawdown lớn, dữ liệu chưa đầy đủ tuyệt đối và chưa có tập test
cuối cùng chưa từng dùng để lựa chọn cấu hình. Chưa đủ bằng chứng để khẳng định
hiệu quả giao dịch tương lai.

**1. Bài toán thực sự đang học là gì?**

Một mẫu là một cặp **mã cổ phiếu – ngày giao dịch**. Ví dụ, FPT ở ngày t có
19 đặc trưng tính từ dữ liệu đến hết phiên t. Mô hình dự đoán một trong ba
nhãn dựa trên lợi nhuận sau 20 nến hợp lệ của cổ phiếu:

`excess_return = return_stock − return_VNINDEX`

- BUY: excess return ≥ 3,6 điểm phần trăm.
- SELL: excess return ≤ −3,6 điểm phần trăm.
- HOLD: nằm giữa hai ngưỡng.

Ngưỡng 3,6% bằng tham số lợi nhuận 3% cộng tham số chi phí nhãn 0,6%.
Đây là cách định nghĩa mục tiêu, không phải bảo đảm lợi nhuận ròng.
Cổ phiếu giảm 2% khi VNINDEX giảm 8% vẫn có nhãn BUY vì vượt chỉ số 6 điểm
phần trăm. SELL cũng không tự động có nghĩa là một giao dịch bán khống.

Vì vậy phải phân biệt ba phép đo: dự đoán đúng **nhãn**, xếp hạng đúng **cổ
phiếu**, và kiếm được **tiền sau chi phí** khi áp dụng luật giao dịch.

**2. Đầu vào và độ phủ dữ liệu**

Nguồn là dữ liệu OHLCV ngày qua vnstock, lưu SQLite; VNINDEX cung cấp bối cảnh
thị trường và mốc so sánh nhãn. Dữ liệu giá trải từ 19/09/2018 đến 17/09/2026.

| Kiểm tra | Kết quả | Ý nghĩa |
|---|---:|---|
| Tổng nến trong SQLite | 85.675 | Gồm cổ phiếu và VNINDEX; chưa phải số mẫu train |
| Mã trong SQLite | 45 cổ phiếu + VNINDEX | Rổ thay đổi qua thời gian; không phải 45 mã cùng thuộc VN30 |
| Nến bị đánh dấu sai | 72, khoảng 0,084% | Không được đưa vào ML; đây chỉ là lỗi phát hiện được bằng quy tắc hiện có |
| Nến hợp lệ có volume bằng 0 | 0 | Không chứng minh thanh khoản luôn đủ để khớp lệnh |
| Cổ phiếu tạo được feature trong đối chứng | 44 | ROS chỉ có 34 dòng nên bị bỏ qua |
| Nến hợp lệ của 44 cổ phiếu | 83.577 | Bao gồm cả ngày không thuộc rổ VN30 |
| Dòng thuộc rổ tại ngày tương ứng | 59.206 | Điều kiện chọn mẫu theo lịch sử |
| Dòng thuộc rổ và đủ feature | 53.416 | 5.790 dòng chưa đủ feature, chủ yếu do cửa sổ khởi tạo |
| Dòng thuộc rổ, đủ feature và nhãn | 52.637 | Kho mẫu khả dụng trên toàn lịch sử; không phải train ban đầu |
| Dòng đủ feature nhưng chưa có nhãn | 779 | 600 thiếu tương lai T+20; 179 thiếu giá VNINDEX tại đầu/cuối nhãn |

Trong 1.512 phiên backtest, có 1.400 phiên có đủ 30 mã và 112 phiên chỉ có dữ
liệu của 29 mã thuộc rổ. Không tự coi một mã thiếu là một mã có lợi nhuận bằng 0
trong đánh giá phân loại.

Số mẫu đủ feature + nhãn và thuộc rổ rất khác nhau: TCX 6, MCH 11, BSR 69,
VPL 119; các mã lâu năm thường khoảng 1.700–1.770. Đây không phải số nến gốc
của các mã đó. Pooled cho phép học từ những mã khác nhưng không chứng minh
khả năng dự đoán riêng cho một mã chỉ có vài mẫu đánh giá.

SQLite đang loại OHLC không hợp lệ, chặn ngày tương lai, chống trùng mã–ngày;
luồng fetch tránh lưu nến hôm nay chưa đóng cửa. Tuy vậy, kiểm tra logic OHLC
không phát hiện được mọi mức giá sai nhưng trông hợp lệ. Chưa có bằng chứng
đối chiếu toàn bộ giá với nguồn độc lập, kiểm chứng đầy đủ điều chỉnh sự kiện
doanh nghiệp hoặc lưu mọi phiên bản dữ liệu đúng như đã biết tại ngày lịch sử.
Rổ trước 03/08/2020 còn dùng xấp xỉ rổ gốc.

**3. Feature và chất lượng mẫu**

| Nhóm | Số feature | Thông tin muốn biểu diễn |
|---|---:|---|
| Xu hướng, SMA và MACD | 5 | Giá đang ở đâu so với xu hướng; động lượng thay đổi ra sao |
| RSI, Stochastic, Williams %R | 4 | Trạng thái dao động của giá |
| Bollinger và ATR | 3 | Vị trí giá trong biên và mức biến động |
| Volume ratio và OBV slope | 2 | Hoạt động giao dịch và thay đổi dòng khối lượng |
| Return 1, 5, 20 phiên | 3 | Đà giá trong quá khứ |
| Đặc trưng VNINDEX | 2 | Bối cảnh thị trường chung |

SMA200 khiến phần đầu chuỗi chưa có feature; T+20 khiến phần cuối chưa có
nhãn. Hai việc này là hệ quả của thiết kế mẫu, không phải mặc định là lỗi.
Kiểm tra hiện tại không thấy giá trị vô cực trong 19 cột feature.

Các điểm cần đánh giá tiếp: tương quan và trùng thông tin giữa chỉ báo, độ ổn
định phân phối theo năm/mã, outlier và mức đóng góp ngoài mẫu của từng nhóm.
MACD histogram và độ dốc MACD vẫn theo đơn vị giá, nên tên gọi “19 feature
tương đối” chưa hoàn toàn chính xác khi pooling. Cây quyết định không bắt buộc
chuẩn hóa z-score, nhưng ý nghĩa kinh tế khác nhau giữa các thang giá vẫn cần
kiểm tra. Hiện chưa có ablation feature đủ để kết luận nhóm nào tạo ra lợi thế.

44.450 mẫu đánh giá không phải 44.450 thử nghiệm độc lập: nhãn T+20 của hai
ngày liền nhau chia sẻ phần lớn tương lai; các cổ phiếu cùng ngày cùng chịu
tác động thị trường. Chưa ước lượng số mẫu độc lập hiệu dụng. Không nên tính
khoảng tin cậy như thể mọi dòng được lấy ngẫu nhiên và độc lập.

**4. Phân phối nhãn hiện tại**

| Nhãn | Số mẫu đánh giá | Tỷ lệ |
|---|---:|---:|
| SELL | 13.326 | 29,98% |
| HOLD | 19.470 | 43,80% |
| BUY | 11.654 | 26,22% |
| Tổng | 44.450 | 100% |

Tỷ lệ HOLD khoảng 66% được nêu trước đây không phải tỷ lệ của tập excess T+20
này. Phân phối cũng thay đổi theo thời gian: HOLD chiếm 37,06% năm 2021 và
56,28% năm 2024. Vì vậy luôn báo cáo support — số mẫu thật của mỗi lớp —
theo từng giai đoạn, không chỉ một tỷ lệ chung.

Tất cả 25 khối dự đoán đã có nhãn trong đối chứng này đều có đủ ba lớp ở cấp
pooled. Khối thứ 26 chưa có nhãn T+20 nào tại ngày dữ liệu kết thúc; không được
diễn giải nó là một khối có precision bằng 0. Một mã riêng hoặc khoảng thời
gian ngắn vẫn có thể thiếu lớp. Metric được quy ước về 0 để tránh phép chia
không xác định không có nghĩa là đã có bằng chứng về chất lượng lớp đó.

**5. Train, validation và test được tổ chức ra sao?**

Train dùng để học tham số. Validation dùng để chọn cấu hình và ngưỡng. Test
cuối cùng dùng để kiểm chứng lựa chọn đã chốt. Nếu xem kết quả test rồi tiếp
tục chọn cấu hình, tập đó đã tham gia quá trình lựa chọn.

Đối chứng chạy expanding walk-forward: học từ quá khứ, dự đoán khối tiếp
theo, rồi mở rộng lịch sử train. Những mẫu từng được kiểm tra có thể đi vào
train ở lần sau khi nhãn đã biết; điều này phù hợp với vận hành tuần tự.

| Thành phần | Thiết kế/số liệu |
|---|---|
| Khoảng backtest | 25/08/2020–17/09/2026; 1.512 phiên |
| Lịch retrain | Mỗi 60 phiên; tổng 26 lần |
| Train lần đầu | 7.607 mẫu; feature dates 10/07/2019–27/07/2020 |
| Nhãn train lần đầu | Kết thúc muộn nhất 24/08/2020, trước ngày retrain |
| Train lần cuối | 52.282 mẫu; feature dates đến 30/07/2026 |
| Nhãn train lần cuối | Kết thúc muộn nhất 27/08/2026, trước retrain 28/08 |
| Loại rò rỉ | Chỉ nhận `label_end < ngày retrain`; các mã cùng ngày nằm cùng khối |
| Đánh giá phân loại | 44.450 dòng, 1.486 ngày có nhãn; đến 17/08/2026 |
| Test cuối cùng chưa từng xem | Chưa có cho việc chọn cấu hình thắng đối chứng |

Trong 63.880 dòng đã chấm điểm, 45.248 dòng thuộc rổ; 19 dòng trong số đó thiếu
dự đoán ML hợp lệ. Còn 45.229 dòng có dự đoán hợp lệ, trừ 779 dòng chưa có nhãn
thì còn 44.450 dòng đánh giá. Các mẫu chưa có nhãn vẫn có thể có tín hiệu để
mô phỏng giao dịch; không dùng chúng để đo precision/recall.

Lỗi lệch ngày hai đầu mút nhãn và bỏ sót phần cuối validation đã được sửa.
Kiểm tra evaluate trên FPT/ACB có 5 fold mỗi mã là kiểm tra luồng per-symbol,
không thay cho kết quả pooled ở trên. Chia ngẫu nhiên dữ liệu thời gian có
thể cho ước lượng tổng quát hóa không đáng tin; tài liệu
[scikit-learn về cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
giải thích lý do cần kiểm tra trên tương lai.

**6. Mô hình và bốn mẫu đối chứng**

RF có 200 cây; XGBoost có 300 vòng boosting, kết hợp xác suất hai model với tỷ
lệ 50/50. Cả hai nhận cùng trọng số mẫu. Balancing tăng đóng góp tương đối của
các lớp ít gặp; không đồng nghĩa phạt mọi dự đoán HOLD. Decay 2 năm giảm một
nửa trọng số sau mỗi 730,5 ngày so với ngày mới nhất trong train: 2 năm trước
có trọng số tương đối 0,5; 4 năm trước 0,25. Trọng số được chuẩn hóa khi fit.

| Cấu hình | BUY precision tại điểm ≥60 | Rank IC | Lợi nhuận cả kỳ | Max drawdown |
|---|---:|---:|---:|---:|
| Balanced, đều | 36,54% | 0,0701 | +109,66% | −49,77% |
| Không balanced, đều | 36,77% | 0,0733 | +205,25% | −49,59% |
| Balanced, decay | 36,06% | 0,0659 | +170,81% | −42,93% |
| Không balanced, decay | 36,95% | 0,0640 | +233,09% | −41,47% |

Giữ cùng dữ liệu, nhãn, seed, mô hình và luật giao dịch giúp so sánh tác động
của trọng số trong phạm vi thử nghiệm. Tuy nhiên, chỉ có một chuỗi lịch sử
được dùng chung; bốn cấu hình không tạo thành bốn mẫu kiểm chứng độc lập.

**7. Đọc metric phân loại và baseline**

Các số sau dùng argmax xác suất ensemble sau hiệu chỉnh trọng số lớp, trước
khi áp ngưỡng điểm giao dịch. Baseline được tính lại trên đúng 44.450 dòng:
luôn HOLD; momentum BUY nếu return 5 phiên ≥1%, SELL nếu ≤−1%, còn lại HOLD.

| Phương pháp | Accuracy | Balanced accuracy | Macro F1 | BUY precision | BUY recall |
|---|---:|---:|---:|---:|---:|
| Luôn HOLD | 43,80% | 33,33% | 0,203 | 0% | 0% |
| Luật momentum đơn giản | 33,33% | 34,77% | 0,333 | 27,59% | 41,99% |
| ML không balanced + decay | 43,74% | 37,19% | 0,345 | 36,71% | 14,04% |

- Accuracy: tỷ lệ nhãn đoán đúng trên tất cả mẫu. Ở đây ML gần bằng luôn
  HOLD, nên không thể chỉ dùng accuracy để quảng bá chất lượng mô hình.
- Precision BUY: trong các mẫu được đoán BUY, bao nhiêu thực sự có nhãn BUY.
  Cao hơn giúp lọc bớt tín hiệu sai theo định nghĩa nhãn.
- Recall BUY: trong tất cả mẫu thật sự BUY, mô hình tìm được bao nhiêu.
  ML hiện tìm được khoảng 14%, bỏ qua nhiều cơ hội theo định nghĩa nhãn.
- F1 của một lớp: trung bình điều hòa precision và recall,
  `2PR/(P+R)`. Macro F1 lấy trung bình F1 của ba lớp với trọng số bằng nhau.
- Balanced accuracy: trung bình recall của các lớp có mặt; ở tập này có đủ
  ba lớp. 37,19% tốt hơn baseline HOLD 33,33%, nhưng mức cải thiện còn nhỏ.
- Support: số mẫu thật của mỗi lớp; predicted count: số mẫu được dự đoán
  thành lớp đó. Hai chỉ số giúp phát hiện mô hình bỏ qua hoặc dự đoán quá
  nhiều một lớp.

BUY F1 của ML là 0,203, thấp hơn momentum 0,333 vì đánh đổi precision lấy
recall. Không có một phương pháp thắng mọi metric. Baseline phân loại ở đây
chưa phải backtest danh mục momentum với cùng giới hạn vốn và chi phí.

Ma trận nhầm lẫn của ML; hàng là nhãn thật, cột là nhãn dự đoán:

| Thật / Dự đoán | SELL | HOLD | BUY |
|---|---:|---:|---:|
| SELL | 2.568 | 9.420 | 1.338 |
| HOLD | 2.749 | 15.239 | 1.482 |
| BUY | 2.013 | 8.005 | 1.636 |

Mô hình dự đoán HOLD 32.664/44.450 mẫu, khoảng 73,5%; phần lớn BUY thật bị
đưa về HOLD. Như vậy, cần kiểm tra cả bỏ sót tín hiệu, không chỉ “mua quá nhiều”.

**8. Từ xác suất sang điểm và tín hiệu**

Điểm ML là `50 + 50 × (P(BUY) − P(SELL))`. Điểm 60 chỉ có nghĩa là chênh lệch
hai xác suất bằng 0,2; không có nghĩa xác suất thắng là 60%.

Với ngưỡng mua ≥60, cấu hình minh họa có:

- 3.580 tín hiệu mã–ngày; 3.570 đã có nhãn, 10 chưa có nhãn.
- 1.319 tín hiệu đúng nhãn BUY; precision = 1.319/3.570 = 36,95%.
- Recall = 1.319/11.654 = 11,32%.
- Tỷ lệ BUY trong cả tập là 26,22%; precision trên cao gấp khoảng 1,41 lần
  tỷ lệ nền. Đây là lift mô tả, chưa phải kết luận có ý nghĩa thống kê.
- 2.251 tín hiệu không đạt nhãn BUY gồm 1.303 HOLD và 948 SELL. Không được
  gọi cả 2.251 tín hiệu này là giao dịch thua tiền.

Precision của nhãn argmax, precision của ngưỡng điểm và win rate giao dịch
có mẫu số khác nhau. Tín hiệu lặp mỗi ngày khi đã có vị thế không tự tạo
thành lệnh mới; còn giới hạn 5 vị thế và thứ tự điểm khi chọn mua.

Chưa có báo cáo log loss, Brier score và reliability diagram cho xác suất
ngoài mẫu của đối chứng. Hiệu chỉnh trọng số lớp không tự chứng minh xác
suất đã được calibration. Brier/log loss đo chất lượng dự báo xác suất;
đường calibration kiểm tra tần suất thật trong các nhóm xác suất dự báo.
Xem [tài liệu calibration](https://scikit-learn.org/stable/modules/calibration.html).

**9. Rank IC đo điều gì?**

Code tính tương quan hạng Spearman giữa điểm và lợi nhuận cổ phiếu T+20 trong
từng ngày, sau đó lấy trung bình. IC dương nghĩa là cổ phiếu có điểm cao có
xu hướng đứng cao hơn về lợi nhuận tương lai. IC 0,0640 không phải accuracy
6,4% và không phải lợi nhuận 6,4%.

Cả bốn cấu hình có IC trung bình dương trong 7/7 năm; 2020 và 2026 không đủ
năm. Tuy vậy, không balanced/không decay có IC cao nhất 0,0733; cấu hình có
lợi nhuận cao nhất chỉ đạt 0,0640. Danh mục chỉ mua một phần cổ phiếu theo
ngưỡng và giới hạn vốn, nên thứ hạng IC và lợi nhuận không bắt buộc giống nhau.

IC hiện dùng lợi nhuận tuyệt đối của cổ phiếu. Với cùng hai ngày đầu/cuối,
trừ cùng lợi nhuận VNINDEX không đổi thứ hạng; khi thiếu nến làm ngày cuối
khác nhau giữa mã thì không còn bảo đảm này. Chưa có khoảng tin cậy xử lý
phụ thuộc thời gian, bảng lợi nhuận các nhóm điểm hay kiểm chứng precision@k.

**10. Backtest: hiệu quả kinh tế và mức rủi ro**

Luật hiện tại: tín hiệu tại close, mua ở open phiên sau, tối đa 5 vị thế;
thoát theo T+20 hoặc điểm <25 khi đủ điều kiện thời gian nắm giữ của simulator.
Phí 0,15% mỗi chiều, thuế bán 0,1%, slippage 0,1% mỗi chiều: tổng chi phí danh
nghĩa khoảng 0,6% một vòng giao dịch.

| Chỉ số cấu hình minh họa | Kết quả | Ý nghĩa |
|---|---:|---|
| Lợi nhuận cả kỳ | +233,09% | Vốn mô phỏng 1 thành 3,3309; không phải 233% mỗi năm |
| Lệnh đã đóng | 264 | Nhỏ hơn nhiều số tín hiệu mã–ngày |
| Win rate | 57,58% | 152 lệnh lãi, 112 lệnh không lãi sau chi phí |
| Lãi trung bình/lệnh | +2,64% | Trung bình số học return lệnh; khác mức tăng của toàn tài khoản |
| Lệnh thắng trung bình | +10,70% | Cần đọc cùng quy mô lệnh thua |
| Lệnh thua trung bình | −8,30% | Win rate riêng lẻ không đủ đánh giá chiến lược |
| Sharpe | 0,989 | Trung bình return ngày / độ lệch chuẩn × √252; code giả định lãi suất phi rủi ro 0 |
| Max drawdown | −41,47% | Mức sụt từ đỉnh vốn xuống đáy lớn nhất; cần tăng khoảng 70,9% để hồi phục từ đáy đó |
| Exposure trung bình | 68,31% | Tỷ trọng vốn đang nằm trong cổ phiếu, không phải tỷ lệ thời gian có tín hiệu |

Benchmark cùng kỳ: danh mục chia đều theo rổ +139,92%, VNINDEX +108,53%.
Danh mục chia đều tái cân bằng hằng ngày không phí; VNINDEX là thay đổi chỉ số.
Cả hai chưa có cấu trúc chi phí và exposure tương đương chiến lược. Chênh lệch
return không tự động là alpha đã điều chỉnh rủi ro.

Backtest chưa mô phỏng đầy đủ lô giao dịch, giới hạn thanh khoản, không khớp
lệnh hoặc phí trượt giá phụ thuộc quy mô. Vị thế mở cuối kỳ được đánh dấu theo
close, chưa trừ chi phí thanh lý giả định. Nhãn close-to-close trên 20 nến hợp
lệ cũng khác giao dịch next-open và thời gian nắm giữ theo lịch chung.

Kết quả phụ thuộc giai đoạn: cấu hình minh họa lỗ 30,97% năm 2022 và 1,15%
trong phần năm 2026 đã quan sát. Giai đoạn 2024–17/09/2026, không balanced/
không decay đạt +102,66%, cao hơn không balanced/decay +84,83%.

**11. Đối chiếu với thực hành đánh giá ML**

Không có ngưỡng accuracy, F1 hay IC chung để tuyên bố mọi bài toán tài chính
đã “đạt chuẩn”. Cần đặt metric theo mục đích sử dụng, baseline phù hợp và cách
kiểm chứng đáng tin. Tài liệu
[scikit-learn về lựa chọn metric](https://scikit-learn.org/stable/modules/model_evaluation.html)
phân biệt đánh giá dự đoán với đánh giá quyết định.

| Hạng mục | Hiện trạng | Bằng chứng còn thiếu |
|---|---|---|
| Đầu vào | Có kiểm tra OHLC, thời gian, trùng và nến lỗi | Đối soát nguồn, độ phủ phiên, dữ liệu điều chỉnh và phiên bản lịch sử |
| Thiết kế mẫu | Có nhãn rõ, feature quá khứ, pooling và membership | Ảnh hưởng mẫu chồng lấn, dữ liệu ít theo mã, độ lệch horizon khi thiếu nến |
| Chống leakage | Đã sửa căn ngày nhãn, purge và validation cuối | Kiểm chứng toàn quy trình dữ liệu theo trạng thái lịch sử |
| Baseline | Có HOLD, momentum cùng tập đánh giá | Đối chứng danh mục đơn giản cùng luật/chi phí/exposure |
| Generalization | Có walk-forward, nhiều giai đoạn thị trường | Tập test cuối chưa xem, dự đoán tương lai đã khóa cấu hình |
| Overfitting | Chưa đủ thông tin để kết luận mức độ | Khoảng cách train–validation, độ ổn định theo seed và lựa chọn cấu hình |
| Xác suất | Có predict_proba và hiệu chỉnh lớp | Calibration, log loss, Brier ngoài mẫu |
| Thống kê | Có metric điểm và theo năm | Khoảng tin cậy theo khối thời gian; kiểm soát lựa chọn sau nhiều thử nghiệm |
| Vận hành | Có cache và artifact cấu hình | Theo dõi freshness, missingness, drift, tỷ lệ dự đoán thiếu và kết quả khi nhãn trưởng thành |

Việc chọn kết quả tốt nhất sau nhiều backtest có thể làm ước lượng hiệu quả
lạc quan; bài nghiên cứu
[The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
phân tích vấn đề này. Đây là lý do coi đối chứng hiện tại là bằng chứng nghiên
cứu, chưa phải xác nhận cuối cùng.

**12. Thứ tự ưu tiên tiếp theo**

1. Khóa định nghĩa mục tiêu: chọn cổ phiếu vượt chỉ số hay lãi ròng tuyệt đối;
   giữ cách đọc nhãn/metric nhất quán với mục tiêu đó.
2. Báo cáo chất lượng dữ liệu theo mã và ngày: nến thiếu, nến sai, độ mới,
   số mẫu usable, phân phối nhãn và ảnh hưởng thay đổi rổ.
3. Chốt ứng viên và luật trước khi kiểm chứng tiếp. Dùng validation cho lựa
   chọn, dành một giai đoạn chưa xem hoặc dự đoán tương lai cho xác nhận cuối.
4. Bổ sung độ bất định và tính ổn định: metric theo năm/mã, bootstrap theo
   khối thời gian phù hợp, độ nhạy chi phí, top-k/nhóm điểm và benchmark cùng
   giả định giao dịch. Không chọn cấu hình chỉ bằng return lớn nhất.
5. Theo dõi dự đoán thật theo thời gian; chỉ đánh giá precision khi nhãn T+20
   đã trưởng thành. Tách điểm đầu tư khỏi xác suất có lãi trong giao diện.

Nguồn nội bộ: `WEIGHTING_EXPERIMENT.md`, `backend/src/features/features.py`,
`backend/src/models/metrics.py`, `backend/src/models/validation.py`,
`backend/src/models/ml_models.py`, `backend/src/backtest/backtester.py`.
Các phép đếm và baseline cùng mẫu vừa được tính lại, lưu tại
`backend/reports/weighting_ablation_20260922/end_to_end_audit.json`.
Không đổi thuật toán, huấn luyện lại hoặc thay cấu hình website trong lượt
rà soát này.
