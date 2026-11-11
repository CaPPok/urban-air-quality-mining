# BÁO CÁO BÀI TẬP LỚN — KHAI PHÁ DỮ LIỆU (CO3029)

**Đề tài 17:** Khai phá Dữ liệu Môi trường & Chất lượng Không khí Đô thị (Urban Air Quality Mining & AQI Prediction)

**Mục tiêu:** Dự báo nồng độ bụi mịn PM2.5 ở giờ tiếp theo và suy ra cấp độ chất lượng không khí theo quy chuẩn AQI được nhóm lựa chọn, phục vụ cảnh báo cộng đồng.

## CHƯƠNG 1. MỞ ĐẦU

### 1.1. Bối cảnh và tính cấp thiết
- Ô nhiễm không khí đô thị và nhu cầu theo dõi bụi mịn PM2.5.
- Vì sao dự báo trước 1 giờ hữu ích cho cảnh báo sớm.

### 1.2. Phát biểu bài toán
- Input: chuỗi quan trắc PM2.5 đến giờ t và khí tượng (nhiệt độ, áp suất, độ ẩm/điểm sương, gió, mưa), thời gian và trạm đo.
- Output 1: PM2.5 ở giờ t+1 (µg/m³).
- Output 2: mức AQI suy ra từ PM2.5 24 giờ khi đủ dữ liệu và có breakpoint chính thức; không đánh đồng với AQI tổng hợp nhiều chất ô nhiễm.

### 1.3. Mục tiêu nghiên cứu
- Tiền xử lý dữ liệu đa trạm/chuỗi thời gian.
- So sánh baseline, Random Forest, XGBoost, LSTM.
- Đo chất lượng dự báo bằng RMSE, MAE, R²; phân loại AQI và báo cáo Macro-F1 khi có nhãn/chuẩn đối chiếu phù hợp.
- Đóng gói demo Streamlit.

### 1.4. Phạm vi, giả định và giới hạn
- Bộ dữ liệu chọn (ví dụ: Beijing Multi-Site Air-Quality, UCI), khoảng thời gian, số trạm, phạm vi địa lý.
- Dự báo 1 giờ và điều kiện cần cho AQI 24h.
- Phân biệt dữ liệu lịch sử với dữ liệu thời gian thực (cần API và cơ chế cập nhật thực tế).

### 1.5. Bố cục báo cáo
- Mô tả ngắn nội dung 5 chương.

**Hình/bảng đề xuất:** Sơ đồ mục tiêu hệ thống, bảng phát biểu input/output.

## CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

### 2.1. Tổng quan Data Mining và CRISP-DM/KDD
- 5 giai đoạn học phần yêu cầu: hiểu bài toán/dữ liệu → tiền xử lý → mô hình hóa/đánh giá → diễn giải/đóng gói → báo cáo.

### 2.2. Các khái niệm môi trường
- PM2.5: định nghĩa, đơn vị µg/m³; đặc trưng biến thiên theo thời gian/khí tượng.
- Chỉ số AQI, 6 mức Tốt/Trung bình/Kém/Xấu/Rất xấu/Nguy hại.
- Nêu tên và nguồn chuẩn AQI đã chọn, quy tắc trung bình 24h, breakpoint, cách làm tròn và giới hạn phạm vi.

### 2.3. Kỹ thuật xử lý dữ liệu chuỗi thời gian
- Missing values; interpolation vs. forward-fill; outliers.
- Time features, lag 1/3/24h, rolling average 24h, one-hot biến hướng gió/trạm.
- Data leakage và chia train/validation/test theo thời gian.

### 2.4. Thuật toán hồi quy
- Baseline persistence: PM2.5(t+1) ≈ PM2.5(t).
- Random Forest Regressor: bagging và tổng hợp cây.
- XGBoost Regressor: gradient boosting, regularization.
- LSTM: cửa nhớ/cổng và lý do phù hợp chuỗi thời gian.

### 2.5. Các thước đo đánh giá
- RMSE = sqrt[(1/n) Σ(y_i−ŷ_i)²].
- MAE = (1/n) Σ|y_i−ŷ_i|.
- R² = 1 − SSE/SST (chú ý có thể âm).
- Precision/Recall/F1 của từng lớp và Macro-F1 = trung bình F1 theo lớp.
- Confusion Matrix; TimeSeriesSplit và vai trò cross-validation.

**Hình/bảng đề xuất:** Sơ đồ RF, gradient boosting, LSTM; bảng đối chiếu thuật toán; công thức các chỉ số.

## CHƯƠNG 3. PHƯƠNG PHÁP NGHIÊN CỨU

### 3.1. Quy trình thực nghiệm tổng thể
- Dataset → làm sạch/EDA → đặc trưng → chia thời gian → baseline/ML/LSTM → đánh giá → tính AQI → dashboard.
- Vẽ pipeline và mô tả mục tiêu từng thành phần.

### 3.2. Thu thập, mô tả và hiểu dữ liệu
- Tên tập, giấy phép/nguồn, số quan sát, trạm đo, thời điểm, chu kỳ lấy mẫu.
- Từ điển thuộc tính: PM2.5, TEMP, PRES, DEWP, RAIN, WSPM, wd, station, time.
- Thống kê missing, outlier, tương quan, độ lệch phân phối, tính mùa vụ.

### 3.3. Làm sạch và feature engineering
- Kiểm tra bản ghi trùng, thời điểm khuyết, PM2.5 âm, dữ liệu thiếu; chiến lược điền khuyết giải thích rõ.
- Chỉ dùng thông tin ≤ t cho dự báo t+1; không nội suy xuyên tập test vào train.
- Lag/rolling, encoding, numerical imputation fit trên train; chỉ rõ cột đích và các cột dự báo.

### 3.4. Thiết kế tập dữ liệu và thực nghiệm
- Train/validation/test theo mốc thời gian, ví dụ 70/15/15, không shuffle.
- TimeSeriesSplit theo timestamp cho mọi trạm, tránh một timestamp nằm đồng thời trong train và validation.
- Baseline và cách chọn mô hình theo validation; test chỉ để báo cáo cuối.

### 3.5. Xây dựng các mô hình
- RF: đặc trưng vào, siêu tham số, phạm vi tuning.
- XGBoost: tương tự.
- LSTM: cách tạo cửa sổ chuỗi, scaler, kiến trúc, epoch, batch size, early stopping.
- Bảng so sánh các thiết lập để nhóm tái lập kết quả.

### 3.6. Tính và phân loại AQI
- Chọn chuẩn AQI chính thức, trích dẫn và lập bảng breakpoint hợp lệ.
- Với dự báo giờ t+1, minh họa cách lập trung bình PM2.5 24 giờ bằng 23 giờ quan trắc gần nhất cộng 1 giờ dự báo (nếu dữ liệu đầy đủ và chuẩn quy định cho phép).
- Phân loại 6 mức, cách xác định nhãn thực tế cho đánh giá Macro-F1.
- Phân biệt AQI từ PM2.5 với AQI tổng hợp của nhiều chất ô nhiễm.

### 3.7. Thiết kế ứng dụng
- Các chức năng: tải dữ liệu/chọn trạm, xem xu hướng, dự báo, mức AQI/cảnh báo, hiển thị vị trí đo khi có tọa độ.
- Phân tách backend inference và giao diện Streamlit; cấu trúc repo và sơ đồ kiến trúc.
- Hệ thống theo thời gian thực: chỉ ghi nhận là có nếu đã triển khai API/live refresh thật.

**Hình/bảng đề xuất:** Pipeline, Data Dictionary, bảng siêu tham số, sơ đồ kiến trúc hệ thống, mockup giao diện.

## CHƯƠNG 4. KẾT QUẢ VÀ THẢO LUẬN

### 4.1. Kết quả EDA và tiền xử lý
- Thống kê missing trước/sau, các biến được chọn, các biểu đồ chính.
- Phân tích biến thiên PM2.5 theo trạm, theo tháng, theo giờ; lưu ý giới hạn dữ liệu.

### 4.2. So sánh hiệu năng mô hình
- Bảng: baseline/RF/XGBoost/LSTM × RMSE, MAE, R² (validation/test).
- Hình thực tế so với dự báo; đường residual và sai số theo trạm hoặc khung giờ.
- Không điền số minh họa thành số liệu kết quả chính thức; chỉ dùng số thực nghiệm.

### 4.3. Tối ưu và phân tích đặc trưng
- Kết quả CV/tuning; phân tích feature importance/SHAP nếu đã thực hiện.
- Thảo luận ưu/nhược RF, XGB, LSTM và độ phức tạp huấn luyện.

### 4.4. Kết quả phân loại AQI
- Confusion Matrix; Precision/Recall/F1 từng lớp, Macro-F1.
- Nêu chuẩn AQI, nhãn tham chiếu, cách chuyển PM2.5 sang cấp ô nhiễm, mức độ bất cân bằng lớp.
- Nếu chưa tính/đánh giá được AQI theo chuẩn, nêu rõ là hạn chế, không tự tạo kết quả.

### 4.5. Kết quả hệ thống demo
- Ảnh chụp màn hình các chức năng thực tế; ví dụ một ca đầu vào/đầu ra; lỗi và thời gian phản hồi nếu đo được.
- Phân biệt hiển thị dữ liệu lịch sử với thời gian thực.

### 4.6. Thảo luận, hạn chế và ý nghĩa
- Khả năng khái quát hóa sang Việt Nam từ dữ liệu Beijing; sai số khi PM2.5 biến động đột ngột.
- Độ chính xác và rủi ro khi suy ra AQI từ chỉ một chất ô nhiễm.
- So sánh với baseline; hạn chế dữ liệu, tài nguyên, giả định.

**Hình/bảng đề xuất:** Bảng tổng hợp thước đo, biểu đồ dự báo, confusion matrix, hình dashboard, bảng ablation.

## CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 5.1. Kết luận
- Tóm tắt các mục tiêu đã đạt **dựa trên kết quả thực tế**, đóng góp kỹ thuật và bài học kinh nghiệm.
- Tóm tắt hiệu năng và ứng dụng, tránh kết luận khẳng định nếu chưa có số liệu.

### 5.2. Hạn chế
- Chất lượng missing/outlier, biến đổi thời tiết/địa lý, nhãn AQI, cold-start trạm, mô hình cập nhật theo thời gian.

### 5.3. Hướng phát triển
- Kết nối nguồn quan trắc Việt Nam/OpenAQ; pipeline dữ liệu và bản đồ thực sự cập nhật theo thời gian.
- Dự báo 24–72 giờ, hiệu chỉnh mô hình theo trạm, kết hợp PM10/O3/NO2/SO2/CO.
- Tính độ bất định/khoảng dự báo; monitoring data/model drift.
- Đưa hệ thống lên hosting và kiểm thử với người dùng.

## TÀI LIỆU THAM KHẢO VÀ PHỤ LỤC

- Nguồn bộ dữ liệu (UCI/OpenAQ), định nghĩa phương pháp AQI chính thức, tài liệu thuật toán, thư viện và sách giáo trình học phần.
- Trích dẫn thống nhất định dạng IEEE/APA theo yêu cầu giảng viên.
- Phụ lục: bảng tham số, link repo GitHub, hướng dẫn tái lập, ảnh demo, bảng phân công 4 người, kê khai AI.
- **Minh bạch AI:** nêu công cụ, phạm vi hỗ trợ (ví dụ: gợi ý cấu trúc, giải thích code, rà soát lỗi) và phần do sinh viên kiểm chứng/chỉnh sửa. Không trình bày nội dung AI tạo ra chưa kiểm chứng như kết quả thực nghiệm.