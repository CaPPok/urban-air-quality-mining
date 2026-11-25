# CHƯƠNG 3. PHƯƠNG PHÁP NGHIÊN CỨU

**Đề tài 17:** Khai phá Dữ liệu Môi trường & Chất lượng Không khí Đô thị  
**Địa bàn nghiên cứu:** Trạm quan trắc tự động liên tục Cách Mạng Tháng 8, Quận 10, TP. Hồ Chí Minh (mã OpenAQ: 3276359)

---

### 3.1. Quy trình thực nghiệm tổng thể
Quy trình nghiên cứu tổng thể của đề tài tuân thủ phương pháp luận khai phá dữ liệu khoa học, bao gồm chuỗi liên hoàn các bước:
1. Thu thập dữ liệu chuỗi thời gian nồng độ bụi từ nền tảng OpenAQ.
2. Tiền xử lý dữ liệu: làm sạch, xử lý dữ liệu khuyết tật nhân quả, kiểm soát ngoại lai và phân tích khám phá.
3. Tạo lập các đặc trưng chuỗi thời gian: trung bình trượt 24 giờ, biến trễ và mã hóa chu kỳ tuần hoàn.
4. Phân chia tập dữ liệu tuần tự theo thời gian nhằm bảo đảm tuyệt đối không rò rỉ dữ liệu.
5. Huấn luyện các mô hình đối kháng cơ sở và các thuật toán học máy phi tuyến.
6. Đánh giá định lượng toàn diện, phân tích chẩn đoán sai số và phần dư.
7. Ánh xạ suy luận cấp chất lượng không khí theo quy chuẩn quốc gia phục vụ hệ thống cảnh báo sớm.

---

### 3.2. Thu thập, mô tả và hiểu dữ liệu
Dữ liệu sử dụng trong đề tài được khai thác từ trạm quan trắc tự động liên tục đặt tại đường Cách Mạng Tháng 8, Quận 10, TP. Hồ Chí Minh, lưu trữ trên kho lưu trữ mở OpenAQ.
* **Thời gian quan trắc:** Chuỗi thời gian liên tục từ ngày 13/02/2025 đến ngày 11/11/2026 với tổng cộng 14,523 giờ đo hợp lệ.
* **Các thuộc tính chính:** Nồng độ bụi mịn $PM_{2.5}$ ($\mu g/m^3$), nhiệt độ môi trường ($^\circ C$), độ ẩm tương đối (%), chỉ số thời gian và tọa độ địa lý trạm đo.
* **Đặc điểm phân phối:** Chuỗi nồng độ có tính tự tương quan thời gian rất cao ở độ trễ 1 giờ ($r \approx 0.88$), biến thiên mạnh theo nhịp ngày đêm và có sự phân hóa sâu sắc giữa mùa mưa và mùa khô tại Nam Bộ.

---

### 3.3. Làm sạch và feature engineering
* **Quy tắc điền khuyết nhân quả:** Nhằm bảo đảm tính nguyên vẹn của dòng thời gian và không đưa thông tin tương lai vào quá khứ, các giá trị thiếu ở đầu vào chỉ được phép điền xuôi nhân quả tối đa 2 giờ liên tiếp. Tuyệt đối không dùng phương pháp nội suy hai phía. Nhãn nồng độ giờ tiếp theo $PM_{2.5}(t+1)$ tuyệt đối không được điền khuyết.
* **Tạo lập đặc trưng chuỗi thời gian:**
  * `pm2_5_lag1h`: Nồng độ quan trắc tức thời tại thời điểm hiện tại.
  * `pm2_5_rolling24h_mean`: Trung bình trượt 24 giờ của $PM_{2.5}$, phản ánh mức nồng độ nền tích lũy dài hạn.
  * `temperature_c`: Nhiệt độ bề mặt, tín hiệu điều khiển đối lưu nhiệt động lực học không khí.
  * `hour_sin`, `hour_cos`: Mã hóa chu kỳ 24 giờ trong ngày bằng hàm lượng giác.
  * `month_sin`, `month_cos`: Mã hóa chu kỳ 12 tháng trong năm.

---

### 3.4. Thiết kế tập dữ liệu và thực nghiệm

#### 3.4.1. Phân chia chuỗi thời gian tuần tự và kiểm soát rò rỉ dữ liệu
Để kết quả đánh giá phản ánh chính xác năng lực dự báo trong thực tế vận hành, toàn bộ 14,523 mẫu quan trắc được phân chia nghiêm ngặt theo thứ tự thời gian tuyến tính, tuyệt đối không xáo trộn ngẫu nhiên:
* **Tập Huấn luyện (chiếm 68.59%):** Gồm 9,961 mẫu, từ `2025-02-13 20:00:00` đến `2026-05-04 05:00:00`. Toàn bộ quá trình tối ưu trọng số mô hình và học tham số chuẩn hóa `scaler.fit` chỉ được phép thực hiện trên tập này.
* **Tập Kiểm định (chiếm 15.69%):** Gồm 2,278 mẫu, từ `2026-05-04 06:00:00` đến `2026-08-07 15:00:00` (rơi vào thời kỳ mùa mưa). Tập này dùng để tinh chỉnh siêu tham số và áp dụng cơ chế dừng sớm.
* **Tập Kiểm thử (chiếm 15.73%):** Gồm 2,284 mẫu, từ `2026-08-07 16:00:00` đến `2026-11-11 00:00:00` (rơi vào thời kỳ giao mùa và đầu mùa khô). Tập này hoàn toàn độc lập, chỉ được đưa vào suy luận nghiệm thu một lần duy nhất ở bước cuối cùng.

**Kiểm toán giao thoa thời gian và bộ chuẩn hóa Scaler:**
$$T_{\text{train\_end}}\ (2026\text{-}05\text{-}04\ 05:00) < T_{\text{val\_start}}\ (2026\text{-}05\text{-}04\ 06:00) < T_{\text{test\_start}}\ (2026\text{-}08\text{-}07\ 16:00)$$
Khoảng giao thoa thời gian giữa các tập bằng đúng 0 giờ. Kiểm toán độc lập tệp `scaler_x.joblib` ghi nhận độ lệch giữa giá trị trung bình lưu trong bộ chuẩn hóa và giá trị trung bình thực tế của tập huấn luyện chỉ là $7.105 \times 10^{-15}$, chứng minh 100% không xảy ra rò rỉ phân phối dữ liệu.

#### 3.4.2. Xác lập hai mô hình đối kháng cơ sở
Theo chuẩn mực nghiên cứu khoa học, các thuật toán học máy phức tạp bắt buộc phải chứng minh được giá trị cải thiện so với hai mô hình cơ sở nền tảng:
1. **Persistence Baseline (Mô hình quán tính ngây thơ $\hat{y}_{t+1} = y_t$):**  
   Lấy trực tiếp giá trị hiện tại làm dự báo cho giờ kế tiếp. Do chuỗi có tính tự tương quan rất cao ($r = 0.8855$), đây là tiêu chuẩn vàng thực tế trong khí tượng ngắn hạn. Bất kỳ thuật toán nào không vượt qua được Persistence đều bị coi là thất bại trong ứng dụng thực tế.
2. **Ridge Regression Baseline (Hồi quy tuyến tính có hiệu chuẩn $L_2$):**  
   Mô hình tối thiểu hóa sai số toàn phương có ràng buộc phạt độ lớn trọng số, nhằm thiết lập vạch đối sánh tối ưu của quan hệ tuyến tính giữa các đặc trưng đầu vào và nồng độ bụi.

#### 3.4.3. Thiết lập ma trận vạch đích đối sánh
* 🎯 **Vạch đích 1 (Tính khả dụng thực tiễn):** Bắt buộc vượt qua Persistence: $\text{Test RMSE} < 7.3299\ \mu g/m^3$.
* 🎯 **Vạch đích 2 (Giá trị phi tuyến thực chất):** Bắt buộc vượt qua Tuyến tính Ridge: $\text{Test RMSE} < 7.0666\ \mu g/m^3$.

---

### 3.5. Xây dựng các mô hình
Nhóm nghiên cứu triển khai song song 5 kiến trúc mô hình đại diện cho các trường phái học thuật:
* **Persistence Baseline:** Mô hình quán tính 0 tham số.
* **Ridge Regression:** Mô hình tuyến tính 8 tham số với hệ số phạt $\alpha = 1.0$.
* **Random Forest Regressor:** Rừng ngẫu nhiên gồm 100 cây quyết định, độ sâu tối đa 8 tầng.
* **XGBoost Regressor:** Tăng cường độ dốc gồm 50 cây, độ sâu tối đa 4 tầng, tốc độ học $\eta = 0.08$ kết hợp cơ chế dừng sớm.
* **LSTM Regressor:** Mạng nơ-ron hồi quy sâu gồm 2 tầng ẩn với 64 nút nhớ, áp dụng kỹ thuật cửa sổ trượt quá khứ 6 giờ và cơ chế dừng sớm tại chu kỳ thứ 13.

---

### 3.6. Tính và phân loại AQI

#### 3.6.1. Quy chuẩn phân loại áp dụng
Đề tài áp dụng nghiêm ngặt theo **Quyết định số 1459/QĐ-TCMT ngày 12/11/2019 của Tổng cục Môi trường** ban hành Hướng dẫn kỹ thuật tính toán và công bố chỉ số chất lượng không khí Việt Nam (VN-AQI), bao gồm 6 cấp độ:
1. **Tốt:** Nồng độ $0 - 25\ \mu g/m^3$ (Chỉ số AQI: $0 - 50$).
2. **Trung bình:** Nồng độ $25 - 50\ \mu g/m^3$ (Chỉ số AQI: $51 - 100$).
3. **Kém:** Nồng độ $50 - 80\ \mu g/m^3$ (Chỉ số AQI: $101 - 150$).
4. **Xấu:** Nồng độ $80 - 150\ \mu g/m^3$ (Chỉ số AQI: $151 - 200$).
5. **Rất xấu:** Nồng độ $150 - 250\ \mu g/m^3$ (Chỉ số AQI: $201 - 300$).
6. **Nguy hại:** Nồng độ $> 250\ \mu g/m^3$ (Chỉ số AQI: $301 - 500$).

#### 3.6.2. Hai hướng tiếp cận phân loại
Nghiên cứu thiết kế thử nghiệm đối sánh giữa hai trường phái:
* **Phương pháp ánh xạ ngưỡng:** Huấn luyện mô hình hồi quy để dự báo nồng độ liên tục $PM_{2.5}(t+1)$, sau đó ánh xạ giá trị số học này qua bảng ngưỡng 6 cấp của VN-AQI.
* **Phương pháp phân loại trực tiếp đa lớp:** Huấn luyện trực tiếp các thuật toán phân loại (Random Forest Classifier, XGBoost Classifier) để dự báo trực tiếp nhãn danh mục của cấp ô nhiễm.

#### 3.6.3. Phân biệt nồng độ tức thời 1 giờ và chỉ số AQI chính thức 24 giờ
Chỉ số VN-AQI chính thống theo quy định của Bộ Tài nguyên và Môi trường được tính toán trên giá trị trung bình trượt 24 giờ của các chất ô nhiễm. Việc áp dụng bảng ngưỡng của Quyết định 1459/QĐ-TCMT cho nồng độ dự báo tức thời 1 giờ trong nghiên cứu này mang tính chất **phát tín hiệu cảnh báo sớm**, giúp người dân và cơ quan quản lý nắm bắt kịp thời các biến động ô nhiễm cấp tính trước khi nồng độ tích lũy 24 giờ đạt ngưỡng nguy hại.

---

### 3.7. Thiết kế ứng dụng
Mô hình sau khi kiểm định đạt chuẩn được đóng gói thành các tệp nhị phân (`artifacts/models/`) và tích hợp vào giao diện ứng dụng web Streamlit. Ứng dụng cung cấp bảng điều khiển trực quan gồm: bản đồ vị trí trạm quan trắc, đồ thị chuỗi thời gian lịch sử, giao diện nhập liệu thông số và huy hiệu cảnh báo cấp độ chất lượng không khí tương ứng.
