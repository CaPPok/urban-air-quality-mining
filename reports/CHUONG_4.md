# CHƯƠNG 4. KẾT QUẢ VÀ THẢO LUẬN

**Đề tài 17:** Khai phá Dữ liệu Môi trường & Chất lượng Không khí Đô thị  
**Địa bàn nghiên cứu:** Trạm quan trắc tự động liên tục Cách Mạng Tháng 8, Quận 10, TP. Hồ Chí Minh (mã OpenAQ: 3276359)  
**Dữ liệu thực nghiệm:** 14,523 giờ quan trắc độc lập (Tập huấn luyện: 9,961 mẫu, Tập kiểm định: 2,278 mẫu, Tập kiểm thử: 2,284 mẫu)

---

### 4.1. Kết quả EDA và tiền xử lý
Quá trình phân tích khám phá dữ liệu trên chuỗi 14,523 giờ quan trắc tại trạm Cách Mạng Tháng 8 đã xác lập các đặc tính phân phối thống kê quan trọng làm cơ sở cho khâu mô hình hóa:
* **Đặc trưng phân phối nồng độ:** Nồng độ $PM_{2.5}$ trung bình toàn chuỗi đạt $31.42\ \mu g/m^3$ với độ lệch chuẩn $15.34\ \mu g/m^3$, phân phối có độ lệch phải rõ rệt và xuất hiện nhiều đỉnh ô nhiễm đột biến vượt ngưỡng $100\ \mu g/m^3$.
* **Sự dịch chuyển phân phối giữa các tập:**
  * Tập huấn luyện (9,961 mẫu): Nồng độ trung bình $33.41\ \mu g/m^3$, tỷ lệ ô nhiễm ($\ge 50\ \mu g/m^3$) chiếm $11.58\%$.
  * Tập kiểm định (2,278 mẫu, thời kỳ mùa mưa): Nồng độ trung bình giảm xuống $24.03\ \mu g/m^3$, tỷ lệ ô nhiễm chỉ chiếm $1.23\%$.
  * Tập kiểm thử (2,284 mẫu, thời kỳ giao mùa sang mùa khô): Nồng độ trung bình tăng lên $30.14\ \mu g/m^3$, tỷ lệ ô nhiễm tăng vọt lên $9.06\%$ (gấp 7.4 lần tập kiểm định).
* **Quy luật chu kỳ thời gian:** Nồng độ bụi biến thiên mạnh theo chu kỳ ngày đêm, hình thành hai đỉnh phát thải trùng với khung giờ cao điểm giao thông buổi sáng ($07:00 - 09:00$) và buổi tối ($18:00 - 20:00$).

---

### 4.2. So sánh hiệu năng mô hình

#### 4.2.1. Kết quả thực nghiệm định lượng bài toán hồi quy
Hiệu năng dự báo nồng độ bụi liên tục $PM_{2.5}$ tại giờ tiếp theo ($t+1$) được đánh giá độc lập trên tập kiểm định và tập kiểm thử, trích xuất trực tiếp từ tệp [artifacts/metrics/regression_comparison.csv](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/metrics/regression_comparison.csv):

| Nhóm mô hình | Thuật toán mô hình hóa | Val RMSE ($\mu g/m^3$) | Val MAE ($\mu g/m^3$) | Val $R^2$ | Test RMSE ($\mu g/m^3$) | Test MAE ($\mu g/m^3$) | Test $R^2$ | Test MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cơ sở đối kháng** | **Persistence Baseline** ($\hat{y}_{t+1} = y_t$) | 6.3930 | 4.2183 | 0.6086 | **7.3299** | 5.1429 | 0.7709 | 20.07% |
| **Tuyến tính chuẩn** | **Ridge Regression** (hiệu chuẩn $L_2$) | 6.0812 | 4.1847 | 0.6459 | **7.0666** | 5.0851 | 0.7871 | 21.26% |
| **Học máy đóng bao** | **Random Forest Regressor** (100 cây) | 5.7770 | 3.9056 | 0.6806 | **7.0721** | 5.0176 | 0.7868 | 20.28% |
| **Học máy Boosting** | **XGBoost Regressor** (dừng sớm) | 5.8679 | 3.9482 | 0.6705 | **6.9911** | 4.9917 | 0.7916 | 20.91% |
| **Học sâu thời gian** | **LSTM Regressor** (2 tầng ẩn PyTorch) | 5.8924 | 3.9442 | 0.6677 | **6.9959** | **4.9290** | **0.7917** | **19.86%** |

```
                       BIỂU ĐỒ SO SÁNH SAI SỐ TEST RMSE VÀ MAE (μg/m³)
    Persistence (Baseline) ─────── 7.33 (RMSE) ─── 5.14 (MAE)  [Vạch mốc cơ sở]
         Ridge Regression  ─────── 7.07 (RMSE) ─── 5.09 (MAE)  [-3.59% RMSE]
            Random Forest  ─────── 7.07 (RMSE) ─── 5.02 (MAE)  [-3.52% RMSE]
        XGBoost Regressor  ─────── 6.99 (RMSE) ─── 4.99 (MAE)  [-4.62% RMSE - Thấp nhất]
           LSTM Regressor  ─────── 7.00 (RMSE) ─── 4.93 (MAE)  [-4.56% RMSE / MAE tốt nhất]
```

#### 4.2.2. Đánh giá đối chiếu các vạch mốc cơ sở
1. **Đánh bại Persistence Baseline:** Toàn bộ 4 thuật toán học máy đều vượt qua mô hình quán tính ngây thơ Persistence. XGBoost kéo giảm sai số toàn phương xuống mức **$6.9911\ \mu g/m^3$** (giảm $4.62\%$), chứng minh mô hình học được các quy luật phi ngẫu nhiên từ dữ liệu.
2. **Đánh bại Ridge Linear Baseline:** Hai mô hình XGBoost và LSTM tiếp tục hạ thấp sai số so với vạch tuyến tính Ridge ($7.0666\ \mu g/m^3$), chứng minh tính hữu ích của việc mô hình hóa các tương tác phi tuyến.
3. **Mô hình có độ chuẩn xác cao nhất toàn diện:** Mô hình mạng nơ-ron hồi quy LSTM đạt **MAE thấp nhất ($4.9290\ \mu g/m^3$)**, **MAPE thấp nhất ($19.86\%$)** và **$R^2$ cao nhất ($0.7917$)** trên tập kiểm thử độc lập.
4. **Minh chứng trực quan:** Biểu đồ phân tán giá trị thực tế so với dự báo được ghi nhận tại [artifacts/figures/actual_vs_predicted_scatter.png](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/figures/actual_vs_predicted_scatter.png).

---

### 4.3. Tối ưu và phân tích đặc trưng

#### 4.3.1. Phân tích tầm quan trọng của đặc trưng
Tầm quan trọng của các biến đầu vào được lượng hóa thông qua Độ giảm tạp chất Gini của Random Forest và Mức tăng thông tin Gain của XGBoost:
* `pm2_5_ug_m3_input_causal` (nồng độ tức thời hiện tại): Chiếm tỷ trọng áp đảo (**$92.53\%$** trên Random Forest và **$85.14\%$** trên XGBoost).
* `pm2_5_rolling24h_mean` (trung bình trượt 24 giờ): Chiếm $1.79\% - 2.25\%$, đóng vai trò giữ mức nền nồng độ dài hạn.
* `temperature_c_input_causal` (nhiệt độ môi trường): Chiếm $1.96\% - 2.65\%$, cung cấp tín hiệu đối lưu nhiệt động lực học.
* Biến chu kỳ giờ (`hour_sin`, `hour_cos`): Chiếm tổng cộng $2.71\% - 6.04\%$, nắm bắt nhịp độ phát thải giao thông.
* Biến chu kỳ tháng (`month_sin`, `month_cos`): Chiếm $1.01\% - 3.92\%$.

Minh chứng trực quan được lưu giữ tại [artifacts/figures/feature_importance_comparison.png](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/figures/feature_importance_comparison.png).

#### 4.3.2. Thử nghiệm bóc tách đặc trưng
Thử nghiệm bóc tách trên Random Forest (trích xuất từ [artifacts/metrics/feature_ablation.csv](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/metrics/feature_ablation.csv)) làm sáng tỏ vai trò nhân quả của từng nhóm biến:
* **Loại bỏ chu kỳ tháng (`month_sin/cos`):** Sai số trên tập kiểm định gần như không đổi ($\Delta\text{RMSE} = \mathbf{+0.0026\ \mu g/m^3}$). Kết luận: Chu kỳ tháng bị **dư thừa thông tin** vì biến trung bình trượt 24 giờ (`pm2_5_rolling24h_mean`) đã tự động hấp thụ mức nồng độ nền của mùa mưa và mùa khô.
* **Loại bỏ biến nhiệt độ (`temperature_c`):** Sai số trên tập kiểm định tăng vọt ($\Delta\text{RMSE} = \mathbf{+0.1143\ \mu g/m^3}$, gấp 44 lần loại bỏ tháng). Kết luận: Nhiệt độ có **đóng góp nhân quả thực sự** vì điều khiển quá trình đối lưu không khí tại TP. Hồ Chí Minh (nắng nóng ban ngày thúc đẩy khuếch tán đối lưu, ban đêm nghịch nhiệt giữ bụi sát mặt đất).

---

### 4.4. Kết quả phân loại AQI

#### 4.4.1. Đối sánh hai hướng tiếp cận phân loại theo chuẩn VN-AQI
Kết quả đối sánh 6 mô hình trên tập kiểm thử độc lập (trích xuất từ [artifacts/metrics/aqi_classification_comparison.csv](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/metrics/aqi_classification_comparison.csv)):

| Hướng tiếp cận | Mô hình cụ thể | Accuracy (%) | Macro-F1 | Weighted-F1 | Đánh giá tổng quan |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Mô hình cơ sở** | **Persistence Baseline ánh xạ** | 80.39% | 0.7286 | 0.8038 | Mốc cơ sở ngây thơ |
| | **Ridge Baseline ánh xạ** | 80.08% | 0.6978 | 0.7994 | Tuyến tính hóa ngưỡng |
| **Phân loại trực tiếp** | **XGBoost Classifier** | 77.89% | 0.7042 | 0.7813 | Phá vỡ quan hệ thứ tự liên tục |
| | **Random Forest Classifier** | 78.42% | 0.7095 | 0.7865 | Kém hơn phương pháp ánh xạ ngưỡng |
| **Phương pháp ánh xạ ngưỡng**| **Random Forest ánh xạ** | 81.39% | 0.7397 | 0.8136 | Vượt trội phân loại trực tiếp |
| | **XGBoost ánh xạ** | 81.65% | 0.7410 | 0.8152 | Cân bằng đa lớp xuất sắc |
| | **LSTM ánh xạ** | **81.91%** | **0.7432** | **0.8164** | **Hiệu năng cao nhất toàn diện** |

*Biện luận học thuật:* **Phương pháp ánh xạ ngưỡng** vượt trội hoàn toàn so với **Phương pháp phân loại trực tiếp đa lớp** (Accuracy cao hơn từ $3.5\% - 4.0\%$). Bản chất nồng độ bụi là đại lượng vật lý liên tục có quan hệ thứ tự tự nhiên. Mô hình phân loại trực tiếp đối xử các cấp như các danh mục rời rạc độc lập, làm thất thoát thông tin về khoảng cách số học giữa các mốc nồng độ.

#### 4.4.2. Phân tích mức độ nghiêm trọng của sai lệch
Phân tích ma trận nhầm lẫn của LSTM trên 2,284 mẫu kiểm thử ([artifacts/figures/aqi_confusion_matrices.png](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/figures/aqi_confusion_matrices.png)):
* **Tỷ lệ dự báo chính xác tuyệt đối:** **81.91%** (1,871 giờ).
* **Tỷ lệ sai lệch an toàn (Lệch tối đa 1 cấp liền kề):** **17.95%** (410 giờ).
  $$\text{Tỷ lệ dự báo an toàn} = 81.91\% + 17.95\% = \mathbf{99.86\%}$$
* **Tỷ lệ sai lệch nguy hiểm (Cách biệt từ 2 cấp trở lên):** Chỉ có **0.14%** (vỏn vẹn 3 ca trên tổng số 2,284 giờ quan sát).

Hệ thống đạt độ tin cậy vận hành rất cao, triệt tiêu gần như hoàn toàn nguy cơ phát sinh phán đoán sai lầm nghiêm trọng gây rủi ro y tế.

---

### 4.5. Kết quả hệ thống demo
Mô hình XGBoost và LSTM tốt nhất đã được kiểm thử tính tương thích trên giao diện ứng dụng Streamlit:
* **Thời gian phản hồi suy luận:** Thời gian nạp dữ liệu và xuất kết quả dự báo trung bình dưới $50\text{ ms}$, đáp ứng hoàn hảo yêu cầu hiển thị tức thời trong các bảng thông tin điện tử đô thị.
* **Chức năng cảnh báo:** Hiển thị trực quan vị trí trạm quan trắc CMT8, đồ thị nồng độ bụi 24 giờ gần nhất, kết quả dự báo giờ tiếp theo kèm huy hiệu màu sắc tương ứng với 6 cấp độ chất lượng không khí của VN-AQI.

---

### 4.6. Thảo luận, hạn chế và ý nghĩa

#### 4.6.1. Phân tích chẩn đoán phân phối phần dư
Phần dư được xác định theo công thức $e_t = y_t - \hat{y}_t$. Kết quả kiểm định thống kê trên tập kiểm thử được trích xuất từ [artifacts/metrics/residuals_summary.csv](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/metrics/residuals_summary.csv):
* **Kiểm định Jarque-Bera:** Giá trị $p < 10^{-50}$ bác bỏ hoàn toàn giả định phần dư tuân theo phân phối chuẩn Gaussian $\mathcal{N}(0, \sigma^2)$.
* **Dạng phân phối Leptokurtic (Hệ số nhọn Kurtosis $> 6$):** Phân phối phần dư có đỉnh tập trung rất nhọn quanh điểm 0 (đa số các giờ không khí bình thường mô hình dự báo rất chuẩn xác) nhưng lại có đuôi phân phối rất dày ở cả hai phía. Phân phối đuôi dày phản ánh hiện tượng không khí đô thị thỉnh thoảng xuất hiện các xung ô nhiễm dị biệt không theo quy luật thống kê thông thường ([artifacts/figures/regression_residuals_distribution.png](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/figures/regression_residuals_distribution.png)).

#### 4.6.2. Biện luận hiện tượng co cụm về trung bình tại dải cực trị
Tại **23 giờ quan sát ô nhiễm cực trị** ($y_{\text{true}} \ge 80\ \mu g/m^3$, thuộc cấp "Xấu" đến "Nguy hại") trong tập kiểm thử:
* Nồng độ thực tế trung bình: **$92.53\ \mu g/m^3$**.
* Nồng độ Random Forest dự báo trung bình: **$76.23\ \mu g/m^3$** (dự báo hụt **$16.30\ \mu g/m^3$**).
* *Biện luận nguyên lý toán học:* Các thuật toán sử dụng hàm mất mát sai số toàn phương trung bình MSE. Nghiệm tối ưu toán học của hàm MSE chính là giá trị kỳ vọng có điều kiện $\hat{y}^* = \mathbb{E}[y \mid \mathbf{X}]$. Trong cấu trúc cây quyết định, giá trị dự báo tại mỗi nút lá là trung bình cộng của các mẫu rơi vào lá. Do các sự kiện cực trị chỉ chiếm $1.01\%$ dữ liệu, các lá cây luôn chứa lẫn các mẫu lân cận có nồng độ thấp hơn, vô tình kéo dự báo lùi về phía trung bình tổng thể ([artifacts/figures/extreme_events_shrinkage.png](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/figures/extreme_events_shrinkage.png)).

#### 4.6.3. Phân tích chuỗi thời gian và khắc phục hiện tượng trễ pha
* **Phân tích theo khung giờ trong ngày:** Sai số dự báo thấp nhất vào ban đêm và rạng sáng ($\text{MAE} \approx 3.8 - 4.2\ \mu g/m^3$), sau đó tăng mạnh vào các khung giờ cao điểm giao thông sáng và chiều tối ($\text{MAE} \approx 6.1 - 6.8\ \mu g/m^3$) do ùn tắc phát thải bụi bất thường ([artifacts/figures/residuals_diurnal_hourly.png](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/figures/residuals_diurnal_hourly.png)).
* **Khắc phục hiện tượng trễ pha của Persistence:** Khảo sát trên **106 giờ xảy ra biến động nồng độ đột ngột** ($|\Delta y| \ge 15\ \mu g/m^3$):
  * Sai số Persistence bùng nổ lên tới $\text{RMSE} = \mathbf{22.4285\ \mu g/m^3}$ do bị trễ pha 1 giờ (dự báo chậm khiến người dân hít phải khói bụi trước khi có cảnh báo).
  * Mô hình XGBoost cắt giảm sai số này xuống còn $\text{RMSE} = \mathbf{20.2672\ \mu g/m^3}$ (giảm $2.16\ \mu g/m^3$, **cải thiện $9.6\%$ sai số**). Điều này khẳng định việc sử dụng Machine Learning là bắt buộc để phát chuông cảnh báo sớm kịp thời.

#### 4.6.4. Đánh giá dưới góc độ nghiệp vụ y tế và sức khỏe cộng đồng
Kết quả phát hiện vượt ngưỡng báo động $50\ \mu g/m^3$ kèm khoảng tin cậy 95% Bootstrap với 1,000 lần lấy mẫu lại (trích xuất từ [artifacts/metrics/threshold_detection_metrics.csv](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/metrics/threshold_detection_metrics.csv)):
* **Random Forest — Ưu tiên y tế dự phòng:** Đạt độ nhạy Recall cao nhất (**$66.18\%$** [95% CI: $59.5\%, 72.5\%$]), bỏ sót ít ca ô nhiễm nhất (chỉ 70 ca). Phù hợp tích hợp cho bệnh viện, trường học và bảo vệ nhóm người nhạy cảm.
* **LSTM — Ưu tiên bảng tin đô thị chống chai lì cảnh báo:** Đạt độ chuẩn xác Precision cao nhất (**$86.27\%$** [95% CI: $80.7\%, 91.8\%$]) và tỷ lệ báo nhầm thấp nhất (**$\text{FAR} = 1.01\%$**, chỉ 21 ca báo nhầm so với 53 ca của Persistence, giảm **$60.4\%$** số ca báo giả).
* **Đường cong Lift:** Nhóm $10\%$ mẫu có nồng độ dự báo cao nhất có tỷ lệ phát hiện trúng đợt ô nhiễm thực tế cao gấp **$8.0$ lần** so với ngẫu nhiên ([artifacts/figures/lift_curve_rf_xgb.png](file:///d:/HCMUT/HK261/CO3029-Data%20Mining/Assignment/urban-air-quality-mining/artifacts/figures/lift_curve_rf_xgb.png)).

#### 4.6.5. Luận giải về trần hiệu năng của bài toán
Sự biến thiên nồng độ chất ô nhiễm không khí tại trạm đô thị CMT8 được chi phối bởi **phương trình truyền - khuếch tán khí quyển Advection-Diffusion**:
$$\frac{\partial C}{\partial t} = -\mathbf{u} \cdot \nabla C + \nabla \cdot (K \nabla C) + S - D$$
Trong đó: $C$ là nồng độ bụi $PM_{2.5}$; $\mathbf{u}$ là vector trường vận tốc gió; $K$ là hệ số khuếch tán rối; $S$ là suất phát thải nguồn; $D$ là tốc độ lắng đọng ướt do mưa; $\text{PBLH}$ là chiều cao lớp biên hành tinh.

Do tập dữ liệu cảm biến OpenAQ chỉ gồm nồng độ quá khứ và nhiệt độ, hoàn toàn thiếu các biến động lực học khí quyển ($\mathbf{u}, D, \text{PBLH}$), mức sai số **$\text{RMSE} \approx 6.99\ \mu g/m^3$ chính là "trần hiệu năng vật lý"** của bài toán. Việc tiếp tục thử nghiệm các kiến trúc học sâu phức tạp hơn trên tập dữ liệu này sẽ không thể tạo ra bước nhảy vọt nào khác mà chỉ gây lãng phí tài nguyên tính toán. Bước đột phá thực sự bắt buộc phải bổ sung dữ liệu khí tượng động lực học.
