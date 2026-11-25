# CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

**Đề tài 17:** Khai phá Dữ liệu Môi trường & Chất lượng Không khí Đô thị  
**Địa bàn nghiên cứu:** Trạm quan trắc tự động liên tục Cách Mạng Tháng 8, Quận 10, TP. Hồ Chí Minh (mã OpenAQ: 3276359)

---

### 5.1. Kết luận

#### 5.1.1. Đối chiếu toàn diện với các Mục tiêu nghiên cứu ban đầu
Nghiên cứu đã hoàn thành trọn vẹn và vượt mức các mục tiêu kỹ thuật được xác lập tại Chương 1 thông qua quy trình thực nghiệm nghiêm ngặt, minh bạch và có thể tái lập:

| Mục tiêu đề ra tại Chương 1 | Kết quả thực nghiệm đạt được | Mức độ hoàn thành & Đánh giá |
| :--- | :--- | :---: |
| **Mục tiêu 1:** Tiền xử lý dữ liệu chuỗi thời gian, kiểm soát rò rỉ dữ liệu tuyệt đối. | Xử lý 14,523 giờ dữ liệu OpenAQ trạm CMT8; áp dụng điền khuyết nhân quả tối đa 2 giờ; phân chia tập huấn luyện, kiểm định và kiểm thử độc lập theo thời gian (70/15/15); kiểm toán bộ chuẩn hóa Scaler xác nhận học tham số hoàn toàn trên tập huấn luyện với độ lệch cực đại $7.105 \times 10^{-15}$. | **Đạt 100%**<br>(Tuyệt đối không rò rỉ dữ liệu) |
| **Mục tiêu 2:** Xây dựng mô hình dự báo nồng độ bụi $PM_{2.5}$ liên tục trước 1 giờ ($t+1$). | Huấn luyện thành công 5 mô hình (Persistence, Ridge, Random Forest, XGBoost, LSTM). XGBoost đạt **$\text{RMSE} = 6.9911\ \mu g/m^3$**; LSTM đạt **$\text{MAE} = 4.9290\ \mu g/m^3$**, **$R^2 = 0.7917$** và **$\text{MAPE} = 19.86\%$**. | **Đạt 100%**<br>(Vượt trội cả hai mô hình đối kháng) |
| **Mục tiêu 3:** Đánh bại các vạch mốc đối kháng cơ sở. | Đánh bại hoàn toàn Persistence Baseline (cải thiện $4.62\%$ RMSE) và Ridge Regression Baseline (cải thiện $1.07\%$ RMSE); đặc biệt cắt giảm $9.6\%$ sai số tại các giờ xảy ra biến động đột ngột. | **Đạt 100%**<br>(Giải quyết triệt để trễ pha) |
| **Mục tiêu 4:** Phân loại cấp chất lượng không khí VN-AQI theo Quyết định số 1459/QĐ-TCMT. | Chứng minh ưu thế của phương pháp ánh xạ ngưỡng: LSTM ánh xạ đạt **Accuracy $81.91\%$**, **Weighted-F1 $0.8164$**; tỷ lệ dự báo an toàn (chính xác hoặc lệch 1 cấp liền kề) đạt **$99.86\%$**; sai lệch nguy hiểm (cách biệt từ 2 cấp trở lên) chỉ **$0.14\%$**. | **Đạt 100%**<br>(Độ an toàn phân loại tối đa) |
| **Mục tiêu 5:** Đánh giá dưới góc độ nghiệp vụ y tế và cảnh báo sớm sức khỏe cộng đồng. | Lượng hóa với khoảng tin cậy 95% Bootstrap: Random Forest đạt **Recall $66.18\%$** (ít bỏ sót ô nhiễm); LSTM đạt **Precision $86.27\%$** và tỷ lệ báo nhầm **$\text{FAR} = 1.01\%$** (giảm $60.4\%$ số ca báo động giả). | **Đạt 100%**<br>(Khả thi cao trong vận hành) |

#### 5.1.2. Khẳng định tính khả thi và Đóng góp kỹ thuật cốt lõi của đề tài
1. **Khẳng định tính khả thi vận hành phân tầng:**  
   Kết quả nghiên cứu chứng minh hệ sinh thái mô hình có khả năng triển khai thực tế rất cao vào hạ tầng đô thị thông minh TP. Hồ Chí Minh theo 2 kịch bản phân tầng rõ ràng:
   * *Kịch bản An toàn y tế dự phòng (Bệnh viện, trường học, người cao tuổi):* Sử dụng **Random Forest** để tối ưu hóa độ nhạy Recall, không bỏ sót các đợt phát tán bụi độc hại.
   * *Kịch bản Bảng tin cảnh báo công cộng (Màn hình giao thông đô thị, cổng thông tin điện tử):* Sử dụng **LSTM** để tối ưu hóa Precision và triệt tiêu báo động giả, giữ vững niềm tin của người dân và tránh hội chứng chai lì cảnh báo.
2. **Khắc phục điểm mù trễ pha của các mô hình quán tính truyền thống:**  
   Nghiên cứu chứng minh luận điểm thực tiễn sắc bén: Dù chuỗi thời gian nồng độ bụi có tính tự tương quan cao, mô hình quán tính ngây thơ Persistence sẽ gặp thảm họa sai số ($\text{RMSE} = 22.43\ \mu g/m^3$) khi xảy ra bùng phát khói bụi bất ngờ. Việc ứng dụng Machine Learning (XGBoost/LSTM) là bắt buộc để kéo giảm sai số xuống $20.27\ \mu g/m^3$, giúp hệ thống cảnh báo sớm phát chuông báo trước khi người dân hít phải nồng độ bụi nguy hại.
3. **Làm sáng tỏ vai trò nhân quả của các biến vi khí hậu:**  
   Thử nghiệm bóc tách đặc trưng chứng minh: Biến chu kỳ tháng là dư thừa vì trung bình trượt 24 giờ (`pm2_5_rolling24h_mean`) đã tự động hấp thụ nồng độ nền của mùa; trong khi nhiệt độ đóng vai trò quyết định đến quá trình đối lưu nhiệt động lực học không khí tại TP. Hồ Chí Minh.

---

### 5.2. Hạn chế
Bên cạnh các kết quả đạt được, việc nhìn nhận khách quan các rào cản kỹ thuật và giới hạn dữ liệu là cơ sở khoa học để định hình các nghiên cứu tiếp theo:

```
                            SƠ ĐỒ 4 HẠN CHẾ CỐT LÕI CỦA ĐỀ TÀI
┌─────────────────────────────────┐               ┌─────────────────────────────────┐
│ 1. HIỆN TƯỢNG CO CỤM CỰC TRỊ    │               │ 2. NÚT THẮT DỮ LIỆU KHÍ HẬU     │
│ Hàm MSE tối ưu kỳ vọng E[y|X],  │               │ Thiếu vector vận tốc gió (u),   │
│ làm dự báo hụt 16.3 μg/m³ tại   │               │ lượng mưa lắng đọng ướt (D) và  │
│ 23 ca ô nhiễm nặng (>80 μg/m³). │               │ chiều cao lớp biên hành tinh.   │
└────────────────┬────────────────┘               └────────────────┬────────────────┘
                 │                                                 │
                 ▼                                                 ▼
┌─────────────────────────────────┐               ┌─────────────────────────────────┐
│ 3. GIỚI HẠN TRẠM ĐƠN LẺ         │               │ 4. GIỚI HẠN QUY CHUẨN ĐỐI CHIẾU │
│ Mô hình hóa trạm CMT8 (Q.10),   │               │ Phân loại cấp độ tức thời 1 giờ │
│ chưa phản ánh tương tác không   │               │ chưa đồng nhất hoàn toàn với    │
│ gian lan truyền toàn TP.HCM.    │               │ chỉ số tích lũy 24h của Bộ TNMT.│
└─────────────────────────────────┘               └─────────────────────────────────┘
```

1. **Hiện tượng co cụm về trung bình tại dải cực trị:**  
   Mô hình có xu hướng dự báo thấp hơn thực tế trung bình $16.30\ \mu g/m^3$ tại các đỉnh ô nhiễm nghiêm trọng ($y_{\text{true}} \ge 80\ \mu g/m^3$). Nguyên nhân do hàm mất mát đối xứng MSE ưu tiên giảm thiểu sai số trên phần đông các mẫu bình thường ($98.99\%$) và kéo giá trị tại các lá cây quyết định về phía kỳ vọng có điều kiện $\mathbb{E}[y \mid \mathbf{X}]$.
2. **Nút thắt dữ liệu và trần hiệu năng vật lý:**  
   Cảm biến OpenAQ tại trạm CMT8 chỉ ghi nhận nồng độ bụi và nhiệt độ, hoàn toàn thiếu các thông số khí tượng động lực học quan trọng: **tốc độ và hướng gió** (thổi dồn hoặc phân tán bụi), **lượng mưa** (rửa trôi bụi tức thời) và **chiều cao lớp biên hành tinh** (quyết định thể tích buồng chứa chất ô nhiễm). Việc thiếu các biến này tạo ra "trần hiệu năng" khiến sai số không thể vượt qua ngưỡng tiệm cận $\text{RMSE} \approx 6.99\ \mu g/m^3$.
3. **Giới hạn không gian trạm đơn lẻ:**  
   Mô hình được tối ưu hóa riêng biệt cho vi khí hậu và đặc thù giao thông trục đường Cách Mạng Tháng 8, Quận 10. Mặc dù đạt độ chính xác cao tại chỗ, mô hình chưa thể khái quát hóa trực tiếp cho các khu vực có hình thái phát thải khác tại TP. Hồ Chí Minh (như khu công nghiệp Cát Lái, bán đảo Thanh Đa hay ven sông Sài Gòn).
4. **Độ lệch giữa nồng độ tức thời 1 giờ và chỉ số AQI chính thức 24 giờ:**  
   Quy chuẩn Quyết định 1459/QĐ-TCMT quy định chỉ số VN-AQI được tính toán trên giá trị trung bình trượt 24 giờ. Đề tài áp dụng bảng ngưỡng breakpoint này lên nồng độ dự báo tức thời 1 giờ ($PM_{2.5}(t+1)$) nhằm mục đích phát chuông cảnh báo sớm. Dù có giá trị vận hành cao, người sử dụng cần phân biệt rõ ràng giữa *"Cấp độ ô nhiễm tức thời 1 giờ"* với *"Chỉ số VN-AQI chính thống công bố 24 giờ"* để tránh nhầm lẫn pháp lý.

---

### 5.3. Hướng phát triển
Dựa trên việc biện luận các hạn chế cốt lõi, nghiên cứu đề xuất 3 hướng mở rộng chiến lược có tính khả thi kỹ thuật cao:

#### 5.3.1. Cải tiến hàm mất mát phi đối xứng
* **Ý tưởng giải pháp:** Thay thế hàm mất mát truyền thống MSE bằng các hàm mất mát có nhận thức rủi ro:
  * **Hàm mất mát phân vị Pinball:** Tối ưu hóa mô hình ở phân vị trên (ví dụ $q = 0.75$ hoặc $q = 0.90$) thay vì giá trị trung bình $q = 0.50$, giúp hệ thống cung cấp khoảng dao động tin cậy thay vì một con số đơn lẻ.
  * **Hàm mất mát phi đối xứng:** Thiết lập hệ số phạt nặng hơn đối với sai số âm ở dải nồng độ cao (phạt nặng trường hợp thực tế ô nhiễm nặng nhưng mô hình đoán sạch):
    $$\mathcal{L}_{\text{asym}}(y, \hat{y}) = \begin{cases} c_1 (y - \hat{y})^2 & \text{khi } y \ge \hat{y} \text{ (dự báo thiếu)} \\ c_2 (y - \hat{y})^2 & \text{khi } y < \hat{y} \text{ (dự báo thừa)} \end{cases} \quad \text{với } c_1 \gg c_2 \text{ tại } y \ge 50\ \mu g/m^3$$
* **Hiệu quả kỳ vọng:** Triệt tiêu hoàn toàn hiện tượng hụt $16.3\ \mu g/m^3$ tại 23 sự kiện cực trị, nâng độ nhạy Recall phát hiện ô nhiễm nặng lên trên $85\%$.

#### 5.3.2. Làm giàu dữ liệu khí quyển đa nguồn
* **Ý tưởng giải pháp:** Vượt qua trần hiệu năng vật lý bằng cách ghép nối dữ liệu trạm mặt đất với các nguồn tái phân tích khí tượng toàn cầu độ trễ thấp (ECMWF ERA5-Land, GFS, hoặc mô hình dự báo thời tiết địa phương WRF):
  * Bổ sung thành phần vector gió bề mặt ($u_{10m}, v_{10m}$) và tốc độ gió tổng hợp.
  * Bổ sung lượng mưa tích lũy hàng giờ.
  * Bổ sung chiều cao lớp biên hành tinh và bức xạ mặt trời.
* **Hiệu quả kỳ vọng:** Cung cấp thông tin động lực học bình lưu - khuếch tán, giúp mô hình bứt phá trần hiệu năng, hạ thấp sai số xuống dưới ngưỡng $\text{RMSE} < 5.5\ \mu g/m^3$.

#### 5.3.3. Mở rộng mạng lưới không - thời gian đa trạm
* **Ý tưởng giải pháp:** Chuyển dịch từ mô hình cục bộ một trạm sang mô hình không - thời gian phân tán toàn đô thị bằng cách tích hợp dữ liệu từ mạng lưới quan trắc đa trạm tại TP. Hồ Chí Minh:
  * Xây dựng đồ thị không gian trong đó các nút là các trạm quan trắc (CMT8, Lãnh Binh Thăng, Thủ Đức, Thảo Điền...), trọng số cạnh phản ánh khoảng cách địa lý và hướng gió thịnh hành.
  * Áp dụng kiến trúc mạng nơ-ron đồ thị không - thời gian kết hợp (ST-GNN, DCRNN hoặc Graph WaveNet).
* **Hiệu quả kỳ vọng:** Mô hình hóa được quá trình lan truyền khói bụi từ các cụm công nghiệp ngoại thành vào trung tâm thành phố, cho phép kéo dài chân trời dự báo từ $1$ giờ lên $6 - 24$ giờ tiếp theo với độ tin cậy vững chắc.
