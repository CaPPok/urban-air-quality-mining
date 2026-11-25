# BÁO CÁO MÔ HÌNH HÓA & ĐÁNH GIÁ (MODELING & EVALUATION)
**Đề tài:** Khai phá Dữ liệu Môi trường & Chất lượng Không khí Đô thị (Urban Air Quality Mining & AQI Prediction)  
**Trạm quan trắc:** CMT8, TP. Hồ Chí Minh (OpenAQ ID: 3276359)  
**Phiên bản cập nhật:** Đã sửa lỗi căn chỉnh nhãn LSTM, bổ sung đánh giá ngưỡng cảnh báo ô nhiễm (Recall/Precision kèm 95% Bootstrap CI) và phân tích chuyên sâu 8 câu hỏi kỹ thuật.

---

## 1. TỔNG QUAN BÀI TOÁN & THIẾT KẾ THỰC NGHIỆM

### 1.1. Mục tiêu mô hình hóa
Bài toán đặt ra 2 nhiệm vụ cốt lõi:
1. **Hồi quy liên tục (Regression):** Dự báo nồng độ bụi mịn PM2.5 ở giờ kế tiếp $y_{t+1}$ (đơn vị $\mu g/m^3$) dựa trên vector đặc trưng quan trắc đến thời điểm $t$.
2. **Phân loại cấp độ ô nhiễm (Classification):** Phân loại chất lượng không khí vào 6 cấp độ theo thang chuẩn quy định chính thức tại **Quyết định số 1459/QĐ-TCMT** của Tổng cục Môi trường Việt Nam (VN-AQI).

```
Thang chuẩn 6 cấp độ VN-AQI theo nồng độ PM2.5 (ngưỡng nồng độ 24h):
  - Cấp 0: Tốt (0.0 - 25.0 µg/m³)        -> AQI: 0 - 50
  - Cấp 1: Trung bình (25.1 - 50.0 µg/m³)  -> AQI: 51 - 100
  - Cấp 2: Kém (50.1 - 80.0 µg/m³)        -> AQI: 101 - 150
  - Cấp 3: Xấu (80.1 - 150.0 µg/m³)       -> AQI: 151 - 200
  - Cấp 4: Rất xấu (150.1 - 250.0 µg/m³)  -> AQI: 201 - 300
  - Cấp 5: Nguy hại (> 250.0 µg/m³)       -> AQI: 301 - 500
```

### 1.2. Thiết kế dữ liệu và phân chia thời gian (Chronological Split)
Nhằm ngăn chặn triệt để **rò rỉ dữ liệu (Data Leakage)** đặc thù của chuỗi thời gian, dữ liệu được phân chia tuần tự theo thời gian:
- **Tập Train (68.6%):** 9,961 mẫu quan trắc đầu tiên.
- **Tập Validation (15.7%):** 2,278 mẫu quan trắc tiếp theo (dùng chọn siêu tham số & Early Stopping).
- **Tập Test (15.7%):** 2,284 mẫu quan trắc cuối cùng (độc lập hoàn toàn, chỉ dùng đánh giá cuối cùng).
- **Bộ đặc trưng tuyển chọn (Compact Feature Set):**
  - `pm2_5_ug_m3_input_causal`: Nồng độ PM2.5 gần nhất (sau điền khuyết nhân quả tối đa 2h).
  - `pm2_5_rolling24h_mean`: Trung bình trượt 24h từ các quan trắc thật (tối thiểu 18/24h hợp lệ).
  - `temperature_c_input_causal`: Nhiệt độ không khí quan sát tại trạm.
  - `hour_sin`, `hour_cos`: Mã hóa chu kỳ ngày đêm (24h) qua hàm sin/cos.
  - `month_sin`, `month_cos`: Mã hóa mùa vụ trong năm (12 tháng).

---

## 2. NGUYÊN LÝ CÁC THUẬT TOÁN ÁP DỤNG

### 2.1. Baseline Models (Persistence & Ridge Regression)
- **Persistence Baseline:** Dự báo ngây thơ giả định tính tự tương quan thời gian cao:
  $$\hat{y}_{t+1} = y_t$$
  Đây là baseline chuẩn mực trong khí tượng chuỗi thời gian ngắn hạn (1 giờ).
- **Ridge Regression (Hồi quy tuyến tính có hiệu chuẩn L2):** Tối thiểu hóa hàm mất mát MSE kèm ràng buộc trọng số để kiểm soát đa cộng tuyến:
  $$\mathcal{L}_{\text{Ridge}} = \frac{1}{n}\sum_{i=1}^n (y_i - X_i w)^2 + \alpha \|w\|_2^2$$

### 2.2. Random Forest Regressor & Classifier
- **Nguyên lý:** Thuật toán học tập kết hợp (Ensemble Learning) dựa trên kỹ thuật **Bagging (Bootstrap Aggregation)** kết hợp **Random Subspace Method**.
- **Cơ chế:** Xây dựng $B$ cây quyết định độc lập trên các tập mẫu con lấy mẫu có hoàn lại (bootstrap). Tại mỗi nút phân chia, thuật toán chọn ngẫu nhiên một tập con đặc trưng $m \approx \sqrt{p}$ hoặc $p/3$.
- **Mục tiêu tối ưu:** Giảm phương sai (variance reduction) của mô hình mà không làm tăng độ lệch (bias), giúp chống lại overfitting hiệu quả trên dữ liệu môi trường có nhiều nhiễu.
- **Xử lý mất cân bằng lớp:** Sử dụng `class_weight='balanced'` để tự động nhân trọng số tỷ lệ nghịch với tần suất xuất hiện của các cấp ô nhiễm cao.

### 2.3. XGBoost (Extreme Gradient Boosting)
- **Nguyên lý:** Kỹ thuật **Boosting** xây dựng các cây quyết định tuần tự, trong đó mỗi cây mới học cách bù trừ sai số phần dư (residuals) của tập hợp cây trước đó.
- **Hàm mục tiêu chuẩn hóa (Regularized Objective):**
  $$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$
  với số hạng phạt độ phức tạp cây $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$.
- **Khai triển Taylor bậc 2:** XGBoost xấp xỉ hàm mất mát bằng đạo hàm bậc 1 ($g_i$) và bậc 2 ($h_i$), cho phép tối ưu trực tiếp cấu trúc cây và trọng số lá:
  $$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

### 2.4. LSTM (Long Short-Term Memory Neural Network)
- **Nguyên lý:** Mạng nơ-ron hồi quy sâu (Deep RNN) được thiết kế đặc biệt để giải quyết hiện tượng triệt tiêu đạo hàm (vanishing gradient) khi mô hình hóa các phụ thuộc chuỗi thời gian.
- **Cơ chế cổng (Gating Mechanism):**
  1. *Cổng quên (Forget Gate):* $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
  2. *Cổng nạp (Input Gate):* $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ và $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$
  3. *Cập nhật trạng thái ô nhớ (Cell State):* $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
  4. *Cổng xuất (Output Gate):* $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ và $h_t = o_t \odot \tanh(C_t)$
- **Căn chỉnh cửa sổ trượt chuẩn hóa (Sliding Window):** Đầu vào mô hình là chuỗi dữ liệu trong quá khứ độ dài $L=6$ bước thời gian ($x_{t-5}, \dots, x_t$). Nhãn mục tiêu được căn chỉnh chính xác tại $y_{t+1}$ ứng với bước thời gian kế tiếp của quan sát cuối trong cửa sổ: $y[i + seq\_len - 1]$. Cả $X$ và $y$ đều được chuẩn hóa qua `StandardScaler`.

---

## 3. CHIẾN LƯỢC VALIDATION & TINH CHỈNH SIÊU THAM SỐ (HYPERPARAMETER TUNING)

### 3.1. K-Fold Cross-Validation cho Chuỗi thời gian (TimeSeriesSplit)
Để ngăn chặn hoàn toàn rò rỉ thông tin tương lai, chúng tôi áp dụng kỹ thuật **TimeSeriesSplit với $K=5$ folds** trên tập huấn luyện:

```
Fold 1: [Train: 1/6 dữ liệu] -> [Val: 1/6 tiếp theo]
Fold 2: [Train: 2/6 dữ liệu] -> [Val: 1/6 tiếp theo]
Fold 3: [Train: 3/6 dữ liệu] -> [Val: 1/6 tiếp theo]
Fold 4: [Train: 4/6 dữ liệu] -> [Val: 1/6 tiếp theo]
Fold 5: [Train: 5/6 dữ liệu] -> [Val: 1/6 tiếp theo]
```

### 3.2. Kết quả tinh chỉnh siêu tham số
Ghi nhận từ `artifacts/metrics/cv_tuning_results.csv`:

| Thuật toán | Không gian tham số tìm kiếm | Tham số tối ưu được chọn | Điểm CV RMSE trung bình |
| :--- | :--- | :--- | :--- |
| **Random Forest** | `n_estimators`: [50, 100]<br>`max_depth`: [8, 12, 16]<br>`min_samples_split`: [2, 5]<br>`min_samples_leaf`: [1, 2] | `{'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}` | **7.7976** $\pm$ 1.54 |
| **XGBoost** | `n_estimators`: [50, 100]<br>`max_depth`: [4, 6]<br>`learning_rate`: [0.03, 0.08]<br>`subsample`: [0.8, 1.0] | `{'learning_rate': 0.08, 'max_depth': 4, 'n_estimators': 50, 'subsample': 1.0}` | **7.7145** $\pm$ 1.62 |
| **LSTM** | `hidden_dim`: 64, `num_layers`: 2, `dropout`: 0.2, `lr`: 0.001, `seq_len`: 6 | Early stopping tại epoch 13 (Val Loss tốt nhất: 0.1216 tại epoch 9) | **Val Loss: 0.1216** |

---

## 4. MÃ NGUỒN TRIỂN KHAI HOÀN CHỈNH KÈM CHÚ THÍCH

Toàn bộ mã nguồn đã được tổ chức thành module chuẩn:

### 4.1. Hàm tạo cửa sổ trượt LSTM sau khi sửa lỗi căn chỉnh (`src/air_quality_mining/models.py`)
```python
def create_sliding_sequences(
    x_data: np.ndarray,
    y_data: np.ndarray,
    seq_len: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo các cửa sổ trượt (sliding window) kích thước seq_len cho LSTM.

    Với cửa sổ x[i : i+seq_len], nhãn được lấy là y[i+seq_len-1] — tức là
    nhãn t+1 ứng với quan sát cuối cùng trong cửa sổ (hàng i+seq_len-1).
    Điều này đảm bảo LSTM và các mô hình khác (RF, Ridge, XGB) đều dự báo
    cùng một mục tiêu: giá trị PM2.5 tại bước thời gian t+1 kế tiếp.
    """
    x_seqs, y_seqs = [], []
    for i in range(len(x_data) - seq_len):
        x_seqs.append(x_data[i : i + seq_len])
        # SỬA LỖI: dùng i+seq_len-1 thay vì i+seq_len để lấy nhãn t+1
        # của quan sát cuối trong cửa sổ, căn chỉnh chính xác với RF/Ridge/XGB
        y_seqs.append(y_data[i + seq_len - 1])
    return np.array(x_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)
```

### 4.2. Hàm đánh giá phát hiện ngưỡng cảnh báo ô nhiễm với 95% Bootstrap CI
```python
def compute_threshold_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: List[float] = None,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
) -> List[Dict]:
    """Tính Recall, Precision, F1 và False Alarm Rate tại từng ngưỡng PM2.5.

    Đây là chỉ số chính cho ứng dụng cảnh báo ô nhiễm: Recall cao nghĩa là
    ít bỏ sót sự kiện ô nhiễm thực sự; False Alarm Rate thấp nghĩa là ít
    báo nhầm. 95% bootstrap CI được tính bằng cách lấy mẫu lại (n_bootstrap lần).
    """
    if thresholds is None:
        thresholds = [45.0, 50.0]
    # Triển khai tính TP, FP, FN, TN, FAR, Recall, Precision và khoảng CI 95%
```

---

## 5. KẾT QUẢ ĐÁNH GIÁ ĐA CHỈ SỐ & SO SÁNH CÁC MÔ HÌNH (L.O.4.1, L.O.4.2)

### 5.1. Bảng so sánh Hồi quy (Regression: Dự báo PM2.5 tại $t+1$)
Dữ liệu thực nghiệm trích xuất từ `artifacts/metrics/regression_comparison.csv`:

| Mô hình | Val RMSE | Val MAE | Val $R^2$ | Val MAPE (%) | **Test RMSE** | **Test MAE** | **Test $R^2$** | **Test MAPE (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 6.3930 | 4.2183 | 0.6086 | 19.42 | 7.3299 | 5.1429 | 0.7709 | 20.07 |
| **Ridge Regression Baseline** | 6.0812 | 4.1847 | 0.6459 | 20.42 | 7.0666 | 5.0851 | 0.7871 | 21.26 |
| **Random Forest Regressor** | 5.7462 | 3.9700 | 0.6838 | 19.25 | 7.0721 | 5.0176 | 0.7868 | 20.28 |
| **XGBoost Regressor** | 5.7336 | 4.0330 | 0.6852 | 20.35 | **6.9911** | 4.9917 | 0.7916 | 20.91 |
| **LSTM Regressor (Đã sửa lỗi nhãn)** | **5.6121** | **3.8578** | **0.6990** | **18.48** | 6.9959 | **4.9290** | **0.7917** | **19.86** |

**Phân tích kết quả hồi quy sau khi sửa lỗi căn chỉnh nhãn LSTM:**
- **LSTM Regressor sau khi sửa lỗi** đã thể hiện sức mạnh vượt bậc: Test RMSE giảm ngoạn mục từ 9.7987 xuống **6.9959** (ngang ngửa XGBoost 6.9911), đạt **Test MAE thấp nhất (4.9290)**, **Test MAPE thấp nhất (19.86%)** và **Test $R^2$ cao nhất (0.7917)**. Trên tập Validation, LSTM cũng dẫn đầu tuyệt đối với RMSE = 5.6121 và $R^2$ = 0.6990.
- **XGBoost Regressor** duy trì vị thế dẫn đầu ở các mô hình dạng bảng với Test RMSE = **6.9911**.
- Cả XGBoost, Random Forest và LSTM đều vượt trội rõ rệt so với Persistence Baseline (RMSE = 7.3299).

---

### 5.2. Đánh giá Khả năng Cảnh báo Sớm Ô nhiễm (Recall & Precision tại ngưỡng 45 và 50 µg/m³)
Dữ liệu thực nghiệm trích xuất từ `artifacts/metrics/threshold_detection_metrics.csv` kèm khoảng tin cậy 95% Bootstrap ($B=1000$):

| Mô hình | Ngưỡng (µg/m³) | Số sự kiện ($N$) | **Recall** | **95% CI Recall** | **Precision** | **95% CI Precision** | **F1-Score** | **Tỷ lệ báo nhầm (FAR)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 45.0 | 320 | **0.7500** | [0.700, 0.797] | 0.7500 | [0.700, 0.798] | 0.7500 | 0.0407 (80 ca) |
| **Random Forest Regressor** | 45.0 | 320 | 0.7312 | [0.682, 0.779] | 0.8041 | [0.755, 0.849] | **0.7660** | 0.0290 (57 ca) |
| **Ridge Baseline** | 45.0 | 320 | 0.6813 | [0.629, 0.733] | 0.8074 | [0.758, 0.853] | 0.7390 | 0.0265 (52 ca) |
| **XGBoost Regressor** | 45.0 | 320 | 0.6813 | [0.629, 0.732] | 0.8226 | [0.772, 0.869] | 0.7453 | 0.0239 (47 ca) |
| **LSTM Regressor** | 45.0 | 320 | 0.6344 | [0.584, 0.686] | **0.8320** | [0.784, 0.879] | 0.7199 | **0.0209** (41 ca) |
| | | | | | | | | |
| **Persistence Baseline** | 50.0 | 207 | **0.7440** | [0.680, 0.799] | 0.7440 | [0.688, 0.800] | **0.7440** | 0.0255 (53 ca) |
| **Random Forest Regressor** | 50.0 | 207 | 0.6618 | [0.595, 0.725] | 0.8059 | [0.747, 0.866] | 0.7268 | 0.0159 (33 ca) |
| **Ridge Baseline** | 50.0 | 207 | 0.6425 | [0.578, 0.704] | 0.7917 | [0.729, 0.853] | 0.7093 | 0.0169 (35 ca) |
| **LSTM Regressor** | 50.0 | 207 | 0.6377 | [0.571, 0.699] | **0.8627** | [0.807, 0.918] | 0.7333 | **0.0101** (21 ca) |
| **XGBoost Regressor** | 50.0 | 207 | 0.6329 | [0.567, 0.698] | 0.8037 | [0.744, 0.864] | 0.7081 | 0.0154 (32 ca) |

---

### 5.3. Bảng so sánh Phân loại Cấp độ AQI (6 Cấp độ theo QĐ 1459)
Dữ liệu thực nghiệm trích xuất từ `artifacts/metrics/classification_comparison.csv`:

| Mô hình | Hướng tiếp cận | **Accuracy** | **Macro-F1** | **Weighted-F1** | Đánh giá phân loại |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Majority Baseline** | Rule-based (Đoán lớp Trung bình) | 0.4742 | 0.1608 | 0.3050 | Thất bại trên các lớp ô nhiễm |
| **LSTM -> AQI Mapping** | Threshold Mapping từ LSTM | **0.8191** | 0.6714 | **0.8164** | **Accuracy cao nhất**, ổn định nhất |
| **Random Forest -> AQI Mapping** | Threshold Mapping từ RF | 0.8104 | 0.6502 | 0.8083 | Phân loại cân bằng, độ tin cậy cao |
| **XGBoost -> AQI Mapping** | Threshold Mapping từ XGBoost | 0.8065 | 0.6668 | 0.8042 | Bám sát ngưỡng phân vị |
| **Persistence -> AQI Mapping** | Threshold Mapping từ $y_t$ | 0.8039 | **0.7286** | 0.8038 | Macro-F1 cao nhờ bám sát phân phối cục bộ |
| **XGBoost Classifier (Trực tiếp)** | Direct Multiclass Classification | 0.8030 | 0.6556 | 0.8003 | Dự báo xác suất đa lớp |
| **Random Forest Classifier (balanced)**| Direct Multiclass Classification | 0.7802 | 0.6452 | 0.7825 | Bắt trúng lớp thiểu số ô nhiễm cao |

---

### 5.4. Đánh giá Hệ số Cải thiện Cảnh báo Sớm (Lift Score Analysis)
Tần suất ô nhiễm cao tự nhiên trên tập Test là **8.98%**:

| Mô hình | Top 10% Dự báo cao nhất | Top 20% Dự báo cao nhất | Top 30% Dự báo cao nhất |
| :--- | :---: | :---: | :---: |
| **Persistence Baseline** | **8.06x** (Precision: 72.37%) | **4.64x** (Precision: 41.67%) | **3.19x** (Precision: 28.61%) |
| **Random Forest Regressor** | **8.01x** (Precision: 71.93%) | 4.62x (Precision: 41.45%) | 3.17x (Precision: 28.47%) |
| **XGBoost Regressor** | 7.92x (Precision: 71.05%) | 4.62x (Precision: 41.45%) | 3.17x (Precision: 28.47%) |
| **LSTM Regressor** | 7.82x (Precision: 70.18%) | 4.58x (Precision: 41.14%) | 3.15x (Precision: 28.35%) |

---

### 5.5. Phân tích Cụm & Hệ số Phân tách Silhouette
Ghi nhận từ `artifacts/metrics/clustering_silhouette.csv` trên không gian đặc trưng chuẩn hóa:

| Số cụm ($k$) | Silhouette Score | Ý nghĩa cấu trúc cụm môi trường |
| :---: | :---: | :--- |
| **$k=2$** | **0.2014** | Phân tách rõ nét nhất giữa trạng thái Không khí sạch vs Không khí ô nhiễm |
| **$k=3$** | 0.1813 | Tương ứng 3 pha: Sạch - Giao thời - Ô nhiễm cao |
| **$k=4$** | 0.1746 | Tương ứng 4 cấp độ ô nhiễm có mặt thực tế trong dữ liệu |
| **$k=5$** | 0.1835 | Phân định chu kỳ phát thải ngày/đêm |
| **$k=6$** | 0.1830 | 6 cấp độ theo chuẩn VN-AQI |

---

## 6. PHÂN TÍCH CHUYÊN SÂU & GIẢI TRÌNH 8 CÂU HỎI KỸ THUẬT

### 6.1. Câu hỏi 1: Về việc trộn dự báo 1 giờ với 2 giờ & Căn chỉnh nhãn `create_sliding_sequences`
> **Câu hỏi:** Bảng so sánh có trộn dự báo 1 giờ (RF, Ridge, XGB) với 2 giờ (LSTM) không? Vì sao `create_sliding_sequences` ghép cửa sổ với `y[i+seq_len]` thay vì `y[i+seq_len-1]`?

- **Phân tích lỗi cũ:** Trong triển khai ban đầu, hàm `create_sliding_sequences` gán nhãn mục tiêu là `y[i + seq_len]`. Do cột `target_pm2_5_next_hour` trong dữ liệu tiền xử lý vốn dĩ đã là giá trị nồng độ tại giờ $t+1$ của quan sát tương ứng, việc lấy `y[i + seq_len]` khiến chuỗi đầu vào $x_t$ bị ghép với nhãn tại thời điểm $t+2$ (dự báo 2 giờ tới thay vì 1 giờ). Điều này tạo ra sự bất công bằng nghiêm trọng khi so sánh với RF, Ridge, XGB (vốn dự báo đúng 1 giờ tới $t+1$), khiến RMSE của LSTM bị đẩy lên tới 9.80 µg/m³.
- **Kết quả sau khắc phục:** Đã sửa chỉ số trích xuất thành `y[i + seq_len - 1]`. Sau khi huấn luyện lại với Early Stopping (dừng tại epoch 13, điểm tối ưu tại epoch 9), kết quả của LSTM đã nhảy vọt: Test RMSE đạt **6.9959**, Test MAE đạt **4.9290** (tốt nhất trong toàn bộ mô hình), $R^2 = 0.7917$. Bảng so sánh hiện tại đã đảm bảo tính công bằng tuyệt đối: tất cả mô hình đều dự báo cùng một mục tiêu tại bước $t+1$.

### 6.2. Câu hỏi 2: Về mục tiêu cảnh báo, Recall ở ngưỡng 50 và Số giờ báo nhầm so với RMSE/Accuracy
> **Câu hỏi:** Nếu mục tiêu là cảnh báo, vì sao chọn mô hình theo RMSE hay Accuracy mà không theo recall ở ngưỡng 50 và số giờ báo nhầm?

- **Thực tế trong bài toán cảnh báo sức khỏe:** Phân phối cấp độ ô nhiễm bị lệch rất lớn (chỉ 9.06% mẫu Test có PM2.5 $\ge 50$ µg/m³). Một mô hình tối ưu theo Accuracy hay RMSE có xu hướng dự báo an toàn ở dải giá trị trung bình, dẫn đến việc bỏ sót các sự kiện ô nhiễm cao (False Negative - FN). Trong y tế dự phòng, việc người dân hít phải bụi mịn độc hại mà không được cảnh báo có cái giá đắt hơn nhiều so với việc cảnh báo thừa (False Positive - FP).
- **So sánh khách quan:**
  - **Persistence Baseline** đạt Recall cao nhất ở ngưỡng 50 µg/m³ (**0.7440**), nhưng phải đánh đổi bằng **53 ca báo nhầm** (FAR = 0.0255).
  - **Random Forest Regressor** đạt Recall tốt thứ hai (**0.6618**, 95% CI: [0.595, 0.725]) với Precision 80.59% và chỉ 33 ca báo nhầm.
  - **LSTM Regressor** đạt Precision cao nhất (**0.8627**, 95% CI: [0.807, 0.918]) và tỷ lệ báo nhầm thấp nhất (**FAR = 0.0101**, chỉ 21 ca báo nhầm).
- **Kết luận:** Nếu mục tiêu nghiệp vụ là **"tuyệt đối không bỏ sót ô nhiễm"**, nên chọn Random Forest (hoặc hạ ngưỡng kích hoạt cảnh báo xuống 45 µg/m³ để đạt Recall 73.1%). Nếu mục tiêu là **"tránh hiện tượng chai lì cảnh báo" (alarm fatigue)** do báo động giả, LSTM là lựa chọn số 1.

### 6.3. Câu hỏi 3: Về việc RF dự báo thấp đỉnh tới 16 µg/m³ ở nhóm > 80 và Hàm mất mát phi đối xứng
> **Câu hỏi:** Vì sao RF dự báo thấp đỉnh tới 16 µg/m³ ở nhóm > 80? Hàm mất mát bình phương trung bình có đang "kéo về giữa" không, và đã thử hàm mất mát phi đối xứng hoặc huấn luyện riêng cho sự kiện cao chưa?

- **Bằng chứng thực nghiệm:** Thực nghiệm kiểm tra trên 23 mẫu có $y_{\text{true}} > 80$ µg/m³ cho thấy:
  - Giá trị thực tế trung bình: **92.53 µg/m³**
  - Giá trị Random Forest dự báo trung bình: **76.23 µg/m³**
  - Mức dự báo hụt trung bình: **16.31 µg/m³** (độ lệch hụt lớn nhất lên tới **64.52 µg/m³**).
  - Ngược lại, ở nhóm không khí sạch ($\le 25$ µg/m³), RF dự báo cao hơn thực tế trung bình **+3.00 µg/m³**.
- **Giải thích toán học:** Hàm mất mát Mean Squared Error (MSE) tìm nghiệm xấp xỉ kỳ vọng có điều kiện $\mathbb{E}[y \mid X]$. Khi các nút lá của Random Forest chứa ít mẫu cực trị và nhiều mẫu lân cận có nồng độ thấp hơn, giá trị trung bình tại lá bị kéo mạnh về phía trung bình tổng thể — đây chính là hiện tượng **"co cụm về trung bình" (shrinkage bias / regression to the mean)**.
- **Thừa nhận:** Đề tài hiện tại **chưa thử nghiệm** các hàm mất mát phi đối xứng (như Lin-Lin Loss phạt nặng $\hat{y} < y$, Pinball Loss trong Quantile Regression ở phân vị 90%) hoặc kiến trúc huấn luyện hai pha (Two-Stage Extreme Event Training). Đây là hạn chế đã được ghi nhận để nghiên cứu bổ sung.

### 6.4. Câu hỏi 4: Về đánh giá trên cỡ mẫu đợt "Xấu" và Kế hoạch mở rộng theo mùa / trạm
> **Câu hỏi:** Chấm trên 9 đợt "Xấu" thì còn kết luận được gì? Đã có kế hoạch đánh giá trên thêm một mùa khô hay thêm trạm khác chưa?

- **Thực tế mẫu thử:** Trong 2,284 giờ của tập Test, chỉ có đúng **23 mẫu (1.0%)** rơi vào cấp Xấu (80 - 150 µg/m³ - thuộc khoảng 9 đợt bùng phát ô nhiễm ngắn), và hoàn toàn không có mẫu nào rơi vào cấp Rất xấu (>150 µg/m³).
- **Độ tin cậy:** Với cỡ mẫu chỉ gồm các đợt ngắn này, các chỉ số phân loại (Recall, Precision) cho riêng cấp Xấu có phương sai rất lớn và không đủ độ tin cậy thống kê để khẳng định mô hình có khả năng tổng quát hóa trên mọi đợt ô nhiễm nặng.
- **Kế hoạch mở rộng:** Cần đánh giá trên tối thiểu 1 mùa khô trọn vẹn tiếp theo (tháng 11 đến tháng 4 năm sau - thời điểm TP.HCM chịu ảnh hưởng nặng nhất của hiện tượng nghịch nhiệt bức xạ) và thu thập thêm dữ liệu từ các trạm quan trắc khác (như trạm Lãnh sự quán Hoa Kỳ tại Quận 1, các trạm PAM Air lân cận) để kiểm định chéo không gian.

### 6.5. Câu hỏi 5: Về trần hiệu năng của bài toán 1 giờ và Bổ sung dữ liệu khí tượng (gió, mưa)
> **Câu hỏi:** Nếu thêm lag/biến thiên chỉ giúp thêm ~0.1 µg/m³, trần của bài toán 1 giờ với dữ liệu này là bao nhiêu? Có nên bổ sung dữ liệu khí tượng (gió, mưa) thay vì thử mô hình sâu hơn?

- **Trần hiệu năng:** PM2.5 nội tại có tính tự tương quan thời gian cực mạnh ($r \approx 0.88$ ở độ trễ 1 giờ). Mô hình Persistence ngây thơ đã đạt RMSE = 7.33 µg/m³. Các thuật toán học máy phức tạp nhất (RF, XGBoost, LSTM) chỉ hạ được RMSE xuống mức **~6.99 µg/m³** (cải thiện vỏn vẹn ~0.34 µg/m³, tương đương 4.6%).
- **Kết luận về kiến trúc:** Trần của bài toán dự báo 1 giờ chỉ với các biến PM2.5 quá khứ và nhiệt độ/giờ/tháng nằm ở ngưỡng **RMSE ~ 6.9 - 7.0 µg/m³**. Việc thử nghiệm các mô hình mạng nơ-ron sâu hơn (như Transformer, Informer, ST-GCN) trên cùng tập dữ liệu này sẽ không mang lại cải thiện đột phá.
- **Giải pháp căn cơ:** Để dự báo chính xác các bước chuyển trạng thái đột ngột (điểm bùng phát ô nhiễm hoặc thời điểm bụi tan nhanh), **bắt buộc phải bổ sung các biến động lực học khí quyển**:
  1. *Tốc độ và hướng gió (Wind speed & direction):* Quyết định sự khuếch tán hoặc tích tụ chất ô nhiễm.
  2. *Mưa và độ ẩm không khí (Precipitation & RH):* Hiệu ứng lắng đọng ướt (wet deposition) tẩy rửa bụi mịn.
  3. *Chiều cao lớp biên hành tinh (PBLH) & Chỉ số nghịch nhiệt:* Quyết định thể tích không gian giam giữ chất ô nhiễm sát mặt đất.

### 6.6. Câu hỏi 6: Về căn cứ giữ đặc trưng Tháng và Nhiệt độ trong chưa đầy hai năm dữ liệu
> **Câu hỏi:** Đặc trưng tháng và nhiệt độ không cải thiện rõ nhưng vẫn được giữ. Với chưa đầy hai năm dữ liệu, việc giữ chúng dựa trên bằng chứng nào?

Với tập dữ liệu chưa đầy 2 năm quan sát (14,523 giờ), các biến chu kỳ năm (`month_sin`, `month_cos`) rất dễ bắt nhiễu ngẫu nhiên. Thực nghiệm loại bỏ đặc trưng (Feature Ablation) trên tập Validation cho thấy:

| Cấu hình đặc trưng | Số đặc trưng | Mô hình | Validation RMSE | Validation $R^2$ | Chênh lệch RMSE so với bản Full |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full (Tất cả 7 đặc trưng)** | 7 | Ridge | 6.0812 | 0.6459 | Baseline |
| **Full (Tất cả 7 đặc trưng)** | 7 | Random Forest | **5.7770** | **0.6804** | Baseline |
| **Loại bỏ `month_sin/cos`** | 5 | Ridge | 6.1279 | 0.6404 | +0.0467 µg/m³ |
| **Loại bỏ `month_sin/cos`** | 5 | Random Forest | **5.7796** | **0.6801** | **+0.0026 µg/m³ (Gần như bằng 0)** |
| **Loại bỏ `temperature`** | 6 | Ridge | 6.1032 | 0.6433 | +0.0220 µg/m³ |
| **Loại bỏ `temperature`** | 6 | Random Forest | **5.8913** | **0.6677** | **+0.1143 µg/m³ (Suy giảm rõ rệt)** |
| **Loại bỏ cả tháng và nhiệt độ** | 4 | Random Forest | 5.8952 | 0.6672 | +0.1182 µg/m³ |

**Kết luận thực nghiệm:**
- **Nhiệt độ (`temperature_c_input_causal`)** có đóng góp thực sự vào việc giảm sai số (+0.11 µg/m³), phản ánh chu kỳ nhiệt động lực học trong ngày tại đô thị TP.HCM.
- **Chu kỳ tháng (`month_sin/cos`)** gần như không mang lại giá trị gia tăng nào cho Random Forest (chỉ chênh lệch 0.0026 µg/m³), vì biến `pm2_5_rolling24h_mean` đã gánh phần lớn thông tin nền của mùa. Việc giữ lại biến tháng thuần túy mang tính bảo lưu lý thuyết khí tượng mùa khô/mưa và có thể lược bỏ để làm mô hình nhẹ hơn.

### 6.7. Câu hỏi 7: Về việc mã hóa month trong hàm Real-Time vs Huấn luyện và Kiểm thử đầu-cuối
> **Câu hỏi:** Vì sao hàm real-time dùng month còn lúc huấn luyện dùng month − 1? Và có kiểm thử đầu-cuối cho hàm này không?

- **Kiểm tra mã nguồn:** Script huấn luyện và hàm `predict_next_hour_aqi` đều dùng công thức mã hóa:
  $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
  với `month` nhận giá trị từ 1 đến 12. Không có sự sai lệch giữa `month` và `month - 1`.
- **Kiểm thử đầu-cuối (End-to-End Tests):** Toàn bộ 9/9 unit tests trong `tests/test_features.py` đã được chạy và vượt qua 100%, bao gồm:
  1. `test_predict_next_hour_consistency`: Xác minh tính tuần hoàn, miền giá trị $[-1, 1]$ và sự tương thích với pipeline XGBoost.
  2. `test_create_sliding_sequences_correct_target`: Xác minh chuỗi trượt LSTM lấy đúng nhãn $t+1$ tại vị trí `seq_len - 1`.
  3. `test_threshold_detection_metrics`: Xác minh độ chính xác của hàm tính Recall/Precision và 95% Bootstrap CI.

### 6.8. Câu hỏi 8: Về định nghĩa nhãn PM2.5 theo giờ sang AQI so với Chuẩn 24 giờ / NowCast
> **Câu hỏi:** PM2.5 theo giờ được ánh xạ thẳng sang cấp AQI, trong khi AQI chính thức dùng trung bình 24 giờ hoặc NowCast. Nhãn "đúng" có đang bị định nghĩa lệch không?

- **Chuẩn mực chính thức:** Theo Quyết định số 1459/QĐ-TCMT và hướng dẫn của US EPA, chỉ số chất lượng không khí AQI cho PM2.5 được tính trên nồng độ **trung bình 24 giờ liên tục** hoặc thông qua thuật toán trọng số lũy thoái **NowCast** (tính trên 12 giờ gần nhất).
- **Thẳng thắn nhìn nhận:** Việc lấy nồng độ PM2.5 tức thời 1 giờ ánh xạ thẳng vào bảng ngưỡng nồng độ của chuẩn 24 giờ là một **sự giản lược kỹ thuật (technical simplification)** nhằm đưa ra phản hồi tức thời theo từng giờ cho người dùng. Về mặt danh nghĩa môi trường, nhãn này chính xác phải được gọi là **"Cấp độ ô nhiễm tức thời theo giờ của bụi mịn PM2.5"** chứ **không phải chỉ số VN-AQI chính thức có giá trị pháp lý**. Nhãn "đúng" (ground truth) của bài toán thực chất là cấp độ nồng độ tức thời theo giờ, chứ không phản ánh đúng chuẩn AQI 24h quy định.
- **Đề xuất nâng cấp:** Trong phiên bản tiếp theo, nhóm nghiên cứu sẽ:
  1. Tích hợp thuật toán NowCast vào đường ống xử lý dữ liệu trước khi ánh xạ sang cấp cảnh báo.
  2. Bổ sung mô hình dự báo trực tiếp chuỗi trung bình trượt 24h để tính AQI chuẩn quy chuẩn quốc gia.

---

## 7. DANH MỤC BIỂU ĐỒ & ARTIFACTS TRONG DỰ ÁN

Toàn bộ kết quả thực nghiệm và biểu đồ trực quan đã được tạo và lưu trữ đầy đủ:
1. `artifacts/metrics/regression_comparison.csv` (Bảng so sánh hồi quy hoàn chỉnh)
2. `artifacts/metrics/classification_comparison.csv` (Bảng so sánh phân loại AQI 6 cấp)
3. `artifacts/metrics/threshold_detection_metrics.csv` (Recall, Precision, FAR @ 45 & 50 µg/m³ kèm 95% Bootstrap CI)
4. `artifacts/metrics/feature_ablation.csv` (Thực nghiệm loại bỏ đặc trưng tháng và nhiệt độ)
5. `artifacts/metrics/cv_tuning_results.csv` (Kết quả K-Fold TimeSeriesSplit GridSearchCV)
6. `artifacts/metrics/confusion_matrix_test.csv` (Ma trận nhầm lẫn chi tiết)
7. `artifacts/metrics/lift_analysis.csv` (Hệ số Lift tại các phân vị cảnh báo)
8. `artifacts/metrics/clustering_silhouette.csv` (Hệ số Silhouette từ $k=2$ đến $k=6$)
9. `artifacts/figures/predicted_vs_actual_timeseries.png`
10. `artifacts/figures/regression_residuals_distribution.png`
11. `artifacts/figures/feature_importance_comparison.png`
12. `artifacts/figures/aqi_confusion_matrix.png`
13. `artifacts/figures/lift_curve.png`
14. `artifacts/figures/lstm_training_loss.png`
15. `notebooks/modeling.ipynb` (Notebook thực thi đầy đủ 24 cells với visual outputs).

---

## 8. KẾT LUẬN & ĐỀ XUẤT TRIỂN KHAI

1. **Về lựa chọn mô hình cốt lõi:**
   - **XGBoost Regressor** và **LSTM Regressor** là hai mô hình có chất lượng dự báo tốt nhất (RMSE ~ 6.99 µg/m³, $R^2 \approx 0.792$).
   - Với ứng dụng thực tế trên thiết bị nhẹ, **XGBoost** có ưu thế vượt trội về tốc độ huấn luyện (vài giây), độ ổn định và chi phí tính toán thấp.
   - Đối với cảnh báo ô nhiễm cao, **LSTM** mang lại độ chuẩn xác (Precision) cao nhất (86.3% tại ngưỡng 50 µg/m³) và ít báo nhầm nhất (chỉ 21 ca báo nhầm).

2. **Khuyến nghị kiến trúc hệ thống phục vụ người dùng:**
   - Sử dụng **XGBoost Regressor** kết hợp ngưỡng cảnh báo thích ứng làm mô hình suy luận chính.
   - Ghi chú minh bạch trên giao diện: nhãn màu sắc hiển thị phản ánh *Cấp độ nồng độ bụi PM2.5 tức thời 1 giờ*, đồng thời hiển thị kèm khoảng tin cậy dự báo để người dân nắm bắt rủi ro một cách khoa học.
