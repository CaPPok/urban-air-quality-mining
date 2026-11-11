# Urban Air Quality Mining & AQI Prediction

> **Bài tập lớn môn Khai phá dữ liệu (Data Mining — CO3029)**  

## 1. Tổng quan

Dự án phân tích dữ liệu quan trắc không khí đô thị, dự báo nồng độ bụi mịn **PM2.5 ở giờ kế tiếp** và định hướng phân loại mức ô nhiễm theo chỉ số **AQI**. có giao diện demo để xem dữ liệu, chọn trạm đo và thực hiện dự báo từ mô hình đã huấn luyện.

**Mục tiêu nghiệp vụ:** hỗ trợ giám sát chất lượng không khí, trực quan hóa xu hướng, đưa ra cảnh báo sớm cho cộng đồng theo chuẩn AQI được xác minh.

**Phạm vi:**

- Dữ liệu lịch sử: **CMT8, HCMC, VietNam** tại [OpenAQ](https://explore.openaq.org/locations/3276359), quan trắc theo giờ, nhiều trạm; có PM2.5, nhiệt độ, khí áp, điểm sương, gió, mưa.
- Hồi quy: dự báo PM2.5 tại `t+1` từ dữ liệu có đến `t`.
- Các mô hình mục tiêu: **baseline persistence, Random Forest Regressor, XGBoost Regressor, LSTM**.
- Phân loại: 6 mức **Tốt / Trung bình / Kém / Xấu / Rất xấu / Nguy hại** theo AQI được lựa chọn và cấu hình có nguồn.
- Thước đo hồi quy: **RMSE, MAE, R²**; phân loại AQI: **Macro-F1, Confusion Matrix**.
- Giao diện demo: Streamlit, biểu đồ PM2.5, dự báo từ mô hình, bản đồ nếu cung cấp tọa độ trạm.

## 2. Cấu trúc thư mục

```text
urban_air_quality_mining/
├── app/                            # Dashboard minh họa
│   └── streamlit_app.py           
├── artifacts/                      # Kết quả huấn luyện và đánh giá mô hình
│   ├── models/                     
│   ├── metrics/                    
│   └── figures/                    
├── configs/                        # Cấu hình dự án
├── data/                           # Dataset
├── notebooks/                      # Notebook EDA và thực nghiệm
│   └── eda.ipynb
├── notes/                          # Ghi chú dự án
│   └── TASK.md
├── reports/                        # Báo cáo tổng hợp
│   └── REPORT.md
├── scripts/                        # Các script chạy pipeline
│   ├── eda.py                     
│   ├── evaluate.py                 
│   └── train.py                    
├── src/                            # Mã nguồn chính  
│   ├── air_quality_mining/                  
│   │   └── features.py     
│   ├── preprocessing/                      
│   │   └── preprocess.py                
├── tests/                          # Test unit       
│   └── test_features.py
├── .gitignore
├── README.md
└── requirements.txt                # Thư viện cài đặt
```

## 3. Chuẩn bị môi trường

```bash
# Tại thư mục gốc repository
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# hoặc 
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 4. Chuẩn bị dữ liệu

Nguồn: **CMT8, HCMC, VietNam** tại [OpenAQ](https://explore.openaq.org/locations/3276359).

1. Tải dữ liệu từ UCI và đọc thông tin giấy phép/nguồn.
2. Giải nén các file `.csv` của từng trạm vào `data/raw/` (không nén ZIP trong đó).
3. File CSV nguồn cần có `year, month, day, hour, PM2.5`; các cột khí tượng thường gồm `TEMP, PRES, DEWP, RAIN, WSPM, wd` và `station`.
4. Để tạo bản đồ, chuẩn bị CSV riêng gồm `station,latitude,longitude` với tọa độ **đã xác minh**. Không tự đoán tọa độ.

## 5. Chạy chương trình


## 6. Quy trình phương pháp

```text
Dữ liệu theo giờ & trạm đo
           │
           ▼
Khám phá / EDA / kiểm tra khuyết & ngoại biên
           │
           ▼
Tiền xử lý và tạo đặc trưng từ quá khứ (lag, rolling 24h)
           │
           ▼
Chia train / validation / test theo thời gian
           │
           ├── Baseline persistence
           ├── Random Forest Regressor
           ├── XGBoost Regressor
           └── LSTM (TODO)
           │
           ▼
TimeSeriesSplit + tuning + RMSE / MAE / R²
           │
           ▼
Chọn mô hình theo validation; báo cáo test cuối
           │
           ▼
PM2.5 dự báo → trung bình 24h → AQI đã chuẩn hóa (TODO kiểm chứng)
           │
           ▼
Macro-F1 / confusion matrix (TODO) → Dashboard Streamlit
```

**Tránh rò rỉ dữ liệu:**

- Đầu vào giờ t chỉ được dùng thông tin ≤ t; nhãn là PM2.5 đo thật của giờ t+1.
- Không shuffle train/test, không dùng dữ liệu tương lai để nội suy mẫu huấn luyện hoặc fit scaler/imputer.
- Với nhiều trạm tại cùng giờ, CV được chia theo *timestamp*, không phải chia ngẫu nhiên theo hàng.
- Không tuning hyperparameter dựa trên test.
- Đối với AQI dự báo của giờ t+1, cần đủ số đo liên tiếp cho trung bình PM2.5 24 giờ và quy tắc tính từ chuẩn chính thức.

## 7. Đánh giá và rubric

| Tiêu chí BTL | Tỉ trọng | Bằng chứng cần nộp |
|---|---:|---|
| EDA & Preprocessing | 20% | Thống kê/biểu đồ, phương pháp điền khuyết, ngoại biên, feature engineering |
| Modeling & Tuning | 30% | **Ít nhất 3 thuật toán thực sự chạy**, CV, tuning, bảng so sánh, giải thích thuật toán |
| Demo / UI | 20% | Streamlit hoặc web hoàn chỉnh có nhập dữ liệu và trả kết quả dự báo/phân loại/khuyến nghị |
| Technical Report | 15% | Báo cáo khoa học **15–25 trang**, trích dẫn đầy đủ, minh bạch công cụ AI |
| Presentation & Q&A | 15% | Slide, demo, thuyết trình, trả lời câu hỏi bản chất thuật toán |

## 8. Báo cáo 5 chương

Báo cáo tại [`reports/REPORT.md`](reports/REPORT.md):

1. **Mở đầu**: tính cấp thiết, mục tiêu, phạm vi, phát biểu bài toán.
2. **Cơ sở lý thuyết**: PM2.5/AQI, xử lý chuỗi thời gian, RF/XGBoost/LSTM, các chỉ số đánh giá.
3. **Phương pháp nghiên cứu**: dữ liệu, EDA, preprocessing, thiết kế thí nghiệm, mô hình, AQI, kiến trúc ứng dụng.
4. **Kết quả và thảo luận**: bảng so sánh, biểu đồ, kết quả AQI, ảnh demo, hạn chế.
5. **Kết luận và hướng phát triển đề tài**: đóng góp đã kiểm chứng, hạn chế, phát triển dữ liệu và triển khai.

## 9. Giới hạn và hướng mở rộng

## 10. Tài liệu tham khảo

- [Dataset: Air Quality Dataset for Ho Chi Minh City](https://zenodo.org/records/18673714): BACKUP DATASET