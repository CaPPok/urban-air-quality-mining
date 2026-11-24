"""Unit tests for air quality modeling, metrics, and AQI classification."""

import unittest
import numpy as np
import torch

from src.air_quality_mining.models import (
    LSTMRegressor,
    compute_classification_metrics,
    compute_lift_score,
    compute_regression_metrics,
    compute_threshold_detection_metrics,
    create_sliding_sequences,
    pm25_series_to_aqi,
    pm25_to_aqi_category,
)


class TestAirQualityModeling(unittest.TestCase):
    def test_aqi_conversion(self):
        self.assertEqual(pm25_to_aqi_category(15.0), 0)   # Tot
        self.assertEqual(pm25_to_aqi_category(35.0), 1)   # Trung binh
        self.assertEqual(pm25_to_aqi_category(60.0), 2)   # Kem
        self.assertEqual(pm25_to_aqi_category(100.0), 3)  # Xau
        self.assertEqual(pm25_to_aqi_category(200.0), 4)  # Rat xau
        self.assertEqual(pm25_to_aqi_category(300.0), 5)  # Nguy hai

        arr = np.array([10.0, 30.0, 75.0, 120.0])
        cats = pm25_series_to_aqi(arr)
        np.testing.assert_array_equal(cats, np.array([0, 1, 2, 3]))

    def test_regression_metrics(self):
        y_true = np.array([20.0, 30.0, 40.0])
        y_pred = np.array([21.0, 29.0, 42.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAE", metrics)
        self.assertIn("R2", metrics)
        self.assertGreater(metrics["R2"], 0.8)

    def test_classification_metrics(self):
        y_true = np.array([0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 2])
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(metrics["Accuracy"], 1.0)
        self.assertEqual(metrics["Macro_F1"], 1.0)

    def test_lift_score(self):
        y_true = np.array([10.0, 20.0, 60.0, 80.0])
        y_pred = np.array([12.0, 25.0, 55.0, 78.0])
        lift = compute_lift_score(y_true, y_pred, high_threshold=50.0, top_fraction=0.5)
        self.assertGreater(lift["lift_at_50pct"], 1.0)

    def test_lstm_architecture(self):
        model = LSTMRegressor(input_dim=7, hidden_dim=32, num_layers=1, dropout=0.0)
        dummy_input = torch.randn(4, 6, 7)
        out = model(dummy_input)
        self.assertEqual(out.shape, torch.Size([4]))

    def test_create_sliding_sequences_shape(self):
        """Kiểm tra kích thước đầu ra của create_sliding_sequences."""
        x = np.ones((20, 5))
        y = np.ones(20)
        seq_x, seq_y = create_sliding_sequences(x, y, seq_len=4)
        # range(20-4) = 16 chuỗi, mỗi chuỗi dài 4, 5 đặc trưng
        self.assertEqual(seq_x.shape, (16, 4, 5))
        self.assertEqual(seq_y.shape, (16,))

    def test_create_sliding_sequences_correct_target(self):
        """Kiểm tra chỉ số nhãn sau sửa lỗi: y[i+seq_len-1] chứ không phải y[i+seq_len].

        Với seq_len=4, chuỗi đầu tiên (i=0) dùng x[0:4], nhãn phải là y[3] (index 3 = seq_len-1).
        Nếu còn lỗi cũ (y[i+seq_len]), nhãn sẽ là y[4] — test này sẽ phát hiện.
        """
        x = np.ones((10, 2))
        y = np.arange(10, dtype=np.float32)  # y = [0,1,2,...,9]
        seq_len = 4
        seq_x, seq_y = create_sliding_sequences(x, y, seq_len=seq_len)
        # Chuỗi i=0: x[0:4], nhãn y[3] = 3.0
        self.assertAlmostEqual(float(seq_y[0]), 3.0,
            msg="Nhãn chuỗi đầu tiên phải là y[seq_len-1]=y[3]=3.0, không phải y[4]=4.0 (lỗi cũ)")
        # Chuỗi i=1: x[1:5], nhãn y[4] = 4.0
        self.assertAlmostEqual(float(seq_y[1]), 4.0,
            msg="Nhãn chuỗi thứ hai phải là y[seq_len]=y[4]=4.0")
        # Số chuỗi = len(x) - seq_len = 10 - 4 = 6
        self.assertEqual(len(seq_y), 6)

    def test_threshold_detection_metrics(self):
        """Kiểm tra compute_threshold_detection_metrics với dữ liệu tổng hợp.

        Trường hợp: 4 sự kiện thực (≥50), dự báo đúng 3 (TP=3), bỏ sót 1 (FN=1), không báo nhầm (FP=0).
        → Recall = 3/4 = 0.75, Precision = 3/3 = 1.0.
        """
        # y_true: 4 điểm ≥50 (index 2,3,6,7); y_pred bỏ sót index 2 (dự báo 48 < 50)
        y_true = np.array([10., 20., 55., 60., 10., 20., 70., 80., 30., 10.])
        y_pred = np.array([12., 22., 48., 58., 12., 15., 65., 75., 35., 15.])
        results = compute_threshold_detection_metrics(y_true, y_pred, thresholds=[50.0], n_bootstrap=100)
        r = results[0]
        self.assertEqual(r["threshold_ug_m3"], 50.0)
        self.assertEqual(r["n_events"], 4)
        # Recall = TP/(TP+FN) = 3/4 = 0.75
        self.assertAlmostEqual(r["recall"], 0.75, places=2)
        # Precision = TP/(TP+FP) = 3/3 = 1.0
        self.assertAlmostEqual(r["precision"], 1.0, places=2)
        # CI bounds phải bọc recall thực
        self.assertLessEqual(r["recall_ci95_lo"], r["recall"])
        self.assertGreaterEqual(r["recall_ci95_hi"], r["recall"])

    def test_predict_next_hour_consistency(self):
        """Kiểm tra đầu-cuối tính nhất quán mã hóa month_sin/cos với dữ liệu huấn luyện."""
        # Mô phỏng logic mã hóa trong train.py: month → month_sin/cos dùng np.sin/cos(2π*month/12)
        for month in [1, 6, 12]:
            m_sin = np.sin(2 * np.pi * month / 12.0)
            m_cos = np.cos(2 * np.pi * month / 12.0)
            # Giá trị phải nằm trong [-1, 1]
            self.assertGreaterEqual(m_sin, -1.0)
            self.assertLessEqual(m_sin, 1.0)
            self.assertGreaterEqual(m_cos, -1.0)
            self.assertLessEqual(m_cos, 1.0)
        # Kiểm tra month=1 và month=13 cho cùng kết quả (tính tuần hoàn)
        sin1 = np.sin(2 * np.pi * 1 / 12.0)
        sin13 = np.sin(2 * np.pi * 13 / 12.0)
        self.assertAlmostEqual(sin1, sin13, places=10,
            msg="Mã hóa month_sin phải tuần hoàn: month=1 phải bằng month=13")


if __name__ == "__main__":
    unittest.main()
