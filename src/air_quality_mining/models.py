"""Model definitions, AQI mapping, and evaluation helpers for air quality mining."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

# Thang phân loại 6 cấp độ ô nhiễm theo Quyết định số 1459/QĐ-TCMT (Tổng cục Môi trường Việt Nam)
VN_AQI_BREAKPOINTS = [
    (0.0, 25.0, 0, "Tot"),
    (25.0, 50.0, 1, "Trung binh"),
    (50.0, 80.0, 2, "Kem"),
    (80.0, 150.0, 3, "Xau"),
    (150.0, 250.0, 4, "Rat xau"),
    (250.0, float("inf"), 5, "Nguy hai"),
]

AQI_CATEGORY_NAMES = ["Tot", "Trung binh", "Kem", "Xau", "Rat xau", "Nguy hai"]
AQI_CATEGORY_VI = ["Tốt", "Trung bình", "Kém", "Xấu", "Rất xấu", "Nguy hại"]


def pm25_to_aqi_category(pm25: float) -> int:
    """Chuyển đổi nồng độ PM2.5 (µg/m³) sang nhãn số 0-5 theo chuẩn VN-AQI."""
    if np.isnan(pm25):
        return 0
    for low, high, cat_id, _ in VN_AQI_BREAKPOINTS:
        if low <= pm25 <= high:
            return cat_id
    return 5


def pm25_series_to_aqi(series: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Chuyển đổi mảng nồng độ PM2.5 sang mảng nhãn AQI cấp độ 0-5."""
    arr = np.asarray(series)
    categories = np.zeros_like(arr, dtype=int)
    for low, high, cat_id, _ in VN_AQI_BREAKPOINTS:
        mask = (arr >= low) & (arr <= high) if high != float("inf") else (arr >= low)
        categories[mask] = cat_id
    return categories


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính các chỉ số đánh giá hồi quy: RMSE, MAE, R², MAPE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE tránh chia cho 0
    non_zero = y_true > 0.1
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    else:
        mape = 0.0

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE_pct": round(mape, 2),
    }


def compute_classification_metrics(
    y_true_cat: np.ndarray,
    y_pred_cat: np.ndarray,
    y_pred_proba: np.ndarray = None,
) -> Dict[str, float]:
    """Tính các chỉ số đánh giá phân loại đa lớp: Accuracy, Macro-F1, Weighted-F1, Multi-class AUC-ROC."""
    y_true_cat = np.asarray(y_true_cat, dtype=int)
    y_pred_cat = np.asarray(y_pred_cat, dtype=int)

    acc = accuracy_score(y_true_cat, y_pred_cat)
    macro_f1 = f1_score(y_true_cat, y_pred_cat, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)

    metrics = {
        "Accuracy": round(float(acc), 4),
        "Macro_F1": round(float(macro_f1), 4),
        "Weighted_F1": round(float(weighted_f1), 4),
    }

    if y_pred_proba is not None:
        try:
            # Multi-class AUC ROC (One-vs-Rest)
            classes_present = np.unique(y_true_cat)
            if len(classes_present) > 1:
                # Lọc cột xác suất tương ứng các lớp xuất hiện
                auc = roc_auc_score(
                    y_true_cat,
                    y_pred_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=classes_present,
                )
                metrics["AUC_ROC_macro"] = round(float(auc), 4)
            else:
                metrics["AUC_ROC_macro"] = None
        except Exception:
            metrics["AUC_ROC_macro"] = None
    else:
        metrics["AUC_ROC_macro"] = None

    return metrics


def compute_lift_score(
    y_true_continuous: np.ndarray,
    y_pred_continuous: np.ndarray,
    high_threshold: float = 50.0,
    top_fraction: float = 0.20,
) -> Dict[str, float]:
    """Tính chỉ số Lift tại phân vị cao nhất (e.g. top 20% dự báo cao nhất) cho sự kiện ô nhiễm cao."""
    y_true_event = (np.asarray(y_true_continuous) > high_threshold).astype(int)
    baseline_rate = float(np.mean(y_true_event))

    k = max(int(len(y_pred_continuous) * top_fraction), 1)
    top_indices = np.argsort(y_pred_continuous)[-k:]
    top_precision = float(np.mean(y_true_event[top_indices]))

    lift = top_precision / baseline_rate if baseline_rate > 0 else 1.0

    return {
        "baseline_event_rate": round(baseline_rate, 4),
        f"top_{int(top_fraction*100)}pct_precision": round(top_precision, 4),
        f"lift_at_{int(top_fraction*100)}pct": round(lift, 2),
    }


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

    Tham số:
        y_true: Mảng giá trị PM2.5 thực tế (µg/m³).
        y_pred: Mảng giá trị PM2.5 dự báo (µg/m³).
        thresholds: Danh sách ngưỡng PM2.5 để xét sự kiện ô nhiễm.
        n_bootstrap: Số lần lấy mẫu bootstrap để tính CI.
        rng_seed: Seed cho random number generator.

    Trả về:
        List of dicts, mỗi dict gồm: threshold, recall, precision, f1,
        false_alarm_rate, recall_ci_lo, recall_ci_hi, precision_ci_lo,
        precision_ci_hi, n_events (số sự kiện thực trong tập).
    """
    if thresholds is None:
        thresholds = [45.0, 50.0]

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    results = []

    for thr in thresholds:
        y_true_bin = (y_true >= thr).astype(int)
        y_pred_bin = (y_pred >= thr).astype(int)
        n_events = int(y_true_bin.sum())
        n_neg = n - n_events

        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate = FPR

        # Bootstrap 95% CI
        boot_recalls, boot_precs = [], []
        indices = np.arange(n)
        for _ in range(n_bootstrap):
            idx = rng.choice(indices, size=n, replace=True)
            bt = y_true_bin[idx]
            bp = y_pred_bin[idx]
            btp = int(np.sum((bt == 1) & (bp == 1)))
            bfp = int(np.sum((bt == 0) & (bp == 1)))
            bfn = int(np.sum((bt == 1) & (bp == 0)))
            boot_recalls.append(btp / (btp + bfn) if (btp + bfn) > 0 else 0.0)
            boot_precs.append(btp / (btp + bfp) if (btp + bfp) > 0 else 0.0)

        boot_recalls = np.array(boot_recalls)
        boot_precs = np.array(boot_precs)

        results.append({
            "threshold_ug_m3": thr,
            "n_events": n_events,
            "n_total": n,
            "event_rate_pct": round(100.0 * n_events / n, 2),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "false_alarm_rate": round(far, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall_ci95_lo": round(float(np.percentile(boot_recalls, 2.5)), 4),
            "recall_ci95_hi": round(float(np.percentile(boot_recalls, 97.5)), 4),
            "precision_ci95_lo": round(float(np.percentile(boot_precs, 2.5)), 4),
            "precision_ci95_hi": round(float(np.percentile(boot_precs, 97.5)), 4),
        })

    return results


class LSTMRegressor(nn.Module):
    """Mô hình hồi quy sâu chuỗi thời gian PM2.5 bằng LSTM kết hợp Fully-Connected Head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass qua LSTM và lấy hidden state ở bước cuối cùng."""
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


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

    Lưu ý: vòng lặp chạy từ i=0 đến i=N-seq_len-1 (N-seq_len phần tử).
    Nhãn y[i+seq_len-1] chạy từ y[seq_len-1] đến y[N-2], tương ứng với
    mảng y_test[seq_len-1 : N-1] khi căn chỉnh ngoài hàm này.
    """
    x_seqs, y_seqs = [], []
    for i in range(len(x_data) - seq_len):
        x_seqs.append(x_data[i : i + seq_len])
        # SỬA LỖI: dùng i+seq_len-1 thay vì i+seq_len để lấy nhãn t+1
        # của quan sát cuối trong cửa sổ, căn chỉnh với RF/Ridge/XGB
        y_seqs.append(y_data[i + seq_len - 1])
    return np.array(x_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)
