"""Huấn luyện các mô hình hồi quy PM2.5 và phân loại cấp độ ô nhiễm AQI.

Bao gồm:
- Baseline Persistence & Ridge Regression
- Random Forest Regressor & Tuning với TimeSeriesSplit (5-fold)
- XGBoost Regressor & Tuning với TimeSeriesSplit (5-fold)
- LSTM Regressor (PyTorch Deep Learning) với Early Stopping
- Random Forest Classifier & XGBoost Classifier trực tiếp cho 6 cấp AQI
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Thêm thư mục gốc vào path để import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.air_quality_mining.models import (
    LSTMRegressor,
    create_sliding_sequences,
    pm25_series_to_aqi,
)


def load_dataset(data_path: Path):
    """Tải tập dữ liệu đã qua tiền xử lý hoàn chỉnh và tách thành các split."""
    print(f"[1/6] Đang đọc dữ liệu từ: {data_path}")
    df = pd.read_csv(data_path)

    features = [
        "pm2_5_ug_m3_input_causal",
        "pm2_5_rolling24h_mean",
        "temperature_c_input_causal",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]
    target = "target_pm2_5_next_hour"

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "validation"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"      - Kích thước tập Train: {len(train_df)} dòng")
    print(f"      - Kích thước tập Validation: {len(val_df)} dòng")
    print(f"      - Kích thước tập Test: {len(test_df)} dòng")

    return train_df, val_df, test_df, features, target


def run_cross_validation_tuning(X_train: pd.DataFrame, y_train: pd.Series, cv_results_path: Path):
    """Thực hiện TimeSeriesSplit 5-fold cross-validation và tinh chỉnh siêu tham số."""
    print("[2/6] Đang thực hiện TimeSeriesSplit K-Fold (5 folds) & GridSearchCV Tuning...")
    tscv = TimeSeriesSplit(n_splits=5)
    tuning_records = []

    # 1. Tinh chỉnh Random Forest Regressor
    print("      -> Tinh chỉnh siêu tham số cho Random Forest Regressor...")
    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [8, 12, 16],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf_grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=rf_param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    rf_grid.fit(X_train, y_train)
    rf_tune_time = time.time() - t0
    best_rf_params = rf_grid.best_params_
    best_rf_score = -rf_grid.best_score_
    print(f"         Tốt nhất RF: RMSE = {best_rf_score:.4f} với tham số: {best_rf_params} ({rf_tune_time:.1f}s)")

    for params, mean_score, std_score in zip(
        rf_grid.cv_results_["params"],
        -rf_grid.cv_results_["mean_test_score"],
        rf_grid.cv_results_["std_test_score"],
    ):
        tuning_records.append({
            "model": "RandomForestRegressor",
            "params": json.dumps(params),
            "cv_mean_rmse": round(mean_score, 4),
            "cv_std_rmse": round(std_score, 4),
            "rank": int(rf_grid.cv_results_["rank_test_score"][len(tuning_records)]),
        })

    # 2. Tinh chỉnh XGBoost Regressor
    print("      -> Tinh chỉnh siêu tham số cho XGBoost Regressor...")
    xgb_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.08],
        "subsample": [0.8, 1.0],
    }
    xgb_grid = GridSearchCV(
        estimator=xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_grid=xgb_param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    xgb_grid.fit(X_train, y_train)
    xgb_tune_time = time.time() - t0
    best_xgb_params = xgb_grid.best_params_
    best_xgb_score = -xgb_grid.best_score_
    print(f"         Tốt nhất XGB: RMSE = {best_xgb_score:.4f} với tham số: {best_xgb_params} ({xgb_tune_time:.1f}s)")

    start_idx = len(tuning_records)
    for i, (params, mean_score, std_score) in enumerate(zip(
        xgb_grid.cv_results_["params"],
        -xgb_grid.cv_results_["mean_test_score"],
        xgb_grid.cv_results_["std_test_score"],
    )):
        tuning_records.append({
            "model": "XGBRegressor",
            "params": json.dumps(params),
            "cv_mean_rmse": round(mean_score, 4),
            "cv_std_rmse": round(std_score, 4),
            "rank": int(xgb_grid.cv_results_["rank_test_score"][i]),
        })

    cv_df = pd.DataFrame(tuning_records)
    cv_df.to_csv(cv_results_path, index=False)
    print(f"      Đã lưu kết quả tuning vào: {cv_results_path}")

    return best_rf_params, best_xgb_params


def train_models():
    """Hàm điều phối toàn bộ quá trình huấn luyện và lưu artifacts."""
    models_dir = PROJECT_ROOT / "artifacts" / "models"
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    data_path = PROJECT_ROOT / "data" / "processed" / "openaq_cmt8_model_ready.csv"
    train_df, val_df, test_df, features, target = load_dataset(data_path)

    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]

    # Chuẩn hóa đặc trưng
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    joblib.dump(scaler_x, models_dir / "scaler_x.joblib")

    scaler_y = StandardScaler()
    scaler_y.fit(train_df[[target]])
    joblib.dump(scaler_y, models_dir / "scaler_y.joblib")

    # TimeSeries Cross-validation Tuning
    cv_csv = metrics_dir / "cv_tuning_results.csv"
    best_rf_params, best_xgb_params = run_cross_validation_tuning(X_train, y_train, cv_csv)

    # 1. Ridge Baseline
    print("[3/6] Huấn luyện Baseline Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    joblib.dump(ridge, models_dir / "ridge_baseline.joblib")

    # 2. Random Forest Regressor tối ưu
    print(f"[4/6] Huấn luyện Random Forest Regressor với tham số tối ưu: {best_rf_params}...")
    rf_best = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    rf_best.fit(X_train, y_train)
    joblib.dump(rf_best, models_dir / "rf_regressor.joblib")

    # 3. XGBoost Regressor tối ưu
    print(f"[5/6] Huấn luyện XGBoost Regressor với tham số tối ưu: {best_xgb_params}...")
    xgb_best = xgb.XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)
    xgb_best.fit(X_train, y_train)
    xgb_best.save_model(str(models_dir / "xgb_regressor.json"))

    # 4. LSTM Regressor
    print("[6/6] Huấn luyện LSTM Regressor (PyTorch) với Early Stopping...")
    seq_len = 6
    X_tr_sc = scaler_x.transform(X_train)
    y_tr_sc = scaler_y.transform(train_df[[target]]).flatten()
    X_val_sc = scaler_x.transform(X_val)
    y_val_sc = scaler_y.transform(val_df[[target]]).flatten()

    X_tr_seq, y_tr_seq = create_sliding_sequences(X_tr_sc, y_tr_sc, seq_len)
    X_val_seq, y_val_seq = create_sliding_sequences(X_val_sc, y_val_sc, seq_len)

    train_dataset = TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32), torch.tensor(y_tr_seq, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    torch.manual_seed(42)
    lstm_model = LSTMRegressor(input_dim=len(features), hidden_dim=64, num_layers=2, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_loss = float("inf")
    patience = 4
    patience_counter = 0
    loss_history = []

    for epoch in range(25):
        lstm_model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = lstm_model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(X_tr_seq)

        lstm_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                preds = lstm_model(bx)
                loss = criterion(preds, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(X_val_seq)

        loss_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"      Epoch {epoch+1:02d} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(lstm_model.state_dict(), models_dir / "lstm_regressor.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Dừng sớm (Early Stopping) tại epoch {epoch+1}")
                break

    pd.DataFrame(loss_history).to_csv(metrics_dir / "lstm_loss_history.csv", index=False)

    # Lưu cấu hình LSTM
    lstm_config = {
        "input_dim": len(features),
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_len": seq_len,
        "features": features,
    }
    with open(models_dir / "lstm_config.json", "w", encoding="utf-8") as f:
        json.dump(lstm_config, f, indent=2)

    # 5. Phân loại AQI trực tiếp: RF Classifier & XGB Classifier
    print("[+] Huấn luyện mô hình phân loại AQI trực tiếp (6 cấp)...")
    y_train_cat = pm25_series_to_aqi(y_train)

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train_cat)
    joblib.dump(rf_clf, models_dir / "rf_classifier.joblib")

    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1)
    xgb_clf.fit(X_train, y_train_cat)
    xgb_clf.save_model(str(models_dir / "xgb_classifier.json"))

    print("\n>>> HOÀN TẤT HUẤN LUYỆN TOÀN BỘ CÁC MÔ HÌNH VÀ LƯU ARTIFACTS THÀNH CÔNG! <<<")


if __name__ == "__main__":
    train_models()
