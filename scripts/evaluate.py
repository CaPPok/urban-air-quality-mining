"""Đánh giá toàn diện các mô hình hồi quy và phân loại theo chuẩn L.O.4.1 và L.O.4.2.

Xuất các bảng chỉ số:
- artifacts/metrics/regression_comparison.csv (RMSE, MAE, R², MAPE)
- artifacts/metrics/classification_comparison.csv (Accuracy, Macro-F1, Weighted-F1, AUC-ROC)
- artifacts/metrics/confusion_matrix_test.csv
- artifacts/metrics/clustering_silhouette.csv (Silhouette scores)
- artifacts/metrics/lift_analysis.csv (Lift @ 10%, 20%, 30%)

Xuất các biểu đồ trực quan:
- artifacts/figures/predicted_vs_actual_timeseries.png
- artifacts/figures/regression_residuals_distribution.png
- artifacts/figures/feature_importance_comparison.png
- artifacts/figures/aqi_confusion_matrix.png
- artifacts/figures/lift_curve.png
- artifacts/figures/lstm_training_loss.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.air_quality_mining.models import (
    AQI_CATEGORY_NAMES,
    AQI_CATEGORY_VI,
    LSTMRegressor,
    compute_classification_metrics,
    compute_lift_score,
    compute_regression_metrics,
    compute_threshold_detection_metrics,
    create_sliding_sequences,
    pm25_series_to_aqi,
)

# Cấu hình font và style vẽ đồ thị
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def evaluate_all():
    """Hàm thực thi đánh giá toàn bộ mô hình và tạo artifacts."""
    models_dir = PROJECT_ROOT / "artifacts" / "models"
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    figures_dir = PROJECT_ROOT / "artifacts" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_path = PROJECT_ROOT / "data" / "processed" / "openaq_cmt8_model_ready.csv"
    print(f"[1/5] Đang nạp dữ liệu từ: {data_path}")
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

    X_train, y_train = train_df[features], train_df[target].values
    X_val, y_val = val_df[features], val_df[target].values
    X_test, y_test = test_df[features], test_df[target].values

    # Nạp scalers và models
    print("[2/5] Đang nạp các mô hình đã huấn luyện...")
    scaler_x = joblib.load(models_dir / "scaler_x.joblib")
    scaler_y = joblib.load(models_dir / "scaler_y.joblib")

    ridge = joblib.load(models_dir / "ridge_baseline.joblib")
    rf_reg = joblib.load(models_dir / "rf_regressor.joblib")
    xgb_reg = xgb.XGBRegressor()
    xgb_reg.load_model(str(models_dir / "xgb_regressor.json"))

    rf_clf = joblib.load(models_dir / "rf_classifier.joblib")
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(str(models_dir / "xgb_classifier.json"))

    with open(models_dir / "lstm_config.json", "r", encoding="utf-8") as f:
        lstm_cfg = json.load(f)

    lstm_reg = LSTMRegressor(
        input_dim=lstm_cfg["input_dim"],
        hidden_dim=lstm_cfg["hidden_dim"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
    )
    lstm_reg.load_state_dict(torch.load(models_dir / "lstm_regressor.pt", weights_only=True))
    lstm_reg.eval()

    # Tạo chuỗi sequence cho LSTM
    seq_len = lstm_cfg["seq_len"]
    X_tr_sc = scaler_x.transform(X_train)
    y_tr_sc = scaler_y.transform(train_df[[target]]).flatten()
    X_val_sc = scaler_x.transform(X_val)
    y_val_sc = scaler_y.transform(val_df[[target]]).flatten()
    X_te_sc = scaler_x.transform(X_test)
    y_te_sc = scaler_y.transform(test_df[[target]]).flatten()

    X_tr_seq, y_tr_seq = create_sliding_sequences(X_tr_sc, y_tr_sc, seq_len)
    X_val_seq, y_val_seq = create_sliding_sequences(X_val_sc, y_val_sc, seq_len)
    X_te_seq, y_te_seq = create_sliding_sequences(X_te_sc, y_te_sc, seq_len)

    # Dự báo LSTM
    with torch.no_grad():
        lstm_pred_val_norm = lstm_reg(torch.tensor(X_val_seq, dtype=torch.float32)).numpy()
        lstm_pred_te_norm = lstm_reg(torch.tensor(X_te_seq, dtype=torch.float32)).numpy()

    lstm_pred_val = scaler_y.inverse_transform(lstm_pred_val_norm.reshape(-1, 1)).flatten()
    lstm_pred_te = scaler_y.inverse_transform(lstm_pred_te_norm.reshape(-1, 1)).flatten()
    # SỬA LỖI căn chỉnh: sau khi fix create_sliding_sequences (dùng y[i+seq_len-1]),
    # nhãn thực tế tương ứng với N-seq_len predictions là y[seq_len-1 : N-1].
    # Dùng slice [seq_len-1:-1] để loại bỏ phần tử cuối (không có prediction tương ứng).
    y_val_aligned = y_val[seq_len - 1 : len(y_val) - 1]
    y_test_aligned = y_test[seq_len - 1 : len(y_test) - 1]

    # Dự báo các mô hình trên Val và Test
    preds_val = {
        "Persistence Baseline": val_df["pm2_5_ug_m3_input_causal"].values,
        "Ridge Baseline": ridge.predict(X_val),
        "Random Forest Regressor": rf_reg.predict(X_val),
        "XGBoost Regressor": xgb_reg.predict(X_val),
        "LSTM Regressor": (lstm_pred_val, y_val_aligned),
    }

    preds_test = {
        "Persistence Baseline": test_df["pm2_5_ug_m3_input_causal"].values,
        "Ridge Baseline": ridge.predict(X_test),
        "Random Forest Regressor": rf_reg.predict(X_test),
        "XGBoost Regressor": xgb_reg.predict(X_test),
        "LSTM Regressor": (lstm_pred_te, y_test_aligned),
    }

    # ==========================================
    # 1. ĐÁNH GIÁ HỒI QUY (L.O.4.1, L.O.4.2)
    # ==========================================
    print("[3/5] Tính toán chỉ số Hồi quy (RMSE, MAE, R², MAPE)...")
    reg_rows = []
    for model_name, p_te in preds_test.items():
        p_val = preds_val[model_name]

        if isinstance(p_te, tuple):
            te_pred, te_true = p_te
            val_pred, val_true = p_val
        else:
            te_pred, te_true = p_te, y_test
            val_pred, val_true = p_val, y_val

        m_val = compute_regression_metrics(val_true, val_pred)
        m_te = compute_regression_metrics(te_true, te_pred)

        reg_rows.append({
            "Model": model_name,
            "Val_RMSE": m_val["RMSE"],
            "Val_MAE": m_val["MAE"],
            "Val_R2": m_val["R2"],
            "Val_MAPE_pct": m_val["MAPE_pct"],
            "Test_RMSE": m_te["RMSE"],
            "Test_MAE": m_te["MAE"],
            "Test_R2": m_te["R2"],
            "Test_MAPE_pct": m_te["MAPE_pct"],
        })

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(metrics_dir / "regression_comparison.csv", index=False)
    print("      Đã lưu bảng so sánh hồi quy:")
    print(reg_df.to_string(index=False))

    # ==========================================
    # 2. ĐÁNH GIÁ PHÂN LOẠI AQI (6 CẤP ĐỘ)
    # ==========================================
    print("\n[4/5] Tính toán chỉ số Phân loại AQI 6 cấp (Accuracy, Macro-F1, Weighted-F1, AUC-ROC)...")
    y_test_cat = pm25_series_to_aqi(y_test)
    y_test_cat_aligned = pm25_series_to_aqi(y_test_aligned)

    clf_rows = []

    # Majority Baseline
    majority_class = int(pd.Series(pm25_series_to_aqi(y_train)).mode()[0])
    maj_pred = np.full_like(y_test_cat, majority_class)
    m_maj = compute_classification_metrics(y_test_cat, maj_pred)
    clf_rows.append({"Model": "Majority Class Baseline", "Approach": "Rule-based", **m_maj})

    # Continuous to AQI Mapping
    for model_name, p_te in preds_test.items():
        if isinstance(p_te, tuple):
            te_pred, _ = p_te
            te_true_cat = y_test_cat_aligned
        else:
            te_pred = p_te
            te_true_cat = y_test_cat

        cat_pred = pm25_series_to_aqi(te_pred)
        m = compute_classification_metrics(te_true_cat, cat_pred)
        clf_rows.append({"Model": f"{model_name} -> AQI Mapping", "Approach": "Threshold Mapping", **m})

    # Direct Classifiers
    rf_clf_pred = rf_clf.predict(X_test)
    rf_clf_proba = rf_clf.predict_proba(X_test)
    m_rf = compute_classification_metrics(y_test_cat, rf_clf_pred, rf_clf_proba)
    clf_rows.append({"Model": "Random Forest Classifier (balanced)", "Approach": "Direct Multiclass", **m_rf})

    xgb_clf_pred = xgb_clf.predict(X_test)
    xgb_clf_proba = xgb_clf.predict_proba(X_test)
    m_xgb = compute_classification_metrics(y_test_cat, xgb_clf_pred, xgb_clf_proba)
    clf_rows.append({"Model": "XGBoost Classifier", "Approach": "Direct Multiclass", **m_xgb})

    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(metrics_dir / "classification_comparison.csv", index=False)
    print("      Đã lưu bảng so sánh phân loại AQI:")
    print(clf_df.to_string(index=False))

    # Confusion matrix cho Test set
    cm_rf = confusion_matrix(y_test_cat, rf_clf_pred, labels=[0, 1, 2, 3])
    cm_xgb = confusion_matrix(y_test_cat, xgb_clf_pred, labels=[0, 1, 2, 3])
    cm_map_xgb = confusion_matrix(y_test_cat, pm25_series_to_aqi(preds_test["XGBoost Regressor"]), labels=[0, 1, 2, 3])

    cm_records = []
    labels_4 = ["Tốt (0-25)", "Trung bình (25-50)", "Kém (50-80)", "Xấu (80-150)"]
    for i, true_label in enumerate(labels_4):
        for j, pred_label in enumerate(labels_4):
            cm_records.append({
                "True_Label": true_label,
                "Pred_Label": pred_label,
                "Count_RF_Direct": int(cm_rf[i, j]),
                "Count_XGB_Direct": int(cm_xgb[i, j]),
                "Count_XGB_Mapped": int(cm_map_xgb[i, j]),
            })
    pd.DataFrame(cm_records).to_csv(metrics_dir / "confusion_matrix_test.csv", index=False)

    # ==========================================
    # 3. THRESHOLD DETECTION METRICS (Cảnh báo ô nhiễm)
    # ==========================================
    print("\n[+] Tính toán chỉ số phát hiện ngưỡng ô nhiễm (Recall/Precision @ 45 & 50 µg/m³)...")
    thr_records = []
    for model_name, p_te in preds_test.items():
        if isinstance(p_te, tuple):
            te_pred, te_true = p_te
        else:
            te_pred, te_true = p_te, y_test
        per_thr = compute_threshold_detection_metrics(te_true, te_pred, thresholds=[45.0, 50.0])
        for row in per_thr:
            thr_records.append({"Model": model_name, **row})
    thr_df = pd.DataFrame(thr_records)
    thr_df.to_csv(metrics_dir / "threshold_detection_metrics.csv", index=False)
    print("      Chỉ số phát hiện ngưỡng ô nhiễm (tập Test):")
    print(thr_df[["Model", "threshold_ug_m3", "n_events", "recall", "precision", "f1",
                  "false_alarm_rate", "recall_ci95_lo", "recall_ci95_hi"]].to_string(index=False))

    # ==========================================
    # 4. FEATURE ABLATION (Q6: month_sin/cos, temperature)
    # ==========================================
    print("\n[+] Chạy Feature Ablation trên tập Validation (loại bỏ từng nhóm đặc trưng)...")
    from sklearn.linear_model import Ridge as _Ridge
    from sklearn.ensemble import RandomForestRegressor as _RF
    ablation_groups = {
        "Full (all 7 features)": features,
        "No month_sin/cos": [f for f in features if "month" not in f],
        "No temperature": [f for f in features if "temperature" not in f],
        "No month + No temp": [f for f in features if "month" not in f and "temperature" not in f],
    }
    ablation_records = []
    for group_name, feat_subset in ablation_groups.items():
        X_tr_ab = train_df[feat_subset].values
        X_va_ab = val_df[feat_subset].values
        y_tr = train_df[target].values
        y_va = val_df[target].values
        # Quick Ridge ablation (fast proxy)
        ridge_ab = _Ridge(alpha=1.0)
        ridge_ab.fit(X_tr_ab, y_tr)
        pred_ab = ridge_ab.predict(X_va_ab)
        m_ab = compute_regression_metrics(y_va, pred_ab)
        ablation_records.append({"features": group_name, "n_features": len(feat_subset),
                                  "model": "Ridge", **m_ab})
        # RF ablation
        rf_ab = _RF(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf_ab.fit(X_tr_ab, y_tr)
        pred_ab = rf_ab.predict(X_va_ab)
        m_ab = compute_regression_metrics(y_va, pred_ab)
        ablation_records.append({"features": group_name, "n_features": len(feat_subset),
                                  "model": "RF", **m_ab})
    abl_df = pd.DataFrame(ablation_records)
    abl_df.to_csv(metrics_dir / "feature_ablation.csv", index=False)
    print("      Kết quả Feature Ablation (Val RMSE):")
    print(abl_df[["model", "features", "n_features", "RMSE", "R2"]].to_string(index=False))

    # ==========================================
    # 5. LIFT ANALYSIS & SILHOUETTE CLUSTERING
    # ==========================================
    print("\n[5/5] Tính toán Lift và Silhouette Score...")
    lift_records = []
    for top_f in [0.10, 0.20, 0.30]:
        for model_name, p_te in preds_test.items():
            if isinstance(p_te, tuple):
                te_pred, te_true = p_te
            else:
                te_pred, te_true = p_te, y_test
            l_dict = compute_lift_score(te_true, te_pred, high_threshold=50.0, top_fraction=top_f)
            lift_records.append({
                "Model": model_name,
                "Top_Fraction": f"{int(top_f*100)}%",
                "Baseline_Rate": l_dict["baseline_event_rate"],
                "Precision": l_dict[f"top_{int(top_f*100)}pct_precision"],
                "Lift": l_dict[f"lift_at_{int(top_f*100)}pct"],
            })
    lift_df = pd.DataFrame(lift_records)
    lift_df.to_csv(metrics_dir / "lift_analysis.csv", index=False)

    # Silhouette analysis
    sil_records = []
    X_full_scaled = scaler_x.transform(df[features].sample(3000, random_state=42))
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X_full_scaled)
        s_score = silhouette_score(X_full_scaled, lbls)
        sil_records.append({"k_clusters": k, "Silhouette_Score": round(float(s_score), 4)})
    sil_df = pd.DataFrame(sil_records)
    sil_df.to_csv(metrics_dir / "clustering_silhouette.csv", index=False)
    print("      Silhouette scores:\n", sil_df.to_string(index=False))

    # ==========================================
    # 4. VẼ VÀ LƯU CÁC BIỂU ĐỒ (ARTIFACTS/FIGURES)
    # ==========================================
    print("\n[+] Đang vẽ và xuất các biểu đồ trực quan...")

    # 1. Predicted vs Actual Time Series (200 giờ đầu của Test set)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    window = 168  # 1 tuần
    ax.plot(y_test[:window], label="Actual PM2.5 (Thực tế)", color="black", linewidth=2.0)
    ax.plot(preds_test["Persistence Baseline"][:window], label="Persistence Baseline", color="gray", linestyle="--", alpha=0.7)
    ax.plot(preds_test["Random Forest Regressor"][:window], label="Random Forest", color="tab:green", alpha=0.85)
    ax.plot(preds_test["XGBoost Regressor"][:window], label="XGBoost", color="tab:orange", alpha=0.9, linewidth=1.5)
    ax.plot(preds_test["LSTM Regressor"][0][:window], label="LSTM Regressor", color="tab:blue", alpha=0.85)
    ax.set_title("So sánh Nồng độ PM2.5 Dự báo vs Thực tế trong 1 tuần (Tập Test)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thời gian (Giờ)", fontsize=11)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "predicted_vs_actual_timeseries.png")
    plt.close(fig)

    # 2. Phân phối sai số phần dư (Residuals Distribution)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150, sharey=True)
    models_to_plot = ["Persistence Baseline", "Random Forest Regressor", "XGBoost Regressor"]
    colors = ["gray", "tab:green", "tab:orange"]

    for ax, m_name, color in zip(axes, models_to_plot, colors):
        res = y_test - preds_test[m_name]
        sns.histplot(res, bins=50, kde=True, ax=ax, color=color)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(f"{m_name}\nMean: {np.mean(res):.2f}, Std: {np.std(res):.2f}", fontsize=11)
        ax.set_xlabel("Phần dư (y_thực tế - y_dựbáo)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phân phối sai số phần dư (Residual Distribution) trên tập Test", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "regression_residuals_distribution.png")
    plt.close(fig)

    # 3. Feature Importance (RF vs XGBoost)
    rf_imp = rf_reg.feature_importances_
    xgb_imp = xgb_reg.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": features,
        "Random Forest": rf_imp,
        "XGBoost": xgb_imp,
    }).sort_values(by="XGBoost", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    y_pos = np.arange(len(features))
    h = 0.35
    ax.barh(y_pos - h/2, imp_df["Random Forest"], height=h, label="Random Forest", color="tab:green", alpha=0.85)
    ax.barh(y_pos + h/2, imp_df["XGBoost"], height=h, label="XGBoost", color="tab:orange", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df["Feature"])
    ax.set_xlabel("Mức độ quan trọng (Feature Importance)", fontsize=11)
    ax.set_title("So sánh Tầm quan trọng Đặc trưng (Feature Importance)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance_comparison.png")
    plt.close(fig)

    # 4. Confusion Matrix cho 3 mô hình phân loại AQI
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    cm_titles = [
        "XGBoost Classifier (Trực tiếp)",
        "RF Classifier (Balanced)",
        "XGBoost Regressor -> AQI Mapping",
    ]
    cms = [cm_xgb, cm_rf, cm_map_xgb]
    short_labels = ["Tốt", "Trung bình", "Kém", "Xấu"]

    for ax, cm, title in zip(axes, cms, cm_titles):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=short_labels, yticklabels=short_labels, cbar=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Nhãn Dự báo (Predicted)", fontsize=10)
        ax.set_ylabel("Nhãn Thực tế (Actual)", fontsize=10)
    fig.suptitle("Ma trận nhầm lẫn (Confusion Matrix) 4 cấp AQI trên tập Test", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "aqi_confusion_matrix.png")
    plt.close(fig)

    # 5. Lift Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for m_name, color in zip(["Persistence Baseline", "Random Forest Regressor", "XGBoost Regressor"], ["gray", "tab:green", "tab:orange"]):
        lifts = []
        for f in fractions:
            l_res = compute_lift_score(y_test, preds_test[m_name], high_threshold=50.0, top_fraction=f)
            lifts.append(l_res[f"lift_at_{int(f*100)}pct"])
        ax.plot([f*100 for f in fractions], lifts, marker="o", label=m_name, color=color, linewidth=2)

    ax.axhline(1.0, color="red", linestyle="--", label="Baseline ngẫu nhiên (Lift = 1.0)")
    ax.set_title("Biểu đồ Lift theo tỷ lệ mẫu cảnh báo ô nhiễm cao (PM2.5 > 50 µg/m³)", fontsize=12, fontweight="bold")
    ax.set_xlabel("% Mẫu được xếp hạng dự báo cao nhất (Top % Samples)", fontsize=11)
    ax.set_ylabel("Hệ số cải thiện (Lift Score)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "lift_curve.png")
    plt.close(fig)

    # 6. LSTM Training Loss
    loss_hist_path = metrics_dir / "lstm_loss_history.csv"
    if loss_hist_path.exists():
        loss_df = pd.read_csv(loss_hist_path)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        ax.plot(loss_df["epoch"], loss_df["train_loss"], marker="o", label="Train Loss (MSE chuẩn hóa)", color="tab:blue")
        ax.plot(loss_df["epoch"], loss_df["val_loss"], marker="s", label="Validation Loss (MSE chuẩn hóa)", color="tab:red")
        ax.set_title("Lịch sử Huấn luyện LSTM Regressor (MSE Loss qua các Epoch)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss (Normalized MSE)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(figures_dir / "lstm_training_loss.png")
        plt.close(fig)

    print("\n>>> ĐÃ HOÀN TẤT ĐÁNH GIÁ VÀ XUẤT TOÀN BỘ ARTIFACTS THÀNH CÔNG! <<<")


if __name__ == "__main__":
    evaluate_all()
