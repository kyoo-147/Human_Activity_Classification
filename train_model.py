# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from collections import deque

# 1. Đọc và gán nhãn dữ liệu từ từng file CSV
def load_and_label_data(file_path, label):
    data = pd.read_csv(file_path)
    data = data[['accel_x', 'accel_y', 'accel_z']]  # Lấy dữ liệu gia tốc cần thiết
    data['label'] = label  # Gán nhãn tư thế
    return data

# Đọc dữ liệu từ từng file và gán nhãn
data_running = load_and_label_data('data_train/running.csv', 'Chạy bộ')
data_walking = load_and_label_data('data_train/walking.csv', 'Đi bộ')
data_standing = load_and_label_data('data_train/standing.csv', 'Đứng')
data_lying = load_and_label_data('data_train/lying.csv', 'Nằm')
data_sitting = load_and_label_data('data_train/sitting.csv', 'Ngồi')

# 2. Kết hợp dữ liệu từ tất cả các tư thế thành một DataFrame
all_data = pd.concat([data_running, data_walking, data_standing, data_lying, data_sitting], ignore_index=True)

# 3. Chuẩn bị dữ liệu bằng cách trích xuất đặc trưng từ cửa sổ trượt
window_size = 6

def extract_features_from_window(window):
    mean_ax = np.mean(window["accel_x"])
    mean_ay = np.mean(window["accel_y"])
    mean_az = np.mean(window["accel_z"])
    sd_ax = np.std(window["accel_x"])
    sd_ay = np.std(window["accel_y"])
    sd_az = np.std(window["accel_z"])
    return [mean_ax, mean_ay, mean_az, sd_ax, sd_ay, sd_az]

# Tạo danh sách để lưu đặc trưng và nhãn
X = []
y = []

# Sử dụng cửa sổ trượt để trích xuất đặc trưng
for i in range(0, len(all_data) - window_size + 1, window_size):
    window = all_data.iloc[i:i + window_size]
    features = extract_features_from_window(window)
    X.append(features)
    y.append(window["label"].values[0])  # Nhãn tương ứng với cửa sổ trượt

# 4. Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Huấn luyện mô hình Decision Tree
print("Huấn luyện mô hình Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 6. Huấn luyện mô hình Random Forest
print("Huấn luyện mô hình Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 7. Đánh giá mô hình Decision Tree
print("\nĐánh giá mô hình Decision Tree:")
y_pred_dt = dt_model.predict(X_test)
print(f"Độ chính xác Decision Tree: {accuracy_score(y_test, y_pred_dt) * 100:.2f}%")
print(classification_report(y_test, y_pred_dt, target_names=["Chạy bộ", "Đi bộ", "Đứng", "Nằm", "Ngồi"]))

# 8. Đánh giá mô hình Random Forest
print("\nĐánh giá mô hình Random Forest:")
y_pred_rf = rf_model.predict(X_test)
print(f"Độ chính xác Random Forest: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print(classification_report(y_test, y_pred_rf, target_names=["Chạy bộ", "Đi bộ", "Đứng", "Nằm", "Ngồi"]))

# 9. Lưu mô hình đã huấn luyện
print("\nLưu mô hình đã huấn luyện...")
joblib.dump(dt_model, 'decision_tree_posture_classification_model.pkl')
joblib.dump(rf_model, 'random_forest_posture_classification_model.pkl')
print("Mô hình đã được lưu thành công.")
