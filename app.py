from flask import Flask, request, jsonify, render_template
import csv
import os
import joblib
import numpy as np
from collections import deque
from datetime import datetime

app = Flask(__name__)

# Tải các mô hình đã huấn luyện
dt_model = joblib.load('models/decision_tree_posture_classification_model.pkl')
rf_model = joblib.load('models/random_forest_posture_classification_model.pkl')

# Cấu hình cửa sổ trượt để lưu dữ liệu gia tốc
window_size = 6
window_ax = deque(maxlen=window_size)
window_ay = deque(maxlen=window_size)
window_az = deque(maxlen=window_size)

# Hàm tạo tên file CSV dựa trên ngày hiện tại
def get_csv_file():
    date_str = datetime.now().strftime('%Y-%m-%d')
    return f'logs_sensor/sensor_data_{date_str}.csv'

# Kiểm tra nếu file CSV chưa tồn tại cho ngày hiện tại và ghi tiêu đề nếu cần
def initialize_csv_file():
    csv_file = get_csv_file()
    if not os.path.exists(csv_file):
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temperature"])
    return csv_file

# Hàm trích xuất đặc trưng từ cửa sổ trượt
def extract_features(window_ax, window_ay, window_az):
    mean_ax = np.mean(window_ax)
    mean_ay = np.mean(window_ay)
    mean_az = np.mean(window_az)
    sd_ax = np.std(window_ax)
    sd_ay = np.std(window_ay)
    sd_az = np.std(window_az)
    return [mean_ax, mean_ay, mean_az, sd_ax, sd_ay, sd_az]

@app.route('/coordinates', methods=['POST'])
def receive_data():
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    required_keys = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temperature"]
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Incomplete data"}), 400

    try:
        print("Data received:", data)
        csv_file = initialize_csv_file()  # Lấy file CSV cho ngày hiện tại

        # Ghi dữ liệu vào file CSV với timestamp hiện tại
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Thời gian hiện tại
                data.get("accel_x"),
                data.get("accel_y"),
                data.get("accel_z"),
                data.get("gyro_x"),
                data.get("gyro_y"),
                data.get("gyro_z"),
                data.get("temperature")
            ])

        # Lưu dữ liệu gia tốc vào cửa sổ trượt
        window_ax.append(float(data.get("accel_x")))
        window_ay.append(float(data.get("accel_y")))
        window_az.append(float(data.get("accel_z")))

        # Kiểm tra số lượng mẫu trong cửa sổ trượt
        if len(window_ax) < window_size:
            return jsonify({
                "message": f"Data received and saved successfully. Waiting for {window_size - len(window_ax)} more samples"
            }), 200

        # Khi cửa sổ trượt đầy, thực hiện dự đoán
        features = extract_features(window_ax, window_ay, window_az)
        dt_prediction = dt_model.predict([features])[0]
        rf_prediction = rf_model.predict([features])[0]

        return jsonify({
            "message": "Data received and saved successfully",
            "decision_tree_prediction": dt_prediction,
            "random_forest_prediction": rf_prediction
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/latest_data', methods=['GET'])
def get_latest_data():
    try:
        csv_file = get_csv_file()
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Bỏ qua tiêu đề
            last_row = list(reader)[-1]
            if last_row:
                data = {
                    "timestamp": last_row[0],
                    "accel_x": float(last_row[1]),
                    "accel_y": float(last_row[2]),
                    "accel_z": float(last_row[3]),
                    "gyro_x": float(last_row[4]),
                    "gyro_y": float(last_row[5]),
                    "gyro_z": float(last_row[6]),
                    "temperature": float(last_row[7])
                }
                return jsonify(data), 200
            else:
                return jsonify({"error": "No data available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)