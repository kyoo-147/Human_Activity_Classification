import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Đường dẫn đến thư mục chứa các tệp CSV
folder_path = 'data_train'
file_names = ['lying.csv', 'walking.csv', 'running.csv', 'standing.csv', 'sitting.csv']
positions = ['Ngủ', 'Đi bộ', 'Chạy bộ', 'Đứng', 'Ngồi']

# Hàm để đọc và xử lý dữ liệu
def read_and_process_data(file_path, accel_threshold=20):
    # Đọc dữ liệu từ tệp CSV
    data = pd.read_csv(file_path)
    
    # Tính độ lớn gia tốc tổng hợp từ các cột `accel_x`, `accel_y`, `accel_z`
    data['accel_magnitude'] = np.sqrt(data['accel_x']**2 + data['accel_y']**2 + data['accel_z']**2)
    
    # Loại bỏ tín hiệu lỗi dựa trên ngưỡng `accel_threshold`
    valid_data = data[data['accel_magnitude'] < accel_threshold]
    invalid_data = data[data['accel_magnitude'] >= accel_threshold]
    
    # Trả về số lượng mẫu hợp lệ và lỗi
    return len(valid_data), len(invalid_data)

# Danh sách chứa số lượng mẫu hợp lệ và lỗi cho tất cả tư thế
valid_counts = []
invalid_counts = []

# Lặp qua các tệp CSV và tính toán số lượng mẫu hợp lệ và lỗi
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    valid_count, invalid_count = read_and_process_data(file_path)
    valid_counts.append(valid_count)
    invalid_counts.append(invalid_count)

# Tính tổng số lượng mẫu hợp lệ và lỗi
total_valid = sum(valid_counts)
total_invalid = sum(invalid_counts)

# Vẽ biểu đồ cột cho tất cả dữ liệu
plt.figure(figsize=(8, 5))
bars = plt.bar(['Dữ liệu hợp lệ', 'Dữ liệu lỗi'], [total_valid, total_invalid], color=['blue', 'red'])

# Nhãn và tiêu đề cho biểu đồ
plt.ylabel('Số lượng mẫu')
plt.title('Tổng số lượng mẫu hợp lệ và lỗi')

# Thêm giá trị số vào các cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.show()
