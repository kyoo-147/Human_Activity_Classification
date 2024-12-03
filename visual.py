import pandas as pd
import matplotlib.pyplot as plt
import os

# Đường dẫn đến thư mục chứa các tệp CSV
folder_path = 'data_train'
file_names = ['lying.csv', 'walking.csv', 'running.csv', 'standing.csv', 'sitting.csv']
positions = ['Ngủ', 'Đi bộ', 'Chạy bộ', 'Đứng', 'Ngồi']

# Đọc và vẽ dữ liệu từ từng tư thế
plt.figure(figsize=(15, 10))
for i, (file_name, position) in enumerate(zip(file_names, positions), 1):
    # Đọc dữ liệu từ tệp CSV
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    
    # Tạo một subplot cho mỗi tư thế
    plt.subplot(3, 2, i)
    plt.plot(data['accel_x'], label='Gia tốc X', color='blue')
    plt.plot(data['accel_y'], label='Gia tốc Y', color='green')
    plt.plot(data['accel_z'], label='Gia tốc Z', color='red')
    
    # Thiết lập nhãn và tiêu đề
    plt.xlabel('Mẫu (Sample)')
    plt.ylabel('Gia tốc (m/s²)')
    plt.title(f'Tư thế: {position}')
    plt.legend()

# Điều chỉnh khoảng cách giữa các subplots
plt.tight_layout()
plt.show()
