import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv("data_train/lying.csv")

# Function to resample data by averaging every 10 samples
def resample_data(data, window_size=10):
    return data.groupby(data.index // window_size).mean()

# Resample the data to reduce size and smooth the signal
resampled_data = resample_data(data)

# Data Augmentation Functions
def clipping(data, threshold=0.9):
    # Clip the signal to a maximum threshold
    clipped_data = data.copy()
    clipped_data = clipped_data.applymap(lambda x: min(max(x, -threshold), threshold))
    return clipped_data

def gain(data, factor=1.1):
    # Apply a gain factor to the signal
    gained_data = data * factor
    return gained_data

def gain_transition(data, factor_range=(0.9, 1.1)):
    # Apply a gradual gain over the signal
    transition_data = data.copy()
    factors = np.linspace(factor_range[0], factor_range[1], len(data))
    for i, col in enumerate(data.columns):
        transition_data[col] *= factors
    return transition_data

def reverse(data):
    # Reverse the signal
    return data[::-1].reset_index(drop=True)

def shift(data, shift_size=5):
    # Shift the signal by a specified number of samples
    shifted_data = data.copy()
    for col in data.columns:
        shifted_data[col] = np.roll(data[col], shift_size)
    return shifted_data

# Apply the data augmentation techniques
augmented_data_clipping = clipping(resampled_data)
augmented_data_gain = gain(resampled_data)
augmented_data_gain_transition = gain_transition(resampled_data)
augmented_data_reverse = reverse(resampled_data)
augmented_data_shift = shift(resampled_data)

# Plot the original and augmented data for visualization
plt.figure(figsize=(12, 8))
plt.plot(resampled_data['accel_x'].values, label='Original accel_x', alpha=0.8)
plt.plot(augmented_data_clipping['accel_x'].values, label='Clipping', linestyle='--', alpha=0.8)
plt.plot(augmented_data_gain['accel_x'].values, label='Gain', linestyle='--', alpha=0.8)
plt.plot(augmented_data_gain_transition['accel_x'].values, label='Gain Transition', linestyle='--', alpha=0.8)
plt.plot(augmented_data_reverse['accel_x'].values, label='Reverse', linestyle='--', alpha=0.8)
plt.plot(augmented_data_shift['accel_x'].values, label='Shift', linestyle='--', alpha=0.8)
plt.legend()
plt.title("Data Augmentation Techniques Applied on accel_x Axis")
plt.xlabel("Sample Index")
plt.ylabel("Acceleration")
plt.show()

# Save augmented data if needed
augmented_data_clipping.to_csv("augmented_data_clipping.csv", index=False)
augmented_data_gain.to_csv("augmented_data_gain.csv", index=False)
augmented_data_gain_transition.to_csv("augmented_data_gain_transition.csv", index=False)
augmented_data_reverse.to_csv("augmented_data_reverse.csv", index=False)
augmented_data_shift.to_csv("augmented_data_shift.csv", index=False)
