import numpy as np
from scipy.interpolate import interp1d

# Sample 3D data with shape (7, 5, 4)
data_3d = np.random.rand(7, 5, 4)  # Replace with your actual 3D data

# Number of frames you want after interpolation
desired_num_frames = 30

# Create a time array for the original data (7 frames)
time_original = np.linspace(0, 1, data_3d.shape[0])

# Create a time array for the desired data (e.g., 30 frames)
time_desired = np.linspace(0, 1, desired_num_frames)

# Initialize an empty 3D array to store the upsampled data
upsampled_data = np.empty((desired_num_frames, data_3d.shape[1], data_3d.shape[2]))

# Loop through each slice along the second and third dimensions and interpolate
for i in range(data_3d.shape[1]):
    for j in range(data_3d.shape[2]):
        slice_data = data_3d[:, i, j]
        interpolator = interp1d(time_original, slice_data, kind='linear', fill_value='extrapolate')
        upsampled_data[:, i, j] = interpolator(time_desired)

print(upsampled_data.shape)