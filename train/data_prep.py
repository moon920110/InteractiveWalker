import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File('../data_collect/test/3.hdf5', 'r') as file:
    tactile = np.clip(file['pressure'][:], 0, 1000)
    imu = file['imu'][:]

    print('hi')


def moving_avarage_smoothing(X,k):
	S = np.zeros(X.shape[0])
	for t in range(X.shape[0]):
		if t < k:
			S[t] = np.mean(X[:t+1])
		else:
			S[t] = np.sum(X[t-k:t])/k
	return S

imu = moving_avarage_smoothing(imu, 5)

index = 0
def update_fig(event):
    global index
    if event.key == 'right':
        index += 1
    elif event.key == 'left':
        index -= 1

    # 인덱스 유효성 검사
    index = max(0, min(index, len(tactile) - 1))

    # 이미지 데이터 업데이트
    img.set_data(tactile[index])
    text.set_text(imu[index])
    ax.set_title(f"Image {index + 1}/{len(tactile)}")
    fig.canvas.draw_idle()

fig, ax = plt.subplots()
img = ax.imshow(tactile[index], cmap='gray')
ax.set_title(f"Image {index+1}/{len(tactile)}")
# ax.colorbar()

text = ax.text(12, 2, str(imu[index]), color='red', ha='center', va='center', fontsize=12)

plt.axis('off')
fig.canvas.mpl_connect('key_press_event', update_fig)
plt.show()