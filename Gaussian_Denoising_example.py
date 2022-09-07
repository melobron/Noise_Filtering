import os
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.restoration import estimate_sigma, denoise_nl_means, denoise_tv_chambolle
import bm3d

# Gaussian Noise
img_path = '../all_datasets/BSD68/3096.jpg'
clean = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.
# clean = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.
std = 0.05
noisy = clean + std * np.random.normal(size=clean.shape)
clean, noisy = np.clip(clean, 0., 1.), np.clip(noisy, 0., 1.)
clean, noisy = np.uint8(clean*255.), np.uint8(noisy*255.)

param = 5
# Mean Filter
mean = cv2.blur(noisy, (param, param))

# Median Filter
median = cv2.medianBlur(noisy, param)

# Gaussian Filter
gaussian = cv2.GaussianBlur(noisy, (param, param), 0)

# Bilateral Filter
bilateral = cv2.bilateralFilter(noisy, param, 75, 75)

# Non-local means (NLM) Filter
nlm = cv2.fastNlMeansDenoising(noisy, None, h=9, templateWindowSize=7, searchWindowSize=21)
# sigma_est = estimate_sigma(noisy)
# nlm = denoise_nl_means(noisy, h=0.6*sigma_est, sigma=sigma_est, fast_mode=False,
#                        patch_size=5, patch_distance=6)

# Total Variation (TV) Filter
tv = denoise_tv_chambolle(noisy, weight=0.2, eps=0.0001, max_num_iter=1000)

# BM3D
bm3d = bm3d.bm3d(noisy, sigma_psd=0.05*255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

# Visualization
fig = plt.figure(figsize=(15, 10))
rows, cols = 3, 4

imgs = [noisy, clean, mean, median, gaussian, bilateral, nlm, tv, bm3d]
titles = ['noisy', 'clean', 'mean', 'median', 'gaussian', 'bilateral', 'nlm', 'tv', 'bm3d']
for i in range(9):
    fig.add_subplot(rows, cols, i+1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
