import os
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import bm3d

# Original SEM
noisy_dir = '../all_datasets/SEM1/test'
noisy_paths = glob(os.path.join(noisy_dir, '*.png'))
clean_dir = '../all_datasets/SEM1/gt'
clean_paths = glob(os.path.join(clean_dir, '*.png'))
noisy = cv2.imread(noisy_paths[0], cv2.IMREAD_GRAYSCALE)
clean = cv2.imread(clean_paths[0], cv2.IMREAD_GRAYSCALE)

param = 11
# Mean Filter
mean = cv2.blur(noisy, (param, param))

# Median Filter
median = cv2.medianBlur(noisy, param)

# Gaussian Filter
gaussian = cv2.GaussianBlur(noisy, (param, param), 0)

# Bilateral Filter
bilateral = cv2.bilateralFilter(noisy, param, 75, 75)

# Non-local means (NLM) Filter: h
nlm = cv2.fastNlMeansDenoising(src=noisy, dst=None, h=25,
                               templateWindowSize=7, searchWindowSize=21)

# Total Variation (TV) Filter: weight
tv = denoise_tv_chambolle(noisy, weight=0.15, eps=0.0001, max_num_iter=200)

# BM3D: sigma_psd
bm3d = bm3d.bm3d(noisy, sigma_psd=0.25*255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

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


