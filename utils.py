import cv2
import numpy as np
from skimage.restoration import estimate_sigma, denoise_nl_means, denoise_tv_chambolle
import bm3d


def crop(img, patch_size):
    if img.ndim == 2:
        h, w = img.shape
        return img[h//2-patch_size//2:h//2+patch_size//2, w//2-patch_size//2:w//2+patch_size//2]
    elif img.ndim == 3:
        c, h, w = img.shape
        return img[:, h//2-patch_size//2:h//2+patch_size//2, w//2-patch_size//2:w//2+patch_size//2]
    else:
        raise NotImplementedError('Wrong image dim')


def denoise(noisy, denoiser):
    method, param = denoiser.split('_')
    if method == 'mean':
        denoised = cv2.blur(noisy, (int(param), int(param)))
    elif method == 'median':
        denoised = cv2.medianBlur(noisy, int(param))
    elif method == 'gaussian':
        denoised = cv2.GaussianBlur(noisy, (int(param), int(param)), 0)
    elif method == 'bilateral':
        denoised = cv2.bilateralFilter(noisy, int(param), 75, 75)
    elif method == 'nlm':
        denoised = cv2.fastNlMeansDenoising(noisy, None, h=float(param), templateWindowSize=7, searchWindowSize=21)
    elif method == 'tv':
        denoised = denoise_tv_chambolle(noisy, weight=float(param), eps=0.0001, max_num_iter=1000)
        denoised = np.uint8(denoised * 255.)
    elif method == 'bm3d':
        denoised = bm3d.bm3d(noisy, sigma_psd=param, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    else:
        raise NotImplementedError('Wrong Denoiser')
    return denoised
