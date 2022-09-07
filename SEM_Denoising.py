import argparse
import random
import os
from glob import glob
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test Traditional SEM')

parser.add_argument('--seed', default=100, type=int)

# Test parameters
parser.add_argument('--dataset', default='SEM5', type=str)  # SEM1, SEM3, SEM5
parser.add_argument('--denoiser', default='tv_0.9', type=str)  # 'filter_param', 'nlm_h', 'tv_weight', 'bm3d_sigma'

# Transformations
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=256)

opt = parser.parse_args()


def generate(args):
    # Random Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Directory
    img_dir = os.path.join('../all_datasets/', args.dataset)
    save_dir = './results/{}_({})'.format(args.dataset, args.denoiser)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Image Directory
    clean_dir = os.path.join(img_dir, 'test_gt')
    clean_paths = sorted(glob(os.path.join(clean_dir, '*.png')))
    noisy_dir = os.path.join(img_dir, 'test')
    noisy_paths = sorted(glob(os.path.join(noisy_dir, '*.png')))

    # Denoising
    noisy_psnr, denoiser_psnr = 0, 0
    noisy_ssim, denoiser_ssim = 0, 0

    avg_time = 0

    dataset_size = len(clean_paths)
    for index in range(dataset_size):
        clean255 = cv2.imread(clean_paths[index], cv2.IMREAD_GRAYSCALE)
        noisy255 = cv2.imread(noisy_paths[index], cv2.IMREAD_GRAYSCALE)

        if args.dataset == 'SEM3':
            clean255, noisy255 = crop(clean255, patch_size=256), crop(noisy255, patch_size=256)

        start = time.time()
        # Denoising
        denoised255 = denoise(noisy255, args.denoiser)
        elapsed = time.time() - start
        avg_time += elapsed / dataset_size

        # Calculate PSNR
        n_psnr = psnr(clean255, noisy255, data_range=255)
        d_psnr = psnr(clean255, denoised255, data_range=255)

        noisy_psnr += n_psnr / dataset_size
        denoiser_psnr += d_psnr / dataset_size

        # Calculate SSIM
        n_ssim = ssim(clean255, noisy255, data_range=255)
        d_ssim = ssim(clean255, denoised255, data_range=255)

        noisy_ssim += n_ssim / dataset_size
        denoiser_ssim += d_ssim / dataset_size

        print('{}th image | PSNR: noisy:{:.3f}, denoiser:{:.3f} |SSIM: noisy:{:.3f}, denoiser:{:.3f}'.format(
            index+1, n_psnr, d_psnr, n_psnr, d_psnr))

        # Save sample images
        if index <= 100000:
            cv2.imwrite(os.path.join(save_dir, '{}th_img_clean.png'.format(index+1)), clean255)
            cv2.imwrite(os.path.join(save_dir, '{}th_img_noisy.png'.format(index+1)), noisy255)
            cv2.imwrite(os.path.join(save_dir, '{}th_img_denoised.png'.format(index+1)), denoised255)

    # Total PSNR, SSIM
    print('{} Average PSNR | noisy:{:.3f}, denoiser:{:.3f}'.format(args.dataset, noisy_psnr, denoiser_psnr))
    print('{} Average SSIM | noisy:{:.3f}, denoiser:{:.3f}'.format(args.dataset, noisy_ssim, denoiser_ssim))
    print('Average Time Elapsed: {}'.format(avg_time))

if __name__ == "__main__":
    generate(opt)
