import argparse
import random
import os
from glob import glob
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test Traditional public')

parser.add_argument('--seed', default=100, type=int)

# Test parameters
parser.add_argument('--noise', default='poisson_50', type=str)  # 'gauss_intensity', 'poisson_intensity'
parser.add_argument('--dataset', default='BSD100', type=str)  # BSD100, Kodak, Set12
parser.add_argument('--denoiser', default='nlm_23', type=str)  # 'filter_param', 'nlm_h', 'tv_weight', 'bm3d_sigma'
parser.add_argument('--aver_num', default=10, type=int)

# Transformations
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)

opt = parser.parse_args()


def generate(args):
    # Random Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Directory
    img_dir = os.path.join('../all_datasets/', args.dataset)
    save_dir = './results/{}_({})_({})'.format(args.dataset, args.noise, args.denoiser)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)

    # Images
    img_paths = glob(os.path.join(img_dir, '*.png'))
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_paths]

    # Noise
    noise_type = args.noise.split('_')[0]
    noise_intensity = int(args.noise.split('_')[1]) / 255.

    # Denoising
    noisy_psnr, denoiser_psnr, overlap_psnr = 0, 0, 0
    noisy_ssim, denoiser_ssim, overlap_ssim = 0, 0, 0

    avg_time1, avg_time2 = 0, 0

    for index, clean255 in enumerate(imgs):
        if args.crop:
            clean255 = crop(clean255, patch_size=args.patch_size)

        clean = clean255/255.
        if noise_type == 'gauss':
            noisy = clean + np.random.normal(size=clean.shape) * noise_intensity
        elif noise_type == 'poisson':
            noisy = np.random.poisson(clean * 255. * noise_intensity) / noise_intensity / 255.
        else:
            raise NotImplementedError('Wrong Noise')

        clean, noisy = np.clip(clean, 0., 1.), np.clip(noisy, 0., 1.)
        clean255, noisy255 = np.uint8(clean * 255.), np.uint8(noisy * 255.)

        start1 = time.time()
        # Denoising
        denoised255 = denoise(noisy255, args.denoiser)
        # noisy = noisy.astype(np.float32)
        # denoised = denoise(noisy, args.denoiser)
        # denoised255 = np.clip(denoised, 0., 1.)
        # denoised255 = np.uint8(denoised255 * 255.)
        elapsed1 = time.time() - start1
        avg_time1 += elapsed1 / len(imgs)

        start2 = time.time()
        overlap255 = np.zeros_like(denoised255).astype(np.float64)
        for i in range(args.aver_num):
            overlap255 += denoise(noisy255, args.denoiser) / args.aver_num
        elapsed2 = time.time() - start2
        avg_time2 += elapsed2 / len(imgs)

        # Calculate PSNR
        n_psnr = psnr(clean255, noisy255, data_range=255)
        d_psnr = psnr(clean255, denoised255, data_range=255)
        o_psnr = psnr(clean255, overlap255, data_range=255)

        noisy_psnr += n_psnr / len(imgs)
        denoiser_psnr += d_psnr / len(imgs)
        overlap_psnr += o_psnr / len(imgs)

        # Calculate SSIM
        n_ssim = ssim(clean255, noisy255, data_range=255)
        d_ssim = ssim(clean255, denoised255, data_range=255)
        o_ssim = ssim(clean255, overlap255, data_range=255)

        noisy_ssim += n_ssim / len(imgs)
        denoiser_ssim += d_ssim / len(imgs)
        overlap_ssim += o_ssim / len(imgs)

        print('{}th image | PSNR: noisy:{:.3f}, denoiser:{:.3f}, overlap:{:.3f} | SSIM: noisy:{:.3f}, denoiser:{:.3f}, overlap:{:.3f}'.format(
            index+1, n_psnr, d_psnr, o_psnr, n_ssim, d_ssim, o_ssim))

        # # Save sample images
        # if index <= 3:
        #     cv2.imwrite(os.path.join(save_dir, '{}th_img_clean.png'.format(index+1)), clean255)
        #     cv2.imwrite(os.path.join(save_dir, '{}th_img_noisy.png'.format(index+1)), noisy255)
        #     cv2.imwrite(os.path.join(save_dir, '{}th_img_denoised.png'.format(index+1)), denoised255)

    # Total PSNR, SSIM
    print('{} Average PSNR | noisy:{:.3f}, denoiser:{:.3f}, overlap:{:.3f}'.format(args.dataset, noisy_psnr, denoiser_psnr, overlap_psnr))
    print('{} Average SSIM | noisy:{:.3f}, denoiser:{:.3f}, overlap:{:.3f}'.format(args.dataset, noisy_ssim, denoiser_ssim, overlap_ssim))
    print('Average Time for Denoising: {}'.format(avg_time1))
    print('Average Time for Overlap: {}'.format(avg_time2))


if __name__ == "__main__":
    generate(opt)
