import os
import sys

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)

# init 
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
sys.stdout.flush()
set_seed(10)
debug = False
if debug:
    save_and_sample_every = 2
    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 200
else:
    save_and_sample_every = 1000
    sampling_timesteps = 5
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 30000

original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = True
    input_condition = True 
    input_condition_mask = True

if condition:
    if input_condition:
        folder = ["/home/shenss/python/dataset/VOC2012_ORI/train/gt",
                "/home/shenss/python/dataset/VOC2012_ORI/train/fs",
                "/home/shenss/python/dataset/VOC2012_ORI/train/wsobel",
                "/home/shenss/python/attacks/images/original",
                "/home/shenss/python/attacks/images/original",
                "/home/shenss/python/attacks/images/original"
                ]
    else:
        folder = ["/home/sss/python/dataset/Celebrity Face Image Dataset/train/gt",
                "/home/sss/python/dataset/Celebrity Face Image Dataset/train/input_ht_rgb",
                "/home/sss/python/dataset/Celebrity Face Image Dataset/test/gt",
                "/home/sss/python/dataset/Celebrity Face Image Dataset/test/input_ht_rgb"]
    train_batch_size = 32
    num_samples = 1
    sum_scale = 1
    image_size = 32
else:
    folder = '/home/sss/python/dataset/CelebA/img_align_celeba'
    train_batch_size = 32
    num_samples = 25
    sum_scale = 1
    image_size = 32

if original_ddim_ddpm:
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        sampling_timesteps=sampling_timesteps_original_ddim_ddpm,
        loss_type='l1',            # L1 or L2
    )
else:
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        share_encoder=0, #1 0 -1，分别对应共享编码器、两个独立的 U-Net 和一个独立的 U-Net。
        condition=condition,
        input_condition=input_condition
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=sampling_timesteps,
        objective='pred_res_noise', # pred_res_noise, pred_res, pred_noise
        loss_type='l1',            # L1 or L2 or huber(SmoothL1)
        condition=condition,
        sum_scale = sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask
    )

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=8e-5,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation = False,
    halftone = 'fs',  # fs, evcs, gmevcs
    gaussian_filter = True,
    get_sobel = 'wsobel',  # None, sobel, canny, wsobel
)

if not trainer.accelerator.is_local_main_process:
    pass
else:
    epoch = 80
    trainer.load(epoch, load_name='DiffSo_GS3_80K_TS5_Model')
    trainer.set_results_folder(f'/home/shenss/python/attacks/images/ht_ori_std_new')
    trainer.test(last=True)

# trainer.set_results_folder('./results/test_sample')
# trainer.test(sample=True)