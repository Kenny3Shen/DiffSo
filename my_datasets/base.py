import os
import random
from pathlib import Path
import pywt
import Augmentor
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=["jpg", "jpeg", "png", "tiff"],
        augment_flip=False,
        convert_image_to=None,
        condition=0,
        equalizeHist=False,
        crop_patch=True,
        halftone=None,
        gaussian_filter=False,
        get_sobel=None,
        sample=False,
    ):
        super().__init__()
        self.equalizeHist = equalizeHist
        self.exts = exts
        self.augment_flip = augment_flip
        self.condition = condition
        self.crop_patch = crop_patch
        self.halftone = halftone
        self.gaussian_filter = gaussian_filter
        self.get_sobel = get_sobel
        self.sample = sample
        if condition == 1:
            # condition
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
        elif condition == 0:
            # generation
            self.paths = self.load_flist(folder)
        elif condition == 2:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            self.input_condition = self.load_flist(folder[2])

        self.image_size = image_size
        self.convert_image_to = convert_image_to

    def __len__(self):
        if self.condition:
            return len(self.input)
        else:
            return len(self.paths)

    def __getitem__(self, index):
        if self.condition == 1:
            # condition
            img0 = Image.open(self.gt[index])
            img1 = Image.open(self.input[index])
            w, h = img0.size
            img0 = (
                convert_image_to_fn(self.convert_image_to, img0)
                if self.convert_image_to
                else img0
            )
            img1 = (
                convert_image_to_fn(self.convert_image_to, img1)
                if self.convert_image_to
                else img1
            )

            img0, img1 = self.pad_img([img0, img1], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1 = self.get_patch([img0, img1], self.image_size)

            img1 = self.ht(img1) if self.halftone else img1

            img1 = self.cv2gaussian_filter(img1) if self.gaussian_filter else img1

            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            images = [[img0, img1]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            if not self.crop_patch:
                # p.resize(1, self.image_size, self.image_size)
                p.crop_by_size(1, self.image_size, self.image_size, centre=False)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            img0 = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)

            return [self.to_tensor(img0), self.to_tensor(img1)]
        elif self.condition == 0:
            # generation
            path = self.paths[index]
            img = Image.open(path)
            img = (
                convert_image_to_fn(self.convert_image_to, img)
                if self.convert_image_to
                else img
            )

            img = self.pad_img([img], self.image_size)[0]

            if self.crop_patch and not self.sample:
                img = self.get_patch([img], self.image_size)[0]

            img = self.cv2equalizeHist(img) if self.equalizeHist else img

            images = [[img]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            if not self.crop_patch:
                p.resize(1, self.image_size, self.image_size)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            img = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)

            return self.to_tensor(img)
        elif self.condition == 2:
            # condition
            img0 = Image.open(self.gt[index])
            img1 = Image.open(self.input[index])
            img2 = Image.open(self.input_condition[index])
            img0 = (
                convert_image_to_fn(self.convert_image_to, img0)
                if self.convert_image_to
                else img0
            )
            img1 = (
                convert_image_to_fn(self.convert_image_to, img1)
                if self.convert_image_to
                else img1
            )
            img2 = (
                convert_image_to_fn(self.convert_image_to, img2)
                if self.convert_image_to
                else img2
            )

            img0, img1, img2 = self.pad_img([img0, img1, img2], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1, img2 = self.get_patch([img0, img1, img2], self.image_size)

            # halftone
            img1 = self.ht(img1) if self.halftone else img1

            # gaussian filter
            img1 = self.cv2gaussian_filter(img1) if self.gaussian_filter else img1

            # equalize histogram
            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            # weighted sobel
            img2 = self.cv2edge(img2) if self.get_sobel else img2
            images = [[img0, img1, img2]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            if not self.crop_patch:
                # p.resize(1, self.image_size, self.image_size)
                p.crop_by_size(1, self.image_size, self.image_size, centre=False)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            img0 = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(augmented_images[0][2], cv2.COLOR_BGR2RGB)

            return [self.to_tensor(img0), self.to_tensor(img1), self.to_tensor(img2)]

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                # file_list = [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]
                # print(file_list)  # 打印列表
                # return file_list

                p = []
                for root, dirs, files in os.walk(Path(f"{flist}")):
                    for file in files:
                        if file.endswith(tuple(self.exts)):
                            p.append(Path(root) / file)  # 使用 Path 类构建路径
                # print(p)
                return p

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding="utf-8")
                except:
                    return [flist]

        return []

    def ht(self, img):
        if self.halftone == "fs":
            (b, g, r) = cv2.split(img)
            b = np.array(Image.fromarray(b).convert("1").convert("L"))
            g = np.array(Image.fromarray(g).convert("1").convert("L"))
            r = np.array(Image.fromarray(r).convert("1").convert("L"))
            img = cv2.merge((b, g, r))
        elif self.halftone == "evcs":
            (b, g, r) = cv2.split(img_bgr)
            b, g, r = (
                (b / 4).astype(np.uint8),
                (g / 4).astype(np.uint8),
                (r / 4).astype(np.uint8),
            )
            b = np.array(Image.fromarray(b).convert("1").convert("L"))
            g = np.array(Image.fromarray(g).convert("1").convert("L"))
            r = np.array(Image.fromarray(r).convert("1").convert("L"))
            img = cv2.merge((b, g, r))
        elif self.halftone == "gmevcs":
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 2] = (img_hsv[:, :, 2] / 4).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            (b, g, r) = cv2.split(img_bgr)
            b = np.array(Image.fromarray(b).convert("1").convert("L"))
            g = np.array(Image.fromarray(g).convert("1").convert("L"))
            r = np.array(Image.fromarray(r).convert("1").convert("L"))
            img = cv2.merge((b, g, r))
        return img

    def remove_high_freq(self, img, wavelet="haar", level=1):
        # 分离RGB通道
        b, g, r = cv2.split(img)
        channels = []

        # 对每个通道进行DWT处理
        for c in [b, g, r]:
            coeffs = pywt.wavedec2(c, wavelet, level=level)
            # 将高频系数置零
            coeffs_H = list(coeffs)
            coeffs_H[1:] = [
                (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD))
                for cH, cV, cD in coeffs_H[1:]
            ]

            # 重构图像
            reconstructed = pywt.waverec2(coeffs_H, wavelet)
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
            channels.append(reconstructed)

        # 合并通道
        img_reconstructed = cv2.merge(channels)
        return img_reconstructed

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def cv2gaussian_filter(self, img):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def cv2edge(self, img):
        if self.get_sobel == "wsobel":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
            img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
            img_edge = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)
        elif self.get_sobel == "sobel":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_edge = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)
        elif self.get_sobel == "canny":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_edge = cv2.Canny(img_gray, 100, 200)
        elif self.get_sobel == "zero":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_edge = np.zeros_like(img_gray)
        return img_edge

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir + "_" + os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h - patch_size)
        cc = random.randint(0, w - patch_size)
        for img in image_list:
            image_list[i] = img[rr : rr + patch_size, cc : cc + patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size - h
                h = patch_size
            if w < patch_size:
                right = patch_size - w
                w = patch_size
            bottom = (
                bottom
                + (h // block_size) * block_size
                + (block_size if h % block_size != 0 else 0)
                - h
            )
            right = (
                right
                + (w // block_size) * block_size
                + (block_size if w % block_size != 0 else 0)
                - w
            )
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size - h
            h = patch_size
        if w < patch_size:
            right = patch_size - w
            w = patch_size
        bottom = (
            bottom
            + (h // block_size) * block_size
            + (block_size if h % block_size != 0 else 0)
            - h
        )
        right = (
            right
            + (w // block_size) * block_size
            + (block_size if w % block_size != 0 else 0)
            - w
        )
        return [bottom, right]
