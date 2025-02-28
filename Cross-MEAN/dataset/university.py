import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
from albumentations.core.transforms_interface import ImageOnlyTransform
import imgaug.augmenters as iaa


def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files

    return data


class U1652DatasetTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]

            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                self.pairs.append((idx, label, query_img, "{}/{}".format(gallery_path, g)))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class U1652DatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):

            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                  file))

                self.sample_ids.append(sample_id)

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if self.mode == "sat":

        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)

        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)

        #    img = np.concatenate([img_0_90, img_180_270], axis=0)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label , img_path

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)


# ********************************************** apply multi-weather setting for U1652 **********************************************************

class ImgAugTransform(ImageOnlyTransform):
    def __init__(self, aug, always_apply=False, p=1.0):
        super(ImgAugTransform, self).__init__(always_apply, p)
        self.aug = aug

    def apply(self, img, **params):
        return self.aug(image=img)


# 自定义云层变换
class CustomCloudLayer(ImgAugTransform):
    def __init__(self, intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2,
                 alpha_min=1.0, alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2,
                 sparsity=0.9, density_multiplier=0.5, seed=None, always_apply=False, p=1.0):
        aug = iaa.CloudLayer(
            intensity_mean=intensity_mean,
            intensity_freq_exponent=intensity_freq_exponent,
            intensity_coarse_scale=intensity_coarse_scale,
            alpha_min=alpha_min,
            alpha_multiplier=alpha_multiplier,
            alpha_size_px_max=alpha_size_px_max,
            alpha_freq_exponent=alpha_freq_exponent,
            sparsity=sparsity,
            density_multiplier=density_multiplier,
            seed=seed
        )
        super(CustomCloudLayer, self).__init__(aug, always_apply, p)


# 自定义雨变换
class CustomRain(ImgAugTransform):
    def __init__(self, drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=None, always_apply=False, p=1.0):
        aug = iaa.Rain(
            drop_size=drop_size,
            speed=speed,
            seed=seed
        )
        super(CustomRain, self).__init__(aug, always_apply, p)


# 自定义雪花变换
class CustomSnowflakes(ImgAugTransform):
    def __init__(self, flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=None, always_apply=False, p=1.0):
        aug = iaa.Snowflakes(
            flake_size=flake_size,
            speed=speed,
            seed=seed
        )
        super(CustomSnowflakes, self).__init__(aug, always_apply, p)


iaa_weather_list = [

    # 0. Normal
    A.NoOp(),
    # 1. Fog
    A.Compose([
        CustomCloudLayer()
    ]),
    # 2. Rain
    A.Compose([
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
    ]),
    # 3. Snow
    A.Compose([
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
    ]),
    # 4. Fog+Rain
    A.Compose([
        CustomCloudLayer(),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)
    ]),
    # 5. Fog+Snow
    A.Compose([
        CustomCloudLayer(),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)
    ]),
    # 6. Rain+Snow
    A.Compose([
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
    ]),
    # 7. Dark
    A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(9, 11), p=0.5),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.5),
        ]),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.15), contrast_limit=(0.2, 0.2), p=1)
    ]),
    # 8. Over-exposure
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0, 0.3), contrast_limit=(1.3, 1.6), p=1)
    ]),
    # 9. Wind
    A.Compose([
        A.MotionBlur(blur_limit=15, p=1)
    ])
]


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    # weather_id = 0
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                # Multi-weather U1652 settings.
                                # iaa_weather_list[weather_id],

                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                      # # Multi-weather U1652 settings.
                                      # A.OneOf(iaa_weather_list, p=1.0),

                                      A.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.3,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size[0]),
                                                          max_width=int(0.2 * img_size[0]),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size[0]),
                                                          min_width=int(0.1 * img_size[0]),
                                                          p=1.0),
                                      ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                        # Multi-weather U1652 settings.
                                        # A.OneOf(iaa_weather_list, p=1.0),

                                        A.ColorJitter(brightness=0.15, contrast=0.7, saturation=0.3, hue=0.3,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms
