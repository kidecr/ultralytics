# Ultralytics YOLO 🚀, AGPL-3.0 license
import os
import contextlib
import math
import random
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import psutil
import glob
import torch
import torchvision
from PIL import Image

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable
from ultralytics.utils.ops import resample_segments
from .augment import Compose, Format, Instances, LetterBox, classify_augmentations, classify_transforms, v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, IMG_FORMATS, get_hash, img2label_paths, verify_image, verify_image_label, rgb2ir_path, rgb2ir_paths

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

class RGBIRDataset(YOLODataset):
    """
    用于读取RGB和IR图片
    
    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).
        data_mode (str): select which type to load, 'RGBT' is read both rgb and ir image and labels, 'RGB' 'IR' 'T' is read single type image, 'RGBT2' is read rgbt type image with npy type 

    Attributes:
        im_files (list): List of image file paths. 为RGB图像路径
        labels (list): List of label data dictionaries. 重新定义为list({rgb_infos, [rgb_bbox, ir_bbox]}), shape初始化为rgb图片大小, rgb和ir应该相同大小
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images. 重新定义为list([rgb_im, ir_im])
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """
    def __init__(self, *args, data_mode="RGBT", data=None, task="detect", **kwargs):
        """Initializes the RGBIRDataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        self.data_mode = data_mode if data_mode in ("RGB", "T", "IR", "RGBT", "RGBT2") else "RGBT" 
        super().__init__(*args, data=data, task=task, **kwargs)
        
    def get_img_files(self, img_path):
        """Read image files. default read RGB path"""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    if self.data_mode in ("T", "IR"):
                        f += glob.glob(str(p / "**"/ "IR" / "*.*"), recursive=True) # 读IR部分，之后会转成RGB路径
                    elif self.data_mode in ("RGBT2"):
                        f += glob.glob(str(p / "**"/ "RGBT" / "*.*"), recursive=True) # 读RGBT部分，之后会转成RGB路径
                    else: 
                        f += glob.glob(str(p / "**"/ "RGB" / "*.*"), recursive=True) # 读RGB部分，之后会转成IR路径
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        return im_files
    
    def load_image(self, i, rect_mode=True):
        """Loads 1 or 2(rgb,ir) image from dataset index 'i', returns (rgb or ir or [rgb,ir])."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    # im = cv2.imread(f)  # BGR
                    # 分别读取RGB和IR图片
                    im = self.read_image(f)
            else:  # read image
                # im = cv2.imread(f)  # BGR
                # 分别读取RGB和IR图片
                im = self.read_image(f)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")


            # 对两张图片分别进行resize
            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == "disk" else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {cache})"
            pbar.close()
            
    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        im = None
        if not f.exists():
            # 分别读取RGB和IR图片
            im = self.read_image(f)
            # 读取后同时保存两份图像
            np.save(f.as_posix(), im, allow_pickle=False)
            
    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = self.read_image(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += (im.nbytes * ratio**2)
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(
                f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                f'with {int(safety_margin * 100)}% safety margin but only '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache
    
    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.
        读取rgb和ir的label, 以rgb的label为主, ir只使用bbox
        
        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            # 分别读取rgb和ir的label
            if self.data_mode in ("RGBT"):
                rgb_results = pool.imap(
                    func=verify_image_label,
                    iterable=zip(
                        self.im_files,
                        self.label_files,
                        repeat(self.prefix),
                        repeat(self.use_keypoints),
                        repeat(len(self.data["names"])),
                        repeat(nkpt),
                        repeat(ndim),
                    ),
                )
                
                ir_results = pool.imap(
                    func=verify_image_label,
                    iterable=zip(
                        rgb2ir_paths(self.im_files),
                        rgb2ir_paths(self.label_files),
                        repeat(self.prefix),
                        repeat(self.use_keypoints),
                        repeat(len(self.data["names"])),
                        repeat(nkpt),
                        repeat(ndim),
                    ),
                )
                # 组织label信息
                ir_lbs = [ir_lb for _, ir_lb, _, _, _, _, _, _, _, _ in ir_results]
                pbar = TQDM(rgb_results, desc=desc, total=total)
                i = 0
                for rgb_im_file, rgb_lb, rgb_shape, rgb_segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                    nm += nm_f
                    nf += nf_f
                    ne += ne_f
                    nc += nc_f
                    if rgb_im_file:
                        x["labels"].append(
                            dict(
                                im_file=rgb_im_file,
                                shape=rgb_shape,
                                cls=np.concatenate((rgb_lb[:, 0:1], ir_lbs[i][:, 0:1]), axis=1).reshape(-1, 1), # n, 1
                                bboxes=np.concatenate((rgb_lb[:, 1:], ir_lbs[i][:, 1:]), axis=1).reshape(-1, 4),  # n, 4
                                segments=rgb_segments,
                                keypoints=keypoint,
                                normalized=True,
                                bbox_format="xywh",
                            )
                        )
                    if msg:
                        msgs.append(msg)
                    # pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
                    pbar.desc = f"{desc} {nf} images, {total} backgrounds, {nc} corrupt"
                    i = i + 1
                pbar.close()
            else:
                results = pool.imap(
                    func=verify_image_label,
                    iterable=zip(
                        self.im_files,
                        self.label_files,
                        repeat(self.prefix),
                        repeat(self.use_keypoints),
                        repeat(len(self.data["names"])),
                        repeat(nkpt),
                        repeat(ndim),
                    ),
                )
                
                pbar = TQDM(rgb_results, desc=desc, total=total)
                for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                    nm += nm_f
                    nf += nf_f
                    ne += ne_f
                    nc += nc_f
                    if im_file:
                        x["labels"].append(
                            dict(
                                im_file=im_file,
                                shape=shape,
                                cls=lb[:, 0:1], # n, 1
                                bboxes=lb[:, 1:],  # n, 4
                                segments=segments,
                                keypoints=keypoint,
                                normalized=True,
                                bbox_format="xywh",
                            )
                        )
                    if msg:
                        msgs.append(msg)
                    # pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
                    pbar.desc = f"{desc} {nf} images, {total} backgrounds, {nc} corrupt"
                pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x
    
    def read_image(self, path):
        # 分别读取RGB和IR图片
        im = None
        if self.data_mode in ("RGB"):
            im = cv2.imread(path) # BGR
        elif self.data_mode in ("T", "IR"):
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis] # T
        elif self.data_mode in ("RGBT2"):
            im = np.load(path) # load BGRT type image in npy format
        else:   # RGBT
            rgb_im = cv2.imread(path) # BGR 
            ir_im = cv2.imread(rgb2ir_path(path), cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis] # T
            # im = [rgb_im, ir_im]
            im = np.concatenate((rgb_im, ir_im), axis=2) 
        return im
    
# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    Extends torchvision ImageFolder to support YOLO classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings. It includes attributes like `imgsz` (image size), `fraction` (fraction
                of data to use), `scale`, `fliplr`, `flipud`, `cache` (disk or RAM caching for faster training),
                `auto_augment`, `hsv_h`, `hsv_s`, `hsv_v`, and `crop_fraction`.
            augment (bool, optional): Whether to apply augmentations to the dataset. Default is False.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification and
                debugging. Default is an empty string.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or args.cache == "ram"  # cache images into RAM
        self.cache_disk = args.cache == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in (-1, 0):
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        # Run scan if *.cache retrieval failed
        nf, nc, msgs, samples, x = 0, 0, [], [], {}
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(results, desc=desc, total=len(self.samples))
            for sample, nf_f, nc_f, msg in pbar:
                if nf_f:
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
            pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        x["hash"] = get_hash([x[0] for x in self.samples])
        x["results"] = nf, nc, len(samples), samples
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return samples


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()
