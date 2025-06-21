import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from PIL import Image
import random
import os
import os.path as osp
import albumentations as albu
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# Define the model (DCSwin small)
class DCSwinSmall(nn.Module):
    def __init__(self, num_classes):
        super(DCSwinSmall, self).__init__()
        # Placeholder for the actual DCSwin model architecture
        # In practice, you would import this from geoseg.models.DCSwin
        self.num_classes = num_classes
        self.dummy_conv = nn.Conv2d(3, num_classes, kernel_size=1)  # Dummy layer for demonstration

    def forward(self, x):
        # Dummy forward pass; replace with actual DCSwin implementation
        return self.dummy_conv(x)

# Define loss functions
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, smooth_factor=0.0, ignore_index=-100):
        super(SoftCrossEntropyLoss, self).__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=1)
        return torch.nn.functional.nll_loss(
            log_prob, target, ignore_index=self.ignore_index, reduction='mean')

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        target_one_hot = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1)
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            input = input * mask.unsqueeze(1)
            target_one_hot = target_one_hot * mask.unsqueeze(1)
        intersection = (input * target_one_hot).sum(dim=(2, 3))
        union = input.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class JointLoss(nn.Module):
    def __init__(self, ce_loss, dice_loss, ce_weight=1.0, dice_weight=1.0):
        super(JointLoss, self).__init__()
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, input, target):
        ce = self.ce_loss(input, target)
        dice = self.dice_loss(input, target)
        return self.ce_weight * ce + self.dice_weight * dice

# Data augmentation classes
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomScale:
    def __init__(self, scale_list, mode='value'):
        self.scale_list = scale_list
        self.mode = mode

    def __call__(self, img, mask):
        scale = random.choice(self.scale_list)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.BILINEAR)
        mask = mask.resize((int(mask.size[0] * scale), int(img.size[1] * scale)), Image.NEAREST)
        return img, mask

class SmartCropV1:
    def __init__(self, crop_size, max_ratio, ignore_index, nopad):
        self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.nopad = nopad

    def __call__(self, img, mask):
        img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)
        return img, mask

# Define classes and palette
CLASSES = ("farmland", "city", "village", "water", "forest", "road", "others", "background")
PALETTE = [[204, 102, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255], [85, 167, 0], [0, 255, 255], [153, 102, 153],
           [255, 255, 255]]

# Image sizes
# Image sizes
ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)

# Training hyperparameters
max_epoch = 5
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 4
lr = 1e-3
weight_decay = 2.5e-4
backbone_lr = 1e-4
backbone_weight_decay = 2.5e-4
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = 'auto'
resume_ckpt_path = None

# Define the network
net = DCSwinSmall(num_classes=num_classes)  # Replace with actual dcswin_small if available

# Define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False

# Define data augmentation
def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def train_aug(img, mask):
    crop_aug = Compose([
        RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)
    ])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def val_aug(img, mask):
    crop_aug = Compose([
        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)
    ])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

# Dataset class for RGB images only
class PenggDataset(torch.utils.data.Dataset):
    def __init__(self, data_root='/home/lzp/FTransMamba/data/pengg/new_crop512', mode='train',
                 img_dir='opt_dir', mask_dir='ann_dir', img_suffix='.tif', mask_suffix='.tif',
                 transform=None, mosaic_ratio=0.0, img_size=ORIGIN_IMG_SIZE):
        self.data_root = osp.join(data_root, mode)  # e.g., '/home/.../new_crop512/train'
        self.img_dir = img_dir  # 'opt_dir' for RGB images
        self.mask_dir = mask_dir  # 'ann_dir' for masks
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform if transform else (train_aug if mode == 'train' else val_aug)
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode in ['val', 'test']:
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_path = osp.join(data_root, img_dir)
        mask_path = osp.join(data_root, mask_dir)
        if not osp.exists(img_path):
            raise FileNotFoundError(f"Image directory not found: {img_path}")
        if not osp.exists(mask_path):
            raise FileNotFoundError(f"Mask directory not found: {mask_path}")
        img_filename_list = os.listdir(img_path)
        mask_filename_list = os.listdir(mask_path)
        print(f"Found {len(img_filename_list)} images and {len(mask_filename_list)} masks in {data_root}")
        if len(img_filename_list) != len(mask_filename_list):
            raise ValueError(f"Number of images ({len(img_filename_list)}) and masks ({len(mask_filename_list)}) do not match in {data_root}")
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        mask_np = np.array(mask)
        if mask_np.max() >= len(CLASSES):
            raise ValueError(f"Mask values exceed num_classes ({len(CLASSES)}): max value found {mask_np.max()}")
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        start_y = h // 4
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(start_y, (h - start_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        return img, mask

# Load datasets
train_dataset = PenggDataset(data_root='/home/lzp/FTransMamba/data/pengg/new_crop512', mode='train',
                             img_dir='opt_dir', mask_dir='ann_dir',
                             mosaic_ratio=0.25, transform=train_aug)

val_dataset = PenggDataset(data_root='/home/lzp/FTransMamba/data/pengg/new_crop512', mode='val',
                           img_dir='opt_dir', mask_dir='ann_dir',
                           transform=val_aug)

test_dataset = PenggDataset(data_root='/home/lzp/FTransMamba/data/pengg/new_crop512', mode='val',
                            img_dir='opt_dir', mask_dir='ann_dir',
                            transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# Define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Basic training loop (for demonstration)
device = torch.device('cuda' if torch.cuda.is_available() and gpus != 0 else 'cpu')
net = net.to(device)
for epoch in range(max_epoch):
    net.train()
    for batch in train_loader:
        imgs = batch['img'].to(device)
        masks = batch['gt_semantic_seg'].to(device)
        optimizer.zero_grad()
        outputs = net(imgs)
        loss_value = loss(outputs, masks)
        loss_value.backward()
        optimizer.step()
    lr_scheduler.step()
    print(f"Epoch {epoch+1}/{max_epoch}, Loss: {loss_value.item()}")