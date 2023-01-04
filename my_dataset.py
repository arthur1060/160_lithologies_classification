from PIL import Image,ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import InterpolationMode
import  random
import torchvision.transforms.functional as ff
from torchvision.transforms import InterpolationMode, RandomRotation, RandomResizedCrop
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class SHDataSets(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list,  transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.crop_size = 224
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image = Image.open(self.images_path[item])

        if random.random()<0.4:
            resize_ratio1, resize_ratio2 = random.randint(260, 280), random.randint(260, 280)
            img1 = image.resize((resize_ratio1, resize_ratio2), Image.BILINEAR)
            rotate_fun = RandomRotation(degrees=(-15, 15))
            img2 = rotate_fun(img1)
            x1 = int(round((resize_ratio1 - self.crop_size) / 2.))
            y1 = int(round((resize_ratio2 - self.crop_size) / 2.))
            image3 = img2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        else:
            resize_size1, resize_size2 = random.randint(224, 260), random.randint(224, 260)
            img2 = image.resize((resize_size1, resize_size2), Image.BILINEAR)

            x1 = random.randint(0, resize_size1 - self.crop_size)
            y1 = random.randint(0,  resize_size2 - self.crop_size)
            image3 = img2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        label = self.images_class[item]

        img = self.transform(image3)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

class MyDataSet_test(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        w, h = img.size

        if w>h:
            padh1 = (w - h)//2
            padh2 = w - h - padh1
            padding = (0, padh1, 0, padh2)
            img = ImageOps.expand(img, border=padding, fill=0)
        else:
            padw1 = (h - w) // 2
            padw2 = h - w - padw1
            padding = (padw1, 0, padw2, 0)
            img = ImageOps.expand(img, border=padding, fill=0)

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels



class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
