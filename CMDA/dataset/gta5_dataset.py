import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
from . import joint_transforms
from . import transforms
import torchvision.transforms as standard_transforms


class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255,
                 ind_file='', semi=False, use_transform=False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.use_transform = use_transform
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if semi:
            np.random.seed(0)
            np.random.shuffle(self.img_ids)
            self.img_ids = self.img_ids[0:1000]
        if not max_iters == None:
            self.img_ids = self.img_ids * \
                int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.ind_file = ind_file
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "gta5_deeplab/images/%s" % name)
            label_file = osp.join(self.root, "gta5_deeplab/labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        transform_list = [
            joint_transforms.RandomSizeAndCrop(
                size=713,
                crop_nopad=True,
                scale_min=0.5,
                scale_max=1.5,
                rec=False
            ),
            joint_transforms.Resize(713)
        ]
        self.joint_transform = joint_transforms.Compose(transform_list)

        image_transform_list = [
            transforms.FlipChannels(),
            transforms.SubMean(),
            transforms.ToTensor()
        ]

        self.image_transform = standard_transforms.Compose(image_transform_list)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        if self.use_transform:
            image, label = self.joint_transform(image, label, None)
            image = self.image_transform(image)
        else:
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)
            image = np.asarray(image, np.float32)

        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label_copy = label_copy.astype(np.int64)

        size = image.shape

        if not self.use_transform:
            image = image[:, :, ::-1]  # change to BGR
            image -= self.mean
            image = image.transpose((2, 0, 1))

        if len(self.ind_file) > 0:
            with open(self.ind_file, 'r') as f:
                lines = f.readlines()
            ind_dict = {line.split()[0]: line.split()[1:] for line in lines}
            ind = ind_dict[name]
            if isinstance(ind, str):
                ind = float(ind)
            elif isinstance(ind, list):
                ind = [float(i) for i in ind]

            return image.copy(), label_copy.copy(), np.array(size), name, ind

        return image.copy(), label_copy.copy(), np.array(size), name


class GTA5ScaleDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(1024, 512), mean=(128, 128, 128),
                 scale=False, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.h = crop_size[1]
        self.w = crop_size[0]
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        if self.scale:
            random_scale = 0.8 + random.random() * 0.4  # 0.8 - 1.2
            image = image.resize((round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)), Image.BICUBIC)
            label = label.resize((round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)), Image.NEAREST)
        else:
            image = image.resize((self.resize_size[0], self.resize_size[1]), Image.BICUBIC)
            label = label.resize((self.resize_size[0], self.resize_size[1]), Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in list(self.id_to_trainid.items()):
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        # print(image.shape, label.shape)
        for i in range(10):  # find hard samples
            x1 = random.randint(0, image.shape[1] - self.h)
            y1 = random.randint(0, image.shape[2] - self.w)
            tmp_label_copy = label_copy[x1:x1 + self.h, y1:y1 + self.w]
            tmp_image = image[:, x1:x1 + self.h, y1:y1 + self.w]
            u = np.unique(tmp_label_copy)
            if len(u) > 10:
                break
            else:
                pass

        image = tmp_image
        label_copy = tmp_label_copy

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis=2)
            label_copy = np.flip(label_copy, axis=1)

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
