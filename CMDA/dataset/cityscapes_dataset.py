import os
import os.path as osp
import numpy as np
import random
from torch.utils import data
from PIL import Image

label2train = [
    [0, 255],
    [1, 255],
    [2, 255],
    [3, 255],
    [4, 255],
    [5, 255],
    [6, 255],
    [7, 0],
    [8, 1],
    [9, 255],
    [10, 255],
    [11, 2],
    [12, 3],
    [13, 4],
    [14, 255],
    [15, 255],
    [16, 255],
    [17, 5],
    [18, 255],
    [19, 6],
    [20, 7],
    [21, 8],
    [22, 9],
    [23, 10],
    [24, 11],
    [25, 12],
    [26, 13],
    [27, 14],
    [28, 15],
    [29, 255],
    [30, 255],
    [31, 16],
    [32, 17],
    [33, 18],
    [-1, 255]
]


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True,
                 ignore_label=255, set='val', lbl=False, ssl_dir=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.lbl = lbl
        self.ssl_dir = ssl_dir
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.len = len(self.img_ids)
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        self.set = set
        for ii, name in enumerate(self.img_ids):
            img_file = osp.join(
                self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbl_name = name.replace('leftImg8bit', 'gtFine_labelIds')
            lbl_file = osp.join(self.root, 'gtFine/%s/%s' %
                                (self.set, lbl_name))
            self.files.append({
                "img": img_file,
                'lbl': lbl_file,
                "name": name,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)

        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        if self.lbl:
            label = Image.open(datafiles['lbl'])
            label = np.array(label, np.int)
            label = label_mapping(label, label2train)
            return image.copy(), label.copy(), np.array(size), name

        if self.ssl_dir:
            label = Image.open(os.path.join(self.ssl_dir, name.split('/')[-1]))
            label = label.resize(self.crop_size, Image.NEAREST)
            label = np.asarray(label, np.int64)

            return image.copy(), label.copy(), np.array(size), name

        return image.copy(), np.array(size), name

class cityscapesPseudoDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(1024, 512), mean=(128, 128, 128),
                 scale=False, mirror=True, ignore_label=255, set='val', ssl_dir=None, confidence_dir=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.ssl_dir = ssl_dir
        self.confidence_dir = confidence_dir
        self.h = crop_size[1]
        self.w = crop_size[0]
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.root, "pseudo_FULL/%s/%s" % (self.set, name))
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        label = Image.open(os.path.join(self.ssl_dir, name.split('/')[-1]))
        confidence = Image.open(os.path.join(self.confidence_dir, name.split('/')[-1]))

        # resize
        if self.scale:
            random_scale = 0.8 + random.random() * 0.4  # 0.8 - 1.2
            image = image.resize((round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)), Image.BICUBIC)
            label = label.resize((round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)), Image.NEAREST)
            confidence = confidence.resize((round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)),
                                 Image.NEAREST)
        else:
            image = image.resize((self.resize_size[0], self.resize_size[1]), Image.BICUBIC)
            label = label.resize((self.resize_size[0], self.resize_size[1]), Image.NEAREST)
            confidence = confidence.resize((self.resize_size[0], self.resize_size[1]), Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)
        confidence = np.asarray(confidence, np.uint8)

        # re-assign labels to match the format of Cityscapes
        label_copy = label
        confidence_copy = confidence

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        #print(image.shape, label.shape)
        for i in range(10):  # find hard samples
            x1 = random.randint(0, image.shape[1] - self.h)
            y1 = random.randint(0, image.shape[2] - self.w)
            tmp_label_copy = label_copy[x1:x1 + self.h, y1:y1 + self.w]
            tmp_confidence_copy = confidence_copy[x1:x1 + self.h, y1:y1 + self.w]
            tmp_image = image[:, x1:x1 + self.h, y1:y1 + self.w]
            u = np.unique(tmp_label_copy)
            if len(u) > 10:
                break
        image = tmp_image
        label_copy = tmp_label_copy
        confidence_copy = tmp_confidence_copy

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis=2)
            label_copy = np.flip(label_copy, axis=1)
            confidence_copy = np.flip(confidence_copy, axis=1)

        label_copy = np.asarray(label_copy, np.int64)
        confidence_copy = np.asarray(confidence_copy, np.int64)

        return image.copy(), label_copy.copy(), np.array(size), name, confidence_copy.copy()