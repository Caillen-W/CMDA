import torch
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from model.deeplab_multi import DeeplabMultiFeature
from dataset.cityscapes_dataset import cityscapesDataSet
from utils.compute_iou import name_classes, intersectionAndUnionGPU
from tqdm import tqdm
from PIL import Image

DATA_DIRECTORY_TARGET = r'/.dataset/Cityscapes'
DATA_LIST_PATH_TARGET_TEST = './dataset/cityscapes_list/val.txt'
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
SCALES = [[1024, 512], [1280, 640], [1536, 768], [1800, 900], [2048, 1024]]
RESUME_PATH = 'snapshot/GTA5_40000.pth'
USE_FLIP = True

print(SCALES)

categories = [0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
mask_colors = {11: [70, 70, 70],  23: [70, 130, 180],  17: [153, 153, 153],  0: [0, 0, 0],
21: [ 107, 142, 35],  15: [100, 100, 150],  5: [111, 74, 0],
22: [152, 251, 152],  13: [190, 153, 153],  12: [102, 102, 156],
24: [220, 20, 60],  6: [81, 0, 81],  27: [0, 0, 70],
7: [128, 64, 128],  19: [250, 170, 30],  20: [220, 220, 0],
4: [20, 20, 20],  26: [0, 0, 142],  32: [0, 0, 230],
8: [244, 35, 232],  34: [0, 0, 142],  1: [0, 0, 0],  16: [150, 120, 90],
14: [180, 165, 180],  28: [0, 60, 100],  31: [0, 80, 100],  25: [255, 0, 0],
33: [ 119, 11, 32],  30: [0, 0, 110]}

def save_to_image(orig, pred, n_image, name, file_dir='result'):
    name = name[0].split('/')[-1]
    pred_img = np.zeros(shape=(pred.shape[0], pred.shape[1], 3), dtype='uint8')
    for cls in range(len(categories)-1):
        pred_img[pred == cls] = mask_colors[categories[cls+1]]
    Image.fromarray(pred_img).save(f"{file_dir}/{name}")

def eval():
    # -------------------------------- init loader ------------------------------- #

    testloader = data.DataLoader(cityscapesDataSet(DATA_DIRECTORY_TARGET, DATA_LIST_PATH_TARGET_TEST,
                                                   crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False,
                                                   set='val', lbl=True),
                                 batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # -------------------------------- init model -------------------------------- #

    model = DeeplabMultiFeature(num_classes=19, multi_level=True)
    saved_state_dict = torch.load(RESUME_PATH)
    if list(saved_state_dict.keys())[0].split('.')[0] == 'module':
        for key in saved_state_dict.keys():
            saved_state_dict['.'.join(key.split('.')[1:])] = saved_state_dict.pop(key)
    model.load_state_dict(saved_state_dict)
    model = model.cuda()
    print('Loaded %s ' % RESUME_PATH)

    # ----------------------------------- test ----------------------------------- #
    intersection_sum = 0
    union_sum = 0
    model.eval()
    for index, batch in enumerate(tqdm(testloader)):
        images, labels, _, name = batch
        images = images.cuda()
        labels = labels.cuda()

        pred_result = []
        for scale in SCALES:
            tmp_images = F.interpolate(images, scale[::-1], mode='bicubic', align_corners=True)
            with torch.no_grad():
                logits = F.softmax(model(tmp_images)[1], dim=1)

            if USE_FLIP:
                with torch.no_grad():
                    flip_logits = F.softmax(model(torch.flip(tmp_images, dims=[3]))[1], dim=1)
                logits += torch.flip(flip_logits, dims=[3])
            logits = F.interpolate(logits, labels.size()[1:], mode='bicubic', align_corners=True)
            pred_result.append(logits)

        result = sum(pred_result)
        label_pred = result.max(dim=1)[1]
        intersection, union = intersectionAndUnionGPU(label_pred, labels, len(name_classes))
        intersection_sum += intersection
        union_sum += union

    intersection_sum = intersection_sum.cpu().numpy()
    union_sum = union_sum.cpu().numpy()
    mIoUs = intersection_sum / union_sum

    mIoU19 = round(np.nanmean(mIoUs) * 100, 2)
    mIoU16 = round(np.mean(mIoUs[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)
    mIoU13 = round(np.mean(mIoUs[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)

    for ind_class in range(len(name_classes)):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU19: %.2f' % mIoU19)
    print('===> mIoU16: %.2f' % mIoU16)
    print('===> mIoU13: %.2f' % mIoU13)

if __name__ == '__main__':
    eval()
