import torch
import torch.nn as nn
from torch.utils import data
from PIL import Image
import os
import numpy as np
from model.deeplab_multi import DeeplabMultiFeature
from dataset.cityscapes_dataset import cityscapesDataSet
import argparse
from config.config import cfg, cfg_from_file
import pprint
from utils.misc import load_sp, load_checkpoint_for_evaluation, mask2onehot

def get_arguments():
    """
    Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--cfg", type=str, default='config/gen_pl_plc_sp.yml',
                        help="available options : DeepLab")
    parser.add_argument("--best-iter", type=int, default=58000,
                        help="iteration with best mIoU")
    return parser.parse_args()

def main():
    args = get_arguments()
    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    print('config:')
    pprint.pprint(cfg)

    if not os.path.exists(r'pseudo_label'):
        os.mkdir('pseudo_label')

    if not os.path.exists(r'pseudo_label_confidence'):
        os.mkdir('pseudo_label_confidence')

    model = DeeplabMultiFeature(num_classes=19, multi_level=True)
    restore_from = os.path.join(cfg.SNAPSHOT_DIR, f'model_{args.best_iter}.pth')
    print('loading the generator:', restore_from)

    load_checkpoint_for_evaluation(model, restore_from, device=0)
    # load data
    target_dataset = cityscapesDataSet(cfg.DATA_DIRECTORY_TARGET, cfg.DATA_LIST_PATH_TARGET, crop_size=cfg.INPUT_SIZE_TARGET,
                                      scale=False, mean=cfg.IMG_MEAN, set='train')

    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.BATCH_SIZE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=False,
                                    pin_memory=True,
                                    worker_init_fn=None)

    predicted_label = np.zeros((len(target_loader), 512, 1024), dtype=np.float32)
    predicted_prob_with_prior = np.zeros((len(target_loader), 512, 1024), dtype=np.float32)
    predicted_prob = np.zeros((len(target_loader), 512, 1024), dtype=np.float32)
    image_name = []
    load_spatial_prior = load_sp('spatial_prior/GTA.npy')

    for index, batch in enumerate(target_loader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, _, name= batch
        with torch.no_grad():
            aux, output = model(image.cuda(0))
            output = nn.functional.softmax(output, dim=1)
            output = nn.functional.interpolate(output, (512, 1024), mode='bilinear', align_corners=True)
            output_with_spatial_prior = (output * load_spatial_prior).cpu().data[0].numpy()
            output = output.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            label = np.argmax(output, axis=2)
            label_one_hot = mask2onehot(label, num_classes=19)
            output_with_spatial_prior = output_with_spatial_prior * label_one_hot
            spatial_prior = np.max(output_with_spatial_prior, axis=0)
            prob = np.max(output, axis=2)
            predicted_label[index] = label.copy()
            predicted_prob[index] = prob.copy()
            predicted_prob_with_prior[index] = spatial_prior.copy()
            image_name.append(name[0])
    thres = []
    for i in range(19):
        y = predicted_prob_with_prior[predicted_label == i]
        if len(y) == 0:
            thres.append(0)
            continue
        y = np.sort(y)
        thres.append(y[int(np.round(len(y) * 0.33))])
    print('prob_thersh:',thres)
    thres = np.array(thres)
    for index in range(len(target_loader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        prob_prior = predicted_prob_with_prior[index]
        for i in range(19):
            label[(prob_prior < thres[i]) * (label == i) * (prob < 0.9)] = 255
        prob[label == 255] = 0
        output = np.asarray(label, dtype=np.uint8)
        prob = prob * 100
        prob = np.asarray(prob, dtype=np.uint8)
        output = Image.fromarray(output)
        prob = Image.fromarray(prob)
        name = name.split('/')[-1]
        output.save('%s/%s' % ('pseudo_label', name))
        prob.save('%s/%s' % ('pseudo_label_confidence', name))
    print('finish')

if __name__ == '__main__':
    main()