import time
from dataset.cityscapes_dataset import cityscapesPseudoDataSet
from dataset.gta5_dataset import GTA5ScaleDataSet
from dataset.synthia_dataset import SYNScaleDataSet
from utils.loss import MaskedBCELoss, ConLoss, bce_loss_image_level
from model.discriminator import FeatDiscriminator
from model.discriminator import FCDiscriminator
from model.deeplab_multi import DeeplabMultiFeature
from torch.utils.tensorboard import SummaryWriter
from config.config import cfg, cfg_from_file
import random
import os.path as osp
import os
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.utils import data, model_zoo
import torch.nn as nn
import argparse
import torch
import pprint
from utils.misc import adjust_learning_rate_D, adjust_learning_rate
try:
    from apex import amp
    APEX_AVAILABLE = True
    print('Use apex to accelerate training')
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def get_arguments():
    """
    Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--cfg", type=str, default='config/test.yml',
                        help="the configs used to train the model from GTA to Cityscapes")
    return parser.parse_args()

args = get_arguments()

def amp_backward(loss, optimizer, retain_graph=False):
    if APEX_AVAILABLE:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)
    else:
        loss.backward(retain_graph=retain_graph)

def main():
    """
    Create the model and start the training.
    """

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    print('config:')
    pprint.pprint(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)

    w, h = map(int, cfg.INPUT_SIZE.split(','))
    input_size = (w, h)
    w, h = map(int, cfg.INPUT_SIZE_TARGET.split(','))
    input_size_target = (w, h)
    w, h = map(int, cfg.CROP_SOURCE.split(','))
    crop_size = (w, h)
    w, h = map(int, cfg.CROP_TARGET.split(','))
    crop_size_target = (w, h)
    Iter = 0

    # ------------------------------ Create network ------------------------------ #

    # init G
    if cfg.MODEL == 'DeepLab':
        model = DeeplabMultiFeature(num_classes=cfg.NUM_CLASSES, multi_level=True)

        if cfg.RESTORE_FROM[:4] == 'http':
            print('Load from internet')
            saved_state_dict = model_zoo.load_url(cfg.RESTORE_FROM)
        else:
            print('Load from local')
            saved_state_dict = torch.load(cfg.RESTORE_FROM)
        if cfg.CONTINUE_TRAIN:
            if list(saved_state_dict.keys())[0].split('.')[0] == 'module':
                for key in saved_state_dict.keys():
                    saved_state_dict['.'.join(key.split('.')[1:])] = saved_state_dict.pop(key)
            model.load_state_dict(saved_state_dict)
        else:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not cfg.NUM_CLASSES == 19 or not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)

    # init D
    model_D = FCDiscriminator(num_classes=cfg.NUM_CLASSES).to(device)
    model_D_aux = FeatDiscriminator().to(device)
    model_D_st = FCDiscriminator(num_classes=cfg.NUM_CLASSES).to(device)

    model.train()
    model.to(device)
    model_D.train()
    model_D.to(device)
    model_D_aux.train()
    model_D_aux.to(device)
    model_D_st.train()
    model_D_st.to(device)

    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.PSEUDO_LABEL_DIR, exist_ok=True)
    os.makedirs(cfg.PSEUDO_LABEL_CONFIDENCE_DIR, exist_ok=True)
    writer = SummaryWriter(cfg.LOG_DIR)

    # ----------------------------- init data loader ----------------------------- #
    if cfg.SOURCE == 'GTA':
        source_set = GTA5ScaleDataSet(cfg.DATA_DIRECTORY, cfg.DATA_LIST_PATH, max_iters=cfg.NUM_STEPS * cfg.BATCH_SIZE,
                                        resize_size=input_size, crop_size=crop_size, scale=True, mirror=True,
                                        mean=cfg.IMG_MEAN)

    elif cfg.SOURCE == 'SYNTHIA':
        source_set = SYNScaleDataSet(cfg.DATA_DIRECTORY, cfg.DATA_LIST_PATH, max_iters=cfg.NUM_STEPS * cfg.BATCH_SIZE,
                                  resize_size=input_size, crop_size=input_size, scale=True, mirror=True,
                                     mean=cfg.IMG_MEAN)

    trainloader = data.DataLoader(source_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    targetset = cityscapesPseudoDataSet(cfg.DATA_DIRECTORY_TARGET, cfg.DATA_LIST_PATH_TARGET,
                                            max_iters=cfg.NUM_STEPS * cfg.BATCH_SIZE, resize_size=input_size_target,
                                            crop_size=crop_size_target, scale=True, mirror=True, mean=cfg.IMG_MEAN,
                                            set=cfg.SET, ssl_dir=cfg.PSEUDO_LABEL_DIR,
                                            confidence_dir=cfg.PSEUDO_LABEL_CONFIDENCE_DIR)

    targetloader = data.DataLoader(targetset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                   num_workers=cfg.NUM_WORKERS, pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    # ------------------------------ init optimizer ------------------------------ #
    optimizer = optim.SGD(model.optim_parameters(cfg),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    optimizer_D_aux = optim.Adam(model_D_aux.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D_aux.zero_grad()

    optimizer_D_st = optim.Adam(model_D_st.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D_st.zero_grad()

    if APEX_AVAILABLE:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1",
            keep_batchnorm_fp32=None, loss_scale="dynamic"
        )

        model_D, optimizer_D = amp.initialize(
            model_D, optimizer_D, opt_level="O1",
            keep_batchnorm_fp32=None, loss_scale="dynamic"
        )

        model_D_aux, optimizer_D_aux = amp.initialize(
            model_D_aux, optimizer_D_aux, opt_level="O1",
            keep_batchnorm_fp32=None, loss_scale="dynamic"
        )
        
        model_D_st, optimizer_D_st = amp.initialize(
            model_D_st, optimizer_D_st, opt_level="O1",
            keep_batchnorm_fp32=None, loss_scale="dynamic"
        )

    # ---------------------------- init loss function ---------------------------- #
    bce_loss = torch.nn.BCEWithLogitsLoss()
    masked_bce_loss = MaskedBCELoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    con_loss = ConLoss(num_classes=cfg.NUM_CLASSES)

    if cfg.BATCH_SIZE == 1:
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
        interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), 
                                    mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(crop_size[1], crop_size[0]), mode='bilinear', align_corners=True)
        interp_target = nn.Upsample(size=(crop_size_target[1], crop_size_target[0]), 
                                    mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    # ------------------------------ start training ------------------------------ #
    for i_iter in range(Iter, cfg.NUM_STEPS):
        loss_seg_src_value = 0
        loss_seg_trg_value = 0
        loss_adv_pl_main_value = 0
        loss_D_value = 0
        loss_adv_pl_aux_value = 0
        loss_D_aux_value = 0
        loss_D_s = 0
        loss_D_t = 0
 
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter, cfg)

        optimizer_D_aux.zero_grad()
        adjust_learning_rate_D(optimizer_D_aux, i_iter, cfg)

        optimizer_D_st.zero_grad()
        adjust_learning_rate_D(optimizer_D_st, i_iter, cfg)

        start_time = time.time()

        # train G
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False
        for param in model_D_aux.parameters():
            param.requires_grad = False
        for param in model_D_st.parameters():
            param.requires_grad = False

        _, src_batch = trainloader_iter.__next__()

        src_images, src_labels, _, _ = src_batch
        src_images = src_images.to(device)
        src_labels = src_labels.long().to(device)

        src_feature, src_pred = model(src_images)
        src_pred = interp(src_pred)
        src_pred_softmax = F.softmax(src_pred, dim=1)
        loss_seg_src = seg_loss(src_pred, src_labels)

        # proper normalization
        amp_backward(loss_seg_src, optimizer)
        loss_seg_src_value += loss_seg_src.item()

        _, batch = targetloader_iter.__next__()
        trg_images, trg_labels, _, _, confidence = batch
        confidence = confidence.cuda()
        trg_labels = trg_labels.cuda()
        trg_images = trg_images.to(device)

        # forward
        trg_feature, pred_target = model(trg_images)
        # interpolation
        pred_target = interp_target(pred_target)
        pred_target_softmax = F.softmax(pred_target, dim=1)

        loss_seg_trg = seg_loss(pred_target, trg_labels)
        loss_con_trg = con_loss(pred_target, trg_labels, confidence)
        loss_seg_trg_value += loss_seg_trg.item()

        D_label = (trg_labels == 255).long()
        D_out = model_D(pred_target_softmax)
        D_out = interp_target(D_out)
        loss_adv_pl_main = masked_bce_loss(D_out, D_label, D_label)

        D_out_aux = model_D_aux(trg_feature)
        D_out_aux = interp_target(D_out_aux)
        loss_adv_pl_aux = masked_bce_loss(D_out_aux, D_label, D_label)

        # adv loss for image-level
        D_out_image_level = model_D_st(pred_target_softmax)
        loss_image_level = bce_loss_image_level(D_out_image_level, source_label)


        loss = loss_seg_trg + cfg.LAMBDA_CONFIDENCE * loss_con_trg + cfg.LAMBDA_ADV_TARGET * loss_adv_pl_main + \
               cfg.LAMBDA_ADV_AUX * loss_adv_pl_aux + cfg.LAMBDA_ADV_ST * loss_image_level

        amp_backward(loss, optimizer)
        loss_adv_pl_main_value += loss_adv_pl_main.item()
        loss_adv_pl_aux_value += loss_adv_pl_aux.item()

        # ---------------------------------- train D --------------------------------- #

        # bring back requires_grad
        for param in model_D.parameters():
            param.requires_grad = True
        for param in model_D_aux.parameters():
            param.requires_grad = True
        for param in model_D_st.parameters():
            param.requires_grad = True

        # train aux D
        trg_feature = trg_feature.detach()
        D_out_aux = model_D_aux(trg_feature)
        D_out_aux = interp_target(D_out_aux)

        loss_D = bce_loss(D_out_aux.squeeze(1), (1 - D_label).float())
        loss_D = loss_D / 2
        amp_backward(loss_D, optimizer_D_aux)
        loss_D_aux_value += loss_D.item()

        # train with target
        pred_target_softmax = pred_target_softmax.detach()
        D_out = model_D(pred_target_softmax)
        D_out = interp_target(D_out)

        loss_D = bce_loss(D_out.squeeze(1), (1 - D_label).float())
        loss_D = loss_D / 2
        amp_backward(loss_D, optimizer_D)
        loss_D_value += loss_D.item()

        # train image level
        src_pred_softmax = src_pred_softmax.detach()
        D_out_image_level = model_D_st(src_pred_softmax)
        loss_D = bce_loss_image_level(D_out_image_level, source_label)
        loss_D = loss_D / 2
        amp_backward(loss_D, optimizer_D_st)
        loss_D_s += loss_D.item()

        D_out_image_level = model_D_st(pred_target_softmax)
        loss_D = bce_loss_image_level(D_out_image_level, target_label)
        loss_D = loss_D / 2
        amp_backward(loss_D, optimizer_D_st)
        loss_D_t += loss_D.item()

        optimizer.step()
        optimizer_D.step()
        optimizer_D_aux.step()
        optimizer_D_st.step()

        batch_time = time.time() - start_time

        # --------------------------------- log info --------------------------------- #

        scalar_info = {
            'loss_seg': loss_seg_src_value,
            'loss_adv_target': loss_adv_pl_main_value,
            'loss_adv_aux': loss_adv_pl_aux_value,
            'loss_image_level':loss_image_level,
            'loss_D': loss_D_value,
            'loss_D_aux': loss_D_aux_value,
            'loss_D_s': loss_D_s,
            'loss_D_t': loss_D_t,
        }

        if i_iter % 1000 == 0:
            for key, val in scalar_info.items():
                writer.add_scalar(key, val, i_iter)

        # log on terminal
        log_str='iter = %8d/%8d, batch time %.4f seg_tar = %.3f,seg_src=%.3f, con = %.3f, adv_pl = %.3f adv_st = %.2f' \
            % (i_iter, cfg.NUM_STEPS, batch_time, loss_seg_trg_value, loss_seg_src_value, cfg.LAMBDA_CONFIDENCE*loss_con_trg,
            loss_adv_pl_main_value, loss_image_level)

        print(log_str)

        # -------------------------------- End training ------------------------------- #

        if i_iter > cfg.NUM_STEPS_STOP-1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(cfg.SNAPSHOT_DIR, 'model_' + str(cfg.NUM_STEPS_STOP) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(cfg.SNAPSHOT_DIR, 'model_' + str(cfg.NUM_STEPS_STOP) + '_D.pth'))
            break

        # -------------------------------- evaluation -------------------------------- #
        if i_iter % cfg.SAVE_PRED_EVERY == 0:
            torch.save(model.state_dict(), osp.join(cfg.SNAPSHOT_DIR, 'model_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(cfg.SNAPSHOT_DIR, 'model_' + str(i_iter) + '_D.pth'))

        model.train()
    writer.close()


if __name__ == '__main__':
    main()
