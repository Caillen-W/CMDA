import numpy as np
import torch

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, cfg):
    lr = lr_poly(cfg.LEARNING_RATE, i_iter, cfg.NUM_STEPS, cfg.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter, cfg):
    lr = lr_poly(cfg.LEARNING_RATE_D, i_iter, cfg.NUM_STEPS, cfg.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    semantic_map = [mask == i for i in range(num_classes)]
    return np.array(semantic_map).astype(np.uint8)

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)

def load_sp(file):
    sp = np.load(file)
    sp = sp[:, :, :19]
    sp = sp / sp.sum(2).reshape(512, 1024, 1)
    sp = sp.transpose(2, 0, 1)
    sp = torch.from_numpy(sp)
    sp = sp.view(1, 19, 512, 1024)
    pooling = torch.nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
    sp = pooling(sp).float().cuda()
    print('load the spatial prior npy successfully')
    return sp