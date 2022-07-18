import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class ConLoss(nn.Module):
    def __init__(self, num_classes=None):
        super(ConLoss, self).__init__()
        self.num_classes = num_classes

    def mask2onehot(self, mask, num_classes):
        """
        Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
        hot encoding vector
        """
        mask = mask.cpu().numpy()
        semantic_map = [mask == i for i in range(num_classes)]
        return torch.from_numpy(np.array(semantic_map).astype(np.uint8))

    def forward(self, predict, target, confidence_mask):
        val_num = target[target != 255].numel()
        target = self.mask2onehot(target, num_classes=self.num_classes).transpose(1, 0).cuda()
        confidence_mask = confidence_mask / 100
        pre_confidence = confidence_mask.unsqueeze(1) * target
        predict = F.softmax(predict, dim=1)
        now_confidence = predict * target
        diff = now_confidence - pre_confidence
        loss = torch.sum(torch.abs(diff)) / val_num
        return loss

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(
            0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(
            1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(
            2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(
            n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(
            predict, target, weight=weight, size_average=self.size_average)
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target, mask):
        num_pixels = mask.sum()
        if num_pixels > 0:
            loss = torch.sum((predict - target) ** 2 * mask) / num_pixels
        else:
            loss = 0
        return loss

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predict, target, mask):
        loss = self.loss_fn(predict.squeeze(1), target.float())
        loss = (loss * mask).sum() / mask.sum()
        return loss

def bce_loss_image_level(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

class EMALoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, pred_s, pred_t, mask_s, mask_t, flip_s, flip_t):
        pred_s = F.softmax(pred_s, dim=1)
        pred_t = F.softmax(pred_t, dim=1)
        if flip_s != flip_t:
            pred_t = pred_t.flip(-1)
            mask_t = mask_t.flip(-1)
        mask = torch.logical_and(mask_s, mask_t)
        loss = torch.mean((pred_s - pred_t) ** 2 * mask)
        return loss


