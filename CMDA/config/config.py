import pprint
import numpy as np
from easydict import EasyDict
import yaml

cfg = EasyDict()

cfg.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.MODEL = 'DeepLab'
cfg.BATCH_SIZE = 4
cfg.ITER_SIZE = 1
cfg.NUM_WORKERS = 4
cfg.SOURCE = 'GTA'
cfg.DATA_DIRECTORY = r'F:/dataset/BDL/gta5_deeplab/gta5_deeplab'
cfg.DATA_LIST_PATH = 'dataset/gta5_list/train.txt'
cfg.IGNORE_LABEL = 255
cfg.INPUT_SIZE = '1280,720'
cfg.CROP_SOURCE = '1024,512'
cfg.DATA_DIRECTORY_TARGET = r'F:/dataset/Cityscapes'
cfg.DATA_LIST_PATH_TARGET = 'dataset/cityscapes_list/train.txt'
cfg.DATA_LIST_PATH_TARGET_TEST = 'dataset/cityscapes_list/val.txt'
cfg.INPUT_SIZE_TARGET = '1024,512'
cfg.CROP_TARGET = '768,384'
cfg.LEARNING_RATE = 5e-4
cfg.MOMENTUM = 0.9
cfg.NUM_CLASSES = 19
cfg.NUM_STEPS = 100000
cfg.NUM_STEPS_STOP = 100000  # early stopping
cfg.POWER = 0.9
cfg.RANDOM_SEED = 1234
cfg.RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
cfg.SAVE_NUM_IMAGES = 2
cfg.SAVE_PRED_EVERY = 2000
cfg.EVAL_PRED_EVERY = 500
cfg.WEIGHT_DECAY = 0.0005
cfg.SNAPSHOT_DIR = 'snapshots'
cfg.LOG_DIR = 'logs'
cfg.PSEUDO_LABEL_DIR = 'pseudo_label'   # the path of saving pseudo labels
cfg.PSEUDO_LABEL_CONFIDENCE_DIR = 'pseudo_label_confidence'   # the path of saving confidence of pseudo labels
cfg.LEARNING_RATE_D = 1e-4
cfg.LAMBDA_ADV_TARGET = 0.05
cfg.LAMBDA_ADV_AUX = 0.01
cfg.LAMBDA_ADV_ST = 0.01
cfg.LAMBDA_CONFIDENCE = 1
cfg.TARGET = 'cityscapes'
cfg.SET = 'train'
cfg.GAN = 'Vanilla'
cfg.CONTINUE_TRAIN = False

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')
        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v

def yaml_load(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)

if __name__ == '__main__':
    pprint.pprint(cfg)