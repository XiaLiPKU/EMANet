import logging
import numpy as np
from torch import Tensor


# Data settings
DATA_ROOT = '/path/to/VOC'
MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
STD = Tensor(np.array([0.229, 0.224, 0.225]))
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
CROP_SIZE = 513
IGNORE_LABEL = 255

# Model definition
N_CLASSES = 21
N_LAYERS = 101
STRIDE = 8
BN_MOM = 3e-4
EM_MOM = 0.9
STAGE_NUM = 3

# Training settings
BATCH_SIZE = 16
ITER_MAX = 30000
ITER_SAVE = 2000

LR_DECAY = 10
LR = 9e-3
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

DEVICE = 0
DEVICES = list(range(0, 4))

LOG_DIR = './logdir' 
MODEL_DIR = './models'
NUM_WORKERS = 16

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
