import os
import sys


sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_mae

from .sgmgnn_arch import LLMEncoder
from .sgmgnn_runner import LLMEncoderRunner
from .sgmgnn_data import PretrainingDataset


CFG = EasyDict()


CFG.DESCRIPTION = "LLMEncoder(DYG_wi) configuration"
CFG.RUNNER = LLMEncoderRunner
CFG.DATASET_CLS = PretrainingDataset
CFG.DATASET_NAME = "DYG_wi"

CFG.DATASET_INPUT_LEN = 288 * 7
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 3


CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True


CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "LLMEncoder"
CFG.MODEL.ARCH = LLMEncoder
CFG.MODEL.PARAM = {
    "patch_size":12,
    "in_channel":1,
    "embed_dim":16,
    "num_heads":2,
    "mlp_ratio":2,
    "dropout":0.1,
    "num_token":288 * 7 / 12,
    "mask_ratio":0.75,
    "encoder_depth":3,
    "decoder_depth":1,
    "mode":"pre-train"

}
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]


CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.0005,
    "weight_decay":0,
    "eps":1.0e-8,
    "betas":(0.9, 0.95)
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[25],
    "gamma":0.5
}


CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 50
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0

CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME

CFG.TRAIN.DATA.BATCH_SIZE = 8
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = True


CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = EasyDict()

CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME

CFG.VAL.DATA.BATCH_SIZE = 4
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = True


CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1


CFG.TEST.DATA = EasyDict()

CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME

CFG.TEST.DATA.BATCH_SIZE = 4
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True
