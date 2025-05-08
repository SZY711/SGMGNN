import os
import sys



sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from .sgmgnn_arch import SGMGNN
from .sgmgnn_runner import SGMGNNRunner
from .sgmgnn_loss import sgmgnn_loss
from .sgmgnn_data import ForecastingDataset


CFG = EasyDict()


CFG.DESCRIPTION = "SGMGNN(DYG_wi) configuration"
CFG.RUNNER = SGMGNNRunner
CFG.DATASET_CLS = ForecastingDataset
CFG.DATASET_NAME = "DYG_wi"

CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.DATASET_ARGS = {
    "seq_len": 288 * 7
    }
CFG.GPU_NUM = 3


CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True


CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "SGMGNN"
CFG.MODEL.ARCH = SGMGNN

CFG.MODEL.PARAM = {
    "dataset_name": CFG.DATASET_NAME,
    "pre_trained_llmencoder_path": "llmencoder_ckpt/LLMEncoder_best_val_MAE.pt",
    "llmencoder_args": {
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
                    "mode":"forecasting"
    },
    "backend_args": {
                    "num_nodes" : 14,
                    "support_len" : 3,
                    "dropout"   : 0.3,
                    "gcn_bool"  : True,
                    "addaptadj" : True,
                    "aptinit"   : None,
                    "in_dim"    : 2,
                    "out_dim"   : 12,
                    "residual_channels" : 32,
                    "dilation_channels" : 32,
                    "skip_channels"     : 256,
                    "end_channels"      : 512,
                    "kernel_size"       : 2,
                    "blocks"            : 4,
                    "layers"            : 2
    },
    "dgl_args": {
                "dataset_name": CFG.DATASET_NAME,
                "k": 3,
                "input_seq_len": CFG.DATASET_INPUT_LEN,
                "output_seq_len": CFG.DATASET_OUTPUT_LEN
    }
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True


CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = sgmgnn_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.005,
    "weight_decay":1.0e-5,
    "eps":1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[1, 18, 36, 54, 72],
    "gamma":0.5
}


CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0

CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME

CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = True

CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 6
CFG.TRAIN.CL.PREDICTION_LENGTH = 12


CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = EasyDict()

CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME

CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = True


CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1


CFG.TEST.DATA = EasyDict()

CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME

CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True
