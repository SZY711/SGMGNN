import os
import sys
from argparse import ArgumentParser
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
from basicts import launch_training

torch.set_num_threads(2) # aviod high cpu avg usage


def parse_args():
    parser = ArgumentParser(description="LLMEncoder")

    parser.add_argument("-c", "--cfg", default="sgmgnn/SGMGNN_DYG_wi.py", help="training config")

    
    parser.add_argument("--gpus", default="5, 6, 7", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus)
