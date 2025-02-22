"""
This script is used to test the pipeline parallelism
"""
import argparse
import torch.nn as nn
import os
from torchgpipe import GPipe
from diffusers import UNet2DModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./output')
    return parser.parse_args()


def main():
    print("Testing pipeline parallelism")
    # model = nn.Sequential(a, b, c, d)
    # model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)
    # output = model(input)


if __name__ == "__main__":
    main()
