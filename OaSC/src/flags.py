import argparse
from distutils import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--cnn")
parser.add_argument("--embs-pred")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--pickle", default=None)
parser.add_argument("--batch", default=256, type=int)
parser.add_argument("--log")
parser.add_argument("--test-dir")
parser.add_argument("--gamma", type=float, default=0.0)
parser.add_argument("--gpu", default="0")
parser.add_argument("--consider-trains", action="store_true")
parser.add_argument("--resnet-50", action="store_true")
parser.add_argument("--graph-type", default="osdd2")
parser.add_argument("--variable", default="open")
parser.add_argument("--exp_type", default="cross")
parser.add_argument("--output", default=None)
parser.add_argument("--change-order-labels", type=int,default=0)
parser.add_argument("--save_to_file", default='None')

parser.add_argument("--classes-ommited",  default=None)
parser.add_argument("--train-dir")
parser.add_argument("--save-path", default="saved_chechkpoints/finetune-osdd")
parser.add_argument("--num-epochs", default=50, type=int)
parser.add_argument("--fold", default=0, type=int)

parser.add_argument("--weights-path")
parser.add_argument("--exp-type")
parser.add_argument("--freeze", type=bool, default='true')

parser.add_argument("--class-ommited",  default=None)

parser.add_argument("--dataset",  default='osdd')


