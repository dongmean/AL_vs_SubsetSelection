import argparse
import numpy as np
import time
from utils import *

parser = argparse.ArgumentParser(description='Parameter Processing')

# Basic arguments
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
parser.add_argument('--n-class', type=str, default=10, help='# of classes')
parser.add_argument('--cycle', type=int, default=10, help='# of AL cycles')
parser.add_argument('--n-query', type=int, default=1000, help='# of query samples')
parser.add_argument('--subset', type=int, default=50000, help='subset')
parser.add_argument('--resolution', type=int, default=32, help='resolution') # 32
parser.add_argument('--model', type=str, default='ResNet18', help='model')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
parser.add_argument('--print_freq', '-p', default=300, type=int, help='print frequency (default: 20)')
parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('-j', '--workers', default=20, type=int, help='number of data loading workers (default: 4)')

# Optimizer and scheduler
parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help=
"Learning rate scheduler")
parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")

# Training
parser.add_argument('--batch-size', "-b", default=128, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument("--test-batch-size", "-tb", default=500, type=int,
                    help="batch size for training, if not specified, it will equal to batch size in argument --batch")
# Testing
parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
"the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                    help="proportion of test dataset used for evaluating the model (default: 1.)")

# Selecting
parser.add_argument('--balance', default=False, type=str_to_bool,
                    help="whether balance selection is performed per class")

# Algorithm
parser.add_argument('--method', default="Uncertainty", help="specifiy AL method to use") #Uncertainty, kCenterGreedy, GTknownUncertainty
parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
parser.add_argument('--submodular_greedy', default="LazyGreedy", help="specifiy greedy algorithm for submodular optimization")
parser.add_argument('--uncertainty', default="Margin", help="specifiy uncertanty score to use") #Margin, LeastConfidence, Entropy, Loss

# Checkpoint and resumption
parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")

args = parser.parse_args()