# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T

# Utils
from tqdm import tqdm
from utils import *

# Custom
from arguments import parser
from ptflops import get_model_complexity_info
import nets
import datasets as datasets
import methods as methods

# Seed
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Main
if __name__ == '__main__':
    # Training settings
    args = parser.parse_args()
    cuda = ""
    if len(args.gpu) > 1:
        cuda = 'cuda'
    elif len(args.gpu) == 1:
        cuda = 'cuda:'+str(args.gpu[0])

    if args.dataset == 'ImageNet':
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    else:
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    print("args: ", args)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_u_all, dst_test = datasets.__dict__[args.dataset](args)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
    print("im_size: ", dst_train[0][0].shape)

    # Initialize Unlabeled Set & Labeled Set
    indices = list(range(len(dst_train)))
    random.shuffle(indices)

    labeled_set = indices[:args.n_query]
    unlabeled_set = indices[args.n_query:]

    dst_subset = torch.utils.data.Subset(dst_train, labeled_set)
    print("Initial set size: ", len(dst_subset))

    # BackgroundGenerator for ImageNet to speed up dataloaders
    if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
        train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # Get Model
    print("| Training on model %s" % args.model)
    network = get_model(args, nets, args.model)

    macs, params = get_model_complexity_info(network, (channel, im_size[0], im_size[1]), as_strings=True, print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # Active learning cycles
    logs = []
    for cycle in range(args.cycle):
        print("====================Cycle: {}====================".format(cycle+1))

        # Get optim configurations for Distrubted SGD
        criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)

        # Training
        print("==========Start Training==========")
        for epoch in range(args.epochs):
            # train for one epoch
            train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)

        acc = test(test_loader, network, criterion, epoch, args, rec)
        print('Cycle {}/{} || Label set size {}: Test acc {}'.format(cycle + 1, args.cycle, len(labeled_set), acc))
        
        # save logs
        logs.append([acc])
        if cycle == args.cycle-1:
            break

        # AL Query Sampling
        print("==========Start Querying==========")

        selection_args = dict(selection_method=args.uncertainty,
                            balance=args.balance,
                            greedy=args.submodular_greedy,
                            function=args.submodular,
                            )
        ALmethod = methods.__dict__[args.method](dst_u_all, unlabeled_set, network, args, **selection_args)
        Q_indices, Q_scores = ALmethod.select()

        # Update the labeled dataset and the unlabeled dataset, respectively
        for idx in Q_indices:
            labeled_set.append(idx)
            unlabeled_set.remove(idx)

        print("# of Labeled: {}, # of Unlabeled: {}".format(len(labeled_set), len(unlabeled_set)))
        assert len(labeled_set) == len(list(set(labeled_set))) and len(unlabeled_set) == len(list(set(unlabeled_set)))
        
        # Re-Configure Training of the Next Cycle
        network = get_model(args, nets, args.model)

        dst_subset = torch.utils.data.Subset(dst_train, labeled_set)
        if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
            train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        else:
            train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    print("Final acc logs")
    logs = np.array(logs).reshape((-1, 1))
    print(logs, flush=True)