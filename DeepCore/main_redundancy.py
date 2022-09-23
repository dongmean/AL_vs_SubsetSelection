import os
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep
import numpy as np

# custom
from arguments import parser
from ptflops import get_model_complexity_info
import copy

def main():
    # parse arguments
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

    args, checkpoint, start_exp, start_epoch = get_more_args(args)

    for exp in range(start_exp, args.num_exp):
        # Get checkpoint if have
        if args.save_path != "":
            checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_se{se}_{dat}_fr{fr}_".format(dst=args.dataset,
                                                                                         net=args.model,
                                                                                         mtd=args.selection,
                                                                                         dat=datetime.now(),
                                                                                         exp=exp,
                                                                                         se=args.selection_epochs,
                                                                                         fr=args.fraction)

        print('\n================== Exp %d ==================' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_ex: ",
              args.num_exp, ", epochs: ", args.epochs, ", fraction: ", args.fraction, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ", resume: ", args.resume, ", device: ", args.device,
              ", checkpoint_name: " + checkpoint_name if args.save_path != "" else "", "\n", sep="")
        
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset](args)
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
        torch.random.manual_seed(exp+args.seed) # Should change this for changing seed

        # Generate Redundancy in Training set
        n_redundant = 10
        '''
        indices_subset = list(range(256))
        dataset_list = []
        for i in range(n_redundant):
            dst_clone = copy.deepcopy(dst_train)
            dst_clone.update_clone_ver(i)
            dst_train_subset = torch.utils.data.Subset(dst_clone, indices_subset)
            
            dataset_list.append(dst_train_subset)
        dataset_list = [dst_train_subset for i in range(n_redundant)]
        '''
        dataset_list = [dst_train for i in range(n_redundant)]
        dst_train = torch.utils.data.ConcatDataset(dataset_list)
        print("len(dst_train) with redundancy", len(dst_train))

        # Core-set Selection
        if "subset" in checkpoint.keys():
            subset = checkpoint['subset']
            selection_args = checkpoint["sel_args"]
        else:
            selection_args = dict(epochs=args.selection_epochs,
                                  selection_method=args.uncertainty,
                                  balance=args.balance,
                                  greedy=args.submodular_greedy,
                                  function=args.submodular,
                                  dst_test = dst_test,
                                  dst_all = dst_train
                                  )
            method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
            start_time = time.time()
            ##### Main Function #####
            subset, warmup_test_acc = method.select()
            assert len(subset["indices"]) == len(list(set(subset["indices"])))
            print("(should be unordered) subset[:10]:", subset["indices"][:10])
            core_selection_time = time.time() - start_time
            print("Elapsed Time: ", core_selection_time)

        logs = []
        n_query = 50000
        n_grid = int(len(dst_train)/n_query)
        print("n_grid: ", n_grid)
        for g in range(n_grid):
            indices = subset["indices"][:(g+1)*n_query]

            indices_no_redun = list(set(indices % len(dst_train)))
            print("len(indices_no_redun): ", len(indices_no_redun))

            dst_subset = torch.utils.data.Subset(dst_train, indices_no_redun) #indices

            print("# of Train: ", len(dst_subset))
            # BackgroundGenerator for ImageNet to speed up dataloaders
            if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
                train_loader = DataLoaderX(dst_subset, batch_size=args.batch, shuffle=True,
                                        num_workers=args.workers, pin_memory=False)
                test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False,
                                        num_workers=args.workers, pin_memory=False)
            else:
                train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch, shuffle=True,
                                                        num_workers=args.workers, pin_memory=False)
                test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch_size, shuffle=False,
                                                        num_workers=args.workers, pin_memory=False)

            # Get configurations for Distrubted SGD
            # Get Model
            print("| Training on model %s" % args.model)
            network = get_model(args, nets, args.model)
            if g == 0:
                macs, params = get_model_complexity_info(network, (3, args.im_size[0], args.im_size[1]), as_strings=True, print_per_layer_stat=False, verbose=False)
                print('{:<30}  {:<8}'.format('MACs: ', macs))
                print('{:<30}  {:<8}'.format('Number of parameters: ', params))

            # Get optim configurations for Distrubted SGD
            criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)

            ##### Training #####
            best_prec1 = 0
            for epoch in range(args.epochs):
                # train for one epoch
                train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec)

                # evaluate on validation set
                prec1 = test(test_loader, network, criterion, epoch, args, rec)

                # remember best prec@1 and save checkpoint
                best_prec1 = max(prec1, best_prec1)
            print('| Best accuracy: ', best_prec1)
            logs.append(best_prec1)

            path = 'result/cifar10/'+args.selection+'.txt' #TODO: remove ../ which is for cifar.sh
            if g == 0:
                f = open(path, 'w')
            else:
                f = open(path, 'a')
            f.write(str(best_prec1)+'\n')  # python will convert \n to os.linesep
            f.close()

            print("len(indices): ", len(indices))
            print("len(indices_no_redun): ", len(indices_no_redun))
            break

        print("Final acc logs")
        logs = np.array(logs).reshape((-1, 1))
        print(logs, flush=True)    


if __name__ == '__main__':
    main()
