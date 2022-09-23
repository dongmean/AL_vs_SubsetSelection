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

def rank_uncertainty(args, model, dst_train, index=None):
    model.eval()
    with torch.no_grad():
        coreset_loader = torch.utils.data.DataLoader(
                dst_train if index is None else torch.utils.data.Subset(dst_train, index),
                batch_size=args.selection_batch, shuffle=False,
                num_workers=args.workers)

        scores = np.array([])
        for i, data in enumerate(coreset_loader):
            input = data[0]
            if i % args.print_freq == 0:
                print("| Selecting for batch [%3d/%3d]" % (i + 1, len(coreset_loader)))
            if args.uncertainty == "LeastConfidence":
                scores = np.append(scores, model(input.to(args.device)).max(axis=1).values.cpu().numpy())
            elif args.uncertainty == "Entropy":
                preds = torch.nn.functional.softmax(model(input.to(args.device)), dim=1).cpu().numpy()
                scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
            elif args.uncertainty == 'Margin':
                preds = torch.nn.functional.softmax(model(input.to(args.device)), dim=1)
                preds_argmax = torch.argmax(preds, dim=1)
                max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                preds_sub_argmax = torch.argmax(preds, dim=1)
                scores = np.append(scores, (max_preds - preds[
                    torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
    return scores

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

        # Get configurations for Distrubted SGD
        # Get Model
        print("| Training on model %s" % args.model)
        network = get_model(args, nets, args.model)

        # Get optim configurations for Distrubted SGD
        criterion, optimizer, scheduler, rec = get_optim_configurations_epochs(args, network)

        # BackgroundGenerator for ImageNet to speed up dataloaders
        if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
            test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        else:
            test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

        n_query = 500
        n_grid = int(len(dst_train)/n_query)
        print("n_grid: ", n_grid)

        coreset_indices = np.array(list(range(len(dst_train))))

        ##### Pruning #####
        pruned_indices = np.array([]).astype(int)
        logs = []
        best_prec1 = 0
        for epoch in range(args.pruning_epochs):
            dst_subset = torch.utils.data.Subset(dst_train, coreset_indices)
            print('==================Epoch: {}, Pruned Ratio: {}=================='.format(epoch, float(len(dst_subset)/len(dst_train))))
            print("# of Train: ", len(dst_subset))

            if len(dst_subset) == 0:
                break

            # BackgroundGenerator for ImageNet to speed up dataloaders
            if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
                train_loader = DataLoaderX(dst_subset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=False)
            else:
                train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=False)
                
            # train for one epoch
            train_epoch(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec)

            # evaluate on validation set
            prec1 = test(test_loader, network, criterion, epoch, args, rec)
            print("acc: ", prec1)

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)

            # Data Pruning
            if epoch >= 10:
                scores = rank_uncertainty(args, network, dst_train, coreset_indices)
                uncertain_indices = np.argsort(scores)

                pruned_indices = np.append(pruned_indices, coreset_indices[uncertain_indices[::-1][:n_query]], 0) #Easy First
                coreset_indices = coreset_indices[uncertain_indices[:-n_query]]

                assert len(coreset_indices) == len(list(set(coreset_indices)))
                assert len(pruned_indices) == len(list(set(pruned_indices)))

            logs.append(prec1)
        print('| Best accuracy: ', best_prec1)
        print("Pruning acc logs")
        logs = np.array(logs).reshape((-1, 1))
        print(logs, flush=True)

        # Training
        logs = []
        n_query = 1000
        n_grid = int(len(dst_train)/n_query)
        print("n_grid: ", n_grid)
        for g in range(n_grid):
            indices = pruned_indices[::-1][:(g+1)*n_query]
            dst_subset = torch.utils.data.Subset(dst_train, indices)

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

            path = 'result/cifar10/'+args.selection+'_gradual_fast_v2_epoch100.txt'
            if g == 0:
                f = open(path, 'w')
            else:
                f = open(path, 'a')
            f.write(str(best_prec1)+'\n')  # python will convert \n to os.linesep
            f.close()

        print("Final acc logs")
        logs = np.array(logs).reshape((-1, 1))
        print(logs, flush=True)    

if __name__ == '__main__':
    main()