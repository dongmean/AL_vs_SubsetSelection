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

# custom
from arguments import parser
from ptflops import get_model_complexity_info

def main():
    # parse arguments
    args = parser.parse_args()
    gpus = ""
    for i, g in enumerate(args.gpu):
        gpus = gpus+str(g)
        if i != len(args.gpu)-1:
            gpus = gpus+","
    state = {k: v for k, v in args._get_kwargs()}
    if args.dataset == 'ImageNet':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        args.device = 'cuda:'+str(gpus) if torch.cuda.is_available() else 'cpu'

    args, checkpoint, start_exp, start_epoch = get_more_args(args)

    for exp in range(start_exp, args.num_exp):

        exp = exp+1 #TOFIXTOFIXTOFIX

        # Get checkpoint if have
        if args.save_path != "":
            checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_se{se}_{dat}_arch{arch}_fr{fr}_".format(dst=args.dataset,
                                                                                         net=args.model,
                                                                                         mtd=args.selection,
                                                                                         dat=datetime.now(),
                                                                                         exp=exp,
                                                                                         se=args.selection_epochs,
                                                                                         fr=args.fraction,
                                                                                         arch=args.core_model)

        print('\n================== Exp %d ==================' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_ex: ",
              args.num_exp, ", epochs: ", args.epochs, ", fraction: ", args.fraction, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ", resume: ", args.resume, ", device: ", args.device,
              ", checkpoint_name: " + checkpoint_name if args.save_path != "" else "", "\n", sep="")
        
        args.resolution = args.core_resolution
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset](args)
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
        torch.random.manual_seed(exp+args.seed) #
        print("im_size for coreset: ", dst_train[0][0].shape)

        # Core-set Selection in low-resolution
        if "subset" in checkpoint.keys():
            subset = checkpoint['subset']
            selection_args = checkpoint["sel_args"]
        else:
            selection_args = dict(epochs=args.selection_epochs,
                                  selection_method=args.uncertainty,
                                  balance=args.balance,
                                  greedy=args.submodular_greedy,
                                  function=args.submodular,
                                  dst_test = dst_test
                                  )
            method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
            start_time = time.time()
            ##### Main Function #####
            subset, warmup_test_acc = method.select()

            core_selection_time = time.time() - start_time
            print("Elapsed Time: ", core_selection_time)

        # Training in high-resolution
        args.resolution = 224
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset](args)
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
        torch.random.manual_seed(exp+args.seed) #

        print("im_size for training: ", dst_train[0][0].shape)

        # Handle weighted subset
        if_weighted = "weights" in subset.keys()
        #if if_weighted:
        #    dst_subset = WeightedSubset(dst_train, subset["indices"], subset["weights"])
        #else:
        dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])

        # BackgroundGenerator for ImageNet to speed up dataloaders
        if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
            train_loader = DataLoaderX(dst_subset, batch_size=args.train_batch, shuffle=True,
                                       num_workers=args.workers, pin_memory=False)
            test_loader = DataLoaderX(dst_test, batch_size=args.train_batch, shuffle=False,
                                      num_workers=args.workers, pin_memory=False)
        else:
            train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                       num_workers=args.workers, pin_memory=False)
            test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                      num_workers=args.workers, pin_memory=False)

        # Listing cross-architecture experiment settings if specified.
        models = [args.model]
        if isinstance(args.cross, list):
            for model in args.cross:
                if model != args.model:
                    models.append(model)

        # Model Training
        for model in models:
            print("| Training on model %s" % model)

            # Get configurations for Distrubted SGD
            network, criterion, optimizer, scheduler, rec = get_fresh_configuration(args, nets, model, train_loader, start_epoch)
            print("Main Model: {}".format(args.model))
            macs, params = get_model_complexity_info(network, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False)
            print('{:<30}  {:<8}'.format('MACs: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
                
            best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys() else 0.0

            # Save the checkpont with only the susbet.
            if args.save_path != "" and args.resume == "":
                save_checkpoint({"exp": exp,
                                 "subset": subset,
                                 "sel_args": selection_args},
                                os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model
                                             + "_") + "unknown.ckpt"), 0, 0.)

            ##### Training #####
            for epoch in range(start_epoch, args.epochs):
                # train for one epoch
                train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False) #if_weighted

                # evaluate on validation set
                if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                    prec1 = test(test_loader, network, criterion, epoch, args, rec)

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1

                    if is_best:
                        best_prec1 = prec1
                        if args.save_path != "":
                            rec = record_ckpt(rec, epoch)
                            save_checkpoint({"exp": exp,
                                             "epoch": epoch + 1,
                                             #"state_dict": network.state_dict(),
                                             #"opt_dict": optimizer.state_dict(),
                                             "warmup_acc": warmup_test_acc,
                                             "best_acc1": best_prec1,
                                             "rec": rec,
                                             "subset": subset,
                                             "elapsed_time": core_selection_time,
                                             "sel_args": selection_args},
                                            os.path.join(args.save_path, checkpoint_name + (
                                                "" if model == args.model else model + "_") + "unknown.ckpt"),
                                            epoch=epoch, prec=best_prec1)

            # Prepare for the next checkpoint
            if args.save_path != "":
                try:
                    os.rename(
                        os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model + "_") +
                                     "unknown.ckpt"), os.path.join(args.save_path, checkpoint_name +
                                     ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1))
                except:
                    save_checkpoint({"exp": exp,
                                     "epoch": args.epochs,
                                     #"state_dict": network.state_dict(),
                                     #"opt_dict": optimizer.state_dict(),
                                     "best_acc1": best_prec1,
                                     "rec": rec,
                                     "subset": subset,
                                     "sel_args": selection_args},
                                    os.path.join(args.save_path, checkpoint_name +
                                                 ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1),
                                    epoch=args.epochs - 1,
                                    prec=best_prec1)

            print('| Best accuracy: ', best_prec1, ", on model " + model if len(models) > 1 else "", end="\n\n")
            start_epoch = 0
            checkpoint = {}
            sleep(2)


if __name__ == '__main__':
    main()
