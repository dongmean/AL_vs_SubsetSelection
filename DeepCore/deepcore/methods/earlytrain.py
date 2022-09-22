from .coresetmethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from copy import deepcopy
from .. import nets
from torchvision import transforms
from ptflops import get_model_complexity_info

class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.
    '''

    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 torchvision_pretrain: bool = False, dst_pretrain_dict: dict = {}, fraction_pretrain=1., dst_test=None,
                 **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model

        if fraction_pretrain <= 0. or fraction_pretrain > 1.:
            raise ValueError("Illegal pretrain fraction value.")
        self.fraction_pretrain = fraction_pretrain

        if dst_pretrain_dict.__len__() != 0:
            dict_keys = dst_pretrain_dict.keys()
            if 'im_size' not in dict_keys or 'channel' not in dict_keys or 'dst_train' not in dict_keys or \
                    'num_classes' not in dict_keys:
                raise AttributeError(
                    'Argument dst_pretrain_dict must contain imszie, channel, dst_train and num_classes.')
            if dst_pretrain_dict['im_size'][0] != args.im_size[0] or dst_pretrain_dict['im_size'][0] != args.im_size[0]:
                raise ValueError("im_size of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['channel'] != args.channel:
                raise ValueError("channel of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['num_classes'] != args.num_classes:
                self.num_classes_mismatch()

        self.dst_pretrain_dict = dst_pretrain_dict
        self.torchvision_pretrain = torchvision_pretrain
        self.if_dst_pretrain = (len(self.dst_pretrain_dict) != 0)

        if torchvision_pretrain:
            # Pretrained models in torchvision only accept 224*224 inputs, therefore we resize current
            # datasets to 224*224.
            if args.im_size[0] != 224 or args.im_size[1] != 224:
                self.dst_train = deepcopy(dst_train)
                self.dst_train.transform = transforms.Compose([self.dst_train.transform, transforms.Resize(224)])
                if self.if_dst_pretrain:
                    self.dst_pretrain_dict['dst_train'] = deepcopy(dst_pretrain_dict['dst_train'])
                    self.dst_pretrain_dict['dst_train'].transform = transforms.Compose(
                        [self.dst_pretrain_dict['dst_train'].transform, transforms.Resize(224)])
        if self.if_dst_pretrain:
            self.n_pretrain = len(self.dst_pretrain_dict['dst_train'])
        self.n_pretrain_size = round(
            self.fraction_pretrain * (self.n_pretrain if self.if_dst_pretrain else self.n_train))
        self.dst_test = dst_test

    def train(self, epoch, list_of_train_idx, **kwargs):
        """ Train model for one epoch """

        self.before_train()
        self.model.train()

        print('=> Training Epoch #%d' % epoch)
        trainset_permutation_inds = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(trainset_permutation_inds, batch_size=self.args.selection_batch,
                                                      drop_last=False)
        trainset_permutation_inds = list(batch_sampler)

        train_loader = torch.utils.data.DataLoader(self.dst_pretrain_dict['dst_train'] if self.if_dst_pretrain
                                                   else self.dst_train, shuffle=False, batch_sampler=batch_sampler,
                                                   num_workers=self.args.workers, pin_memory=False)

        for i, data in enumerate(train_loader):
            inputs, targets = data[0], data[1]
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.after_loss(outputs, loss, targets, trainset_permutation_inds[i], epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

            loss.backward()
            self.model_optimizer.step()
        return self.finish_train()

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        # Setup model and loss
        self.model = nets.__dict__[self.args.core_model if self.specific_model is None else self.specific_model](
            self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
            pretrained=self.torchvision_pretrain,
            im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)

        print("Warm-up Model: {}".format(self.args.core_model))
        print("resolution: ", self.args.resolution)
        macs, params = get_model_complexity_info(self.model, (self.args.channel, self.args.resolution, self.args.resolution), as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('MACs: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        if self.args.device == "cpu":
            print("Using CPU.")
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu[0])
            self.model = nets.nets_utils.MyDataParallel(self.model, device_ids=self.args.gpu)
        elif torch.cuda.device_count() > 1:
            self.model = nets.nets_utils.MyDataParallel(self.model).cuda()

        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.criterion.__init__()

        # Setup optimizer
        if self.args.selection_optimizer == "SGD":
            self.model_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.selection_lr,
                                                   momentum=self.args.selection_momentum,
                                                   weight_decay=self.args.selection_weight_decay,
                                                   nesterov=self.args.selection_nesterov)
        elif self.args.selection_optimizer == "Adam":
            self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.selection_lr,
                                                    weight_decay=self.args.selection_weight_decay)
        else:
            self.model_optimizer = torch.optim.__dict__[self.args.selection_optimizer](self.model.parameters(),
                                                                       lr=self.args.selection_lr,
                                                                       momentum=self.args.selection_momentum,
                                                                       weight_decay=self.args.selection_weight_decay,
                                                                       nesterov=self.args.selection_nesterov)

        self.before_run()
        start_train_time = time.time() #
        for epoch in range(self.epochs):
            list_of_train_idx = np.random.choice(np.arange(self.n_pretrain if self.if_dst_pretrain else self.n_train),
                                                 self.n_pretrain_size, replace=False)
            self.before_epoch()
            self.train(epoch, list_of_train_idx)
            self.after_epoch()
            
        if self.dst_test is not None and self.args.selection_test_interval > 0 :
            warmup_test_acc = self.test(epoch)
        print("Warmup Training Time: ", time.time()-start_train_time) #

        return self.finish_run(), warmup_test_acc

    def test(self, epoch):
        self.model.no_grad = True
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(self.dst_test if self.args.selection_test_fraction == 1. else
                                                  torch.utils.data.Subset(self.dst_test, np.random.choice(
                                                      np.arange(len(self.dst_test)),
                                                      round(len(self.dst_test) * self.args.selection_test_fraction),
                                                      replace=False)),
                                                  batch_size=self.args.selection_batch, shuffle=False,
                                                  num_workers=self.args.workers, pin_memory=False)
        correct = 0.
        total = 0.

        print('=> Warm-up Testing Epoch #%d' % epoch)
        for batch_idx, data in enumerate(test_loader):
            input, target = data[0], data[1]
            output = self.model(input.to(self.args.device))
            loss = self.criterion(output, target.to(self.args.device)).sum()

            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
        warmup_test_acc = 100. * correct / total
        print(' Test Acc: %.3f%%' % warmup_test_acc)
        self.model.no_grad = False
        return warmup_test_acc

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass
    
    def select(self, **kwargs):
        selection_result, warmup_test_acc = self.run()
        return selection_result, warmup_test_acc
