from .earlytrain import EarlyTrain
import torch
#import torch.nn.functional as F
import numpy as np
import time
import copy
from utils import *
import deepcore.nets as nets
from ptflops import get_model_complexity_info

class WeightNet(nn.Module):
    def __init__(self, input, hidden1, output): # size
        super(WeightNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)

        return (torch.tanh(out)+1)/2 # 0~1

class GetIndicators(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
            In the forward pass we receive a Tensor containing the input and return
            a Tensor containing the output. ctx is a context object that can be used
            to stash information for backward computation. You can cache arbitrary
            objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # this is the correct way to do it (torch.gte also works)
        mask = torch.autograd.Variable((x >= 0.5).float(), requires_grad=True)
        # if you do this: mask = (x > 5).int(), then pytorch breaks the computational graph
        # and no longer calls .backward()!
        return mask
    
    @staticmethod
    def backward(ctx, g):
        """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
            NOTE: We need only "g" here because we have only one input -> x
        """
        # NOTE: This isn't getting printed! I don't know why
        # print("Incoming grad = Outgoing grad = {}".format(g))
        # send the gradient g straight-through on the backward pass.
        return g


class BiLevelMeta(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model=None, balance=False, dst_all=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        self.balance = balance
        #self.W = torch.nn.Parameter(torch.zeros(len(dst_train), device=self.args.device)+0.5) # Constant 0.5
        #self.W = torch.nn.Parameter(torch.rand(len(dst_train), device=self.args.device))# Uniform [0,1)
        #self.opt_W = torch.optim.SGD(self.W,lr=1e-3,weight_decay=5e-4)

        self.dst_all = dst_all

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def get_model(self, args, nets, model):
        network = nets.__dict__[model](args.channel, args.num_classes, args.im_size, penultimate=args.penultimate).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu[0])
            network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
        elif torch.cuda.device_count() > 1:
            network = nets.nets_utils.MyDataParallel(network).cuda()

        return network

    # NOTE: overriding run() with cosine_lr_decay
    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        # Setup model and loss
        self.model = nets.__dict__[self.args.core_model if self.specific_model is None else self.specific_model](
            self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
            pretrained=self.torchvision_pretrain,
            im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size,
            penultimate=self.args.penultimate).to(self.args.device)

        print("Warm-up Model: {}".format(self.args.core_model))
        print("resolution: ", self.args.resolution)
        macs, params = get_model_complexity_info(self.model, (3, self.args.resolution, self.args.resolution), as_strings=True,
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

        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.args.device)

        # Setup dataloader
        train_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=True, batch_size=self.args.train_batch, num_workers=self.args.workers, pin_memory=False)
        
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
        # LR scheduler
        if self.args.meta_scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, len(train_loader) * self.args.meta_epochs,
                                                                    eta_min=self.args.min_lr)
        elif self.args.meta_scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.model_optimizer, step_size=len(train_loader) * int(self.args.meta_epochs+1),
                                                        gamma=self.args.gamma)
        else:
            self.scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](self.model_optimizer)        

        self.before_run()
        start_train_time = time.time() #
        for epoch in range(self.epochs):
            self.before_epoch()
            self.train(epoch, train_loader)
            self.after_epoch()
            
        if self.dst_test is not None and self.args.selection_test_interval > 0 :
            warmup_test_acc = self.test(epoch)
        print("Warmup Training Time: ", time.time()-start_train_time) #

        return self.finish_run(), warmup_test_acc

    # NOTE: overriding train() with scheduler.step()
    def train(self, epoch, train_loader, **kwargs):
        """ Train model for one epoch """
        self.before_train()
        self.model.train()

        print('=> Training Epoch #%d' % epoch)
        for i, data in enumerate(train_loader):
            inputs, targets = data[0], data[1]
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

            loss.backward()
            self.model_optimizer.step()
            self.scheduler.step()
        return self.finish_train()
    
    # NOTE: overriding test() with out_cnn
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
        for batch_idx, data in enumerate(test_loader):
            input, target = data[0], data[1]
            output, _ = self.model(input.to(self.args.device))
            loss = self.criterion(output, target.to(self.args.device)).sum()

            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
        warmup_test_acc = 100. * correct / total
        print('Epoch: {}, Test Acc: {}'.format(epoch, warmup_test_acc))
        self.model.no_grad = False
        return warmup_test_acc

    def finish_run(self):
        self.bilevel_optimization()
        
        mask, score = self.get_selected_indices()
        print("# selected_indices: ", sum(mask))

        indices = np.arange(len(self.dst_train))
        # TODO:
        if self.balance:
            print("No balance version yet")
            selection_result = indices[mask]
        else:
            selection_result = indices[mask]
        return {"indices": selection_result, "scores": score}

    def get_selected_indices(self):
        select_loader = torch.utils.data.DataLoader(self.dst_train, batch_size=self.args.test_batch_size, num_workers=self.args.workers, pin_memory=False)
     
        score, mask = np.array([]), np.array([], dtype=bool)
        for i, data in enumerate(select_loader):
            inputs = data[0].to(self.args.device)

            _, embs = self.model(inputs)

            score_batch = self.W(embs)
            mask_batch = score_batch.ge(0.5).reshape(-1).cpu()
            
            score = np.append(score, score_batch.reshape(-1).cpu().data)
            mask = np.append(mask, mask_batch)

        return mask, score
        
            
    def bilevel_optimization(self):
        self.model.train()
        emb_dim = 512 #TODO: Need to fix with .shape[]
        self.W = WeightNet(emb_dim, 100, 1).to(self.args.device)
        self.W.train()
        self.opt_W = torch.optim.SGD(self.W.parameters() ,lr=0.01, weight_decay=0)

        self.inner_train_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=True, drop_last=False,
                                                batch_size=self.args.train_batch, num_workers=self.args.workers, pin_memory=False)
        self.outer_train_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=True, drop_last=False, 
                                                batch_size=self.args.train_batch, num_workers=self.args.workers, pin_memory=False)

        cur_lamda = self.args.lamda
        print("meta_epochs: ", self.args.meta_epochs)
        for epoch in range(self.epochs, self.args.meta_epochs):
            self.bilevel_train(self.inner_train_loader, self.outer_train_loader, epoch, cur_lamda)
            self.test(epoch)

    def bilevel_train(self, inner_loader, outer_loader, epoch, cur_lamda):
        self.model.train()
        total_selected = 0
        for i, data in enumerate(inner_loader):
            ### Update w ###
            try:
                inputs_val, targets_val, idxs_val = next(outer_loader_iter)
            except:
                outer_loader_iter = iter(outer_loader)
                inputs_val, targets_val, idxs_val = next(outer_loader_iter)

            inputs_val, targets_val, idxs_val = inputs_val.to(self.args.device), targets_val.to(self.args.device), idxs_val.to(self.args.device)
            
            model_copy = self.get_model(self.args, nets, self.args.model)
            model_copy.load_state_dict(self.model.state_dict())

            outputs_val, embs_val = model_copy(inputs_val)
            ce_loss_val = self.criterion(outputs_val, targets_val)

            #print(self.W(embs_val)[:10])

            # loss
            mask_val = GetIndicators.apply(self.W(embs_val)).reshape(-1)

            masked_ce_loss_val = (mask_val*ce_loss_val).mean()

            model_copy.zero_grad()
            grads = torch.autograd.grad(masked_ce_loss_val, (model_copy.params()), create_graph=True)
            backbone_lr = self.scheduler.get_last_lr()[0]

            if i ==0:
                print("backbone_lr: ", backbone_lr)

            # Making Gradient Flow
            model_copy.update_params(lr_inner=backbone_lr, source_params=grads)
            del grads

            outputs_val, embs_val = model_copy(inputs_val)
            ce_loss_val = self.criterion(outputs_val, targets_val)

            reg_loss = torch.norm(self.W(embs_val), p=1)

            loss = ce_loss_val.mean() + cur_lamda*reg_loss #(1/backbone_lr)*

            self.opt_W.zero_grad()
            loss.backward()
            self.opt_W.step()

            ### Update \theta ###
            inputs, targets, idxs = data[0], data[1], data[2]
            inputs, targets, idxs = inputs.to(self.args.device), targets.to(self.args.device), idxs.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs, embs = self.model(inputs)

            # loss
            ce_loss = self.criterion(outputs, targets)
            mask = GetIndicators.apply(self.W(embs)).reshape(-1)
            masked_ce_loss = (mask*ce_loss).mean()

            loss = masked_ce_loss

            # backward propagate, update optimizer & scheduler
            loss.backward()
            self.model_optimizer.step()
            self.scheduler.step()

            total_selected += sum(mask)
        print("# selected: ", total_selected)

        debug_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=False, batch_size=10, num_workers=self.args.workers, pin_memory=False)
        for i, data in enumerate(debug_loader):
            inputs = data[0].to(self.args.device)
            outputs, embs = self.model(inputs)
            print(self.W(embs).reshape(-1))
            break