from .earlytrain import EarlyTrain
import torch
import numpy as np
import time
import copy
from utils import *
import deepcore.nets as nets
from ptflops import get_model_complexity_info

# Pairwise Loss
def PairwiseMaskingLoss(losses, masks):
    loss = 0
    t = time.time()
    for i, li in enumerate(losses):
        mi = masks[i]
        for j, lj in enumerate(losses):
            mj = masks[j]
            mij = mi*mj

            loss += mij*(li+lj) + (1-mij)*(mi*li + mj*lj)
    loss = loss/(len(losses)*len(losses))
    print("PairwiseMaskingLoss: ", loss)
    print("Elapsed Time: ", time.time()-t)
    return loss

def FastPairwiseMaskingLoss(losses, masks):
    relu_masks = torch.relu(masks-0.5)
    mask_matrix = GetIndicators.apply(relu_masks*(relu_masks.reshape(-1,1))+0.5)
    masks = GetIndicators.apply(masks)

    loss = 0
    n_batch = len(losses)
    t = time.time()
    for i, li in enumerate(losses):
        mij_vec = mask_matrix[i]
        mi = masks[i].repeat(n_batch)
        mj = masks
        li_vec = li.repeat(n_batch)
        lj_vec = losses
        
        losses_batch = mij_vec*(li_vec+lj_vec)+(1-mij_vec)*(mi*li_vec+mj*lj_vec)
        losses_batch[i] = 0

        loss += losses_batch.sum()

    loss = loss/(len(losses)*len(losses))
    #print("FasterPairwiseMaskingLoss: ", loss)
    #print("Elapsed Time: ", time.time()-t)
    return loss

def DiversityLoss(embs, masks, n_class):
    relu_masks = torch.relu(masks-0.5)
    mask_matrix = GetIndicators.apply(relu_masks*(relu_masks.reshape(-1,1))+0.5)
    masks = GetIndicators.apply(masks)

    loss = 0
    n_batch = len(embs)
    t = time.time()
    for i, emb_i in enumerate(embs):
        mij_vec = mask_matrix[i] #[B]
        mi = masks[i].repeat(n_batch)
        mj = masks
        emb_i_vec = emb_i.repeat(n_batch, 1) #[B,emb_dim]
        emb_j_vec = embs

        Cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        dis = 1-Cos(emb_i_vec, emb_j_vec) # [B]
        dis = torch.max(dis, torch.tensor(1/n_class))

        losses_batch = mij_vec*(-torch.log(dis)) # [128]
        losses_batch[i] = 0

        loss += losses_batch.sum()

    loss = loss/(len(embs)*(len(embs)-1))
    #print("DiversityLoss: ", loss)
    #print("Elapsed Time: ", time.time()-t)
    return loss

class MaskWeight(nn.Module):
    def __init__(self, num_train, num_class):
        super().__init__()
        self.num_train = num_train
        self.W = torch.zeros(self.num_train)+0.5
        #const_idx = torch.randperm(num_train)[:int(num_train/100)]
        #self.W[const_idx] = 1
        self.W = torch.nn.Parameter(self.W)

        #self.W = torch.nn.Parameter(torch.rand(num_train)/10+0.5)

    def forward(self, x, idx): # x is L_i
        mask = GetIndicators.apply(self.W[idx])

        # TODO: Gradient not fluids, What's the problem? Update params?
        #if 0 in idx:
        #    print(self.W[0:10])
        #print(idx)

        out = mask*x
        assert mask.sum() % 1 == 0

        #out = x*self.W[idx]*self.W[idx]
        return out
    
    def get_l1_norm(self, idx):
        loss = torch.norm(self.W[idx], p=1)
        loss = loss/len(idx)
        return loss
    
    def get_weight(self):
        return self.W
    
    def get_weight_by_idx(self, idx):
        return self.W[idx]

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
        mask = torch.autograd.Variable((x > 0.5).float(), requires_grad=True)
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


class BiLevel2Step(EarlyTrain):
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

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

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
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.model_optimizer, step_size=len(train_loader) * self.args.step_size,
                                                        gamma=self.args.gamma)
        else:
            self.scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](self.model_optimizer)        

        self.before_run()
        start_train_time = time.time()
        for epoch in range(self.epochs): #selection_epochs
            self.before_epoch()
            self.train(epoch, train_loader)
            self.after_epoch()
        
        warmup_test_acc = 0
        if self.dst_test is not None and self.args.selection_test_interval > 0 and self.epochs>0:
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
        
        selected = self.W.get_weight().ge(0.5).cpu()
        print("# selected_indices: ", sum(selected))

        indices = np.arange(len(self.dst_train))
        # TODO:
        if self.balance:
            print("No balance version yet")
            selection_result = indices[selected]
        else:
            selection_result = indices[selected]
        return {"indices": selection_result, "scores": self.W.get_weight()} #TODO: score[indices[selected]]

    def bilevel_optimization(self):
        self.model.train()
        self.W = MaskWeight(len(self.dst_train), self.args.n_class).to(self.args.device)
        self.W.train()
        self.opt_W = torch.optim.SGD(self.W.parameters(), lr=0.1, weight_decay=0)

        self.inner_train_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=True, drop_last=False,
                                                batch_size=self.args.train_batch, num_workers=self.args.workers, pin_memory=False)
        self.outer_train_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=True, drop_last=False, 
                                                batch_size=self.args.train_batch, num_workers=self.args.workers, pin_memory=False)

        self.scheduler_W = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_W, len(self.outer_train_loader) * self.args.meta_epochs,
                                                                    eta_min=0.01)

        cur_lamda = self.args.lamda
        for epoch in range(self.epochs, self.args.meta_epochs): #self.args.meta_epochs
            self.bilevel_train(self.inner_train_loader, self.outer_train_loader, epoch, cur_lamda)
            if self.args.lamda_scheduling == True:
                cur_lamda = self.update_lambda(self.W.get_weight(), cur_lamda, epoch)
            self.test(epoch)
        
    def bilevel_train(self, inner_loader, outer_loader, epoch, cur_lamda):
        self.model.train()
        for i, data in enumerate(inner_loader):
            ### Update w ###
            try:
                inputs_val, targets_val, idxs_val = next(outer_loader_iter)
            except:
                outer_loader_iter = iter(outer_loader)
                inputs_val, targets_val, idxs_val = next(outer_loader_iter)

            inputs_val, targets_val, idxs_val = inputs_val.to(self.args.device), targets_val.to(self.args.device), idxs_val.to(self.args.device)
            
            model_copy = self.get_model(self.args, nets, self.args.model)
            model_copy.load_state_dict(self.model.state_dict()) #ERROR

            for i in range(2):
                outputs_val, embs_val = model_copy(inputs_val)
                ce_loss_val = self.criterion(outputs_val, targets_val)

                # loss
                if self.args.pairwise_mask == True:
                    masks_val = self.W.get_weight_by_idx(idxs_val)
                    masked_ce_loss_val = FastPairwiseMaskingLoss(ce_loss_val, masks_val)
                else:
                    masked_ce_loss_val = self.W(ce_loss_val, idxs_val).mean()

                model_copy.zero_grad()
                grads = torch.autograd.grad(masked_ce_loss_val, (model_copy.params()), create_graph=True)
                #print(grads)
                meta_lr = self.scheduler.get_last_lr()[0]

                # Making Gradient Flow
                model_copy.update_params(lr_inner=meta_lr, source_params=grads)
                del grads

            outputs_val, _ = model_copy(inputs_val)
            ce_loss_val = self.criterion(outputs_val, targets_val)

            div_loss = torch.tensor(0)
            if self.args.diversity == True:
                masks_val = self.W.get_weight_by_idx(idxs_val)
                div_loss = DiversityLoss(embs_val, masks_val, self.args.n_class)

            reg_loss = self.W.get_l1_norm(idxs_val)

            #if epoch > self.epochs:
            #    print(ce_loss_val.mean().data, div_loss.data, reg_loss.data)

            loss = ce_loss_val.mean() + self.args.lamda2*div_loss + cur_lamda*reg_loss

            # NOTE: to check if self.W is not connected in loss.backward graph
            #grads = torch.autograd.grad(loss, (self.W.parameters()), create_graph=True)
            #print(grads[0][idxs_val]) # NOTE: This is non-zero now

            self.opt_W.zero_grad()
            loss.backward()
            self.opt_W.step()
            self.scheduler_W.step()

            ### Update \theta ###
            inputs, targets, idxs = data[0], data[1], data[2]
            inputs, targets, idxs = inputs.to(self.args.device), targets.to(self.args.device), idxs.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs, embs = self.model(inputs)

            # loss
            ce_loss = self.criterion(outputs, targets)

            if self.args.pairwise_mask == True:
                masks = self.W.get_weight_by_idx(idxs)
                masked_ce_loss = FastPairwiseMaskingLoss(ce_loss, masks)
            else:
                masked_ce_loss = self.W(ce_loss, idxs).mean()
            loss = masked_ce_loss

            # backward propagate, update optimizer & scheduler
            loss.backward()
            self.model_optimizer.step()
            #self.scheduler.step()

        print("# selected: ", self.W.get_weight().ge(0.5).sum())
        print(self.W.get_weight()[:10])


    def update_lambda(self, W, cur_lamda, epoch):
        mask = GetIndicators.apply(W)
        cur_fraction = mask.mean()

        if cur_fraction > self.args.fraction:
            print("Epoch: {}, Cur_Fr: {}, Cur_Lambda: {}".format(epoch, cur_fraction, cur_lamda))
            cur_lamda = cur_lamda * 2
        else:
            print("On Target Fraction!!! Epoch: {}, Cur_Fr: {}, Cur_Lambda: {}".format(epoch, cur_fraction, cur_lamda))
            cur_lamda = self.args.lamda
            print("new cur_lambda: ", cur_lamda)
        return cur_lamda
        
