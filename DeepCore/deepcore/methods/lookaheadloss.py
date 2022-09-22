from .earlytrain import EarlyTrain
import torch
import numpy as np
import time
import copy
from utils import *
import deepcore.nets as nets

class LookAheadLoss(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model=None, balance=False, dst_all=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        self.balance = balance
        self.history = 1
        self.LAloss = np.zeros((len(dst_train), self.history))
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

    def finish_run(self):
        if self.balance:
            print("No balance version yet")
            scores = self.rank_lookaheadloss()
            selection_result = np.argsort(scores)[:self.coreset_size]
        else:
            st = time.time()
            scores = self.rank_lookaheadloss()
            et = time.time()
            print("Elapsed Time for LookAheadLoss: ", et-st)

            selection_result = np.argsort(scores)[:self.coreset_size]
        return {"indices": selection_result, "scores": scores}

    def rank_lookaheadloss(self, index=None):
        loss_batch_size = 1
        train_loader = torch.utils.data.DataLoader(self.dst_train, shuffle=False, batch_size=loss_batch_size, num_workers=self.args.workers)
        
        val_frac = 0.1
        indices = np.random.choice(len(self.dst_all), int(len(self.dst_all)*val_frac))
        dst_subset = torch.utils.data.Subset(self.dst_all, indices)
        val_loader = torch.utils.data.DataLoader(dst_subset, batch_size=self.args.test_batch_size, num_workers=self.args.workers)      
        #val_loader = torch.utils.data.DataLoader(self.dst_all, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        
        for h in range(self.history):
            print("history: ", h)
            st = time.time()
            for i, data in enumerate(train_loader):
                if i % 1000 == 0:
                    print("batch: ", i)
                    print("Appr time: ", time.time()-st)
                    st = time.time()
                # model copy
                model_copy = get_model(self.args, nets, self.args.model)
                model_copy.load_state_dict(self.model.state_dict())
                opt_copy = torch.optim.SGD(model_copy.parameters(), 0.01, weight_decay=self.args.weight_decay)
                #opt_copy.load_state_dict(self.model_optimizer.state_dict()) #When batch size=1, model_copy becomes like a random classifier after updating
                
                # input processing
                inputs, targets, index = data[0].to(self.args.device), data[1].to(self.args.device), data[2]
                
                opt_copy.zero_grad()
                outputs = model_copy(inputs)

                loss = self.criterion(outputs, targets)
                loss = loss.mean()

                loss.backward()
                opt_copy.step()

                # update LookAhead loss
                self.update_LAloss(index, h, model_copy, val_loader)

        #assert len(np.where(self.LAloss==0)[0])==0 # acc may not be zero
        scores = self.LAloss.sum(axis=1)
        print("scores.shape: ", scores.shape)
        print(scores[0:30])

        return scores
    
    def update_LAloss(self, index, cur_history, model_copy, val_loader):
        mode = 'acc' #acc, loss
        if mode == 'acc':
            val_acc = self.calc_val_acc(model_copy, val_loader)
            self.LAloss[index, cur_history] = val_acc
        elif mode == 'loss':
            val_loss = self.calc_val_loss(model_copy, val_loader)
            self.LAloss[index, cur_history] = val_loss
    
    def calc_val_acc(self, model_copy, val_loader):
        total, correct = 0, 0
        t = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets = data[0].to(self.args.device), data[1].to(self.args.device)

            outs = model_copy(inputs)

            _, preds = torch.max(torch.nn.functional.softmax(outs, dim=1), 1)

            total += targets.size(0)
            correct += (preds == targets).sum().item()
        #print("Time for calculating val acc: ", time.time()-t)
        
        acc = correct/total
        return acc 
    
    def calc_val_loss(self, model_copy, val_loader):
        loss= 0
        criterion = nn.CrossEntropyLoss().to(self.args.device)
        t = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets = data[0].to(self.args.device), data[1].to(self.args.device)

            outs = model_copy(inputs)

            loss += criterion(outs, targets)
        loss = loss/len(val_loader)

        return loss.cpu().detach()

    def select(self, **kwargs):
        selection_result, warmup_test_acc = self.run()
        return selection_result, warmup_test_acc
