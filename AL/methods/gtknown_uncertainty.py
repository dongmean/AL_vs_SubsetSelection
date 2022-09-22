from .almethod import ALMethod
import torch
import torch.nn as nn
import numpy as np

class GTknownUncertainty(ALMethod):
    def __init__(self, dst_u_all, unlabeled_set, model, args, selection_method="LeastConfidence", balance=False, **kwargs):
        super().__init__(dst_u_all, unlabeled_set, model, args, **kwargs)

        selection_choices = ["LeastConfidence", "Entropy", "Margin", "Loss"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method
        self.balance = balance

    def run(self):
        scores, pred_y, true_y = self.rank_uncertainty()
        
        if self.selection_method == 'Loss':
            q_indices = np.argsort(-scores)[:self.args.n_query]
            #print(scores[q_indices])
        else:
            wrong_idx = np.where(true_y != pred_y)[0]
            correct_idx = np.where(true_y == pred_y)[0]
            assert len(wrong_idx)+len(correct_idx) == len(self.unlabeled_set)

            q_indices = wrong_idx[np.argsort(scores[wrong_idx])[:self.args.n_query]]
            print("# of wrongly predicted examples: ", len(wrong_idx))
            if len(q_indices) < self.args.n_query:
                correct_q_indices = correct_idx[np.argsort(scores[correct_idx])[:(self.args.n_query-len(q_indices))]]
                q_indices = np.append(q_indices, correct_q_indices)

        return q_indices, scores
    
    def rank_uncertainty(self, index=None):
        self.model.eval()
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(
                self.dst_unlabeled,
                batch_size=self.args.test_batch_size,
                shuffle=False,
                num_workers=self.args.workers)

            scores = np.array([])
            pred_y = np.array([])
            true_y = np.array([])
            batch_num = len(train_loader)

            for i, (input, label) in enumerate(train_loader):
                true_y = np.append(true_y, label)
                if i % self.args.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                if self.selection_method == "LeastConfidence":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    pred_y = np.append(pred_y, torch.argmax(preds, dim=1).cpu().numpy())
                    scores = np.append(scores, preds.max(axis=1).values.cpu().numpy())
                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                    pred_y = np.append(pred_y, torch.argmax(preds, dim=1).cpu().numpy())
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    pred_y = np.append(pred_y, preds_argmax.cpu().numpy())
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[
                        torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
                elif self.selection_method == 'Loss':
                    criterion = nn.CrossEntropyLoss(reduction='none').to(self.args.device)

                    output = self.model(input.to(self.args.device))
                    loss = criterion(output, label.to(self.args.device))

                    scores = np.append(scores, loss.cpu().numpy())

        return scores, pred_y, true_y

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_indices = [self.unlabeled_set[idx] for idx in selected_indices]

        return Q_indices, scores
