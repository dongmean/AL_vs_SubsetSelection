from .almethod import ALMethod
import torch
import numpy as np

class Uncertainty(ALMethod):
    def __init__(self, dst_u_all, unlabeled_set, model, args, selection_method="LeastConfidence", balance=False, **kwargs):
        super().__init__(dst_u_all, unlabeled_set, model, args, **kwargs)

        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method
        self.balance = balance

    def run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            scores = []
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_unlabeled)[self.dst_unlabeled.targets == c]
                scores.append(self.rank_uncertainty(class_index))
                selection_result = np.append(selection_result, class_index[np.argsort(scores[-1])[
                                                               :round(len(class_index) * self.args.n_query)]])
        else:
            scores = self.rank_uncertainty()
            selection_result = np.argsort(scores)[:self.args.n_query]
        return selection_result, scores

    def rank_uncertainty(self, index=None):
        self.model.eval()
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(
                self.dst_unlabeled if index is None else torch.utils.data.Subset(self.dst_unlabeled, index),
                batch_size=self.args.test_batch_size,
                num_workers=self.args.workers)

            scores = np.array([])
            batch_num = len(train_loader)

            for i, (input, _) in enumerate(train_loader):
                if i % self.args.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                if self.selection_method == "LeastConfidence":
                    scores = np.append(scores, self.model(input.to(self.args.device)).max(axis=1).values.cpu().numpy())
                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[
                        torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
        return scores

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_indices = [self.unlabeled_set[idx] for idx in selected_indices]

        return Q_indices, scores
