from .almethod import ALMethod
import torch
import numpy as np
from .methods_utils import euclidean_dist

def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None,
                    print_freq: int = 200):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
    #print("matrix.shape: ", matrix.shape) # [1300, 512]

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    with torch.no_grad():
        np.random.seed(random_seed)
        select_result = np.zeros(sample_num, dtype=bool)
        # Randomly select one initial point.
        already_selected = [np.random.randint(0, sample_num)]
        budget -= 1
        select_result[already_selected] = True

        num_of_already_selected = np.sum(select_result)

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            if i % print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, budget))
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]

class kCenterGreedy(ALMethod):
    def __init__(self, dst_u_all, unlabeled_set, model, args, balance=False, metric="euclidean", **kwargs):
        super().__init__(dst_u_all, unlabeled_set, model, args, **kwargs)
        self.balance = balance
        if metric == "euclidean":
            self.metric = euclidean_dist
        self.random_seed = 0

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
            selection_result = np.argsort(scores)[::-1][:self.args.n_query]
        return selection_result, scores

    def construct_matrix(self):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                matrix = []

                data_loader = torch.utils.data.DataLoader(self.dst_unlabeled,
                                    batch_size=self.args.test_batch_size,
                                    num_workers=self.args.workers)

                for i, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix.append(self.model.embedding_recorder.embedding)

        self.model.no_grad = False
        return torch.cat(matrix, dim=0)

    def select(self, **kwargs):
        matrix = self.construct_matrix() # [#unlabeled, 512]
        #:TODO
        selected_indices = k_center_greedy(matrix, budget=self.args.n_query,
                                            metric=self.metric, device=self.args.device,
                                            random_seed=self.random_seed)

        scores = list(np.ones(len(selected_indices)))
        Q_indices = [self.unlabeled_set[idx] for idx in selected_indices]

        return Q_indices, scores
