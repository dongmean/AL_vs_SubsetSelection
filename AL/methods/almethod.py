import torch
class ALMethod(object):
    def __init__(self, dst_u_all, unlabeled_set, model, args, **kwargs):
        self.dst_u_all = dst_u_all
        self.unlabeled_set = unlabeled_set
        self.dst_unlabeled = torch.utils.data.Subset(dst_u_all, unlabeled_set)
        self.n_unlabeled = len(self.dst_unlabeled)
        self.num_classes = args.n_class
        self.model = model
        self.index = []
        self.args = args

    def select(self, **kwargs):
        return