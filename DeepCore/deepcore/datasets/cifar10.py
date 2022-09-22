from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch import tensor, long

class MyCIFAR10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = datasets.CIFAR10(file_path, train=train, download=download, transform=transform)
        self.targets = self.cifar10.targets
        self.classes = self.cifar10.classes
        self.clone_ver = 0      

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index + int(256*self.clone_ver)

    def __len__(self):
        return len(self.cifar10)

    def update_clone_ver(self, cur_ver):
        self.clone_ver = cur_ver

def CIFAR10(args):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    #dst_train = datasets.CIFAR10(args.data_path+'/cifar10', train=True, download=False, transform=train_transform)
    dst_train = MyCIFAR10(args.data_path+'/cifar10', train=True, download=False, transform=train_transform)
    dst_test = datasets.CIFAR10(args.data_path+'/cifar10', train=False, download=False, transform=test_transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test