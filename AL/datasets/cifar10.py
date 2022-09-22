from torchvision import datasets
from torchvision import transforms as T
from torch import tensor, long


def CIFAR10(args):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(args.data_path+'/cifar10', train=True, download=False, transform=train_transform)
    dst_unlabeled = datasets.CIFAR10(args.data_path+'/cifar10', train=True, download=False, transform=test_transform)
    dst_test = datasets.CIFAR10(args.data_path+'/cifar10', train=False, download=False, transform=test_transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_unlabeled, dst_test
