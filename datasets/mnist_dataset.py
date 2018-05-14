from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, config):
        super(MNISTDataset, self).__init__()

        func = [transforms.ToTensor()]

        if config.augment:
            func.append( transforms.Normalize((0.1307,), (0.3081,)) )

        self.transforms = transforms.Compose(func)
        self.dataset = MNIST(download=config.download, root=config.data_path, transform=self.transforms, train=config.download)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def get_validation_split(self):
        raise NotImplementedError