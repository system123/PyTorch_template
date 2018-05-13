from torchvision import transforms
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    def __init__(self, config):

        # Specify your transforms in the dataset
        # Transforms shouldn't be generic over all datasets
        self.transforms = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor()
            ])

        pass

    def __getitem__(self, index):
        x1 = x2 = y = 5
        return (x2, y)

    def __len__(self):
        count = 100
        return count

    def get_validation_split(self):
        return None
